import ctypes
import bpy

MESHES = {}

def get_mesh_name(object):
    name = object.name_full
    if len(object.modifiers) == 0 and object.data:
        name = object.type + '_' + object.data.name_full
    return name

def get_mesh(object):
    key = get_mesh_name(object)
    if key not in MESHES.keys() or MESHES[key] is None:
        MESHES[key] = load_mesh(object, key)
    return MESHES[key]

def _find_armature(object):
    """Find the armature associated with an object via the 'malt_armature' custom property.
    Returns the armature object, or None if not set or not an armature."""
    arm_obj = object.get('malt_armature')
    if arm_obj is not None and hasattr(arm_obj, 'type') and arm_obj.type == 'ARMATURE':
        return arm_obj
    return None

def _needs_smooth_data(object):
    """Check if the object's compute tree has any smooth barrier nodes.
    Returns True if smooth-related SSBOs (adjacency, cotangent, edge_metadata)
    need to be built for this mesh."""
    mesh_data = object.original.data if object.original else object.data
    compute_tree_name = getattr(mesh_data, 'malt_compute_nodes', '') if mesh_data else ''
    if not compute_tree_name:
        return False
    compute_tree = bpy.data.node_groups.get(compute_tree_name)
    if compute_tree is None:
        return False
    dispatch_plan = compute_tree.get('dispatch_plan')
    if not dispatch_plan:
        return False
    return any(step.get('type') == 'smooth' for step in dispatch_plan)

def load_mesh(object, name):
    from . import CBlenderMalt
    import time as _time

    m = object.data
    if object.type != 'MESH' or object.mode == 'EDIT':
        m = object.to_mesh()

    if m is None or len(m.polygons) == 0:
        return None

    needs_smooth = _needs_smooth_data(object)

    _t0 = _time.perf_counter()
    m.calc_loop_triangles()

    def to_ptr(ptr, ctype):
        return (ctype * 1).from_address(ptr)

    def attribute_ptr(name, ctype):
        ptr = 0
        if name in m.attributes:
            ptr = m.attributes[name].data[0].as_pointer()
        return to_ptr(ptr, ctype)
    
    mesh_ptr = ctypes.c_void_p(m.as_pointer())

    loop_count = len(m.loops)
    loop_tri_count = len(m.loop_triangles)
    material_count = max(1, len(m.materials))

    positions = get_load_buffer('positions', ctypes.c_float, (loop_count * 3))
    indices = []
    indices_ptrs = (ctypes.c_void_p * material_count)()
    for i in range(material_count):
        indices.append(get_load_buffer('indices'+str(i), ctypes.c_uint32, (loop_tri_count * 3)))
        indices_ptrs[i] = ctypes.cast(indices[i].buffer(), ctypes.c_void_p)
    
    indices_lengths = (ctypes.c_uint32 * material_count)()

    CBlenderMalt.retrieve_mesh_data(
        attribute_ptr("position", ctypes.c_float),
        attribute_ptr(".corner_vert", ctypes.c_int), loop_count,
        to_ptr(m.loop_triangles[0].as_pointer(), ctypes.c_int),
        to_ptr(m.loop_triangle_polygons[0].as_pointer(), ctypes.c_int), loop_tri_count,
        attribute_ptr("material_index", ctypes.c_int),
        positions.buffer(), indices_ptrs, indices_lengths)

    for i in range(material_count):
        indices[i]._size = indices_lengths[i]

    _t1 = _time.perf_counter()
    # Rest positions and normals are uploaded as vec4 (4 floats per element, w=0) rather than
    # vec3 so the GPU buffer stride matches std430's vec4[] layout (16 bytes/element).
    # vec3[] in std430 also has 16-byte stride per the GLSL spec, but the CPU data is tightly
    # packed at 12 bytes/element, causing a read/write offset mismatch on the GPU.

    # Rest positions: loop-indexed positions padded to vec4 for SSBO binding 8.
    rest_positions = get_load_buffer('rest_positions', ctypes.c_float, loop_count * 4)
    pos_src = ctypes.cast(positions.buffer(), ctypes.POINTER(ctypes.c_float))
    pos_dst = ctypes.cast(rest_positions.buffer(), ctypes.POINTER(ctypes.c_float))
    CBlenderMalt.pad_vec3_to_vec4(pos_src, pos_dst, loop_count)

    # corner_vert: loop index → unique vertex index mapping (binding 10).
    # Lets compute shaders work in vertex space and scatter results back to corners.
    corner_vert = get_load_buffer('corner_vert', ctypes.c_int, loop_count)
    corner_vert_src = attribute_ptr(".corner_vert", ctypes.c_int)
    ctypes.memmove(corner_vert.buffer(), corner_vert_src, corner_vert.size_in_bytes())
    cv_ptr = ctypes.cast(corner_vert.buffer(), ctypes.POINTER(ctypes.c_int))

    vertex_count = len(m.vertices)

    _t2 = _time.perf_counter()

    # Smooth-related data: adjacency CSR, vert_corner CSR, cotangent weights, edge metadata.
    # Only built when the compute graph has a smooth barrier node (~15s for 500K verts).
    adjacency_data_buf = None
    vert_corner_data_buf = None
    cotangent_weights_buf = None
    edge_meta_buf = None

    if needs_smooth:
        # Adjacency CSR: vertex -> neighbor vertices (from mesh edges + quad diagonals)
        from collections import defaultdict
        neighbors = defaultdict(set)
        for edge in m.edges:
            v0, v1 = edge.vertices
            neighbors[v0].add(v1)
            neighbors[v1].add(v0)

        # Add quad 0-2 diagonals (matching Blender's loop_triangles triangulation).
        diagonal_edges = set()  # (min(v0,v2), max(v0,v2)) for tracking
        for poly in m.polygons:
            if poly.loop_total == 4:
                v0 = cv_ptr[poly.loop_start]
                v2 = cv_ptr[poly.loop_start + 2]
                if v2 not in neighbors[v0]:  # skip if already a real edge
                    neighbors[v0].add(v2)
                    neighbors[v2].add(v0)
                    diagonal_edges.add((min(v0, v2), max(v0, v2)))

        adjacency_offsets_list = [0]
        adjacency_indices_list = []
        edge_to_adj_idx = {}
        for v in range(vertex_count):
            adj = sorted(neighbors.get(v, []))
            for j, neighbor in enumerate(adj):
                edge_to_adj_idx[(v, neighbor)] = len(adjacency_indices_list) + j
            adjacency_indices_list.extend(adj)
            adjacency_offsets_list.append(len(adjacency_indices_list))

        # Track which adjacency indices are quad diagonals (for IS_DIAGONAL flag).
        diagonal_adj_indices = set()
        for (va, vb) in diagonal_edges:
            idx_ab = edge_to_adj_idx.get((va, vb))
            idx_ba = edge_to_adj_idx.get((vb, va))
            if idx_ab is not None: diagonal_adj_indices.add(idx_ab)
            if idx_ba is not None: diagonal_adj_indices.add(idx_ba)

        # Pack adjacency into single buffer: [offsets (vertex_count+1) | indices (2E)]
        adj_total = len(adjacency_offsets_list) + max(len(adjacency_indices_list), 1)
        adjacency_data_buf = get_load_buffer('adjacency_data', ctypes.c_int, adj_total)
        adj_ptr = ctypes.cast(adjacency_data_buf.buffer(), ctypes.POINTER(ctypes.c_int))
        for i, val in enumerate(adjacency_offsets_list):
            adj_ptr[i] = val
        base = len(adjacency_offsets_list)
        for i, val in enumerate(adjacency_indices_list):
            adj_ptr[base + i] = val

        # Vertex-to-corner CSR: vertex -> loop/corner indices (inverse of corner_vert)
        vert_corners = defaultdict(list)
        for loop_idx in range(loop_count):
            vert_corners[cv_ptr[loop_idx]].append(loop_idx)

        vert_corner_offsets_list = [0]
        vert_corner_indices_list = []
        for v in range(vertex_count):
            corners = sorted(vert_corners.get(v, []))
            vert_corner_indices_list.extend(corners)
            vert_corner_offsets_list.append(len(vert_corner_indices_list))

        # Pack vert-corner into single buffer: [offsets (vertex_count+1) | indices (L)]
        vc_total = len(vert_corner_offsets_list) + max(len(vert_corner_indices_list), 1)
        vert_corner_data_buf = get_load_buffer('vert_corner_data', ctypes.c_int, vc_total)
        vc_ptr2 = ctypes.cast(vert_corner_data_buf.buffer(), ctypes.POINTER(ctypes.c_int))
        for i, val in enumerate(vert_corner_offsets_list):
            vc_ptr2[i] = val
        base = len(vert_corner_offsets_list)
        for i, val in enumerate(vert_corner_indices_list):
            vc_ptr2[base + i] = val

    _t3 = _time.perf_counter()

    if needs_smooth:
        # Cotangent weights: one float per entry in adjacency_indices_list.
        # For each edge (v, neighbor), the weight is (cot(α) + cot(β)) / 2 where α and β
        # are the angles opposite the edge in the two adjacent triangles.  Boundary edges
        # get a single cot(α)/2.  Negative cotangents (obtuse angles) are clamped to 0.
        import math
        pos_ptr = ctypes.cast(positions.buffer(), ctypes.POINTER(ctypes.c_float))
        # Build one representative corner per vertex for position lookup.
        vert_rep_corner = [0] * vertex_count
        for v in range(vertex_count):
            corners = vert_corners.get(v)
            if corners:
                vert_rep_corner[v] = corners[0]

        def vert_pos(v):
            c = vert_rep_corner[v]
            return (pos_ptr[c*3], pos_ptr[c*3+1], pos_ptr[c*3+2])

        cotangent_weights_list = [0.0] * max(len(adjacency_indices_list), 1)
        for tri in m.loop_triangles:
            l0, l1, l2 = tri.loops
            v0, v1, v2 = cv_ptr[l0], cv_ptr[l1], cv_ptr[l2]
            p0, p1, p2 = vert_pos(v0), vert_pos(v1), vert_pos(v2)
            # For each edge, compute cot of opposite angle, clamped to >= 0.
            # Edge (va, vb): opposite vertex vc at position pc.
            for (va, vb, pc) in ((v0, v1, p2), (v1, v2, p0), (v0, v2, p1)):
                # pa, pb from the edge endpoints
                pa, pb = vert_pos(va), vert_pos(vb)
                e1 = (pa[0]-pc[0], pa[1]-pc[1], pa[2]-pc[2])
                e2 = (pb[0]-pc[0], pb[1]-pc[1], pb[2]-pc[2])
                dot = e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2]
                cx = e1[1]*e2[2] - e1[2]*e2[1]
                cy = e1[2]*e2[0] - e1[0]*e2[2]
                cz = e1[0]*e2[1] - e1[1]*e2[0]
                cross_len = math.sqrt(cx*cx + cy*cy + cz*cz)
                if cross_len < 1e-10:
                    continue  # degenerate triangle
                cot_val = max(dot / cross_len, 0.0) * 0.5
                idx_ab = edge_to_adj_idx.get((va, vb))
                if idx_ab is not None:
                    cotangent_weights_list[idx_ab] += cot_val
                idx_ba = edge_to_adj_idx.get((vb, va))
                if idx_ba is not None:
                    cotangent_weights_list[idx_ba] += cot_val

        cot_count = max(len(adjacency_indices_list), 1)
        cotangent_weights_buf = get_load_buffer('cotangent_weights', ctypes.c_float, cot_count)
        cot_ptr = ctypes.cast(cotangent_weights_buf.buffer(), ctypes.POINTER(ctypes.c_float))
        for i, val in enumerate(cotangent_weights_list):
            cot_ptr[i] = val

    _t4 = _time.perf_counter()
    # Normals: loop-indexed corner normals padded to vec4 (4 floats, w=0).
    # This buffer serves as the Normal VBO (read by the vertex shader with stride=16 to
    # skip the w padding) and as the writable Normal SSBO (binding 11).  The compute
    # shader writes its output normal here each frame.
    # Using GL_DYNAMIC_DRAW so the compute shader can write updated normals back.
    normals = get_load_buffer('normals', ctypes.c_float, loop_count * 4)
    norm_src = (ctypes.c_float * (loop_count * 3)).from_address(m.corner_normals[0].as_pointer())
    norm_dst = ctypes.cast(normals.buffer(), ctypes.POINTER(ctypes.c_float))
    CBlenderMalt.pad_vec3_to_vec4(norm_src, norm_dst, loop_count)

    # Rest normals: read-only copy of the original Blender normals (binding 12).
    # main() in NPR_ComputeShader.glsl initialises the 'normal' inout parameter from
    # this buffer, so the Compute Input node always delivers the original mesh normal
    # regardless of what the compute shader wrote to normals[] on the previous frame.
    rest_normals = get_load_buffer('rest_normals', ctypes.c_float, loop_count * 4)
    ctypes.memmove(rest_normals.buffer(), normals.buffer(), normals.size_in_bytes())

    uvs_list = []
    tangents_buffer = None
    for i, uv_layer in enumerate(m.uv_layers):
        if i >= 4: break
        uvs = (ctypes.c_float * (loop_count * 2)).from_address(uv_layer.data[0].as_pointer())
        uv_buffer = get_load_buffer('uv'+str(i), ctypes.c_float, loop_count * 2)
        ctypes.memmove(uv_buffer.buffer(), uvs, uv_buffer.size_in_bytes())
        uvs_list.append(uv_buffer)
        if i == 0 and object.original.data.malt_parameters.bools['precomputed_tangents'].boolean:
            tangents_buffer = get_load_buffer('tangents'+str(i), ctypes.c_float, (loop_count * 4))
            CBlenderMalt.mesh_tangents(
                to_ptr(m.loop_triangles[0].as_pointer(), ctypes.c_int),
                loop_tri_count * 3,
                positions.buffer(),
                normals.buffer(),
                uv_buffer.buffer(),
                tangents_buffer.buffer())
    
    colors_list = [None]*4
    if object.type == 'MESH':
        for i in range(4):
            override = getattr(object.original.data, f'malt_vertex_color_override_{i}')
            attribute = m.color_attributes.get(override)
            #if attribute is None and i < len(m.color_attributes):
            #    attribute = m.color_attributes[i]
            if attribute and attribute.domain == 'CORNER':
                type = None
                if attribute.data_type == 'BYTE_COLOR':
                    type = ctypes.c_uint8
                elif attribute.data_type == 'FLOAT_COLOR':
                    type = ctypes.c_float
                else:
                    continue
                color = (type * (loop_count * 4)).from_address(attribute.data[0].as_pointer())
                color_buffer = get_load_buffer('colors'+str(i), type, loop_count*4)
                ctypes.memmove(color_buffer.buffer(), color, color_buffer.size_in_bytes())
                colors_list[i] = color_buffer

    ssbo_colors_list = [None]*8
    if object.type == 'MESH':
        for i in range(8):
            attr_name = f'malt_ssbo_{i}'
            attribute = m.attributes.get(attr_name)
            if attribute and attribute.domain == 'FACE':
                if attribute.data_type == 'FLOAT_COLOR':
                    face_data = (ctypes.c_float * (len(m.polygons) * 4)).from_address(attribute.data[0].as_pointer())
                    expanded = get_load_buffer('ssbo_color'+str(i), ctypes.c_float, loop_count * 4)
                    expanded_ptr = ctypes.cast(expanded.buffer(), ctypes.POINTER(ctypes.c_float))
                    for poly in m.polygons:
                        fi = poly.index
                        r, g, b, a = face_data[fi*4], face_data[fi*4+1], face_data[fi*4+2], face_data[fi*4+3]
                        for j in range(poly.loop_start, poly.loop_start + poly.loop_total):
                            expanded_ptr[j*4]   = r
                            expanded_ptr[j*4+1] = g
                            expanded_ptr[j*4+2] = b
                            expanded_ptr[j*4+3] = a
                    ssbo_colors_list[i] = expanded
                elif attribute.data_type == 'BYTE_COLOR':
                    face_data = (ctypes.c_uint8 * (len(m.polygons) * 4)).from_address(attribute.data[0].as_pointer())
                    expanded = get_load_buffer('ssbo_color'+str(i), ctypes.c_float, loop_count * 4)
                    expanded_ptr = ctypes.cast(expanded.buffer(), ctypes.POINTER(ctypes.c_float))
                    for poly in m.polygons:
                        fi = poly.index
                        r = face_data[fi*4]   / 255.0
                        g = face_data[fi*4+1] / 255.0
                        b = face_data[fi*4+2] / 255.0
                        a = face_data[fi*4+3] / 255.0
                        for j in range(poly.loop_start, poly.loop_start + poly.loop_total):
                            expanded_ptr[j*4]   = r
                            expanded_ptr[j*4+1] = g
                            expanded_ptr[j*4+2] = b
                            expanded_ptr[j*4+3] = a
                    ssbo_colors_list[i] = expanded
            elif attribute and attribute.domain == 'CORNER':
                if attribute.data_type == 'FLOAT_COLOR':
                    corner_data = (ctypes.c_float * (loop_count * 4)).from_address(attribute.data[0].as_pointer())
                    buf = get_load_buffer('ssbo_color'+str(i), ctypes.c_float, loop_count * 4)
                    ctypes.memmove(buf.buffer(), corner_data, buf.size_in_bytes())
                    ssbo_colors_list[i] = buf
                elif attribute.data_type == 'BYTE_COLOR':
                    corner_data = (ctypes.c_uint8 * (loop_count * 4)).from_address(attribute.data[0].as_pointer())
                    buf = get_load_buffer('ssbo_color'+str(i), ctypes.c_float, loop_count * 4)
                    buf_ptr = ctypes.cast(buf.buffer(), ctypes.POINTER(ctypes.c_float))
                    for j in range(loop_count * 4):
                        buf_ptr[j] = corner_data[j] / 255.0
                    ssbo_colors_list[i] = buf
            elif attribute and attribute.domain == 'POINT':
                # Expand vertex-domain data to face-corner domain using corner_vert mapping.
                cv_ptr = m.attributes[".corner_vert"].data[0].as_pointer() if ".corner_vert" in m.attributes else 0
                if not cv_ptr:
                    continue
                cv_src = (ctypes.c_int * loop_count).from_address(cv_ptr)
                if attribute.data_type == 'FLOAT_COLOR':
                    vtx_data = (ctypes.c_float * (vertex_count * 4)).from_address(attribute.data[0].as_pointer())
                    expanded = get_load_buffer('ssbo_color'+str(i), ctypes.c_float, loop_count * 4)
                    expanded_ptr = ctypes.cast(expanded.buffer(), ctypes.POINTER(ctypes.c_float))
                    for j in range(loop_count):
                        v = cv_src[j]
                        expanded_ptr[j*4]   = vtx_data[v*4]
                        expanded_ptr[j*4+1] = vtx_data[v*4+1]
                        expanded_ptr[j*4+2] = vtx_data[v*4+2]
                        expanded_ptr[j*4+3] = vtx_data[v*4+3]
                    ssbo_colors_list[i] = expanded
                elif attribute.data_type == 'BYTE_COLOR':
                    vtx_data = (ctypes.c_uint8 * (vertex_count * 4)).from_address(attribute.data[0].as_pointer())
                    expanded = get_load_buffer('ssbo_color'+str(i), ctypes.c_float, loop_count * 4)
                    expanded_ptr = ctypes.cast(expanded.buffer(), ctypes.POINTER(ctypes.c_float))
                    for j in range(loop_count):
                        v = cv_src[j]
                        expanded_ptr[j*4]   = vtx_data[v*4]   / 255.0
                        expanded_ptr[j*4+1] = vtx_data[v*4+1] / 255.0
                        expanded_ptr[j*4+2] = vtx_data[v*4+2] / 255.0
                        expanded_ptr[j*4+3] = vtx_data[v*4+3] / 255.0
                    ssbo_colors_list[i] = expanded

    _t5 = _time.perf_counter()

    if needs_smooth:
        # Per-edge metadata for smooth kernel.
        # For each directed adjacency entry (va→vb), stores 3 ints:
        #   [owning_fan_id, gather_corner_index, flags]
        # flags bit 0 = IS_DIAGONAL (quad 0-2 diagonal, not a real mesh edge).
        # When fan groups are present (ssbo_data_2.x), owning_fan and gather_corner
        # are derived from fan group IDs + mesh topology.  Otherwise they default to -1.
        from collections import defaultdict
        adj_count = max(len(adjacency_indices_list), 1)
        edge_meta_count = adj_count * 3
        edge_meta_list = [0] * edge_meta_count
        # Default: owning_fan=-1, gather_corner=-1, flags=0
        for i in range(adj_count):
            edge_meta_list[i * 3]     = -1
            edge_meta_list[i * 3 + 1] = -1
            edge_meta_list[i * 3 + 2] = 0

        # Set IS_DIAGONAL flag for quad diagonal edges.
        for adj_idx in diagonal_adj_indices:
            edge_meta_list[adj_idx * 3 + 2] = 1

        # If fan groups present, compute owning_fan and gather_corner per edge.
        if ssbo_colors_list[2] is not None and len(adjacency_indices_list) > 0:
            fan_ptr = ctypes.cast(ssbo_colors_list[2].buffer(), ctypes.POINTER(ctypes.c_float))

            # Build edge→face mapping: undirected edge → list of (va, loop_at_va, vb, loop_at_vb)
            edge_face_loops = defaultdict(list)
            for poly in m.polygons:
                nverts = poly.loop_total
                for i in range(nverts):
                    loop_i = poly.loop_start + i
                    loop_j = poly.loop_start + ((i + 1) % nverts)
                    vi = cv_ptr[loop_i]
                    vj = cv_ptr[loop_j]
                    key = (min(vi, vj), max(vi, vj))
                    edge_face_loops[key].append((vi, loop_i, vj, loop_j))
                # Add quad diagonal to edge_face_loops for fan group lookup.
                if nverts == 4:
                    l0 = poly.loop_start
                    l2 = poly.loop_start + 2
                    vi = cv_ptr[l0]
                    vj = cv_ptr[l2]
                    key = (min(vi, vj), max(vi, vj))
                    edge_face_loops[key].append((vi, l0, vj, l2))

            for (va, vb), adj_idx in edge_to_adj_idx.items():
                key = (min(va, vb), max(va, vb))
                faces = edge_face_loops.get(key, [])
                if not faces:
                    continue  # isolated edge (no faces) — leave as -1, -1
                fan_groups_at_va = set()
                gather_corner = -1
                for (v0, l0, v1, l1) in faces:
                    if v0 == va:
                        loop_va, loop_vb = l0, l1
                    else:
                        loop_va, loop_vb = l1, l0
                    fan_groups_at_va.add(int(fan_ptr[loop_va * 4]))  # .x channel
                    gather_corner = loop_vb
                if len(fan_groups_at_va) == 1:
                    owning_fan = fan_groups_at_va.pop()
                else:
                    owning_fan = -1
                    gather_corner = -1
                edge_meta_list[adj_idx * 3]     = owning_fan
                edge_meta_list[adj_idx * 3 + 1] = gather_corner
                # flags (adj_idx * 3 + 2) already set above

        edge_meta_buf = get_load_buffer('edge_metadata', ctypes.c_int, edge_meta_count)
        em_ptr = ctypes.cast(edge_meta_buf.buffer(), ctypes.POINTER(ctypes.c_int))
        for i, val in enumerate(edge_meta_list):
            em_ptr[i] = val

    _t6 = _time.perf_counter()
    # Bone indices and weights for GPU skinning (per-vertex, max 4 influences).
    # Extracted from vertex groups that match armature bone names.
    bone_indices_buf = None
    bone_weights_buf = None
    bone_count = 0
    armature_obj = _find_armature(object)
    if armature_obj is not None and armature_obj.type == 'ARMATURE':
        armature_data = armature_obj.data
        bone_count = len(armature_data.bones)

        # Map bone names to indices (enumeration order of armature.bones).
        bone_name_to_index = {}
        for i, bone in enumerate(armature_data.bones):
            bone_name_to_index[bone.name] = i

        # Pre-map vertex group index → bone index (avoids per-vertex name lookup).
        vg_to_bone = {}
        for vg in object.vertex_groups:
            if vg.name in bone_name_to_index:
                vg_to_bone[vg.index] = bone_name_to_index[vg.name]

        # Pre-extract all bone influences into flat arrays (CSR format).
        # Python API iteration over v.groups is the irreducible cost; the
        # sort/top-4/normalize/pack step is offloaded to C for speed.
        influence_bones = []
        influence_weights = []
        offsets = [0]
        for v in m.vertices:
            for g in v.groups:
                bone_idx = vg_to_bone.get(g.group)
                if bone_idx is not None and g.weight > 0.0:
                    influence_bones.append(bone_idx)
                    influence_weights.append(g.weight)
            offsets.append(len(influence_bones))

        total_influences = len(influence_bones)

        # Build ctypes arrays for the C function.
        c_offsets = (ctypes.c_int * (vertex_count + 1))(*offsets)
        c_bones = (ctypes.c_int * max(total_influences, 1))(*influence_bones)
        c_weights = (ctypes.c_float * max(total_influences, 1))(*influence_weights)

        # Allocate output buffers.
        bone_indices_buf = get_load_buffer('bone_indices', ctypes.c_int, vertex_count * 4)
        bone_weights_buf = get_load_buffer('bone_weights', ctypes.c_float, vertex_count * 4)

        # Pack bone data in C (sort, top-4, normalize, write 4 slots per vertex).
        CBlenderMalt.pack_bone_data(
            c_offsets, c_bones, c_weights,
            vertex_count,
            bone_indices_buf.buffer(), bone_weights_buf.buffer())

    _t7 = _time.perf_counter()

    _smooth_label = "built" if needs_smooth else "SKIPPED"
    print(f"[MaltMeshes] load_mesh timing for '{name}' "
          f"(V={vertex_count}, L={loop_count}, smooth={_smooth_label}):")
    print(f"  retrieve_mesh_data + setup : {_t1 - _t0:.3f}s")
    print(f"  rest_pos + corner_vert     : {_t2 - _t1:.3f}s")
    print(f"  adjacency CSR + vert_corner: {_t3 - _t2:.3f}s")
    print(f"  cotangent weights          : {_t4 - _t3:.3f}s")
    print(f"  normals + UVs + colors     : {_t5 - _t4:.3f}s")
    print(f"  edge metadata              : {_t6 - _t5:.3f}s")
    print(f"  bone data                  : {_t7 - _t6:.3f}s")
    print(f"  TOTAL                      : {_t7 - _t0:.3f}s")

    mesh_data = {
        'positions': positions,
        'indices': indices,
        'normals': normals,
        'uvs': uvs_list,
        'tangents': tangents_buffer,
        'colors': colors_list,
        'ssbo_colors': ssbo_colors_list,
        'vertex_count': vertex_count,
        'loop_count': loop_count,
        'rest_positions': rest_positions,
        'rest_normals': rest_normals,
        'corner_vert': corner_vert,
        'adjacency_data': adjacency_data_buf,
        'vert_corner_data': vert_corner_data_buf,
        'cotangent_weights': cotangent_weights_buf,
        'edge_metadata': edge_meta_buf,
        'bone_indices': bone_indices_buf,
        'bone_weights': bone_weights_buf,
        'bone_count': bone_count,
    }

    from . import MaltPipeline
    MaltPipeline.get_bridge().load_mesh(name, mesh_data)

    from Bridge.Proxys import MeshProxy
    return [MeshProxy(name, i) for i in range(material_count)]

def get_load_buffer(name, ctype, size):
    from . import MaltPipeline
    return MaltPipeline.get_bridge().get_shared_buffer(ctype, size)

def unload_mesh(object):
    MESHES[get_mesh_name(object)] = None

def reset_meshes():
    global MESHES
    MESHES = {}

def draw_vertex_color_overrides(self, context):
    if context.scene.render.engine != 'MALT':
        return
    mesh = context.object.data
    self.layout.use_property_split = True
    self.layout.label(text='Malt Vertex Colors')
    def draw_color_override(key):
        value = getattr(mesh, key)
        layout = self.layout
        if value in mesh.color_attributes and mesh.color_attributes[value].domain != 'CORNER':
            layout = self.layout.box()
            layout.label(text='Only Face Corner attributes are supported', icon='ERROR')
        layout.prop_search(mesh, key, mesh, 'color_attributes')    

    draw_color_override('malt_vertex_color_override_0')
    draw_color_override('malt_vertex_color_override_1')
    draw_color_override('malt_vertex_color_override_2')
    draw_color_override('malt_vertex_color_override_3')


def register():
    bpy.types.Mesh.malt_compute_nodes = bpy.props.StringProperty(
        name='Compute Node Tree',
        description='Malt Compute node tree to run on this mesh each frame',
        options={'LIBRARY_EDITABLE'}, override={'LIBRARY_OVERRIDABLE'})

    bpy.types.Mesh.malt_vertex_color_override_0 = bpy.props.StringProperty(name='0',
        options={'LIBRARY_EDITABLE'}, override={'LIBRARY_OVERRIDABLE'})
    bpy.types.Mesh.malt_vertex_color_override_1 = bpy.props.StringProperty(name='1',
        options={'LIBRARY_EDITABLE'}, override={'LIBRARY_OVERRIDABLE'})
    bpy.types.Mesh.malt_vertex_color_override_2 = bpy.props.StringProperty(name='2',
        options={'LIBRARY_EDITABLE'}, override={'LIBRARY_OVERRIDABLE'})
    bpy.types.Mesh.malt_vertex_color_override_3 = bpy.props.StringProperty(name='3',
        options={'LIBRARY_EDITABLE'}, override={'LIBRARY_OVERRIDABLE'})
    bpy.types.DATA_PT_vertex_colors.append(draw_vertex_color_overrides)


def unregister():
    del bpy.types.Mesh.malt_compute_nodes

    bpy.types.DATA_PT_vertex_colors.remove(draw_vertex_color_overrides)
    del bpy.types.Mesh.malt_vertex_color_override_0
    del bpy.types.Mesh.malt_vertex_color_override_1
    del bpy.types.Mesh.malt_vertex_color_override_2
    del bpy.types.Mesh.malt_vertex_color_override_3
