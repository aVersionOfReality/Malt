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

def _get_compute_requirements(object):
    """Return compute requirements derived from mesh-level boolean properties.
    Returns a dict with flags:
      'smooth_data' — adjacency CSR, vert_corner CSR, fan_groups, cotangent weights, edge metadata
      'bone_data'   — per-vertex bone indices and weights
      'curvature_data' — (implied by smooth_data, but tracked separately)
    Returns an empty dict if no compute operations are enabled."""
    mesh_data = object.original.data if object.original else object.data
    if mesh_data is None:
        return {}
    reqs = {}
    has_smooth = getattr(mesh_data, 'malt_compute_smooth_normals', False)
    has_curvature = getattr(mesh_data, 'malt_compute_curvature', False)
    has_skin = getattr(mesh_data, 'malt_compute_skin', False)
    if has_smooth or has_curvature:
        reqs['smooth_data'] = 1
    if has_curvature:
        reqs['curvature_data'] = 1
    if has_skin:
        reqs['bone_data'] = 1
    return reqs

def load_mesh(object, name):
    from . import CBlenderMalt
    import time as _time

    m = object.data
    if object.type != 'MESH' or object.mode == 'EDIT':
        m = object.to_mesh()

    if m is None or len(m.polygons) == 0:
        return None

    compute_reqs = _get_compute_requirements(object)
    needs_curvature = bool(compute_reqs.get('curvature_data', 0))
    # Curvature node needs adjacency CSR + cotangent weights, so imply smooth_data.
    needs_smooth = bool(compute_reqs.get('smooth_data', 0)) or needs_curvature
    needs_bones = bool(compute_reqs.get('bone_data', 0))

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
    # Only built when the compute graph has a smooth barrier node.
    # All four structures are built in a single C call (build_smooth_data).
    adjacency_data_buf = None
    vert_corner_data_buf = None
    fan_groups_buf = None
    cotangent_weights_buf = None
    edge_meta_buf = None

    _t3 = _time.perf_counter()
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
    # Used by the smooth kernel for mix_factor blend and by _reset_buffers()
    # to restore normals[] to rest pose at the start of each compute frame.
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
        # Build all smooth data structures in a single C call.
        # Pre-extract Blender data into flat arrays using fast bulk APIs.
        _ec = len(m.edges)
        _pc = len(m.polygons)
        _tc = len(m.loop_triangles)

        # Edge vertex pairs: flat [v0, v1, v0, v1, ...]
        c_edges = (ctypes.c_int * (_ec * 2))()
        m.edges.foreach_get('vertices', c_edges)

        # Polygon loop_start and loop_total arrays.
        c_poly_ls = (ctypes.c_int * _pc)()
        c_poly_lt = (ctypes.c_int * _pc)()
        m.polygons.foreach_get('loop_start', c_poly_ls)
        m.polygons.foreach_get('loop_total', c_poly_lt)

        # Triangle loop indices: flat [l0, l1, l2, l0, l1, l2, ...]
        c_tri_loops = (ctypes.c_int * (_tc * 3))()
        m.loop_triangles.foreach_get('loops', c_tri_loops)

        # Worst-case adjacency index count: 2 * edges + 2 * quads (each quad adds a diagonal).
        # Each edge contributes 2 directed entries; each quad diagonal adds 2 more.
        max_adj_indices = 2 * _ec + 2 * _pc  # generous upper bound
        max_adj_buf = (vertex_count + 1) + max_adj_indices

        # Allocate worst-case output buffers (C function writes actual sizes).
        out_adj = (ctypes.c_int * max_adj_buf)()
        out_adj_total = ctypes.c_int(0)
        out_vc = (ctypes.c_int * ((vertex_count + 1) + loop_count))()
        out_vc_total = ctypes.c_int(0)
        out_fg = (ctypes.c_int * loop_count)()
        out_cot = (ctypes.c_float * max(max_adj_indices, 1))()
        out_adj_idx_count = ctypes.c_int(0)
        out_emeta = (ctypes.c_int * (max_adj_indices * 3))()

        # Pass normals (vec4-padded) for fan group computation from normal similarity.
        normals_ptr = ctypes.cast(normals.buffer(), ctypes.POINTER(ctypes.c_float))

        CBlenderMalt.build_smooth_data(
            c_edges, _ec,
            cv_ptr, loop_count,
            c_poly_ls, c_poly_lt, _pc,
            ctypes.cast(positions.buffer(), ctypes.POINTER(ctypes.c_float)),
            c_tri_loops, _tc,
            normals_ptr,
            vertex_count,
            out_adj, ctypes.byref(out_adj_total),
            out_vc, ctypes.byref(out_vc_total),
            out_fg,
            out_cot, ctypes.byref(out_adj_idx_count),
            out_emeta)

        adj_total_val = out_adj_total.value
        vc_total_val = out_vc_total.value
        adj_idx_count_val = out_adj_idx_count.value

        # Copy C results into shared load buffers.
        adjacency_data_buf = get_load_buffer('adjacency_data', ctypes.c_int, adj_total_val)
        ctypes.memmove(adjacency_data_buf.buffer(), out_adj, adj_total_val * ctypes.sizeof(ctypes.c_int))

        vert_corner_data_buf = get_load_buffer('vert_corner_data', ctypes.c_int, vc_total_val)
        ctypes.memmove(vert_corner_data_buf.buffer(), out_vc, vc_total_val * ctypes.sizeof(ctypes.c_int))

        fan_groups_buf = get_load_buffer('fan_groups', ctypes.c_int, loop_count)
        ctypes.memmove(fan_groups_buf.buffer(), out_fg, loop_count * ctypes.sizeof(ctypes.c_int))

        cot_count = max(adj_idx_count_val, 1)
        cotangent_weights_buf = get_load_buffer('cotangent_weights', ctypes.c_float, cot_count)
        ctypes.memmove(cotangent_weights_buf.buffer(), out_cot, cot_count * ctypes.sizeof(ctypes.c_float))

        edge_meta_count = adj_idx_count_val * 3
        edge_meta_buf = get_load_buffer('edge_metadata', ctypes.c_int, max(edge_meta_count, 1))
        ctypes.memmove(edge_meta_buf.buffer(), out_emeta, max(edge_meta_count, 1) * ctypes.sizeof(ctypes.c_int))

    # Laplacian smooth per-corner weight attributes (face corner color, vec4).
    # Two attributes, matching the two smooth_weights SSBOs:
    #   avr_malt_laplacian1: R=mix_factor, G=momentum, B=application, A=unused
    #   avr_malt_laplacian2: R=contribution, G=own_normal, B=unused, A=unused
    # Accepts FLOAT_COLOR (vec4) or BYTE_COLOR (uint8 vec4).
    # The attributes may be created by Geometry Nodes (only on evaluated mesh).
    laplacian1_buf = None
    laplacian2_buf = None
    if needs_smooth:
        for attr_name, buf_name in [('avr_malt_laplacian1', 'laplacian1'),
                                     ('avr_malt_laplacian2', 'laplacian2')]:
            attr = m.attributes.get(attr_name)
            if attr and attr.domain == 'CORNER':
                if attr.data_type == 'FLOAT_COLOR':
                    src = (ctypes.c_float * (loop_count * 4)).from_address(attr.data[0].as_pointer())
                    buf = get_load_buffer(buf_name, ctypes.c_float, loop_count * 4)
                    ctypes.memmove(buf.buffer(), src, buf.size_in_bytes())
                elif attr.data_type == 'BYTE_COLOR':
                    src = (ctypes.c_uint8 * (loop_count * 4)).from_address(attr.data[0].as_pointer())
                    buf = get_load_buffer(buf_name, ctypes.c_float, loop_count * 4)
                    dst = ctypes.cast(buf.buffer(), ctypes.POINTER(ctypes.c_float))
                    for j in range(loop_count * 4):
                        dst[j] = src[j] / 255.0
                else:
                    buf = None
                if buf is not None:
                    if buf_name == 'laplacian1':
                        laplacian1_buf = buf
                    else:
                        laplacian2_buf = buf

    _t6 = _time.perf_counter()
    # Bone indices and weights for GPU skinning (per-vertex, max 4 influences).
    # Only extracted when the compute graph has a skin barrier connected to the output.
    bone_indices_buf = None
    bone_weights_buf = None
    bone_count = 0
    armature_obj = _find_armature(object) if needs_bones else None
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

    if bpy.context.preferences.addons['BlenderMalt'].preferences.show_timings:
        _smooth_label = "C ext" if needs_smooth else "SKIPPED"
        _bone_label = f"{_t7 - _t6:.3f}s" if needs_bones else "SKIPPED"
        print(f"[MaltMeshes] load_mesh timing for '{name}' "
              f"(V={vertex_count}, L={loop_count}, smooth={_smooth_label}, bones={_bone_label}):")
        print(f"  retrieve_mesh_data + setup : {_t1 - _t0:.3f}s")
        print(f"  rest_pos + corner_vert     : {_t2 - _t1:.3f}s")
        print(f"  (smooth data placeholder)  : {_t3 - _t2:.3f}s")
        print(f"  (cotangent placeholder)    : {_t4 - _t3:.3f}s")
        print(f"  normals + UVs + colors     : {_t5 - _t4:.3f}s")
        print(f"  smooth C ext (all 4)       : {_t6 - _t5:.3f}s")
        print(f"  bone data                  : {_bone_label}")
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
        'fan_groups': fan_groups_buf,
        'cotangent_weights': cotangent_weights_buf,
        'edge_metadata': edge_meta_buf,
        'laplacian1': laplacian1_buf,
        'laplacian2': laplacian2_buf,
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
    # Compute pipeline enable/disable toggles.
    bpy.types.Mesh.malt_compute_skin = bpy.props.BoolProperty(
        name='GPU Skinning', default=False,
        description='Enable GPU Linear Blend Skinning')
    bpy.types.Mesh.malt_compute_curvature = bpy.props.BoolProperty(
        name='Curvature', default=False,
        description='Compute per-corner curvature for tessellation and shading')
    bpy.types.Mesh.malt_compute_smooth_normals = bpy.props.BoolProperty(
        name='Laplacian Smooth Normals', default=False,
        description='Laplacian smooth normals for stylized shading')

    # Smooth normals parameters.
    bpy.types.Mesh.malt_smooth_iterations = bpy.props.IntProperty(
        name='Iterations', default=10, min=1, max=500,
        description='Number of Laplacian smoothing iterations')
    bpy.types.Mesh.malt_smooth_cotangent_factor = bpy.props.FloatProperty(
        name='Cotangent Factor', default=1.0, soft_min=0.0, soft_max=1.0,
        description='Blend between uniform (0) and cotangent (1) weights')
    bpy.types.Mesh.malt_smooth_quad_mode = bpy.props.IntProperty(
        name='Quad Mode', default=2, min=0, max=2,
        description='0 = standard, 1 = quad virtual triangulation, 2 = quad full')
    bpy.types.Mesh.malt_smooth_mix_factor = bpy.props.FloatProperty(
        name='Smooth Mix Factor', default=1.0, soft_min=0.0, soft_max=1.0,
        description='Blend between original (0) and smoothed (1) normals')
    bpy.types.Mesh.malt_smooth_momentum_factor = bpy.props.FloatProperty(
        name='Momentum Factor', default=0.25, soft_min=0.0, soft_max=0.5,
        description='Velocity-based smoothing momentum per iteration')
    bpy.types.Mesh.malt_smooth_application_strength = bpy.props.FloatProperty(
        name='Application Strength', default=0.5, soft_min=0.0, soft_max=1.0,
        description='How much smoothing is applied per iteration')
    bpy.types.Mesh.malt_smooth_contribution_strength = bpy.props.FloatProperty(
        name='Contribution Strength', default=1.0, soft_min=0.0, soft_max=1.0,
        description='Weight of neighbor contributions')
    bpy.types.Mesh.malt_smooth_own_normal_strength = bpy.props.FloatProperty(
        name='Own Normal Strength', default=1.0, soft_min=0.0, soft_max=1.0,
        description='Weight of own normal in the smoothing blend')
    bpy.types.Mesh.malt_smooth_groups_enabled = bpy.props.BoolProperty(
        name='Fan Groups', default=True,
        description='Respect smooth/sharp edge groups during smoothing')

    # Per-parameter attribute toggles.
    # avr_malt_laplacian1: R=mix_factor, G=momentum, B=application
    bpy.types.Mesh.malt_smooth_use_attr_mix_factor = bpy.props.BoolProperty(
        name='Use Attribute', default=False,
        description='Use avr_malt_laplacian1.R instead of uniform value')
    bpy.types.Mesh.malt_smooth_use_attr_momentum = bpy.props.BoolProperty(
        name='Use Attribute', default=False,
        description='Use avr_malt_laplacian1.G instead of uniform value')
    bpy.types.Mesh.malt_smooth_use_attr_application = bpy.props.BoolProperty(
        name='Use Attribute', default=False,
        description='Use avr_malt_laplacian1.B instead of uniform value')
    # avr_malt_laplacian2: R=contribution, G=own_normal
    bpy.types.Mesh.malt_smooth_use_attr_contribution = bpy.props.BoolProperty(
        name='Use Attribute', default=False,
        description='Use avr_malt_laplacian2.R instead of uniform value')
    bpy.types.Mesh.malt_smooth_use_attr_own_normal = bpy.props.BoolProperty(
        name='Use Attribute', default=False,
        description='Use avr_malt_laplacian2.G instead of uniform value')

    # Tessellation toggle.
    bpy.types.Mesh.malt_tessellation = bpy.props.BoolProperty(
        name='Tessellation', default=False,
        description='Enable GPU tessellation for this mesh')

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
    del bpy.types.Mesh.malt_compute_skin
    del bpy.types.Mesh.malt_compute_curvature
    del bpy.types.Mesh.malt_compute_smooth_normals
    del bpy.types.Mesh.malt_smooth_iterations
    del bpy.types.Mesh.malt_smooth_cotangent_factor
    del bpy.types.Mesh.malt_smooth_quad_mode
    del bpy.types.Mesh.malt_smooth_mix_factor
    del bpy.types.Mesh.malt_smooth_momentum_factor
    del bpy.types.Mesh.malt_smooth_application_strength
    del bpy.types.Mesh.malt_smooth_contribution_strength
    del bpy.types.Mesh.malt_smooth_own_normal_strength
    del bpy.types.Mesh.malt_smooth_groups_enabled
    del bpy.types.Mesh.malt_smooth_use_attr_mix_factor
    del bpy.types.Mesh.malt_smooth_use_attr_momentum
    del bpy.types.Mesh.malt_smooth_use_attr_application
    del bpy.types.Mesh.malt_smooth_use_attr_contribution
    del bpy.types.Mesh.malt_smooth_use_attr_own_normal
    del bpy.types.Mesh.malt_tessellation

    bpy.types.DATA_PT_vertex_colors.remove(draw_vertex_color_overrides)
    del bpy.types.Mesh.malt_vertex_color_override_0
    del bpy.types.Mesh.malt_vertex_color_override_1
    del bpy.types.Mesh.malt_vertex_color_override_2
    del bpy.types.Mesh.malt_vertex_color_override_3
