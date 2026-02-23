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

def load_mesh(object, name):
    from . import CBlenderMalt

    m = object.data
    if object.type != 'MESH' or object.mode == 'EDIT':
        m = object.to_mesh()
    
    if m is None or len(m.polygons) == 0:
        return None
    
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

    vertex_count = len(m.vertices)

    # Adjacency CSR: vertex -> neighbor vertices (from mesh edges)
    from collections import defaultdict
    neighbors = defaultdict(set)
    for edge in m.edges:
        v0, v1 = edge.vertices
        neighbors[v0].add(v1)
        neighbors[v1].add(v0)

    adjacency_offsets_list = [0]
    adjacency_indices_list = []
    for v in range(vertex_count):
        adj = sorted(neighbors.get(v, []))
        adjacency_indices_list.extend(adj)
        adjacency_offsets_list.append(len(adjacency_indices_list))

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
    cv_ptr = ctypes.cast(corner_vert.buffer(), ctypes.POINTER(ctypes.c_int))
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
    vc_ptr = ctypes.cast(vert_corner_data_buf.buffer(), ctypes.POINTER(ctypes.c_int))
    for i, val in enumerate(vert_corner_offsets_list):
        vc_ptr[i] = val
    base = len(vert_corner_offsets_list)
    for i, val in enumerate(vert_corner_indices_list):
        vc_ptr[base + i] = val

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

def draw_compute_shader(self, context):
    if context.scene.render.engine != 'MALT':
        return
    if context.object is None or context.object.type != 'MESH':
        return
    mesh = context.object.data
    self.layout.use_property_split = True
    self.layout.label(text='Malt Compute')
    self.layout.prop_search(mesh, 'malt_compute_nodes', bpy.data, 'node_groups',
                            text='Compute Node Tree')


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
    bpy.types.DATA_PT_vertex_colors.append(draw_compute_shader)

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
    bpy.types.DATA_PT_vertex_colors.remove(draw_compute_shader)
    del bpy.types.Mesh.malt_compute_nodes

    bpy.types.DATA_PT_vertex_colors.remove(draw_vertex_color_overrides)
    del bpy.types.Mesh.malt_vertex_color_override_0
    del bpy.types.Mesh.malt_vertex_color_override_1
    del bpy.types.Mesh.malt_vertex_color_override_2
    del bpy.types.Mesh.malt_vertex_color_override_3
