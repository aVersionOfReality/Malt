import ctypes
import bpy

MAX_VERTEX_COLORS = 9

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

    normals = get_load_buffer('normals', ctypes.c_float, (loop_count * 3))
    ctypes.memmove(normals.buffer(), m.corner_normals[0].as_pointer(), normals.size_in_bytes())

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
    
    colors_list = [None] * MAX_VERTEX_COLORS
    if object.type == 'MESH':
        orig = object.original.data
        override_count = orig.malt_vertex_color_override_count
        # Build a map: slot_index -> attribute_name from active override fields
        overrides = {}
        for f in range(override_count):
            slot = getattr(orig, f'malt_vertex_color_override_index_{f}')
            name = getattr(orig, f'malt_vertex_color_override_{f}')
            if name:
                overrides[slot] = name
        for i in range(MAX_VERTEX_COLORS):
            # Use explicit override if set, otherwise fall back to malt_vcol-N
            override = overrides.get(i, '')
            if not override:
                default_name = f'malt_vcol-{i}'
                if default_name in m.color_attributes:
                    override = default_name
            if not override:
                continue
            attribute = m.color_attributes.get(override)
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

    mesh_data = {
        'positions': positions,
        'indices': indices,
        'normals': normals,
        'uvs': uvs_list,
        'tangents': tangents_buffer,
        'colors': colors_list,
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

def unload_mesh_data(mesh_data):
    """Invalidate cached mesh entries for all objects using mesh_data."""
    # Fast path: objects with no modifiers share the key 'MESH_<data name>'
    direct_key = 'MESH_' + mesh_data.name_full
    if direct_key in MESHES:
        MESHES[direct_key] = None
    # Slow path: objects with modifiers use their own object name as key —
    # only iterate objects that both use this mesh AND have modifiers
    for obj in bpy.data.objects:
        if obj.data is mesh_data and len(obj.modifiers) > 0:
            MESHES[get_mesh_name(obj)] = None

def reset_meshes():
    global MESHES
    MESHES = {}

class MALT_OT_pick_color_attribute(bpy.types.Operator):
    bl_idname = 'malt.pick_color_attribute'
    bl_label = 'Pick Color Attribute'
    bl_property = 'attribute_name'

    property_name: bpy.props.StringProperty()
    attribute_name: bpy.props.EnumProperty(
        items=lambda self, context: [
            (attr.name, attr.name, '')
            for attr in context.object.data.color_attributes
            if attr.domain == 'CORNER'
        ] or [('', 'No CORNER color attributes', '')]
    )

    def execute(self, context):
        setattr(context.object.data, self.property_name, self.attribute_name)
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}


class MALT_OT_add_color_override(bpy.types.Operator):
    bl_idname = 'malt.add_color_override'
    bl_label = 'Add Override Field'
    bl_description = (
        "Max 9 Vertex Colors. Must be Face Corner. "
        "Defaults to 'malt_vcol-0', 'malt_vcol-1', etc. "
        "Add a field to override a slot with a different "
        "Face Corner Color attribute."
    )

    def execute(self, context):
        mesh = context.object.data
        count = mesh.malt_vertex_color_override_count
        if count >= MAX_VERTEX_COLORS:
            self.report({'ERROR'}, f'Max fields already in use ({MAX_VERTEX_COLORS})')
            return {'CANCELLED'}
        # Check if all 9 slot indices are already claimed
        claimed = set()
        for f in range(count):
            claimed.add(getattr(mesh, f'malt_vertex_color_override_index_{f}'))
        if len(claimed) >= MAX_VERTEX_COLORS:
            self.report({'ERROR'}, f'All {MAX_VERTEX_COLORS} color slots are already targeted')
            return {'CANCELLED'}
        # Pick the lowest unclaimed index as the default for the new field
        new_index = next(i for i in range(MAX_VERTEX_COLORS) if i not in claimed)
        f = count
        setattr(mesh, f'malt_vertex_color_override_index_{f}', new_index)
        setattr(mesh, f'malt_vertex_color_override_{f}', '')
        mesh.malt_vertex_color_override_count += 1
        return {'FINISHED'}


class MALT_OT_remove_color_override(bpy.types.Operator):
    bl_idname = 'malt.remove_color_override'
    bl_label = 'Remove Override Field'
    bl_description = 'Remove this vertex color override slot'

    index: bpy.props.IntProperty()

    def execute(self, context):
        mesh = context.object.data
        count = mesh.malt_vertex_color_override_count
        # Compact: shift fields above the removed one down by one
        for i in range(self.index, count - 1):
            src_idx = getattr(mesh, f'malt_vertex_color_override_index_{i + 1}')
            src_name = getattr(mesh, f'malt_vertex_color_override_{i + 1}')
            setattr(mesh, f'malt_vertex_color_override_index_{i}', src_idx)
            setattr(mesh, f'malt_vertex_color_override_{i}', src_name)
        # Clear the now-vacant last slot and decrement
        setattr(mesh, f'malt_vertex_color_override_index_{count - 1}', 0)
        setattr(mesh, f'malt_vertex_color_override_{count - 1}', '')
        mesh.malt_vertex_color_override_count -= 1
        return {'FINISHED'}


def draw_vertex_color_overrides(self, context):
    if context.scene.render.engine != 'MALT':
        return
    mesh = context.object.data
    self.layout.use_property_split = True
    self.layout.label(text='Malt Vertex Color Overrides')

    count = mesh.malt_vertex_color_override_count

    # Collect claimed indices to detect duplicates
    claimed_indices = [getattr(mesh, f'malt_vertex_color_override_index_{i}') for i in range(count)]
    duplicate_indices = {idx for idx in claimed_indices if claimed_indices.count(idx) > 1}

    for i in range(count):
        key = f'malt_vertex_color_override_{i}'
        index_key = f'malt_vertex_color_override_index_{i}'
        value = getattr(mesh, key)
        slot_index = getattr(mesh, index_key)

        layout = self.layout

        # Warn if the attribute exists but isn't a CORNER domain
        if value in mesh.color_attributes and mesh.color_attributes[value].domain != 'CORNER':
            layout = self.layout.box()
            layout.label(text='Only Face Corner attributes are supported', icon='ERROR')

        # Warn if this slot index is targeted by more than one field
        if slot_index in duplicate_indices:
            if layout is self.layout:
                layout = self.layout.box()
            layout.label(text=f'Duplicate index {slot_index}', icon='ERROR')

        row = layout.row(align=True)
        # Integer drag field for the target slot index
        row.prop(mesh, index_key, text='')
        # Free-text attribute name field
        row.prop(mesh, key, text='', icon='GROUP_VCOL')
        # Searchable dropdown to pick from existing CORNER attributes
        op = row.operator('malt.pick_color_attribute', text='', icon='DOWNARROW_HLT')
        op.property_name = key
        # Remove button
        rm = row.operator('malt.remove_color_override', text='', icon='X')
        rm.index = i

    self.layout.operator('malt.add_color_override', icon='ADD')


def _vcol_override_update(self, context):
    unload_mesh_data(self)

def register():
    bpy.utils.register_class(MALT_OT_pick_color_attribute)
    bpy.utils.register_class(MALT_OT_add_color_override)
    bpy.utils.register_class(MALT_OT_remove_color_override)
    for i in range(MAX_VERTEX_COLORS):
        setattr(bpy.types.Mesh, f'malt_vertex_color_override_{i}',
            bpy.props.StringProperty(name=str(i), update=_vcol_override_update,
                options={'LIBRARY_EDITABLE'}, override={'LIBRARY_OVERRIDABLE'}))
        setattr(bpy.types.Mesh, f'malt_vertex_color_override_index_{i}',
            bpy.props.IntProperty(name=str(i), default=i, min=0, max=MAX_VERTEX_COLORS - 1,
                update=_vcol_override_update,
                options={'LIBRARY_EDITABLE'}, override={'LIBRARY_OVERRIDABLE'}))
    bpy.types.Mesh.malt_vertex_color_override_count = bpy.props.IntProperty(
        name='Malt VCol Override Count', default=0, min=0, max=MAX_VERTEX_COLORS,
        update=_vcol_override_update,
        options={'LIBRARY_EDITABLE'}, override={'LIBRARY_OVERRIDABLE'})
    bpy.types.DATA_PT_vertex_colors.append(draw_vertex_color_overrides)


def unregister():
    bpy.types.DATA_PT_vertex_colors.remove(draw_vertex_color_overrides)
    for i in range(MAX_VERTEX_COLORS):
        delattr(bpy.types.Mesh, f'malt_vertex_color_override_{i}')
        delattr(bpy.types.Mesh, f'malt_vertex_color_override_index_{i}')
    del bpy.types.Mesh.malt_vertex_color_override_count
    bpy.utils.unregister_class(MALT_OT_remove_color_override)
    bpy.utils.unregister_class(MALT_OT_add_color_override)
    bpy.utils.unregister_class(MALT_OT_pick_color_attribute)
