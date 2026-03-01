import bpy


class OT_MaltRefreshMeshes(bpy.types.Operator):
    bl_idname = "wm.malt_refresh_meshes"
    bl_label = "Refresh Meshes"
    bl_description = "Clear the mesh cache so all meshes are re-extracted on the next frame"

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'MALT'

    def execute(self, context):
        from . import MaltMeshes
        MaltMeshes.reset_meshes()
        for screen in bpy.data.screens:
            for area in screen.areas:
                area.tag_redraw()
        self.report({'INFO'}, "Mesh cache cleared")
        return {'FINISHED'}


class OT_MaltRefreshShaders(bpy.types.Operator):
    bl_idname = "wm.malt_refresh_shaders"
    bl_label = "Refresh Shaders"
    bl_description = ("Regenerate all node tree sources and recompile all "
                      "material and compute shaders")

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'MALT'

    def execute(self, context):
        from . import MaltMaterial

        # Force all node trees to regenerate their GLSL source.
        for tree in bpy.data.node_groups:
            if tree.bl_idname == 'MaltTree':
                if hasattr(tree, 'update_ext'):
                    tree.update_ext(force_track_shader_changes=False, force_update=True)

        # Reset material/compute shader caches and recompile.
        MaltMaterial.reset_materials()
        MaltMaterial.reset_compute_materials()
        MaltMaterial.track_shader_changes(force_update=True)
        MaltMaterial.track_compute_shader_changes()

        for screen in bpy.data.screens:
            for area in screen.areas:
                area.tag_redraw()
        self.report({'INFO'}, "Shaders recompiled")
        return {'FINISHED'}


class OT_MaltFullRefresh(bpy.types.Operator):
    bl_idname = "wm.malt_full_refresh"
    bl_label = "Full Refresh"
    bl_description = ("Restart the Malt server and reinitialize all caches. "
                      "Picks up server-side code changes (Malt/, Bridge/) "
                      "without restarting Blender")

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'MALT' and context.scene.world is not None

    def execute(self, context):
        # Restart server, reset all caches, reinitialize parameters.
        # This spawns a fresh server process that loads current Malt/
        # and Bridge/ code from disk, so server-side changes are picked up.
        # Client-side (BlenderMalt/) changes require a Blender restart
        # because importlib.reload() breaks isinstance() checks across
        # modules that hold references to old class objects.
        context.scene.world.malt.update_pipeline(context)

        self.report({'INFO'}, "Malt full refresh complete")
        return {'FINISHED'}


class VIEW3D_PT_AVR_Malt(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "AVR Malt"

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == 'MALT'

    def draw(self, context):
        layout = self.layout
        layout.operator('wm.console_toggle', icon='CONSOLE')
        layout.operator('wm.malt_refresh_meshes', icon='MESH_DATA')
        layout.operator('wm.malt_refresh_shaders', icon='NODE_MATERIAL')
        layout.operator('wm.malt_full_refresh', icon='RECOVER_LAST')

        preferences = context.preferences.addons['BlenderMalt'].preferences
        layout.separator()
        layout.label(text="Debug")
        layout.prop(preferences, 'show_timings')


classes = (
    OT_MaltRefreshMeshes,
    OT_MaltRefreshShaders,
    OT_MaltFullRefresh,
    VIEW3D_PT_AVR_Malt,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
