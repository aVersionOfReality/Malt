from Malt.GL.Mesh import Mesh
from Malt.GL.Texture import Texture
from Malt.GL.Texture import Gradient
from Malt.Scene import Material

class MeshProxy(Mesh):

    def  __init__(self, name, submesh_index):
        self.name = name
        self.mesh = None
        self.submesh_index = submesh_index
    
    def resolve(self):
        import Bridge.Mesh
        self.mesh = Bridge.Mesh.MESHES[self.name][self.submesh_index]
        self.__dict__.update(self.mesh.__dict__)
    
    def __del__(self):
        pass

class TextureProxy(Texture):

    def  __init__(self, name):
        self.name = name
        self.texture = None
    
    def resolve(self):
        import Bridge.Texture
        self.texture = Bridge.Texture.TEXTURES[self.name]
        self.__dict__.update(self.texture.__dict__)
    
    def __del__(self):
        pass

class GradientProxy(Gradient):

    def  __init__(self, name):
        self.name = name
        self.gradient = None
    
    def resolve(self):
        import Bridge.Texture
        self.gradient = Bridge.Texture.GRADIENTS[self.name]
        self.__dict__.update(self.gradient.__dict__)
    
    def __del__(self):
        pass

class MaterialProxy(Material):

    def __init__(self, path, shader_parameters, parameters):
        self.path = path
        self.shader_parameters = shader_parameters
        super().__init__(None, parameters)

    def resolve(self):
        import Bridge.Material
        self.shader = Bridge.Material.get_shader(self.path, self.shader_parameters)


class ComputeShaderProxy():
    """Proxy that assigns a compiled ComputeShader to a MeshCustomLoad.

    Created on the Blender side (client process) and resolved on the server
    side, where it sets mesh.compute_shader so run_compute_pass() picks it up.

    When *dispatch_plan* is provided (graph has barrier nodes), the proxy
    resolves each segment shader and attaches the full plan to the mesh
    for Pipeline.run_compute_pass() to execute step by step.
    """

    def __init__(self, mesh_name, submesh_index, compute_path,
                 shader_parameters=None, dispatch_plan=None,
                 bone_matrices=None):
        self.mesh_name = mesh_name
        self.submesh_index = submesh_index
        self.compute_path = compute_path
        self.shader_parameters = shader_parameters or {}
        self.dispatch_plan = dispatch_plan
        self.bone_matrices = bone_matrices

    def resolve(self):
        import Bridge.Mesh
        import Bridge.ComputeMaterial
        meshes = Bridge.Mesh.MESHES.get(self.mesh_name)
        if not meshes:
            return
        mesh = meshes[self.submesh_index]
        if mesh is None:
            return

        if self.dispatch_plan:
            # Multi-segment: resolve each segment shader in the plan.
            resolved_plan = []
            for step in self.dispatch_plan:
                if step['type'] == 'segment':
                    shader = Bridge.ComputeMaterial.get_compute_shader(step['path'])
                    resolved_plan.append({**step, 'shader': shader})
                else:
                    resolved_plan.append(step)
            mesh.dispatch_plan = resolved_plan
            # For backward compat checks, set compute_shader to first segment.
            first_seg = next((s for s in resolved_plan if s['type'] == 'segment'), None)
            mesh.compute_shader = first_seg.get('shader') if first_seg else None
        else:
            # Single-segment: existing behavior.
            shader = Bridge.ComputeMaterial.get_compute_shader(self.compute_path)
            mesh.compute_shader = shader
            mesh.dispatch_plan = None

        mesh.compute_shader_parameters = self.shader_parameters

        # Upload per-frame bone matrices to the mesh's SSBO.
        # bone_matrices is a bytes object (pickle-safe); reconstruct a ctypes
        # array on the server side for the GL upload.
        if self.bone_matrices is not None:
            bone_ssbo = getattr(mesh, 'bone_matrices_ssbo', None)
            if bone_ssbo is not None:
                import ctypes
                from Malt.GL.GL import glBindBuffer, glBufferData, glBufferSubData
                from Malt.GL.GL import GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW
                data_size = len(self.bone_matrices)
                num_floats = data_size // ctypes.sizeof(ctypes.c_float)
                c_arr = (ctypes.c_float * num_floats).from_buffer_copy(self.bone_matrices)
                if data_size > bone_ssbo.size:
                    # Reallocate if bone count changed.
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bone_ssbo.buffer[0])
                    glBufferData(GL_SHADER_STORAGE_BUFFER, data_size,
                                 ctypes.pointer(c_arr), GL_DYNAMIC_DRAW)
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
                    bone_ssbo.size = data_size
                else:
                    bone_ssbo.load_sub_data(
                        ctypes.pointer(c_arr), data_size)
