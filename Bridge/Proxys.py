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
    """

    def __init__(self, mesh_name, submesh_index, compute_path):
        self.mesh_name = mesh_name
        self.submesh_index = submesh_index
        self.compute_path = compute_path

    def resolve(self):
        import Bridge.Mesh
        import Bridge.ComputeMaterial
        meshes = Bridge.Mesh.MESHES.get(self.mesh_name)
        if not meshes:
            return
        mesh = meshes[self.submesh_index]
        if mesh is None:
            return
        mesh.compute_shader = Bridge.ComputeMaterial.get_compute_shader(self.compute_path)
