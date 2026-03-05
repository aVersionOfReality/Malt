import math, os, ctypes
from os import path

from Malt.Utils import LOG

from Malt.GL.GL import *
from Malt.GL.Mesh import Mesh, MeshCustomLoad
from Malt.GL.Shader import Shader, UBO, SSBO, shader_preprocessor
from Malt.GL.ComputeShader import ComputeShader

from Malt.Render import Common
from Malt.PipelineParameters import *

SHADER_DIR = path.join(path.dirname(__file__), 'Shaders')

class Pipeline():

    SHADER_INCLUDE_PATHS = []

    BLEND_SHADER = None
    COPY_SHADER = None

    def __init__(self, plugins=[]):
        from multiprocessing.dummy import Pool
        self.pool = Pool(16)

        if SHADER_DIR not in Pipeline.SHADER_INCLUDE_PATHS:
            Pipeline.SHADER_INCLUDE_PATHS.append(SHADER_DIR)

        self.resolution = None
        self.sample_count = 0
        self.result = None
        self.is_final_render = None
        
        plugins = [plugin for plugin in plugins if plugin.poll_pipeline(self)]
        self.setup_parameters()
        for plugin in plugins:
            plugin.register_pipeline_parameters(self.parameters)
        self.setup_graphs()
        for plugin in plugins:
            for graph in plugin.register_pipeline_graphs():
                self.add_graph(graph)
        for plugin in plugins:
            plugin.register_graph_libraries(self.graphs)
        for graph in self.graphs.values():
            graph.setup_reflection()
        self.setup_resources()
    
    def setup_parameters(self):
        self.parameters = PipelineParameters()
        
        self.parameters.mesh['double_sided'] = Parameter(False, Type.BOOL, doc=
            "Disables backface culling, so geometry is rendered from both sides.")
        
        self.parameters.mesh['precomputed_tangents'] = Parameter(False, Type.BOOL, doc="""
            Load precomputed mesh tangents *(needed for improving normal mapping quality on low poly meshes)*. 
            It's disabled by default since it slows down mesh loading in Blender.  
            When disabled, the *tangents* are calculated on the fly from the *pixel shader*.""")
        
        self.parameters.world['Material.Default'] = MaterialParameter('', '.mesh.glsl', 'Mesh', doc=
            "The default material, used for objects with no material assigned.")
        
        self.parameters.world['Material.Override'] = MaterialParameter('', '.mesh.glsl', 'Mesh', doc=
            "When set, overrides all scene materials with this one.")
        
        self.parameters.world['Viewport.Resolution Scale'] = Parameter(1.0 , Type.FLOAT, doc="""
            A multiplier for the viewport resolution.
            It can be lowered to improve viewport performance or for specific styles, like *pixel art*."""
        )
        self.parameters.world['Viewport.Smooth Interpolation'] = Parameter(True , Type.BOOL, doc="""
            The interpolation mode used when *Resolution Scale* is not 1.
            Toggles between *Nearest/Bilinear* interpolation.""")
    
    def get_parameters(self):
        return self.parameters

    def setup_graphs(self):
        self.graphs = {}
    
    def add_graph(self, graph):
        if graph.file_extension.endswith('glsl'):
            graph.include_paths += self.SHADER_INCLUDE_PATHS
        self.graphs[graph.name] = graph
    
    def get_graphs(self):
        result = {}
        for name, graph in self.graphs.items():
            result[name] = graph.get_serializable_copy()
        return result

    def setup_resources(self):
        self.common_buffer = Common.CommonBuffer()
        positions=[
             1.0,  1.0, 0.0,
             1.0, -1.0, 0.0,
            -1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
        ]
        indices=[
            0, 1, 3,
            1, 2, 3,
        ]
        self.quad = Mesh(positions, indices)
        
        if Pipeline.BLEND_SHADER is None:
            source='''#include "Passes/BlendTexture.glsl"'''
            Pipeline.BLEND_SHADER = self.compile_shader_from_source(source)
        self.blend_shader = Pipeline.BLEND_SHADER

        if Pipeline.COPY_SHADER is None:
            source = '''#include "Passes/CopyTextures.glsl"'''
            Pipeline.COPY_SHADER = self.compile_shader_from_source(source)
        self.copy_shader = Pipeline.COPY_SHADER
    
    def get_render_outputs(self):
        return {
            'COLOR' : GL_RGBA32F,
            'DEPTH' : GL_R32F,
        }
    
    def get_samples(self):
        return [(0,0)]
    
    def needs_more_samples(self):
        return self.sample_count < len(self.get_samples())
    
    def setup_render_targets(self, resolution):
        pass
    
    def find_shader_path(self, path, search_paths=[]):
        if os.path.exists(path):
            return path
        else:
            for shader_path in self.SHADER_INCLUDE_PATHS + search_paths:
                full_path = os.path.join(shader_path, path)
                if os.path.exists(full_path):
                    return full_path
        return None
    
    def compile_shader_from_source(self, source, include_paths=[], defines=[]):
        vertex_src = shader_preprocessor(source, include_paths + self.SHADER_INCLUDE_PATHS, defines + ['VERTEX_SHADER'])
        pixel_src = shader_preprocessor(source, include_paths + self.SHADER_INCLUDE_PATHS, defines + ['PIXEL_SHADER'])
        return Shader(vertex_src, pixel_src)

    def compile_compute_shader_from_source(self, source, include_paths=[], defines=[]):
        compute_src = shader_preprocessor(source, include_paths + self.SHADER_INCLUDE_PATHS, defines + ['COMPUTE_STAGE'])
        return ComputeShader(compute_src)

    def compile_material_from_source(self, material_type, source, include_paths=[]):
        return self.graphs[material_type].compile_material(source, include_paths)
    
    def compile_material(self, shader_path, search_paths=[]):
        try:
            file_dir = path.dirname(shader_path)
            source = '#include "{}"'.format(path.basename(shader_path))
            material_type = shader_path.split('.')[-2]
            for graph in self.graphs.values():
                if shader_path.endswith(graph.file_extension):
                    material_type = graph.name
            return self.compile_material_from_source(material_type, source, [file_dir] + search_paths)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return str(e)
    
    def load_mesh(self, position, indices, normal, tangent=None, uvs=[], colors=[], ssbo_colors=[None]*8, vertex_count=0, loop_count=0, rest_positions=None, rest_normals=None, corner_vert=None, adjacency_data=None, vert_corner_data=None, fan_groups=None, cotangent_weights=None, edge_metadata=None, laplacian1=None, laplacian2=None, bone_indices=None, bone_weights=None, bone_count=0):
        # Each parameter implements the Malt.Utils.IBuffer interface
        # Indices is an array of index buffers corresponding to each of the materials a mesh has
        # VBOs are shared for all the materials
          
        def load_VBO(data):
            VBO = gl_buffer(GL_INT, 1)
            glGenBuffers(1, VBO)
            glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
            glBufferData(GL_ARRAY_BUFFER, data.size_in_bytes(), data.buffer(), GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            return VBO

        position_vbo = load_VBO(position)

        # Deformed position buffer: vec4 per loop (16 bytes), GPU-writable.
        # Uses vec4 instead of vec3 so the buffer stride (16 bytes) matches std430 vec4[]
        # layout in the compute shader SSBO. The VBO reads only xyz (element_size=3) with
        # an explicit stride of 16 to skip the w padding float.
        deformed_position_buffer = gl_buffer(GL_INT, 1)
        glGenBuffers(1, deformed_position_buffer)
        glBindBuffer(GL_ARRAY_BUFFER, deformed_position_buffer[0])
        if rest_positions is not None:
            glBufferData(GL_ARRAY_BUFFER, rest_positions.size_in_bytes(), rest_positions.buffer(), GL_DYNAMIC_DRAW)
        else:
            glBufferData(GL_ARRAY_BUFFER, loop_count * 16, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # Normal VBO: GL_DYNAMIC_DRAW so the compute shader can write updated normals back.
        # Expected to be 4-float per loop element (vec4, w=0) when coming from MaltMeshes.py.
        normal_vbo = gl_buffer(GL_INT, 1)
        glGenBuffers(1, normal_vbo)
        glBindBuffer(GL_ARRAY_BUFFER, normal_vbo[0])
        glBufferData(GL_ARRAY_BUFFER, normal.size_in_bytes(), normal.buffer(), GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        tangent_vbo = load_VBO(tangent) if tangent else None
        uv_vbos = [load_VBO(e) for e in uvs]
        color_vbos = [load_VBO(e) if e else None for e in colors]

        ssbo_objects = [None]*8
        for i, ssbo_data in enumerate(ssbo_colors):
            if ssbo_data is not None and i < 8:
                ssbo = SSBO()
                ssbo.load_raw(ssbo_data.buffer(), ssbo_data.size_in_bytes())
                ssbo_objects[i] = ssbo

        rest_position_ssbo = None
        if rest_positions is not None:
            rest_position_ssbo = SSBO()
            rest_position_ssbo.load_raw(rest_positions.buffer(), rest_positions.size_in_bytes())

        rest_normals_ssbo = None
        if rest_normals is not None:
            rest_normals_ssbo = SSBO()
            rest_normals_ssbo.load_raw(rest_normals.buffer(), rest_normals.size_in_bytes())

        corner_vert_ssbo = None
        if corner_vert is not None:
            corner_vert_ssbo = SSBO()
            corner_vert_ssbo.load_raw(corner_vert.buffer(), corner_vert.size_in_bytes())

        adjacency_data_ssbo = None
        if adjacency_data is not None:
            adjacency_data_ssbo = SSBO()
            adjacency_data_ssbo.load_raw(adjacency_data.buffer(), adjacency_data.size_in_bytes())

        vert_corner_data_ssbo = None
        if vert_corner_data is not None:
            vert_corner_data_ssbo = SSBO()
            vert_corner_data_ssbo.load_raw(vert_corner_data.buffer(), vert_corner_data.size_in_bytes())

        fan_groups_ssbo = None
        if fan_groups is not None:
            fan_groups_ssbo = SSBO()
            fan_groups_ssbo.load_raw(fan_groups.buffer(), fan_groups.size_in_bytes())

        cotangent_weights_ssbo = None
        if cotangent_weights is not None:
            cotangent_weights_ssbo = SSBO()
            cotangent_weights_ssbo.load_raw(cotangent_weights.buffer(), cotangent_weights.size_in_bytes())

        edge_metadata_ssbo = None
        if edge_metadata is not None:
            edge_metadata_ssbo = SSBO()
            edge_metadata_ssbo.load_raw(edge_metadata.buffer(), edge_metadata.size_in_bytes())

        # Scratch buffer for Laplacian smooth ping-pong (same size as deformed_positions).
        smooth_scratch_ssbo = None
        if loop_count > 0:
            smooth_scratch_ssbo = SSBO()
            scratch_size = loop_count * 16  # vec4 per loop
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_scratch_ssbo.buffer[0])
            glBufferData(GL_SHADER_STORAGE_BUFFER, scratch_size, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            smooth_scratch_ssbo.size = scratch_size

        # Smoothed normals output buffer (written by smooth kernel, read by post-smooth segments).
        # Keeps normals[] (binding 11) untouched so downstream nodes can read original normals.
        smoothed_normals_ssbo = None
        if loop_count > 0:
            smoothed_normals_ssbo = SSBO()
            sn_size = loop_count * 16  # vec4 per loop
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, smoothed_normals_ssbo.buffer[0])
            glBufferData(GL_SHADER_STORAGE_BUFFER, sn_size, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            smoothed_normals_ssbo.size = sn_size

        # Per-corner weight parameters for smooth kernel (two vec4 SSBOs).
        # smooth_weights (binding 18): mix, momentum, application
        # smooth_weights_2 (binding 25): contribution, own_normal
        smooth_weights_ssbo = None
        smooth_weights_2_ssbo = None
        smooth_prev_ssbo = None  # Previous iteration buffer for momentum
        if loop_count > 0:
            weights_size = loop_count * 16  # vec4 per loop
            smooth_weights_ssbo = SSBO()
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_weights_ssbo.buffer[0])
            glBufferData(GL_SHADER_STORAGE_BUFFER, weights_size, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            smooth_weights_ssbo.size = weights_size

            smooth_weights_2_ssbo = SSBO()
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_weights_2_ssbo.buffer[0])
            glBufferData(GL_SHADER_STORAGE_BUFFER, weights_size, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            smooth_weights_2_ssbo.size = weights_size

            smooth_prev_ssbo = SSBO()
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, smooth_prev_ssbo.buffer[0])
            glBufferData(GL_SHADER_STORAGE_BUFFER, weights_size, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            smooth_prev_ssbo.size = weights_size

        # Per-corner laplacian weight attributes (CPU-side copies).
        # When present, _fill_smooth_weights can read per-corner values from these
        # instead of broadcasting the uniform value.
        # laplacian1 maps to smooth_weights (binding 18): mix, momentum, application
        # laplacian2 maps to smooth_weights_2 (binding 25): contribution, own_normal
        laplacian1_data = None
        if laplacian1 is not None:
            n = loop_count * 4
            laplacian1_data = (ctypes.c_float * n)()
            ctypes.memmove(laplacian1_data, laplacian1.buffer(), n * 4)
        laplacian2_data = None
        if laplacian2 is not None:
            n = loop_count * 4
            laplacian2_data = (ctypes.c_float * n)()
            ctypes.memmove(laplacian2_data, laplacian2.buffer(), n * 4)

        # GPU skinning SSBOs: per-vertex bone indices/weights (static) and
        # per-bone matrices (pre-allocated empty, filled per-frame).
        bone_indices_ssbo = None
        if bone_indices is not None:
            bone_indices_ssbo = SSBO()
            bone_indices_ssbo.load_raw(bone_indices.buffer(), bone_indices.size_in_bytes())

        bone_weights_ssbo = None
        if bone_weights is not None:
            bone_weights_ssbo = SSBO()
            bone_weights_ssbo.load_raw(bone_weights.buffer(), bone_weights.size_in_bytes())

        bone_matrices_ssbo = None
        if bone_count > 0:
            bone_matrices_ssbo = SSBO()
            matrices_size = bone_count * 64  # mat4 = 16 floats = 64 bytes per bone
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bone_matrices_ssbo.buffer[0])
            glBufferData(GL_SHADER_STORAGE_BUFFER, matrices_size, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            bone_matrices_ssbo.size = matrices_size

        # Per-corner curvature values (written by Compute_Curvature node, read by mesh shader).
        curvature_ssbo = None
        if loop_count > 0:
            curvature_ssbo = SSBO()
            curvature_size = loop_count * 4  # float per loop
            zero_data = (ctypes.c_float * loop_count)()
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, curvature_ssbo.buffer[0])
            glBufferData(GL_SHADER_STORAGE_BUFFER, curvature_size, zero_data, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            curvature_ssbo.size = curvature_size

        results = []

        for i, index in enumerate(indices):
            result = MeshCustomLoad()
            
            result.VAO = gl_buffer(GL_INT, 1)
            glGenVertexArrays(1, result.VAO)
            glBindVertexArray(result.VAO[0])
            
            result.EBO = gl_buffer(GL_INT, 1)
            glGenBuffers(1, result.EBO)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, result.EBO[0])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, index.size_in_bytes(), index.buffer(), GL_STATIC_DRAW)
            
            result.index_count = len(index)

            result.position = position_vbo
            result.normal = normal_vbo
            result.tangent = tangent_vbo
            result.uvs = uv_vbos
            result.colors = color_vbos
            result.ssbo_list = ssbo_objects
            result.vertex_count = vertex_count
            result.loop_count = loop_count
            result.rest_position_ssbo = rest_position_ssbo
            result.rest_normals_ssbo = rest_normals_ssbo
            result.deformed_position_buffer = deformed_position_buffer
            result.corner_vert_ssbo = corner_vert_ssbo
            result.adjacency_data_ssbo = adjacency_data_ssbo
            result.vert_corner_data_ssbo = vert_corner_data_ssbo
            result.fan_groups_ssbo = fan_groups_ssbo
            result.cotangent_weights_ssbo = cotangent_weights_ssbo
            result.edge_metadata_ssbo = edge_metadata_ssbo
            result.smooth_scratch_ssbo = smooth_scratch_ssbo
            result.smoothed_normals_ssbo = smoothed_normals_ssbo
            result.smooth_weights_ssbo = smooth_weights_ssbo
            result.smooth_weights_2_ssbo = smooth_weights_2_ssbo
            result.smooth_prev_ssbo = smooth_prev_ssbo
            result.laplacian1_data = laplacian1_data
            result.laplacian2_data = laplacian2_data
            result.bone_indices_ssbo = bone_indices_ssbo
            result.bone_weights_ssbo = bone_weights_ssbo
            result.bone_matrices_ssbo = bone_matrices_ssbo
            result.bone_count = bone_count
            result.curvature_ssbo = curvature_ssbo

            def bind_VBO(VBO, index, element_size, gl_type=GL_FLOAT, gl_normalize=GL_FALSE, stride=0):
                glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
                glEnableVertexAttribArray(index)
                glVertexAttribPointer(index, element_size, gl_type, gl_normalize, stride, None)
            
            # stride=16: read 3 floats (xyz) with 16-byte spacing to skip the vec4 w padding
            bind_VBO(result.deformed_position_buffer, 0, 3, stride=16)
            if loop_count > 0 and normal.size_in_bytes() == loop_count * 16:
                # 4-float vec4 normals (writable, stride=16 to skip w padding)
                bind_VBO(result.normal, 1, 3, stride=16)
            elif position.size_in_bytes() == normal.size_in_bytes():
                # 3-float float normals (legacy tightly-packed)
                bind_VBO(result.normal, 1, 3)
            else:
                # Compressed short normals
                bind_VBO(result.normal, 1, 3, GL_SHORT, GL_TRUE)
            
            if tangent:
                bind_VBO(result.tangent, 2, 4)
            
            max_uv = 4
            max_vertex_colors = 9
            uv0_index = 3
            color0_index = uv0_index + max_uv
            for i, uv in enumerate(result.uvs):
                if i >= max_uv:
                    LOG.warning('{} : UV count exceeds max supported UVs ({})'.format(name, max_uv))
                    break
                bind_VBO(uv, uv0_index + i, 2)
            for i, color in enumerate(result.colors):
                if i >= max_vertex_colors:
                    LOG.warning('{} : Vertex Color Layer count exceeds max supported layers ({})'.format(name, max_uv))
                    break
                if color:
                    if colors[i]._ctype == ctypes.c_uint8:
                        bind_VBO(color, color0_index + i, 4, GL_UNSIGNED_BYTE, GL_TRUE)
                        result.color_is_srgb[i] = True
                    if colors[i]._ctype == ctypes.c_float:
                        bind_VBO(color, color0_index + i, 4, GL_FLOAT)

            glBindVertexArray(0)
            results.append(result)

        return results
    
    def draw_screen_pass(self, shader, target, blend = False):
        #Allow screen passes draw to gl_FragDepth
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_ALWAYS)
        glDisable(GL_CULL_FACE)
        if blend:
            glEnable(GL_BLEND)
        else:
            glDisable(GL_BLEND)
        target.bind()
        shader.bind()
        self.quad.draw()
    
    def blend_texture(self, blend_texture, target, opacity):
        self.blend_shader.textures['blend_texture'] = blend_texture
        glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        glBlendColor(0, 0, 0, opacity)
        self.draw_screen_pass(self.blend_shader, target, True)
    
    def copy_textures(self, target, color_sources=[], depth_source=None):
        for i, texture in enumerate(color_sources):
            self.copy_shader.textures[f'IN[{str(i)}]'] = texture
        self.copy_shader.textures['IN_DEPTH'] = depth_source
        self.draw_screen_pass(self.copy_shader, target)
    
    def build_scene_batches(self, objects):
        result = {}
        for obj in objects:
            if obj.material not in result:
                result[obj.material] = {}
            if obj.mesh not in result[obj.material]:
                result[obj.material][obj.mesh] = {}
            mesh_dict = result[obj.material][obj.mesh]
            if obj.mirror_scale:
                if 'mirror_scale' not in mesh_dict:
                    mesh_dict['mirror_scale'] = []
                mesh_dict['mirror_scale'].append(obj)
            else:
                if 'normal_scale' not in mesh_dict:
                    mesh_dict['normal_scale'] = []
                mesh_dict['normal_scale'].append(obj)
        
        # Assume at least 64kb of UBO storage (d3d11 requirement) and max element size of mat4
        max_instances = 1000
        models = (max_instances * (ctypes.c_float * 16))()
        ids = (max_instances * ctypes.c_uint)()

        for material, meshes in result.items():
            for mesh, scale_groups in meshes.items():
                for scale_group, objs in scale_groups.items():
                    batches = []
                    scale_groups[scale_group] = batches
                    
                    i = 0
                    batch_length = len(objs)
                    
                    while i < batch_length:
                        instance_i = i % max_instances
                        models[instance_i] = objs[i].matrix
                        ids[instance_i] = objs[i].parameters['ID']

                        i+=1
                        instances_count = instance_i + 1

                        if i == batch_length or instances_count == max_instances:
                            local_models = ((ctypes.c_float * 16) * instances_count).from_address(ctypes.addressof(models))
                            # IDs are stored as uvec4, so we make sure the buffer count is a multiple of 4,
                            # since some drivers will only bind a full uvec4 (see issue #319)
                            id_buffer_count = math.ceil(instances_count/4)*4
                            local_ids = (ctypes.c_uint * id_buffer_count).from_address(ctypes.addressof(ids))

                            models_UBO = UBO()
                            ids_UBO = UBO()

                            models_UBO.load_data(local_models)
                            ids_UBO.load_data(local_ids)

                            batches.append({
                                'instances_count': instances_count,
                                'BATCH_MODELS':models_UBO,
                                'BATCH_IDS':ids_UBO,
                            })
            
        return result
    
    _smooth_kernel = None

    def _get_smooth_kernel(self):
        """Lazy-compile the Laplacian smoothing kernel (cached on the class)."""
        if Pipeline._smooth_kernel is not None:
            return Pipeline._smooth_kernel
        kernel_path = path.join(SHADER_DIR, 'ComputeKernels', 'Laplacian_Smooth_Kernel.glsl')
        try:
            with open(kernel_path) as f:
                source = '#define COMPUTE_STAGE\n' + f.read()
            Pipeline._smooth_kernel = ComputeShader(source)
            if Pipeline._smooth_kernel.error:
                LOG.error(f'SMOOTH KERNEL ERROR: {Pipeline._smooth_kernel.error}')
        except Exception as e:
            LOG.error(f'Failed to load smooth kernel: {e}')
            Pipeline._smooth_kernel = ComputeShader(None)
        return Pipeline._smooth_kernel

    _skin_kernel = None

    def _get_skin_kernel(self):
        """Lazy-compile the LBS skinning kernel (cached on the class)."""
        if Pipeline._skin_kernel is not None:
            return Pipeline._skin_kernel
        kernel_path = path.join(SHADER_DIR, 'ComputeKernels', 'LBS_Skin_Kernel.glsl')
        try:
            with open(kernel_path) as f:
                source = '#define COMPUTE_STAGE\n' + f.read()
            Pipeline._skin_kernel = ComputeShader(source)
            if Pipeline._skin_kernel.error:
                LOG.error(f'SKIN KERNEL ERROR: {Pipeline._skin_kernel.error}')
        except Exception as e:
            LOG.error(f'Failed to load skin kernel: {e}')
            Pipeline._skin_kernel = ComputeShader(None)
        return Pipeline._skin_kernel

    def _bind_compute_ssbos(self, m):
        """Bind all compute-related SSBOs for the fixed pipeline."""
        # Core compute SSBOs (bindings 8–14, 16–24)
        if m.rest_position_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m.rest_position_ssbo.buffer[0])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m.deformed_position_buffer[0])
        if m.corner_vert_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, m.corner_vert_ssbo.buffer[0])
        if m.normal is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, m.normal[0])
        if m.rest_normals_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, m.rest_normals_ssbo.buffer[0])
        if m.adjacency_data_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, m.adjacency_data_ssbo.buffer[0])
        if m.vert_corner_data_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, m.vert_corner_data_ssbo.buffer[0])
        if m.smooth_scratch_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, m.smooth_scratch_ssbo.buffer[0])
        if m.cotangent_weights_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, m.cotangent_weights_ssbo.buffer[0])
        if m.edge_metadata_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, m.edge_metadata_ssbo.buffer[0])
        if m.smooth_weights_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 18, m.smooth_weights_ssbo.buffer[0])
        if getattr(m, 'bone_matrices_ssbo', None) is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 19, m.bone_matrices_ssbo.buffer[0])
        if getattr(m, 'bone_indices_ssbo', None) is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, m.bone_indices_ssbo.buffer[0])
        if getattr(m, 'bone_weights_ssbo', None) is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, m.bone_weights_ssbo.buffer[0])
        if m.fan_groups_ssbo is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 22, m.fan_groups_ssbo.buffer[0])
        if getattr(m, 'curvature_ssbo', None) is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 23, m.curvature_ssbo.buffer[0])
        if getattr(m, 'smoothed_normals_ssbo', None) is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 24, m.smoothed_normals_ssbo.buffer[0])
        if getattr(m, 'smooth_weights_2_ssbo', None) is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 25, m.smooth_weights_2_ssbo.buffer[0])
        if getattr(m, 'smooth_prev_ssbo', None) is not None:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 26, m.smooth_prev_ssbo.buffer[0])

    def _reset_buffers(self, m):
        """Reset working buffers from immutable rest buffers via GPU copy."""
        if m.rest_position_ssbo is not None:
            copy_size = m.rest_position_ssbo.size
            glBindBuffer(GL_COPY_READ_BUFFER, m.rest_position_ssbo.buffer[0])
            glBindBuffer(GL_COPY_WRITE_BUFFER, m.deformed_position_buffer[0])
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, copy_size)
        if m.rest_normals_ssbo is not None:
            copy_size = m.rest_normals_ssbo.size
            glBindBuffer(GL_COPY_READ_BUFFER, m.rest_normals_ssbo.buffer[0])
            glBindBuffer(GL_COPY_WRITE_BUFFER, m.normal[0])
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, copy_size)
        glBindBuffer(GL_COPY_READ_BUFFER, 0)
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

    _curvature_kernel = None

    def _get_curvature_kernel(self):
        """Lazy-compile the curvature compute kernel (cached on the class)."""
        if Pipeline._curvature_kernel is not None:
            return Pipeline._curvature_kernel
        kernel_path = path.join(SHADER_DIR, 'ComputeKernels', 'Calculate_Curvature_Kernel.glsl')
        try:
            with open(kernel_path) as f:
                source = '#define COMPUTE_STAGE\n' + f.read()
            Pipeline._curvature_kernel = ComputeShader(source)
            if Pipeline._curvature_kernel.error:
                LOG.error(f'CURVATURE KERNEL ERROR: {Pipeline._curvature_kernel.error}')
        except Exception as e:
            LOG.error(f'Failed to load curvature kernel: {e}')
            Pipeline._curvature_kernel = ComputeShader(None)
        return Pipeline._curvature_kernel

    def _run_curvature_step(self, m, workgroups):
        """Dispatch the standalone curvature kernel."""
        curvature_kernel = self._get_curvature_kernel()
        if curvature_kernel is None or curvature_kernel.error:
            return False

        if 'LOOP_COUNT' in curvature_kernel.uniforms:
            curvature_kernel.uniforms['LOOP_COUNT'].set_value(m.loop_count)
        if 'VERTEX_COUNT' in curvature_kernel.uniforms:
            curvature_kernel.uniforms['VERTEX_COUNT'].set_value(m.vertex_count)

        curvature_kernel.bind()
        curvature_kernel.dispatch(workgroups)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        return True

    def _fill_smooth_weights(self, m, params):
        """Write smooth weight values to all corners in the two smooth_weights SSBOs.

        SSBO 1 (binding 18, smooth_weights): maps to avr_malt_laplacian1
          .x = mix_factor,  .y = momentum_factor,  .z = application_strength,  .w = 0
        SSBO 2 (binding 25, smooth_weights_2): maps to avr_malt_laplacian2
          .x = contribution_strength,  .y = own_normal_strength,  .z = 0,  .w = 0

        For each channel, use either the uniform value from params or the per-corner
        value from the laplacian attribute, depending on the use_attr_* toggles.
        """
        if m.smooth_weights_ssbo is None:
            return

        # SSBO 1 channels: mix, momentum, application (indices 0,1,2 in vec4)
        uniform1 = [
            params.get('smooth_mix_factor', 1.0),
            params.get('smooth_momentum_factor', 0.25),
            params.get('smooth_application_strength', 0.5),
            0.0,  # unused .w
        ]
        use_attr1 = [
            params.get('use_attr_mix_factor', False),
            params.get('use_attr_momentum', False),
            params.get('use_attr_application', False),
            False,  # .w never from attribute
        ]
        attr1 = getattr(m, 'laplacian1_data', None)
        any_attr1 = any(use_attr1) and attr1 is not None

        # SSBO 2 channels: contribution, own_normal (indices 0,1 in vec4)
        uniform2 = [
            params.get('smooth_contribution_strength', 1.0),
            params.get('smooth_own_normal_strength', 1.0),
            0.0,  # unused .z
            0.0,  # unused .w
        ]
        use_attr2 = [
            params.get('use_attr_contribution', False),
            params.get('use_attr_own_normal', False),
            False,
            False,
        ]
        attr2 = getattr(m, 'laplacian2_data', None)
        any_attr2 = any(use_attr2) and attr2 is not None

        if (any(use_attr1) and attr1 is None) or (any(use_attr2) and attr2 is None):
            if not getattr(self, '_warned_no_laplacian_attr', False):
                missing = []
                if any(use_attr1) and attr1 is None:
                    missing.append('avr_malt_laplacian1')
                if any(use_attr2) and attr2 is None:
                    missing.append('avr_malt_laplacian2')
                print(f"[Pipeline] WARNING: 'Use Attribute' enabled but {', '.join(missing)} "
                      "data missing on mesh. Falling back to uniform values. "
                      "Refresh meshes to reload.")
                self._warned_no_laplacian_attr = True

        n = m.loop_count

        # Fill SSBO 1.
        buf1 = (ctypes.c_float * (n * 4))()
        if any_attr1:
            for i in range(n):
                for c in range(4):
                    buf1[i*4+c] = attr1[i*4+c] if use_attr1[c] else uniform1[c]
        else:
            for i in range(n):
                buf1[i*4]   = uniform1[0]
                buf1[i*4+1] = uniform1[1]
                buf1[i*4+2] = uniform1[2]
                buf1[i*4+3] = uniform1[3]
        m.smooth_weights_ssbo.load_sub_data(ctypes.pointer(buf1), n * 16)

        # Fill SSBO 2.
        if getattr(m, 'smooth_weights_2_ssbo', None) is not None:
            buf2 = (ctypes.c_float * (n * 4))()
            if any_attr2:
                for i in range(n):
                    for c in range(4):
                        buf2[i*4+c] = attr2[i*4+c] if use_attr2[c] else uniform2[c]
            else:
                for i in range(n):
                    buf2[i*4]   = uniform2[0]
                    buf2[i*4+1] = uniform2[1]
                    buf2[i*4+2] = uniform2[2]
                    buf2[i*4+3] = uniform2[3]
            m.smooth_weights_2_ssbo.load_sub_data(ctypes.pointer(buf2), n * 16)

    def _run_smooth_normals_step(self, m, params, workgroups):
        """Run Laplacian normal smoothing (ping-pong into smoothed_normals)."""
        si = int(params.get('smooth_iterations', 10))
        cf = float(params.get('smooth_cotangent_factor', 1.0))
        qm = int(params.get('smooth_quad_mode', 0))
        groups_enabled = params.get('smooth_groups_enabled', True) and (m.fan_groups_ssbo is not None)

        if si <= 0 or m.smooth_scratch_ssbo is None:
            return False
        if getattr(m, 'smoothed_normals_ssbo', None) is None:
            return False

        self._fill_smooth_weights(m, params)

        smooth_kernel = self._get_smooth_kernel()
        if smooth_kernel is None or smooth_kernel.error:
            return False

        if 'LOOP_COUNT' in smooth_kernel.uniforms:
            smooth_kernel.uniforms['LOOP_COUNT'].set_value(m.loop_count)
        if 'VERTEX_COUNT' in smooth_kernel.uniforms:
            smooth_kernel.uniforms['VERTEX_COUNT'].set_value(m.vertex_count)
        if 'COTANGENT_FACTOR' in smooth_kernel.uniforms:
            smooth_kernel.uniforms['COTANGENT_FACTOR'].set_value(cf)
        if 'GROUPS_ENABLED' in smooth_kernel.uniforms:
            smooth_kernel.uniforms['GROUPS_ENABLED'].set_value(1 if groups_enabled else 0)
        if 'QUAD_MODE' in smooth_kernel.uniforms:
            smooth_kernel.uniforms['QUAD_MODE'].set_value(qm)
        if 'LAST_ITERATION' in smooth_kernel.uniforms:
            smooth_kernel.uniforms['LAST_ITERATION'].set_value(0)
        if 'ITERATION_INDEX' in smooth_kernel.uniforms:
            smooth_kernel.uniforms['ITERATION_INDEX'].set_value(0)

        # Copy normals[] → smoothed_normals[] as starting point.
        buf_a = m.smoothed_normals_ssbo.buffer[0]
        copy_size = m.smoothed_normals_ssbo.size
        glBindBuffer(GL_COPY_READ_BUFFER, m.normal[0])
        glBindBuffer(GL_COPY_WRITE_BUFFER, buf_a)
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, copy_size)
        glBindBuffer(GL_COPY_READ_BUFFER, 0)
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

        # Also copy normals into smooth_prev for first iteration's momentum baseline.
        buf_prev = getattr(m, 'smooth_prev_ssbo', None)
        if buf_prev is not None:
            glBindBuffer(GL_COPY_READ_BUFFER, m.normal[0])
            glBindBuffer(GL_COPY_WRITE_BUFFER, buf_prev.buffer[0])
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, copy_size)
            glBindBuffer(GL_COPY_READ_BUFFER, 0)
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 26, buf_prev.buffer[0])

        buf_b = m.smooth_scratch_ssbo.buffer[0]
        for s_iter in range(si):
            if s_iter == si - 1:
                if 'LAST_ITERATION' in smooth_kernel.uniforms:
                    smooth_kernel.uniforms['LAST_ITERATION'].set_value(1)
            if 'ITERATION_INDEX' in smooth_kernel.uniforms:
                smooth_kernel.uniforms['ITERATION_INDEX'].set_value(s_iter)

            # Determine source/dest for this iteration's ping-pong.
            if s_iter % 2 == 0:
                src_buf, dst_buf = buf_a, buf_b
            else:
                src_buf, dst_buf = buf_b, buf_a

            # Momentum: bind previous iteration's source to binding 26.
            # For iter 0, smooth_prev already holds the initial normals.
            # For iter > 0, smooth_prev holds the source from the previous iteration.
            # (We snapshot the current source into smooth_prev AFTER dispatch.)

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, src_buf)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, dst_buf)

            smooth_kernel.bind()
            smooth_kernel.dispatch(workgroups)

            if s_iter < si - 1:
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            # Snapshot current source into smooth_prev for next iteration's momentum.
            if buf_prev is not None and s_iter < si - 1:
                glBindBuffer(GL_COPY_READ_BUFFER, src_buf)
                glBindBuffer(GL_COPY_WRITE_BUFFER, buf_prev.buffer[0])
                glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, copy_size)
                glBindBuffer(GL_COPY_READ_BUFFER, 0)
                glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        if si % 2 == 1:
            copy_size = m.smooth_scratch_ssbo.size
            glBindBuffer(GL_COPY_READ_BUFFER, buf_b)
            glBindBuffer(GL_COPY_WRITE_BUFFER, buf_a)
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, copy_size)
            glBindBuffer(GL_COPY_READ_BUFFER, 0)
            glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

        # Restore bindings.
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 24, buf_a)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m.deformed_position_buffer[0])
        return True

    def _run_skin_step(self, m, workgroups):
        """Run the LBS skinning kernel (single dispatch)."""
        bone_matrices_ssbo = getattr(m, 'bone_matrices_ssbo', None)
        bone_indices_ssbo = getattr(m, 'bone_indices_ssbo', None)
        bone_weights_ssbo = getattr(m, 'bone_weights_ssbo', None)

        if bone_matrices_ssbo is None or bone_indices_ssbo is None or bone_weights_ssbo is None:
            return False

        skin_kernel = self._get_skin_kernel()
        if skin_kernel is None or skin_kernel.error:
            return False

        if 'LOOP_COUNT' in skin_kernel.uniforms:
            skin_kernel.uniforms['LOOP_COUNT'].set_value(m.loop_count)
        if 'SKIN_POSITION' in skin_kernel.uniforms:
            skin_kernel.uniforms['SKIN_POSITION'].set_value(True)
        if 'SKIN_NORMAL' in skin_kernel.uniforms:
            skin_kernel.uniforms['SKIN_NORMAL'].set_value(True)

        skin_kernel.bind()
        skin_kernel.dispatch(workgroups)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        return True

    def run_compute_pass(self, scene_batches):
        """Fixed-order compute pipeline: Reset → Skin → Curvature → Smooth.

        Call this before draw_scene_pass(). Only dispatches once per frame.
        """
        if getattr(self, '_compute_dispatched_this_frame', False):
            return

        any_dispatched = False

        for material, meshes in scene_batches.items():
            for mesh_key in meshes.keys():
                m = mesh_key.mesh
                if not hasattr(m, 'deformed_position_buffer') or m.deformed_position_buffer is None:
                    continue

                compute_params = getattr(m, 'compute_params', None)
                if compute_params is None:
                    continue

                compute_skin = compute_params.get('compute_skin', False)
                compute_curvature = compute_params.get('compute_curvature', False)
                compute_smooth = compute_params.get('compute_smooth_normals', False)

                if not compute_skin and not compute_curvature and not compute_smooth:
                    continue

                workgroup_size = 64
                workgroups = math.ceil(m.loop_count / workgroup_size)

                # 1. Reset working buffers from rest buffers.
                self._reset_buffers(m)

                # Bind all SSBOs for the compute pipeline.
                self._bind_compute_ssbos(m)

                # 2. Skinning (transforms positions + normals in-place).
                if compute_skin:
                    self._run_skin_step(m, workgroups)
                    any_dispatched = True

                # 3. Curvature (reads posed normals + positions → curvature_ssbo).
                if compute_curvature:
                    self._run_curvature_step(m, workgroups)
                    any_dispatched = True

                # 4. Smooth normals (reads posed normals → smoothed_normals).
                if compute_smooth:
                    self._run_smooth_normals_step(m, compute_params, workgroups)
                    any_dispatched = True

        if any_dispatched:
            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)
            self._compute_dispatched_this_frame = True

    def draw_scene_pass(self, render_target, scene_batches, pass_name=None, default_shader=None, shader_resources={}, depth_test_function=GL_LEQUAL):
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(depth_test_function)
        glDepthMask(GL_TRUE)
        glDepthRange(0,1)

        render_target.bind()

        _double_sided = None

        for material in scene_batches.keys():
            shader = default_shader
            if material and pass_name in material.shader and material.shader[pass_name]:
                shader = material.shader[pass_name]

            for resource in shader_resources.values():
                resource.shader_callback(shader)

            shader.bind()

            _has_tess = getattr(shader, 'has_tessellation', False)
            if _has_tess:
                glPatchParameteri(GL_PATCH_VERTICES, 3)

            precomputed_tangents_uniform = shader.uniforms.get('PRECOMPUTED_TANGENTS')
            _precomputed_tangents = None
            _scale_group = None
            _color_is_srgb = None
            _ssbo_active = None
            _ssbo_active_high = None
            
            meshes = scene_batches[material]
            for mesh in meshes.keys():
                mesh.mesh.bind()

                double_sided = mesh.parameters['double_sided']
                if double_sided != _double_sided:
                    _double_sided = double_sided
                    if _double_sided:
                        glDisable(GL_CULL_FACE)
                    else:
                        glEnable(GL_CULL_FACE)
                        glCullFace(GL_BACK)
                
                srgb_mask = 0
                for _i, _v in enumerate(mesh.mesh.color_is_srgb):
                    if _v:
                        srgb_mask |= (1 << _i)
                if srgb_mask != _color_is_srgb:
                    if 'COLOR_IS_SRGB' in shader.uniforms:
                        shader.uniforms['COLOR_IS_SRGB'].bind(srgb_mask)
                        _color_is_srgb = srgb_mask

                if precomputed_tangents_uniform:
                    precomputed_tangents = mesh.parameters['precomputed_tangents']
                    if _precomputed_tangents != precomputed_tangents:
                        _precomputed_tangents = precomputed_tangents
                        precomputed_tangents_uniform.bind(precomputed_tangents)

                if hasattr(mesh.mesh, 'ssbo_list'):
                    ssbo_active = tuple(s is not None for s in mesh.mesh.ssbo_list[:4])
                    ssbo_active_high = tuple(s is not None for s in mesh.mesh.ssbo_list[4:8])
                    if ssbo_active != _ssbo_active:
                        if 'SSBO_ACTIVE' in shader.uniforms:
                            shader.uniforms['SSBO_ACTIVE'].bind(ssbo_active)
                        _ssbo_active = ssbo_active
                    if ssbo_active_high != _ssbo_active_high:
                        if 'SSBO_ACTIVE_HIGH' in shader.uniforms:
                            shader.uniforms['SSBO_ACTIVE_HIGH'].bind(ssbo_active_high)
                        _ssbo_active_high = ssbo_active_high
                    for i, ssbo in enumerate(mesh.mesh.ssbo_list):
                        block_name = f'SSBO_DATA_{i}'
                        if ssbo is not None and block_name in shader.storage_blocks:
                            ssbo.bind(shader.storage_blocks[block_name])

                curvature_ssbo = getattr(mesh.mesh, 'curvature_ssbo', None)
                has_curvature = curvature_ssbo is not None
                if has_curvature:
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 23, curvature_ssbo.buffer[0])
                if 'CURVATURE_SSBO_ACTIVE' in shader.uniforms:
                    shader.uniforms['CURVATURE_SSBO_ACTIVE'].bind(has_curvature)

                smoothed_normals_ssbo = getattr(mesh.mesh, 'smoothed_normals_ssbo', None)
                has_smoothed_normals = smoothed_normals_ssbo is not None
                if has_smoothed_normals:
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 24, smoothed_normals_ssbo.buffer[0])
                if 'SMOOTHED_NORMALS_SSBO_ACTIVE' in shader.uniforms:
                    shader.uniforms['SMOOTHED_NORMALS_SSBO_ACTIVE'].bind(has_smoothed_normals)

                # If the shader has tessellation stages, we MUST use GL_PATCHES
                # (GL_TRIANGLES is an invalid operation with TCS/TES present).
                # The mesh toggle controls tessellation levels via a uniform
                # instead of switching the primitive type.
                _primitive = GL_PATCHES if _has_tess else GL_TRIANGLES

                cp = getattr(mesh.mesh, 'compute_params', None)
                mesh_tess = cp.get('tessellation_enabled', False) if cp else False
                if 'TESS_DISABLED' in shader.uniforms:
                    shader.uniforms['TESS_DISABLED'].bind(0 if mesh_tess else 1)

                for scale_group, batches in meshes[mesh].items():
                    if scale_group != _scale_group:
                        _scale_group = scale_group
                        if scale_group == 'normal_scale':
                            glFrontFace(GL_CCW)
                            if 'MIRROR_SCALE' in shader.uniforms:
                                shader.uniforms['MIRROR_SCALE'].bind(False)
                        else:
                            glFrontFace(GL_CW)
                            if 'MIRROR_SCALE' in shader.uniforms:
                                shader.uniforms['MIRROR_SCALE'].bind(True)

                    for batch in batches:
                        batch['BATCH_MODELS'].bind(shader.uniform_blocks['BATCH_MODELS'])
                        batch['BATCH_IDS'].bind(shader.uniform_blocks['BATCH_IDS'])
                        glDrawElementsInstanced(_primitive, mesh.mesh.index_count, GL_UNSIGNED_INT, NULL, batch['instances_count'])


    def render(self, resolution, scene, is_final_render, is_new_frame):
        self.is_final_render = is_final_render
        if self.resolution != resolution:
            self.resolution = resolution
            self.setup_render_targets(resolution)
            self.sample_count = 0
        
        if is_new_frame:
            self.sample_count = 0
        
        if self.needs_more_samples() == False:
            return self.result
        
        self.common_buffer.load(scene, resolution)
        self.result = self.do_render(resolution, scene, is_final_render, is_new_frame)
        
        self.sample_count += 1

        return self.result

    def do_render(self, resolution, scene, is_final_render, is_new_frame):
        return {}
