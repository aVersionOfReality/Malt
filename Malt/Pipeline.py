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
    
    def load_mesh(self, position, indices, normal, tangent=None, uvs=[], colors=[], ssbo_colors=[None]*8, vertex_count=0, loop_count=0, rest_positions=None, rest_normals=None, corner_vert=None):
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
            max_vertex_colors = 4
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
    
    def run_compute_pass(self, scene_batches):
        """Dispatch compute shaders for all meshes that have one assigned.

        Call this before draw_scene_pass().

        Supports multi-dispatch iteration: if the shader declares a
        COMPUTE_ITERATIONS uniform with value > 1, the shader is dispatched
        that many times with GL_SHADER_STORAGE_BARRIER_BIT between each
        dispatch.  The ITERATION uniform (0, 1, 2, ...) is set before each
        dispatch so main() can choose whether to read from rest buffers
        (iteration 0) or from the previous dispatch's output (iteration 1+).

        The memory barrier issued at the end ensures the deformed position
        buffer writes are visible to the subsequent vertex shader reads via
        in_position.
        """
        any_dispatched = False

        for material, meshes in scene_batches.items():
            for mesh_key in meshes.keys():
                m = mesh_key.mesh
                if not hasattr(m, 'deformed_position_buffer') or m.deformed_position_buffer is None:
                    continue

                compute_shader = getattr(m, 'compute_shader', None)
                if compute_shader is None:
                    continue

                if compute_shader.error:
                    continue

                # Cache all uniform values BEFORE bind() so that bind()'s bulk upload
                # sends the current frame's values rather than last frame's.
                if 'LOOP_COUNT' in compute_shader.uniforms:
                    compute_shader.uniforms['LOOP_COUNT'].set_value(m.loop_count)
                if 'TIME' in compute_shader.uniforms:
                    compute_shader.uniforms['TIME'].set_value(self.common_buffer.data.TIME)

                compute_params = getattr(m, 'compute_shader_parameters', {})
                for name, value in compute_params.items():
                    if name in compute_shader.uniforms:
                        compute_shader.uniforms[name].set_value(value)

                # Bind mesh attribute SSBOs at standard binding points.
                # Bindings 0–7 mirror the render pass (loop-domain vec4 face-corner attributes).
                # Bindings 8–12 are compute-specific:
                #   8  = rest_positions  (loop-indexed vec4[], read-only original positions)
                #   9  = deformed_positions (loop-indexed vec4[], read-write compute output)
                #   10 = corner_vert     (loop-indexed int[], loop→vertex mapping)
                #   11 = normals         (loop-indexed vec4[], read-write compute output)
                #   12 = rest_normals    (loop-indexed vec4[], read-only original normals)

                # Face-corner attribute SSBOs (bindings 0–7)
                ssbo_active = tuple(s is not None for s in m.ssbo_list[:4])
                ssbo_active_high = tuple(s is not None for s in m.ssbo_list[4:8])
                if 'SSBO_ACTIVE' in compute_shader.uniforms:
                    compute_shader.uniforms['SSBO_ACTIVE'].set_value(ssbo_active)
                if 'SSBO_ACTIVE_HIGH' in compute_shader.uniforms:
                    compute_shader.uniforms['SSBO_ACTIVE_HIGH'].set_value(ssbo_active_high)

                # Bind SSBOs once -- these are context-level state and persist
                # across bind() calls within the iteration loop below.
                for i, ssbo in enumerate(m.ssbo_list):
                    if ssbo is not None:
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, ssbo.buffer[0])

                # Compute-specific SSBOs
                if m.rest_position_ssbo is not None:
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, m.rest_position_ssbo.buffer[0])
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, m.deformed_position_buffer[0])
                if m.corner_vert_ssbo is not None:
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, m.corner_vert_ssbo.buffer[0])
                if m.normal is not None:
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, m.normal[0])
                if m.rest_normals_ssbo is not None:
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, m.rest_normals_ssbo.buffer[0])

                # Determine iteration count.
                # Node parameters are always registered in the node tree's
                # malt_parameters (and thus in compute_params) regardless
                # of whether the generated GLSL uses a literal or a uniform.
                # Scanning compute_params is therefore reliable; scanning
                # shader uniforms is not (the value may be baked as a literal).
                iterations = 1
                for pname, pval in compute_params.items():
                    if 'compute_iterations' in pname.lower():
                        val = pval
                        if hasattr(val, '__len__'):
                            val = val[0]
                        if isinstance(val, (int, float)) and val >= 0:
                            iterations = int(val)
                        break

                # Tell the shader how many iterations were requested so
                # main() can skip COMPUTE_SHADER when iterations == 0.
                if 'COMPUTE_ITERATIONS' in compute_shader.uniforms:
                    compute_shader.uniforms['COMPUTE_ITERATIONS'].set_value(iterations)

                workgroup_size = 64
                workgroups = math.ceil(m.loop_count / workgroup_size)

                if iterations == 0:
                    # Dispatch once to copy rest→deformed (reset to rest pose).
                    # main() reads rest buffers (ITERATION=0) and skips
                    # COMPUTE_SHADER (COMPUTE_ITERATIONS=0).
                    if 'ITERATION' in compute_shader.uniforms:
                        compute_shader.uniforms['ITERATION'].set_value(0)
                    compute_shader.bind()
                    compute_shader.dispatch(workgroups)
                    any_dispatched = True
                else:
                    for iteration in range(iterations):
                        # ITERATION is the only uniform that changes per dispatch.
                        # bind() re-uploads all cached uniform values (including the
                        # ITERATION we just set and the unchanging ones from above).
                        if 'ITERATION' in compute_shader.uniforms:
                            compute_shader.uniforms['ITERATION'].set_value(iteration)

                        compute_shader.bind()
                        compute_shader.dispatch(workgroups)
                        any_dispatched = True

                        # Issue a barrier between iterations so the next dispatch sees
                        # this dispatch's writes to deformed_positions/normals.
                        # The final barrier (for vertex shader visibility) is issued
                        # after the outer loop, not here.
                        if iteration < iterations - 1:
                            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        if any_dispatched:
            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)

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
                
                color_is_srgb = tuple(mesh.mesh.color_is_srgb)
                if color_is_srgb != _color_is_srgb :
                    if 'COLOR_IS_SRGB' in shader.uniforms:
                        shader.uniforms['COLOR_IS_SRGB'].bind(color_is_srgb)
                        _color_is_srgb = color_is_srgb

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
                        glDrawElementsInstanced(GL_TRIANGLES, mesh.mesh.index_count, GL_UNSIGNED_INT, NULL, batch['instances_count'])


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
