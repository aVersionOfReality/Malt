"""Server-side cache and compilation for Compute shader node trees.

Mirrors Bridge/Material.py but compiles .compute.glsl files into a single
ComputeShader object instead of a {pass_name: Shader} dict.

The compiled ComputeShader is stored in COMPUTE_SHADERS[path] and retrieved
by ComputeShaderProxy.resolve() to assign mesh.compute_shader on each frame.
"""

COMPUTE_SHADERS = {}  # path → ComputeShader | None


class ComputeMaterial():
    """Compiled compute shader, analogous to Bridge.Material.Material."""

    def __init__(self, path, pipeline):
        self.path = path
        self.compiler_error = ''

        compiled = pipeline.compile_material(path)

        if isinstance(compiled, str):
            # compile_material returned an error string
            self.compiler_error = compiled
        else:
            shader = compiled.get('COMPUTE')
            if shader is None:
                self.compiler_error = 'No COMPUTE pass in compiled material'
            elif shader.error:
                self.compiler_error = shader.error

        global COMPUTE_SHADERS
        if self.compiler_error == '':
            COMPUTE_SHADERS[self.path] = compiled['COMPUTE']
        else:
            COMPUTE_SHADERS[self.path] = None


def get_compute_shader(path):
    """Return the compiled ComputeShader for *path*, or None if not compiled."""
    return COMPUTE_SHADERS.get(path, None)
