import os

from Malt.GL.GL import *
from Malt.GL.Shader import (
    shader_preprocessor,
    reflect_program_uniforms,
    reflect_program_uniform_blocks,
    reflect_program_storage_blocks,
    fix_line_directive_paths,
    hasGLExtension,
    buffer_to_string,
)
from Malt.Utils import LOG


def compile_compute_program(source):
    """Compile a single GL_COMPUTE_SHADER stage into a linked program.

    Mirrors compile_gl_program() in Shader.py but for compute-only programs.
    Uses the same binary cache folder; the cache key is a SHA1 of the
    preprocessed compute source (with #line directives stripped, same as the
    vertex+fragment path).

    Returns:
        (program, error_string)  — error_string is '' on success.
    """
    import textwrap

    # Prepend the version/extension header, identical to finalize_source() in
    # compile_gl_program so that cached binaries use a consistent format.
    bindless_setup = '#define OPTIONALLY_BINDLESS\n'
    if hasGLExtension('GL_ARB_bindless_texture'):
        bindless_setup = (
            '#extension GL_ARB_bindless_texture : enable\n'
            '#define OPTIONALLY_BINDLESS layout(bindless_sampler)\n'
        )

    source = textwrap.dedent(f'''
        #version 450 core
        #extension GL_ARB_shading_language_include : enable
        {bindless_setup}
        #line 1 "src"
    ''') + source
    source = fix_line_directive_paths(source)

    # ---- Binary cache (same folder as vertex/fragment programs) ----
    import hashlib, tempfile
    hash_src = ''.join(
        line for line in source.splitlines(True)
        if not line.startswith('#line')
    )
    shader_hash = hashlib.sha1(hash_src.encode()).hexdigest()
    cache_folder = os.path.join(tempfile.gettempdir(), 'MALT_SHADERS_CACHE')
    os.makedirs(cache_folder, exist_ok=True)
    cache_path  = os.path.join(cache_folder, shader_hash + '.bin')
    format_path = os.path.join(cache_folder, shader_hash + '.fmt')

    cache, fmt = None, None
    if os.path.exists(cache_path) and os.path.exists(format_path):
        from pathlib import Path
        Path(cache_path).touch()
        with open(cache_path, 'rb') as f:
            raw = f.read()
            cache = (GLubyte * len(raw)).from_buffer_copy(raw)
        Path(format_path).touch()
        with open(format_path, 'rb') as f:
            fmt = GLuint.from_buffer_copy(f.read())

    status  = gl_buffer(GL_INT, 1)
    program = glCreateProgram()
    error   = ''

    # Try loading from cache first
    if cache:
        try:
            glProgramBinary(program, fmt, cache, len(cache))
            glGetProgramiv(program, GL_LINK_STATUS, status)
            if status[0] != GL_FALSE:
                return (program, error)
        except Exception:
            LOG.error(f'Failed to load cached compute program binary: {shader_hash} ({fmt})')

    # ---- Compile from source ----
    shader = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(shader, source)
    glCompileShader(shader)

    glGetShaderiv(shader, GL_COMPILE_STATUS, status)
    if status[0] == GL_FALSE:
        info_log = glGetShaderInfoLog(shader)
        error += 'COMPUTE SHADER COMPILER ERROR :\n' + buffer_to_string(info_log)

    glAttachShader(program, shader)
    glLinkProgram(program)
    glDeleteShader(shader)

    glGetProgramiv(program, GL_LINK_STATUS, status)
    if status[0] == GL_FALSE:
        info_log = glGetProgramInfoLog(program)
        error += 'COMPUTE SHADER LINKER ERROR :\n' + buffer_to_string(info_log)
    else:
        # Save binary to cache
        length = gl_buffer(GL_INT, 1)
        glGetProgramiv(program, GL_PROGRAM_BINARY_LENGTH, length)
        fmt_buf = gl_buffer(GL_UNSIGNED_INT, 1)
        bin_buf = gl_buffer(GL_UNSIGNED_BYTE, length[0])
        glGetProgramBinary(program, length[0], NULL, fmt_buf, bin_buf)
        with open(cache_path, 'wb') as f:
            f.write(bin_buf)
        with open(format_path, 'wb') as f:
            f.write(fmt_buf)

    return (program, error)


class ComputeShader:
    """A compiled OpenGL compute shader program.

    Exposes the same ``uniforms`` and ``storage_blocks`` interface as
    ``Shader``, plus a ``dispatch(x, y, z)`` method.  There is no
    ``copy()`` because compute shaders are not per-material; a single
    instance is shared.
    """

    def __init__(self, source):
        if source:
            self.source  = source
            self.program, self.error = compile_compute_program(source)
        else:
            self.source  = source
            self.program = None
            self.error   = 'NO SOURCE'

        self.uniforms       = {}
        self.textures       = {}
        self.uniform_blocks = {}
        self.storage_blocks = {}

        if self.error == '':
            self.error = None
            self.uniforms = reflect_program_uniforms(self.program)
            texture_index = 0
            for name, uniform in self.uniforms.items():
                if uniform.is_sampler():
                    uniform.set_value(texture_index)
                    texture_index += 1
                    self.textures[name] = None
            self.uniform_blocks = reflect_program_uniform_blocks(self.program)
            self.storage_blocks = reflect_program_storage_blocks(self.program)
        elif self.error != 'NO SOURCE':
            LOG.error(self.error)

    def bind(self):
        """Bind the compute program and upload all current uniform values."""
        glUseProgram(self.program)
        for uniform in self.uniforms.values():
            uniform.bind()
        for name, texture in self.textures.items():
            if name not in self.uniforms:
                LOG.debug(f'Compute texture uniform {name} not found')
                continue
            uniform = self.uniforms[name]
            glActiveTexture(GL_TEXTURE0 + uniform.value[0])
            if texture:
                if hasattr(texture, 'bind'):
                    texture.bind()
                else:
                    glBindTexture(uniform.texture_type(), texture)
            else:
                glBindTexture(uniform.texture_type(), 0)

    def dispatch(self, x, y=1, z=1):
        """Dispatch the compute shader with the given workgroup counts.

        Call ``bind()`` (and bind any SSBOs) before calling this.
        Issue ``glMemoryBarrier`` after this if the results will be read
        by a subsequent vertex/fragment draw or another compute pass.
        """
        glDispatchCompute(x, y, z)
