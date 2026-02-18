"""
Compute shader readback test — run from Blender's scripting workspace.

Instructions:
  1. Open Blender with the Malt addon loaded and active.
  2. Switch to the Scripting workspace.
  3. Open this file (or paste it into a new text block).
  4. Click Run Script.

What it does:
  - Compiles a trivial compute shader that writes gl_GlobalInvocationID.x
    as a float into an output SSBO.
  - Allocates an SSBO for 256 floats, dispatches 4 workgroups of 64 threads.
  - Reads the buffer back to CPU with glGetBufferSubData.
  - Prints PASS or FAIL with details to the system console.

Open the system console (Window > Toggle System Console) to see the output.
"""

import ctypes
import sys

from Malt.GL.GL import *
from Malt.GL.Shader import SSBO
from Malt.GL.ComputeShader import ComputeShader

# ---- Compute shader source ----
# Writes float(gl_GlobalInvocationID.x) into result[gl_GlobalInvocationID.x].
# Note: no #version directive here -- compile_compute_program() prepends it,
# exactly as compile_gl_program() does for vertex/fragment shaders.
COMPUTE_SOURCE = """\
layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer RESULT_BLOCK {
    float result[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    result[idx] = float(idx);
}
"""

# ---- Test parameters ----
N = 256
WORKGROUP_SIZE = 64

# ---- Compile ----
shader = ComputeShader(COMPUTE_SOURCE)

if shader.error:
    print("FAIL: shader compilation error:")
    print(shader.error)
    raise Exception("Compute shader compilation failed -- see output above")

print("OK  : compute shader compiled successfully")

# ---- Allocate output SSBO ----
FloatArray = ctypes.c_float * N
zeros = FloatArray(*([0.0] * N))

ssbo = SSBO()
ssbo.load_data(zeros, GL_DYNAMIC_READ)

# ---- Bind and dispatch ----
shader.bind()
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo.buffer[0])

workgroups = N // WORKGROUP_SIZE
shader.dispatch(workgroups)

# Ensure compute writes are visible to the CPU readback below.
glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT)

# ---- Read back ----
result = FloatArray()
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo.buffer[0])
glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ctypes.sizeof(result), result)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

# ---- Verify ----
errors = []
for i in range(N):
    if result[i] != float(i):
        errors.append(f"  result[{i}] = {result[i]}, expected {float(i)}")

if errors:
    print(f"FAIL: {len(errors)} value(s) wrong:")
    for e in errors[:10]:
        print(e)
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")
else:
    print(f"OK  : all {N} values correct (result[i] == float(i))")
    print("PASS: compute shader dispatch and SSBO readback working correctly")
