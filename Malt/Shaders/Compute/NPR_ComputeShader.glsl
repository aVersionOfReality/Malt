// Standard header for Malt Compute node trees (.compute.glsl files).
// Included automatically by the Compute graph's default_global_scope.
//
// Binding point reservations (must not be used by individual compute shaders):
//   0–7  = SSBO_DATA_0–7        — loop-domain vec4[] (face corner attributes, all domains expanded)
//   8    = rest_positions        — loop-indexed vec4[], read-only original positions
//   9    = deformed_positions    — loop-indexed vec4[], read-write; written back to Position VBO
//   10   = corner_vert           — loop-indexed int[], maps each loop to its unique vertex index
//   11   = normals               — loop-indexed vec4[], read-write; written back to Normal VBO
//   12   = rest_normals          — loop-indexed vec4[], read-only original normals (never overwritten)
//
// Every .compute.glsl node tree must implement:
//   void COMPUTE_SHADER(inout vec3 position, inout vec3 normal)
//
// main() initialises position from rest_positions and normal from rest_normals on the
// first iteration (ITERATION == 0), and from deformed_positions/normals on subsequent
// iterations.  After COMPUTE_SHADER returns, the updated position/normal are written
// back to deformed_positions/normals.
// Unconnected inout sockets pass through the original values unchanged.
// Use gl_GlobalInvocationID.x inside COMPUTE_SHADER to get the current loop index.
//
// Multi-dispatch iteration:
//   COMPUTE_ITERATIONS controls how many times Python dispatches this shader per frame.
//   ITERATION is set by Python before each dispatch (0, 1, 2, ...).
//   On ITERATION 0, position/normal are initialised from the immutable rest buffers.
//   On ITERATION 1+, they are read from the previous dispatch's output buffers.
//
// NOTE: The conditional compilation guard is COMPUTE_STAGE (not COMPUTE_SHADER), because
// COMPUTE_SHADER is also the GLSL function name. Defining a macro with the same name as a
// function causes the C preprocessor to erase the function name before the GLSL compiler
// sees it. COMPUTE_STAGE avoids this conflict.
// Use gl_GlobalInvocationID.x to get the current loop/corner index inside COMPUTE_SHADER.
// Use corner_vert[gl_GlobalInvocationID.x] to get the unique vertex index.
// Out-of-range invocations are guarded by the LOOP_COUNT check in main().

#ifndef NPR_COMPUTE_SHADER_GLSL
#define NPR_COMPUTE_SHADER_GLSL

#ifdef COMPUTE_STAGE

layout(local_size_x = 64) in;

// Face-corner attribute SSBOs (bindings 0–7, all domains expanded to loop/face-corner)
layout(std430, binding = 0) buffer SSBO_DATA_0 { vec4 ssbo_data_0[]; };
layout(std430, binding = 1) buffer SSBO_DATA_1 { vec4 ssbo_data_1[]; };
layout(std430, binding = 2) buffer SSBO_DATA_2 { vec4 ssbo_data_2[]; };
layout(std430, binding = 3) buffer SSBO_DATA_3 { vec4 ssbo_data_3[]; };
layout(std430, binding = 4) buffer SSBO_DATA_4 { vec4 ssbo_data_4[]; };
layout(std430, binding = 5) buffer SSBO_DATA_5 { vec4 ssbo_data_5[]; };
layout(std430, binding = 6) buffer SSBO_DATA_6 { vec4 ssbo_data_6[]; };
layout(std430, binding = 7) buffer SSBO_DATA_7 { vec4 ssbo_data_7[]; };

uniform bvec4 SSBO_ACTIVE = bvec4(false);
uniform bvec4 SSBO_ACTIVE_HIGH = bvec4(false);

// Compute-specific SSBOs (bindings 8–11)
// vec4 is used for positions and normals (instead of vec3) because std430 gives vec3[]
// a 16-byte stride which would mismatch the tightly-packed 12-byte data uploaded from
// Python/CPU. vec4 has unambiguous 16-byte stride and the .xyz swizzle recovers the data.
layout(std430, binding = 8) readonly buffer REST_POSITIONS {
    vec4 rest_positions[];
};

layout(std430, binding = 9) buffer DEFORMED_POSITIONS {
    vec4 deformed_positions[];
};

layout(std430, binding = 10) readonly buffer CORNER_VERT {
    int corner_vert[];
};

layout(std430, binding = 11) buffer NORMALS {
    vec4 normals[];
};

// Read-only original normals — never overwritten by the compute shader.
// main() reads from here so the 'normal' inout parameter always starts as
// the original Blender mesh normal, regardless of what was written to
// normals[] on the previous frame.
layout(std430, binding = 12) readonly buffer REST_NORMALS {
    vec4 rest_normals[];
};

uniform uint LOOP_COUNT = 0u;
uniform uint COMPUTE_ITERATIONS = 1u;  // Total iterations — read by Python dispatch loop
uniform uint ITERATION = 0u;           // Current iteration — set by Python before each dispatch

// When the user's compute graph provides an implementation, CUSTOM_COMPUTE_SHADER is
// defined by generate_source() and the forward declaration below is used (the actual
// definition comes from the generated .compute.glsl file).  When there is no user
// graph (e.g. no output node), the fallback no-op definition is used so the shader
// still links correctly.
#ifdef CUSTOM_COMPUTE_SHADER
void COMPUTE_SHADER(inout vec3 position, inout vec3 normal);
#else
void COMPUTE_SHADER(inout vec3 position, inout vec3 normal) { }
#endif

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= LOOP_COUNT) return;
    vec3 position;
    vec3 normal;
    if (ITERATION == 0u) {
        // First iteration: start from the immutable rest-pose data.
        position = rest_positions[idx].xyz;
        normal   = rest_normals[idx].xyz;
    } else {
        // Subsequent iterations: build on the previous dispatch's output.
        position = deformed_positions[idx].xyz;
        normal   = normals[idx].xyz;
    }
    // COMPUTE_ITERATIONS == 0 means "skip": copy rest→deformed without
    // running the user's node graph.  Python still dispatches once so the
    // buffers are reset to rest pose.
    if (COMPUTE_ITERATIONS > 0u) {
        COMPUTE_SHADER(position, normal);
    }
    deformed_positions[idx] = vec4(position, 0.0);
    normals[idx]            = vec4(normal,   0.0);
}

#else // not COMPUTE_STAGE — reflection/vertex/pixel context

// Forward declaration for non-compute contexts (e.g. vertex/pixel shader includes).
// The GLSLParser ignores bare forward declarations, so the stub body that makes
// COMPUTE_SHADER discoverable lives in _DEFAULT_COMPUTE_SHADER_SRC instead.
void COMPUTE_SHADER(inout vec3 position, inout vec3 normal);

#endif // COMPUTE_STAGE

#endif // NPR_COMPUTE_SHADER_GLSL
