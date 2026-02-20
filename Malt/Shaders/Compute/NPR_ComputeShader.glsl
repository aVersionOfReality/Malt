// Standard header for Malt Compute node trees (.compute.glsl files).
// Included automatically by the Compute graph's default_global_scope.
//
// Binding point reservations (must not be used by individual compute shaders):
//   0–3  = SSBO_DATA_0–3        — loop-domain vec4[] (face corner color attributes)
//   4–7  = SSBO_VTX_DATA_0–3   — vertex-domain vec4[] (vertex attributes, indexed via corner_vert)
//   8    = rest_positions        — loop-indexed vec3[], read-only undeformed positions
//   9    = deformed_positions    — loop-indexed vec3[], read-write; current position in and out
//   10   = corner_vert           — loop-indexed int[], maps each loop to its unique vertex index
//   11   = normals               — loop-indexed vec3[], read-only current normals
//
// Every .compute.glsl node tree must implement:
//   void COMPUTE_SHADER(inout uint loop_index)
//
// loop_index is inout so the Blender node graph Output node has an input socket,
// allowing compute nodes to be chained from Input through to Output.
// The value is not modified; the inout qualifier is used only for execution-order wiring.
//
// NOTE: The conditional compilation guard is COMPUTE_STAGE (not COMPUTE_SHADER), because
// COMPUTE_SHADER is also the GLSL function name. Defining a macro with the same name as a
// function causes the C preprocessor to erase the function name before the GLSL compiler
// sees it. COMPUTE_STAGE avoids this conflict.
// The loop_index argument is the current invocation's loop/corner index.
// Use corner_vert[loop_index] to get the unique vertex index.
// Guard against out-of-range access using LOOP_COUNT.

#ifndef NPR_COMPUTE_SHADER_GLSL
#define NPR_COMPUTE_SHADER_GLSL

#ifdef COMPUTE_STAGE

layout(local_size_x = 64) in;

// Loop-domain color/attribute SSBOs (match render pass bindings 0–3)
layout(std430, binding = 0) buffer SSBO_DATA_0 { vec4 ssbo_data_0[]; };
layout(std430, binding = 1) buffer SSBO_DATA_1 { vec4 ssbo_data_1[]; };
layout(std430, binding = 2) buffer SSBO_DATA_2 { vec4 ssbo_data_2[]; };
layout(std430, binding = 3) buffer SSBO_DATA_3 { vec4 ssbo_data_3[]; };

// Vertex-domain SSBOs (match render pass bindings 4–7, indexed via corner_vert)
layout(std430, binding = 4) buffer SSBO_VTX_DATA_0 { vec4 ssbo_vtx_data_0[]; };
layout(std430, binding = 5) buffer SSBO_VTX_DATA_1 { vec4 ssbo_vtx_data_1[]; };
layout(std430, binding = 6) buffer SSBO_VTX_DATA_2 { vec4 ssbo_vtx_data_2[]; };
layout(std430, binding = 7) buffer SSBO_VTX_DATA_3 { vec4 ssbo_vtx_data_3[]; };

// Compute-specific SSBOs (bindings 8–11)
layout(std430, binding = 8) readonly buffer REST_POSITIONS {
    vec3 rest_positions[];
};

layout(std430, binding = 9) buffer DEFORMED_POSITIONS {
    vec3 deformed_positions[];
};

layout(std430, binding = 10) readonly buffer CORNER_VERT {
    int corner_vert[];
};

layout(std430, binding = 11) readonly buffer NORMALS {
    vec3 normals[];
};

uniform uint LOOP_COUNT = 0u;

void COMPUTE_SHADER(inout uint loop_index);

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= LOOP_COUNT) return;
    deformed_positions[idx] = rest_positions[idx];
    COMPUTE_SHADER(idx);
}

#else // not COMPUTE_STAGE — reflection/vertex/pixel context

// Forward declaration for non-compute contexts (e.g. vertex/pixel shader includes).
// The GLSLParser ignores bare forward declarations, so the stub body that makes
// COMPUTE_SHADER discoverable lives in _DEFAULT_COMPUTE_SHADER_SRC instead.
void COMPUTE_SHADER(inout uint loop_index);

#endif // COMPUTE_STAGE

#endif // NPR_COMPUTE_SHADER_GLSL
