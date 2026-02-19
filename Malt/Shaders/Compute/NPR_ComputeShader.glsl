// Standard header for Malt Compute node trees (.compute.glsl files).
// Included automatically by the Compute graph's default_global_scope.
//
// Binding point reservations (must not be used by individual compute shaders):
//   8  = REST_POSITIONS      — loop-indexed vec3[], read-only undeformed positions
//   9  = DEFORMED_POSITIONS  — loop-indexed vec3[], read-write deformed positions
//   10 = CORNER_VERT         — loop-indexed int[], maps each loop to its unique vertex index
//
// Every .compute.glsl node tree must implement:
//   void COMPUTE_SHADER(uint loop_index)
//
// The loop_index argument is the current invocation's loop/corner index.
// Use corner_vert[loop_index] to get the unique vertex index.
// Guard against out-of-range access using LOOP_COUNT.

#ifdef COMPUTE_SHADER

layout(local_size_x = 64) in;

layout(std430, binding = 8) readonly buffer REST_POSITIONS {
    vec3 rest_positions[];
};

layout(std430, binding = 9) buffer DEFORMED_POSITIONS {
    vec3 deformed_positions[];
};

layout(std430, binding = 10) readonly buffer CORNER_VERT {
    int corner_vert[];
};

uniform uint LOOP_COUNT = 0u;

void COMPUTE_SHADER(uint loop_index);

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= LOOP_COUNT) return;
    COMPUTE_SHADER(idx);
}

#else // not COMPUTE_SHADER — reflection/vertex/pixel context

// Forward declaration visible to the reflection tool so that the node editor
// can discover the entry-point signature without requiring COMPUTE_SHADER.
void COMPUTE_SHADER(uint loop_index);

#endif // COMPUTE_SHADER
