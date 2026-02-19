// Sine wave vertex displacement compute shader.
// Reads loop-indexed rest positions from binding 8, writes displaced positions
// to the deformed position buffer at binding 9.
// Dispatched with ceil(loop_count / 64) workgroups before draw_scene_pass.

#ifdef COMPUTE_SHADER

layout(local_size_x = 64) in;

layout(std430, binding = 8) readonly buffer REST_POSITIONS {
    vec3 rest_positions[];
};

layout(std430, binding = 9) writeonly buffer DEFORMED_POSITIONS {
    vec3 deformed_positions[];
};

uniform float TIME = 0.0;

void main() {
    uint idx = gl_GlobalInvocationID.x;

    // Guard against over-dispatch from ceil rounding.
    if (idx >= rest_positions.length()) return;

    vec3 pos = rest_positions[idx];
    pos.z += sin(pos.x * 4.0 + TIME) * 0.2;
    deformed_positions[idx] = pos;
}

#endif // COMPUTE_SHADER
