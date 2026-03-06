// Delta Mush apply kernel: reconstruct tangent frame from smoothed deformed
// geometry, transform precomputed rest deltas back to world space, and write
// corrected positions to deformed_positions (binding 9).
//
// Dispatched once after all smoothing iterations complete.
//
// The tangent frame construction must match the CPU precomputation exactly:
//   N = normal from normals buffer (post-skin)
//   T = normalize(smoothed_neighbor0_pos - smoothed_pos), orthogonalized against N
//   B = cross(N, T)
//   T = cross(B, N)

#ifdef COMPUTE_STAGE

layout(local_size_x = 64) in;

// Corrected positions written here (deformed_positions).
layout(std430, binding = 9) buffer DEFORMED_POSITIONS { vec4 deformed_positions[]; };

// Smoothed deformed positions (result of ping-pong smoothing).
layout(std430, binding = 15) readonly buffer SMOOTHED_DEFORMED { vec4 smoothed_deformed[]; };

// Per-vertex tangent-space deltas, precomputed at mesh load.
layout(std430, binding = 27) readonly buffer REST_DELTAS { vec4 rest_deltas[]; };

layout(std430, binding = 10) readonly buffer CORNER_VERT { int corner_vert[]; };
layout(std430, binding = 11) readonly buffer NORMALS { vec4 normals_buf[]; };
layout(std430, binding = 13) readonly buffer ADJACENCY_DATA { int adjacency_data[]; };
layout(std430, binding = 14) readonly buffer VERT_CORNER_DATA { int vert_corner_data[]; };

// Original skinned positions (saved before smoothing).
layout(std430, binding = 26) readonly buffer SKINNED_POSITIONS { vec4 skinned_positions[]; };

// Per-corner weight parameters — avr_malt_data2.
// .z (B channel) = delta mush factor when USE_ATTR_FACTOR is enabled.
// .w (A channel) = delta mush scale when USE_ATTR_SCALE is enabled.
layout(std430, binding = 25) readonly buffer SMOOTH_WEIGHTS_2 { vec4 smooth_weights_2[]; };

uniform uint LOOP_COUNT = 0u;
uniform uint VERTEX_COUNT = 0u;
uniform float FACTOR = 1.0;       // 0 = unsmoothed, 1 = fully smoothed positions
uniform int USE_ATTR_FACTOR = 0;  // 1 = read factor from avr_malt_data2.B per corner
uniform float SCALE = 1.0;        // 0 = raw skinned, 1 = fully corrected
uniform int USE_ATTR_SCALE = 0;   // 1 = read scale from avr_malt_data2.A per corner

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= LOOP_COUNT) return;

    int vert = corner_vert[idx];

    vec3 fully_smoothed_pos = smoothed_deformed[idx].xyz;
    vec3 tangent_delta = rest_deltas[vert].xyz;

    // Build tangent frame from smoothed deformed geometry.
    vec3 N = normalize(normals_buf[idx].xyz);

    // Compute the corrected position (before factor/scale blend).
    int adj_start = adjacency_data[vert];
    int adj_end   = adjacency_data[vert + 1];

    vec3 corrected_pos;
    if (adj_start < adj_end && dot(N, N) > 0.5) {
        int adj_base = int(VERTEX_COUNT) + 1;
        int vc_base  = int(VERTEX_COUNT) + 1;
        int neighbor_vert = adjacency_data[adj_base + adj_start];
        int neighbor_corner = vert_corner_data[vc_base + vert_corner_data[neighbor_vert]];
        vec3 neighbor_pos = smoothed_deformed[neighbor_corner].xyz;

        vec3 T = normalize(neighbor_pos - fully_smoothed_pos);
        // Gram-Schmidt orthogonalize T against N.
        T = T - dot(T, N) * N;
        float T_len = length(T);

        if (T_len > 1e-6) {
            T = T / T_len;
            vec3 B = cross(N, T);
            T = cross(B, N);

            // Transform delta from tangent space to world space.
            vec3 world_delta = tangent_delta.x * T + tangent_delta.y * B + tangent_delta.z * N;
            corrected_pos = fully_smoothed_pos + world_delta;
        } else {
            corrected_pos = fully_smoothed_pos + tangent_delta;
        }
    } else {
        corrected_pos = fully_smoothed_pos + tangent_delta;
    }

    // Scatter to all corners of this vertex, applying per-corner factor and scale.
    int vc_base = int(VERTEX_COUNT) + 1;
    int my_start = vert_corner_data[vert];
    int my_end   = vert_corner_data[vert + 1];
    for (int i = my_start; i < my_end; i++) {
        int corner = vert_corner_data[vc_base + i];
        vec4 w = smooth_weights_2[corner];
        float f = (USE_ATTR_FACTOR != 0) ? w.z : FACTOR;
        float s = (USE_ATTR_SCALE != 0) ? w.w : SCALE;
        vec3 skinned_pos = skinned_positions[corner].xyz;
        // Factor: blend between skinned and fully corrected position.
        vec3 factored_pos = mix(skinned_pos, corrected_pos, f);
        // Scale: blend between skinned and factored result.
        deformed_positions[corner] = vec4(mix(skinned_pos, factored_pos, s), 0.0);
    }
}

#endif // COMPUTE_STAGE
