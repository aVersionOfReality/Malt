// Laplacian position smoothing for Delta Mush.
// Dispatched N times with ping-pong buffer swapping between binding 9 (src)
// and binding 15 (dst).  Python rebinds these each iteration.
//
// Simplified version of Laplacian_Smooth_Kernel.glsl:
//   - Operates on positions (not normals) — no normalize, no fan groups
//   - No momentum, no per-corner weight attributes
//   - Uses uniform or cotangent weights

#ifdef COMPUTE_STAGE

layout(local_size_x = 64) in;

// Ping-pong buffers — Python rebinds these each iteration.
layout(std430, binding = 9)  readonly buffer SMOOTH_SRC { vec4 smooth_src[]; };
layout(std430, binding = 15) writeonly buffer SMOOTH_DST { vec4 smooth_dst[]; };

layout(std430, binding = 10) readonly buffer CORNER_VERT { int corner_vert[]; };
layout(std430, binding = 13) readonly buffer ADJACENCY_DATA { int adjacency_data[]; };
layout(std430, binding = 14) readonly buffer VERT_CORNER_DATA { int vert_corner_data[]; };
layout(std430, binding = 16) readonly buffer COTANGENT_WEIGHTS { float cotangent_weights[]; };

uniform uint LOOP_COUNT = 0u;
uniform uint VERTEX_COUNT = 0u;
uniform float COTANGENT_FACTOR = 0.0;  // 0 = uniform weights, 1 = cotangent weights

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= LOOP_COUNT) return;

    int vert = corner_vert[idx];

    vec3 current_pos = smooth_src[idx].xyz;

    // Gather neighbor positions via adjacency CSR.
    int adj_start = adjacency_data[vert];
    int adj_end   = adjacency_data[vert + 1];
    int adj_count = adj_end - adj_start;

    if (adj_count == 0) {
        smooth_dst[idx] = vec4(current_pos, 0.0);
        return;
    }

    int adj_base = int(VERTEX_COUNT) + 1;
    int vc_base  = int(VERTEX_COUNT) + 1;
    vec3 weighted_sum = vec3(0.0);
    float total_weight = 0.0;

    for (int i = adj_start; i < adj_end; i++) {
        int neighbor_vert = adjacency_data[adj_base + i];
        // Read from first corner of neighbor vertex.
        int neighbor_corner = vert_corner_data[vc_base + vert_corner_data[neighbor_vert]];

        float w = mix(1.0, cotangent_weights[i], COTANGENT_FACTOR);
        weighted_sum += w * smooth_src[neighbor_corner].xyz;
        total_weight += w;
    }

    vec3 smoothed;
    if (total_weight > 0.0) {
        smoothed = weighted_sum / total_weight;
    } else {
        smoothed = current_pos;
    }

    // Scatter to all corners of this vertex.
    int my_start = vert_corner_data[vert];
    int my_end   = vert_corner_data[vert + 1];
    for (int i = my_start; i < my_end; i++) {
        int corner = vert_corner_data[vc_base + i];
        smooth_dst[corner] = vec4(smoothed, 0.0);
    }
}

#endif // COMPUTE_STAGE
