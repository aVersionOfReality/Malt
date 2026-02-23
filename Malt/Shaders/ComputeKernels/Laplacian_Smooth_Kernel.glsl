// Standalone Laplacian smoothing compute shader (not a graph node).
// Dispatched by Pipeline.py after the graph shader, N times with ping-pong
// buffer swapping between each dispatch.
//
// Python swaps which buffer is bound at binding 9 (src) and 15 (dst) each
// iteration.  The shader always reads from binding 9 and writes to binding 15.
//
// Position mode: all corners of the same vertex converge to the same smoothed
// position.  Each invocation gathers neighbor positions via the adjacency CSR,
// computes the Laplacian average, blends with current position by STRENGTH,
// then scatters the result to all corners of its vertex.

#ifdef COMPUTE_STAGE

layout(local_size_x = 64) in;

// Ping-pong position buffers — Python rebinds these each iteration.
layout(std430, binding = 9)  readonly buffer SMOOTH_SRC { vec4 smooth_src[]; };
layout(std430, binding = 15) writeonly buffer SMOOTH_DST { vec4 smooth_dst[]; };

layout(std430, binding = 10) readonly buffer CORNER_VERT { int corner_vert[]; };
layout(std430, binding = 13) readonly buffer ADJACENCY_DATA { int adjacency_data[]; };
layout(std430, binding = 14) readonly buffer VERT_CORNER_DATA { int vert_corner_data[]; };

uniform uint LOOP_COUNT = 0u;
uniform uint VERTEX_COUNT = 0u;
uniform float SMOOTH_STRENGTH = 0.5;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= LOOP_COUNT) return;

    int vert = corner_vert[idx];

    // Read current position from source buffer
    vec3 current_pos = smooth_src[idx].xyz;

    // --- Gather neighbor positions via adjacency CSR ---
    int adj_start = adjacency_data[vert];
    int adj_end   = adjacency_data[vert + 1];
    int adj_count = adj_end - adj_start;

    if (adj_count == 0) {
        smooth_dst[idx] = vec4(current_pos, 0.0);
        return;
    }

    int adj_base = int(VERTEX_COUNT) + 1;
    vec3 neighbor_avg = vec3(0.0);
    for (int i = adj_start; i < adj_end; i++) {
        int neighbor_vert = adjacency_data[adj_base + i];
        // Pick one representative corner for this neighbor vertex
        int neighbor_corner = vert_corner_data[int(VERTEX_COUNT) + 1 + vert_corner_data[neighbor_vert]];
        neighbor_avg += smooth_src[neighbor_corner].xyz;
    }
    neighbor_avg /= float(adj_count);

    // Blend toward neighbor average
    vec3 smoothed = mix(current_pos, neighbor_avg, SMOOTH_STRENGTH);

    // --- Scatter to ALL corners of this vertex (position mode) ---
    int vc_base = int(VERTEX_COUNT) + 1;
    int my_start = vert_corner_data[vert];
    int my_end   = vert_corner_data[vert + 1];
    for (int i = my_start; i < my_end; i++) {
        smooth_dst[vert_corner_data[vc_base + i]] = vec4(smoothed, 0.0);
    }
}

#endif // COMPUTE_STAGE
