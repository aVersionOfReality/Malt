// Standalone Laplacian smoothing compute shader (not a graph node).
// Dispatched by Pipeline.py after the graph shader, N times with ping-pong
// buffer swapping between each dispatch.
//
// Python swaps which buffer is bound at binding 9 (src) and 15 (dst) each
// iteration.  The shader always reads from binding 9 and writes to binding 15.
//
// Each invocation gathers neighbor values via the adjacency CSR, computes a
// weighted sum, normalizes it, then blends with the current corner's scaled
// value using APPLICATION_STRENGTH.  The result is normalized and scattered.
//
// Per-corner weight parameters (binding 18, smooth_weights vec4[]):
//   .x = application_strength: per-iteration blend (0 = no change, 1 = full)
//   .y = contribution_strength: scales neighbor values during gather
//   .z = own_normal_strength: scales a corner's normal everywhere it appears
//   Written by the barrier node function body, read by this kernel per-corner.
//   In gather, neighbor's own_normal_strength and contribution_strength are used.
//   In self blend, current corner's application_strength and own_normal_strength.
//
//   .w = mix_factor: post-all-iterations blend between rest normal and smoothed
//     result (0 = rest, 1 = fully smoothed).  Only applied on the last
//     iteration (LAST_ITERATION == 1).
//
// Cotangent weights (binding 16) are pre-computed CPU-side from the mesh
// geometry.  They are indexed in parallel with the adjacency indices.
//
// Per-edge metadata (binding 17, always present):
//   For each adjacency entry i, edge_meta stores 3 ints:
//     [i*3]   = owning fan ID at the source vertex (-1 if sharp or no groups)
//     [i*3+1] = corner index at the neighbor vertex to read from (-1 if sharp)
//     [i*3+2] = flags (bit 0 = IS_DIAGONAL: quad 0-2 diagonal, not a real edge)
//
// Group-based smoothing (GROUPS_ENABLED == 1):
//   Fan group IDs (binding 2, ssbo_data_2.x) control scatter: only write
//   to corners at the same vertex that share the current corner's fan group.
//   Gather: compare edge's owning fan with current corner's fan group.
//   If they match, read from the precomputed corner index.  If not, skip.
//
// Quad diagonal handling (QUAD_MODE):
//   Mode 0: skip diagonal edges (default, original mesh edges only)
//   Mode 1: include diagonals with uniform weight 1.0
//   Mode 2: include diagonals with cotangent weight (virtual triangulation)

#ifdef COMPUTE_STAGE

layout(local_size_x = 64) in;

// Ping-pong buffers — Python rebinds these each iteration.
layout(std430, binding = 9)  readonly buffer SMOOTH_SRC { vec4 smooth_src[]; };
layout(std430, binding = 15) writeonly buffer SMOOTH_DST { vec4 smooth_dst[]; };

layout(std430, binding = 10) readonly buffer CORNER_VERT { int corner_vert[]; };
layout(std430, binding = 13) readonly buffer ADJACENCY_DATA { int adjacency_data[]; };
layout(std430, binding = 14) readonly buffer VERT_CORNER_DATA { int vert_corner_data[]; };
layout(std430, binding = 16) readonly buffer COTANGENT_WEIGHTS { float cotangent_weights[]; };

// Per-edge metadata: triplets of [owning_fan_id, gather_corner_index, flags].
// Built CPU-side from fan group IDs + mesh topology.
layout(std430, binding = 17) readonly buffer EDGE_META { int edge_meta[]; };

// Fan group IDs for scatter filtering.  ssbo_data_2.x = fan group (float → int).
layout(std430, binding = 2)  readonly buffer SSBO_DATA_2 { vec4 ssbo_data_2[]; };

// Rest normals for post-iteration mix (binding 12, already bound by Pipeline).
layout(std430, binding = 12) readonly buffer REST_NORMALS { vec4 rest_normals_buf[]; };

// Per-corner weight parameters (written by barrier node in graph shader).
layout(std430, binding = 18) readonly buffer SMOOTH_WEIGHTS { vec4 smooth_weights[]; };

uniform uint LOOP_COUNT = 0u;
uniform uint VERTEX_COUNT = 0u;
uniform float COTANGENT_FACTOR = 1.0;         // 0 = uniform weights, 1 = cotangent weights
uniform int LAST_ITERATION = 0;              // 1 on the final iteration — triggers per-corner mix with rest normal
uniform int GROUPS_ENABLED = 0;               // 0 = legacy per-vertex, 1 = group-aware
uniform int QUAD_MODE = 0;                    // 0 = skip diagonals, 1 = include (uniform), 2 = include (cotangent)

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= LOOP_COUNT) return;

    int vert = corner_vert[idx];

    // Read current value from source buffer
    vec3 current_val = smooth_src[idx].xyz;

    // Read fan group ID for this corner (if group-based smoothing is enabled).
    int my_fan_group = -1;
    if (GROUPS_ENABLED > 0) {
        my_fan_group = int(ssbo_data_2[idx].x);
    }

    // --- Gather neighbor values via adjacency CSR with cotangent weights ---
    int adj_start = adjacency_data[vert];
    int adj_end   = adjacency_data[vert + 1];
    int adj_count = adj_end - adj_start;

    if (adj_count == 0) {
        smooth_dst[idx] = vec4(current_val, 0.0);
        return;
    }

    int adj_base = int(VERTEX_COUNT) + 1;
    int vc_base  = int(VERTEX_COUNT) + 1;
    vec3 raw_sum = vec3(0.0);
    bool any_gathered = false;
    for (int i = adj_start; i < adj_end; i++) {
        int owning_fan    = edge_meta[i * 3];
        int gather_corner = edge_meta[i * 3 + 1];
        int flags         = edge_meta[i * 3 + 2];
        bool is_diagonal  = (flags & 1) != 0;

        // Skip diagonals in mode 0.
        if (is_diagonal && QUAD_MODE == 0) continue;

        int neighbor_vert = adjacency_data[adj_base + i];
        int neighbor_corner;

        if (GROUPS_ENABLED > 0) {
            if (owning_fan != my_fan_group) continue;  // not our fan's edge, or sharp
            neighbor_corner = gather_corner;
        } else {
            // Legacy: pick first corner of neighbor vertex.
            neighbor_corner = vert_corner_data[vc_base + vert_corner_data[neighbor_vert]];
        }

        // Weight: mode 1 diagonals use uniform weight 1.0, all others use cotangent blend.
        float w;
        if (is_diagonal && QUAD_MODE == 1) {
            w = 1.0;
        } else {
            w = mix(1.0, cotangent_weights[i], COTANGENT_FACTOR);
        }

        // Per-corner neighbor weights: own_normal_strength and contribution_strength
        // from the NEIGHBOR corner's smooth_weights entry.
        float n_own  = smooth_weights[neighbor_corner].z;
        float n_cont = smooth_weights[neighbor_corner].y;
        raw_sum += w * smooth_src[neighbor_corner].xyz * n_own * n_cont;
        any_gathered = true;
    }

    // Per-corner weights for the current corner.
    float my_app = smooth_weights[idx].x;
    float my_own = smooth_weights[idx].z;

    vec3 smoothed;
    if (any_gathered && length(raw_sum) > 1e-10) {
        vec3 avg = normalize(raw_sum);
        // Scale current value by own_normal_strength (shrinks anchor when < 1).
        vec3 my_scaled = current_val * my_own;
        // Blend and normalize (my_scaled may not be unit length).
        smoothed = normalize(my_scaled + my_app * (avg - my_scaled));
    } else {
        // No neighbors gathered or degenerate — pass through
        smoothed = current_val;
    }

    // --- Post-iteration mix: blend smoothed result with rest normal ---
    // Only applied on the last iteration.  Per-corner mix_factor is in
    // smooth_weights[idx].w (written by barrier node function body).
    if (LAST_ITERATION != 0) {
        float my_mix = smooth_weights[idx].w;
        if (my_mix < 1.0) {
            vec3 rest_val = rest_normals_buf[idx].xyz;
            smoothed = normalize(mix(rest_val, smoothed, my_mix));
        }
    }

    // --- Scatter to corners of this vertex (filtered by fan group if enabled) ---
    int my_start = vert_corner_data[vert];
    int my_end   = vert_corner_data[vert + 1];
    for (int i = my_start; i < my_end; i++) {
        int corner = vert_corner_data[vc_base + i];
        if (my_fan_group >= 0 && int(ssbo_data_2[corner].x) != my_fan_group) {
            continue;  // Different fan group — skip
        }
        smooth_dst[corner] = vec4(smoothed, 0.0);
    }
}

#endif // COMPUTE_STAGE
