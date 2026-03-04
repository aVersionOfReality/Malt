// Standalone per-corner curvature compute kernel.
// Dispatched by Pipeline._run_curvature_step().
//
// Computes curvature as the weighted average of dot(n_i - n_j, edge_dir)
// across all edge-connected neighbors, using cotangent weights for geometric
// accuracy. The result is a signed scalar (positive = convex, negative = concave).
//
// Reads posed positions (binding 9) and posed normals (binding 11) — after
// skinning, if skinning ran before curvature. Writes per-corner curvature
// to curvature_ssbo (binding 23), which the mesh shader reads via the
// Compute_Curvature_Input node.

#ifdef COMPUTE_STAGE

layout(local_size_x = 64) in;

// Position and normal buffers (post-skinning if skin ran first).
layout(std430, binding = 9)  readonly buffer DEFORMED_POSITIONS { vec4 deformed_positions[]; };
layout(std430, binding = 11) readonly buffer NORMALS            { vec4 normals[]; };

// Loop-to-vertex mapping.
layout(std430, binding = 10) readonly buffer CORNER_VERT { int corner_vert[]; };

// Adjacency CSR data.
// adjacency_data: [offsets (VERTEX_COUNT+1 ints) | neighbor vertex indices]
layout(std430, binding = 13) readonly buffer ADJACENCY_DATA   { int adjacency_data[]; };
// vert_corner_data: [offsets (VERTEX_COUNT+1 ints) | corner/loop indices]
layout(std430, binding = 14) readonly buffer VERT_CORNER_DATA { int vert_corner_data[]; };
// Per-edge cotangent weights (parallel to adjacency indices).
layout(std430, binding = 16) readonly buffer COTANGENT_WEIGHTS { float cotangent_weights[]; };

// Output: per-corner curvature.
layout(std430, binding = 23) writeonly buffer CURVATURE_DATA { float curvature_ssbo[]; };

uniform uint LOOP_COUNT = 0u;
uniform uint VERTEX_COUNT = 0u;

// ── Adjacency helper functions ──

int adjacency_count(int v) {
    return adjacency_data[v + 1] - adjacency_data[v];
}

int adjacency_neighbor(int v, int i) {
    return adjacency_data[int(VERTEX_COUNT) + 1 + adjacency_data[v] + i];
}

int vert_corner_index(int v, int i) {
    return vert_corner_data[int(VERTEX_COUNT) + 1 + vert_corner_data[v] + i];
}

// ── Main ──

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= LOOP_COUNT) return;

    int vert = corner_vert[idx];
    vec3 position = deformed_positions[idx].xyz;
    vec3 normal = normals[idx].xyz;

    int adj_count = adjacency_count(vert);
    float curvature_sum = 0.0;
    float weight_sum = 0.0;

    for (int i = 0; i < adj_count; i++)
    {
        int neighbor_vert = adjacency_neighbor(vert, i);
        int n_corner = vert_corner_index(neighbor_vert, 0);

        vec3 n_pos = deformed_positions[n_corner].xyz;
        vec3 n_nrm = normals[n_corner].xyz;

        vec3 edge = n_pos - position;
        float edge_len = length(edge);
        if (edge_len < 1e-8) continue;

        vec3 edge_dir = edge / edge_len;
        float w = cotangent_weights[adjacency_data[vert] + i];
        curvature_sum += w * dot(normal - n_nrm, edge_dir);
        weight_sum += w;
    }

    curvature_ssbo[idx] = (weight_sum > 1e-8) ? curvature_sum / weight_sum : 0.0;
}

#endif // COMPUTE_STAGE
