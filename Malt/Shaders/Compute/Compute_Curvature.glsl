// Compute-graph node: per-vertex curvature via normal-difference method.
//
// Computes curvature as the weighted average of dot(n_i - n_j, edge_dir)
// across all edge-connected neighbors, using cotangent weights for geometric
// accuracy. The result is a signed scalar (positive = convex, negative = concave).
//
// Inputs default to the compute shader's built-in position/normal variables
// when disconnected, but can be overridden (e.g. with smooth normals from a
// barrier alias).
//
// The computed curvature is written to a dedicated SSBO (binding 23) so the
// mesh shader can read it independently via the Compute Curvature Input node.

#ifndef COMPUTE_CURVATURE_GLSL
#define COMPUTE_CURVATURE_GLSL

/* META GLOBAL
    @meta: category=Compute; subcategory=Curvature;
*/

#ifndef COMPUTE_STAGE

/*  META
    @meta: label=Compute Curvature;
    @normal: default_initialization=normal;
    @position: default_initialization=position;
    @result: label=Curvature;
*/
void Compute_Curvature(vec3 normal, vec3 position, out float result) {}

#else // COMPUTE_STAGE

void Compute_Curvature(vec3 normal, vec3 position, out float result)
{
    uint idx = gl_GlobalInvocationID.x;
    int vert = corner_vert[idx];

    // Gather normal-difference curvature from vertex neighbors.
    int adj_count = adjacency_count(vert);

    float curvature_sum = 0.0;
    float weight_sum = 0.0;

    for (int i = 0; i < adj_count; i++)
    {
        int neighbor_vert = adjacency_neighbor(vert, i);
        // Read neighbor data from a representative corner.
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

    result = (weight_sum > 1e-8) ? curvature_sum / weight_sum : 0.0;

    // Write to dedicated curvature SSBO for mesh shader access.
    curvature_ssbo[idx] = result;
}

#endif // COMPUTE_STAGE
#endif // COMPUTE_CURVATURE_GLSL
