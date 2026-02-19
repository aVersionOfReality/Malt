// Displace — compute shader node
//
// Displaces each loop position along its current normal, scaled by a per-vertex
// weight read from ssbo_vtx_data_0.r (the "malt_ssbo_vtx_0" FLOAT_COLOR attribute).
//
// Usage:
//   - Create a FLOAT_COLOR attribute named "malt_ssbo_vtx_0" on the mesh (e.g. via
//     Geometry Nodes). Set the R channel to the desired per-vertex weight (0.0–1.0).
//   - Call Displace(loop_index, strength) from within COMPUTE_SHADER().

#ifndef COMPUTE_SHADER
// Forward declarations for node graph reflection (non-compute context).
void Displace(uint loop_index, float strength);
#else

void Displace(uint loop_index, float strength)
{
    int vert = corner_vert[loop_index];
    float weight = ssbo_vtx_data_0[vert].r;
    vec3 n = normalize(normals[loop_index]);
    deformed_positions[loop_index] += n * weight * strength;
}

#endif // COMPUTE_SHADER
