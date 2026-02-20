// Displace — compute shader node
//
// Displaces each loop position along its current normal, scaled by a per-vertex
// weight read from ssbo_vtx_data_0.r (the "malt_ssbo_vtx_0" FLOAT_COLOR attribute).
//
// Usage:
//   - Create a FLOAT_COLOR attribute named "malt_ssbo_vtx_0" on the mesh (e.g. via
//     Geometry Nodes). Set the R channel to the desired per-vertex weight (0.0–1.0).
//   - Connect: COMPUTE SHADER Input.Loop Index → Displace.Loop Index → COMPUTE SHADER Output.Loop Index

/* META GLOBAL
    @meta: category=Compute;
*/

#ifndef COMPUTE_STAGE
// Stub body for the reflection pass. GLSLParser only reports functions that have
// a body, so a bare forward declaration would make Displace invisible as a node.
// loop_index is inout so it appears as both an input and output socket on the node,
// enabling execution flow wiring to the COMPUTE SHADER Output node.
void Displace(inout uint loop_index, float strength) { }
#else

void Displace(inout uint loop_index, float strength)
{
    int vert = corner_vert[loop_index];
    float weight = ssbo_vtx_data_0[vert].r;
    vec3 n = normalize(normals[loop_index]);
    deformed_positions[loop_index] += n * weight * strength;
}

#endif // COMPUTE_STAGE
