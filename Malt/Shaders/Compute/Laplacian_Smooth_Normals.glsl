// Laplacian Smooth Normals graph node.
//
// This is a marker node — it does not generate smoothing code in the graph
// shader.  Instead, its parameters are read by Pipeline.py, which dispatches
// a dedicated smoothing kernel after the graph shader finishes.
//
// Usage: place this node in the compute graph and connect the normal
// pass-through.  Set smooth_iterations > 0 to enable smoothing.
//
// Group-based dispatch: fan group IDs from ssbo_data_2.x (hardwired).
// Quad diagonals: quad_mode 0=ignore, 1=include (uniform weight), 2=virtual triangles.

#ifndef COMPUTE_LAPLACIAN_SMOOTH_NORMALS_GLSL
#define COMPUTE_LAPLACIAN_SMOOTH_NORMALS_GLSL

/* META GLOBAL
    @meta: category=Compute;
*/

#ifndef COMPUTE_STAGE

/*  META
    @meta: label=Laplacian Smooth Normals; barrier=true; smooth_target=normal;
    @quad_mode: default=0; min=0; max=2;
    @cotangent_factor: default=1.0; min=0.0; max=1.0;
    @smooth_iterations: default=0; min=0; max=1000;
    @mix_factor: default=1.0; min=0.0; max=1.0;
    @application_strength: default=0.5; min=0.0; max=1.0;
    @contribution_strength: default=1.0; min=0.0; max=1.0;
    @own_normal_strength: default=1.0; min=0.0; max=1.0;
*/
void Laplacian_Smooth_Normals(inout vec3 normal, int quad_mode, float cotangent_factor, int smooth_iterations, float mix_factor, float application_strength, float contribution_strength, float own_normal_strength) {}

#else // COMPUTE_STAGE

void Laplacian_Smooth_Normals(inout vec3 normal, int quad_mode, float cotangent_factor, int smooth_iterations, float mix_factor, float application_strength, float contribution_strength, float own_normal_strength)
{
    // Write per-corner weight parameters to the smooth_weights buffer
    // (binding 18, declared in NPR_ComputeShader.glsl).  The smooth kernel
    // reads these per-corner values instead of uniforms.
    smooth_weights[gl_GlobalInvocationID.x] = vec4(
        application_strength, contribution_strength, own_normal_strength, mix_factor);
}

#endif // COMPUTE_STAGE

#endif // COMPUTE_LAPLACIAN_SMOOTH_NORMALS_GLSL
