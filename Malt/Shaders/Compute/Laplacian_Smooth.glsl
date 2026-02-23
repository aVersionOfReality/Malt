// Laplacian Smooth graph node.
//
// This is a marker node — it does not generate smoothing code in the graph
// shader.  Instead, its parameters (smooth_iterations, smooth_strength) are
// read by Pipeline.py, which dispatches a dedicated smoothing kernel after
// the graph shader finishes.
//
// Usage: place this node in the compute graph and connect the position
// pass-through.  Set smooth_iterations > 0 to enable smoothing.

#ifndef COMPUTE_LAPLACIAN_SMOOTH_GLSL
#define COMPUTE_LAPLACIAN_SMOOTH_GLSL

/* META GLOBAL
    @meta: category=Compute;
*/

#ifndef COMPUTE_STAGE

/*  META
    @meta: label=Laplacian Smooth; subcategory=Smooth; barrier=true;
    @smooth_iterations: default=0; min=0; max=1000;
    @smooth_strength: default=0.5; min=0.0; max=1.0;
*/
void Laplacian_Smooth(int smooth_iterations, float smooth_strength, inout vec3 position) {}

#else // COMPUTE_STAGE

void Laplacian_Smooth(int smooth_iterations, float smooth_strength, inout vec3 position)
{
    // No-op in the graph shader.  Pipeline.py reads smooth_iterations and
    // smooth_strength from compute_shader_parameters and dispatches the
    // Laplacian_Smooth_Kernel.glsl shader after this graph dispatch.
}

#endif // COMPUTE_STAGE

#endif // COMPUTE_LAPLACIAN_SMOOTH_GLSL
