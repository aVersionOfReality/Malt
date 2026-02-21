// Iterative compute nodes.
//
// These nodes are designed to test and use multi-dispatch iteration.
// When COMPUTE_ITERATIONS > 1 in the shader properties, the compute shader
// is dispatched multiple times per frame.  On iteration 0, position/normal
// start from the rest-pose data; on subsequent iterations, they build on
// the previous dispatch's output.
//
// Iterative_Displace adds an offset to the position each dispatch.
// With COMPUTE_ITERATIONS = N, the total displacement is N * offset.

#ifndef COMPUTE_ITERATIVE_GLSL
#define COMPUTE_ITERATIVE_GLSL

/* META GLOBAL
    @meta: category=Compute;
*/

#ifndef COMPUTE_STAGE

/*  META
    @meta: label=Iterative Displace; subcategory=Iterative;
    @offset: subtype=Vector;
*/
void Iterative_Displace(vec3 offset, inout vec3 position) {}

#else // COMPUTE_STAGE

void Iterative_Displace(vec3 offset, inout vec3 position)
{
    position += offset;
}

#endif // COMPUTE_STAGE

#endif // COMPUTE_ITERATIVE_GLSL
