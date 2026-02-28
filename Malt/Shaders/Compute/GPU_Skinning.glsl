// GPU Skinning (Linear Blend Skinning) barrier node.
//
// Place this node in a compute graph to enable GPU skeletal skinning.
// The node creates a segment boundary: pre-skin operations run first,
// then the LBS kernel transforms rest-pose positions/normals to posed
// using bone matrices uploaded each frame from Blender's armature.
//
// Requires:
//   - An armature modifier on the object (can be disabled)
//   - Vertex groups matching bone names
//   - A compute node tree assigned to the mesh
//
// The armature modifier identifies which armature is associated with
// the object. Its state (enabled/disabled) is irrelevant — bone data
// is extracted from the armature's pose bones directly.

#ifndef COMPUTE_GPU_SKINNING_GLSL
#define COMPUTE_GPU_SKINNING_GLSL

/* META GLOBAL
    @meta: category=Compute;
*/

#ifndef COMPUTE_STAGE

/*  META
    @meta: label=GPU Skinning; barrier=true; skin=true;
*/
void GPU_Skinning(inout vec3 position, inout vec3 normal) {}

#else // COMPUTE_STAGE

void GPU_Skinning(inout vec3 position, inout vec3 normal)
{
    // v1: no per-corner parameters.
    // The LBS kernel applies full skinning to all vertices.
}

#endif // COMPUTE_STAGE

#endif // COMPUTE_GPU_SKINNING_GLSL
