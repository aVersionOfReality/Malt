// Compute-graph utility nodes.
//
// Current_Position reads from the live deformed_positions buffer (last frame's compute output).
// Current_Normal  reads from the normals buffer (last frame's compute output).
// Both use gl_GlobalInvocationID.x directly.
//
// The Compute Input node's Position/Normal outputs are always the ORIGINAL mesh values
// (from rest_positions / rest_normals), because main() in NPR_ComputeShader.glsl
// initialises the inout parameters from those immutable buffers.
// Use Current_Position / Current_Normal only when you intentionally want feedback from
// the previous frame (e.g. iterative simulation).
//
// Separate_Color and XYZ are pure-math helpers for unpacking SSBO/color data.

#ifndef COMPUTE_MATH_GLSL
#define COMPUTE_MATH_GLSL

/* META GLOBAL
    @meta: category=Compute;
*/

/*  META
    @meta: label=Separate Color; subcategory=Color;
    @color: subtype=Color;
*/
void Separate_Color(vec4 color, out float r, out float g, out float b, out float a)
{
    r = color.r;
    g = color.g;
    b = color.b;
    a = color.a;
}

/*  META
    @meta: label=XYZ; subcategory=Vector;
    @vector: subtype=Vector;
*/
void XYZ(vec4 vector, out vec3 xyz)
{
    xyz = vector.xyz;
}

/*  META
    @meta: label=Separate Color Vec3; subcategory=Color;
    @color: subtype=Color;
*/
void Separate_Color_Vec3(vec4 color, out vec3 rgb, out float a)
{
    rgb = color.rgb;
    a   = color.a;
}

#ifndef COMPUTE_STAGE

/*  META
    @meta: label=Current Position; subcategory=Position;
    @position: subtype=Vector;
*/
void Current_Position(out vec3 position) {}

/*  META
    @meta: label=Current Normal; subcategory=Position;
    @normal: subtype=Vector;
*/
void Current_Normal(out vec3 normal) {}

#else // COMPUTE_STAGE

void Current_Position(out vec3 position)
{
    position = deformed_positions[gl_GlobalInvocationID.x].xyz;
}

void Current_Normal(out vec3 normal)
{
    normal = normals[gl_GlobalInvocationID.x].xyz;
}

#endif // COMPUTE_STAGE

#endif // COMPUTE_MATH_GLSL
