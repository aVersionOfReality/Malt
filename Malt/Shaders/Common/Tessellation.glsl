#ifndef TESSELLATION_GLSL
#define TESSELLATION_GLSL

// Tessellation interface declarations for TCS and TES stages.
// Included from Common.glsl only when TESS_CONTROL_SHADER or TESS_EVAL_SHADER is defined.

#define TESS_INTERPOLATE_3(a0, a1, a2) \
    (gl_TessCoord.x * (a0) + gl_TessCoord.y * (a1) + gl_TessCoord.z * (a2))

#ifdef TESS_CONTROL_SHADER

// TCS reads VS outputs as arrays, writes TCS_* arrays for TES.
in vec3 IO_POSITION[];
in vec3 IO_NORMAL[];
in vec3 IO_TANGENT[];
in vec3 IO_BITANGENT[];
in vec2 IO_UV[][4];
in vec4 IO_COLOR[][4];
flat in uvec4 IO_ID[];
flat in int IO_VERTEX_ID[];
in mat4 MODEL[];
in vec3 IO_BARYCENTRIC[];
in float IO_TESS_STRENGTH[];
in vec3 IO_TESS_NORMAL[];
in float IO_TESS_SCALE_BY_CURVATURE[];
flat in int IO_TESS_CURVATURE_MODE[];

layout(vertices = 3) out;

out vec3 TCS_POSITION[];
out vec3 TCS_NORMAL[];
out vec3 TCS_TANGENT[];
out vec3 TCS_BITANGENT[];
out vec2 TCS_UV[][4];
out vec4 TCS_COLOR[][4];
flat out uvec4 TCS_ID[];
flat out int TCS_VERTEX_ID[];
out mat4 TCS_MODEL[];
out vec3 TCS_BARYCENTRIC[];
out float TCS_TESS_STRENGTH[];
out vec3 TCS_TESS_NORMAL[];
out float TCS_TESS_SCALE_BY_CURVATURE[];
flat out int TCS_TESS_CURVATURE_MODE[];

// Pass all per-vertex attributes from VS to TES.
#define TCS_PASSTHROUGH() \
    TCS_POSITION[gl_InvocationID] = IO_POSITION[gl_InvocationID]; \
    TCS_NORMAL[gl_InvocationID] = IO_NORMAL[gl_InvocationID]; \
    TCS_TANGENT[gl_InvocationID] = IO_TANGENT[gl_InvocationID]; \
    TCS_BITANGENT[gl_InvocationID] = IO_BITANGENT[gl_InvocationID]; \
    TCS_UV[gl_InvocationID] = IO_UV[gl_InvocationID]; \
    TCS_COLOR[gl_InvocationID] = IO_COLOR[gl_InvocationID]; \
    TCS_ID[gl_InvocationID] = IO_ID[gl_InvocationID]; \
    TCS_VERTEX_ID[gl_InvocationID] = IO_VERTEX_ID[gl_InvocationID]; \
    TCS_MODEL[gl_InvocationID] = MODEL[gl_InvocationID]; \
    TCS_BARYCENTRIC[gl_InvocationID] = IO_BARYCENTRIC[gl_InvocationID]; \
    TCS_TESS_STRENGTH[gl_InvocationID] = IO_TESS_STRENGTH[gl_InvocationID]; \
    TCS_TESS_NORMAL[gl_InvocationID] = IO_TESS_NORMAL[gl_InvocationID]; \
    TCS_TESS_SCALE_BY_CURVATURE[gl_InvocationID] = IO_TESS_SCALE_BY_CURVATURE[gl_InvocationID]; \
    TCS_TESS_CURVATURE_MODE[gl_InvocationID] = IO_TESS_CURVATURE_MODE[gl_InvocationID];

#endif // TESS_CONTROL_SHADER


#ifdef TESS_EVAL_SHADER

layout(triangles, fractional_odd_spacing, ccw) in;

// TES reads TCS outputs as arrays.
in vec3 TCS_POSITION[];
in vec3 TCS_NORMAL[];
in vec3 TCS_TANGENT[];
in vec3 TCS_BITANGENT[];
in vec2 TCS_UV[][4];
in vec4 TCS_COLOR[][4];
flat in uvec4 TCS_ID[];
flat in int TCS_VERTEX_ID[];
in mat4 TCS_MODEL[];
in vec3 TCS_BARYCENTRIC[];
in float TCS_TESS_STRENGTH[];
in vec3 TCS_TESS_NORMAL[];
in float TCS_TESS_SCALE_BY_CURVATURE[];
flat in int TCS_TESS_CURVATURE_MODE[];

// TES writes scalar outputs to FS (same names VS normally writes).
out vec3 IO_POSITION;
out vec3 IO_NORMAL;
out vec3 IO_TANGENT;
out vec3 IO_BITANGENT;
out vec2 IO_UV[4];
out vec4 IO_COLOR[4];
flat out uvec4 IO_ID;
flat out int IO_VERTEX_ID;
out mat4 MODEL;
out vec3 IO_BARYCENTRIC;
out float IO_TESS_STRENGTH;
out vec3 IO_TESS_NORMAL;
out float IO_TESS_SCALE_BY_CURVATURE;
flat out int IO_TESS_CURVATURE_MODE;

// Interpolate all attributes using barycentric coordinates and write to IO outputs.
#define TES_INTERPOLATE_ALL() \
    IO_POSITION = TESS_INTERPOLATE_3(TCS_POSITION[0], TCS_POSITION[1], TCS_POSITION[2]); \
    IO_NORMAL = normalize(TESS_INTERPOLATE_3(TCS_NORMAL[0], TCS_NORMAL[1], TCS_NORMAL[2])); \
    IO_TANGENT = normalize(TESS_INTERPOLATE_3(TCS_TANGENT[0], TCS_TANGENT[1], TCS_TANGENT[2])); \
    IO_BITANGENT = normalize(TESS_INTERPOLATE_3(TCS_BITANGENT[0], TCS_BITANGENT[1], TCS_BITANGENT[2])); \
    for (int _i = 0; _i < 4; _i++) { \
        IO_UV[_i] = TESS_INTERPOLATE_3(TCS_UV[0][_i], TCS_UV[1][_i], TCS_UV[2][_i]); \
        IO_COLOR[_i] = TESS_INTERPOLATE_3(TCS_COLOR[0][_i], TCS_COLOR[1][_i], TCS_COLOR[2][_i]); \
    } \
    IO_ID = TCS_ID[0]; \
    IO_VERTEX_ID = TCS_VERTEX_ID[0]; \
    MODEL = TCS_MODEL[0]; \
    IO_BARYCENTRIC = gl_TessCoord; \
    IO_TESS_STRENGTH = TESS_INTERPOLATE_3(TCS_TESS_STRENGTH[0], TCS_TESS_STRENGTH[1], TCS_TESS_STRENGTH[2]); \
    IO_TESS_NORMAL = normalize(TESS_INTERPOLATE_3(TCS_TESS_NORMAL[0], TCS_TESS_NORMAL[1], TCS_TESS_NORMAL[2])); \
    IO_TESS_SCALE_BY_CURVATURE = TESS_INTERPOLATE_3(TCS_TESS_SCALE_BY_CURVATURE[0], TCS_TESS_SCALE_BY_CURVATURE[1], TCS_TESS_SCALE_BY_CURVATURE[2]); \
    IO_TESS_CURVATURE_MODE = TCS_TESS_CURVATURE_MODE[0];

// Estimate patch curvature from the three corner normals.
// Returns 0 for flat patches, 1 for strongly curved.
// mode 0 = minimum edge dot (conservative), mode 1 = average edge dot.
float curvature_from_normals(vec3 n0, vec3 n1, vec3 n2, int mode)
{
    float d01 = dot(n0, n1);
    float d12 = dot(n1, n2);
    float d20 = dot(n2, n0);
    if (mode == 1)
    {
        float avg = (d01 + d12 + d20) / 3.0;
        return clamp(1.0 - avg, 0.0, 1.0);
    }
    // mode 0: minimum edge dot (default)
    float minDot = min(min(d01, d12), d20);
    return clamp(1.0 - minDot, 0.0, 1.0);
}

// Phong tessellation: project interpolated position onto tangent planes at each
// original vertex, blend projections by barycentric weights.
vec3 phong_tessellate(vec3 linear_pos, vec3 p0, vec3 p1, vec3 p2,
                      vec3 n0, vec3 n1, vec3 n2, float strength)
{
    vec3 proj0 = linear_pos - dot(linear_pos - p0, n0) * n0;
    vec3 proj1 = linear_pos - dot(linear_pos - p1, n1) * n1;
    vec3 proj2 = linear_pos - dot(linear_pos - p2, n2) * n2;
    vec3 phong_pos = TESS_INTERPOLATE_3(proj0, proj1, proj2);
    return mix(linear_pos, phong_pos, strength);
}

#endif // TESS_EVAL_SHADER

#endif // TESSELLATION_GLSL
