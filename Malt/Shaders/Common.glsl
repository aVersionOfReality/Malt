#ifndef COMMON_GLSL
#define COMMON_GLSL

#ifndef MAX_VERTEX_COLORS
    #define MAX_VERTEX_COLORS 9
#endif

#ifdef VERTEX_SHADER
#define vertex_out out
#else
#define vertex_out in
#endif

vec3 POSITION;
vec3 NORMAL;
vec3 TANGENT;
vec3 BITANGENT;
vec2 UV[4];
vec4 COLOR[MAX_VERTEX_COLORS];
uvec4 ID;

vertex_out mat4 MODEL;

layout(std140) uniform COMMON_UNIFORMS
{
    uniform mat4 CAMERA;
    uniform mat4 PROJECTION;
    uniform ivec2 RESOLUTION;
    uniform vec2 SAMPLE_OFFSET; //Sample offset is already baked into PROJECTION.
    uniform int SAMPLE_COUNT;
    uniform int FRAME;
    uniform float TIME;
};

uniform bool MIRROR_SCALE = false;
uniform bool PRECOMPUTED_TANGENTS = false;

uniform uint COLOR_IS_SRGB = 0u;

#ifndef MAX_BATCH_SIZE
    // Assume at least 64kb of UBO storage (d3d11 requirement) and max element size of mat4
    #define MAX_BATCH_SIZE 1000
#endif

layout(std140) uniform BATCH_MODELS
{
    uniform mat4 BATCH_MODEL[MAX_BATCH_SIZE];
};
layout(std140) uniform BATCH_IDS
{
    //Use uvec4 so the uints are tightly packed
    uniform uvec4 BATCH_ID[MAX_BATCH_SIZE/4+1];
};
#define BATCH_ID(index) BATCH_ID[(index)/4][(index)%4]

vertex_out vec3 IO_POSITION;
vertex_out vec3 IO_NORMAL;
vertex_out vec3 IO_TANGENT;
vertex_out vec3 IO_BITANGENT;
vertex_out vec2 IO_UV[4];
vertex_out vec4 IO_COLOR[MAX_VERTEX_COLORS];
flat vertex_out uvec4 IO_ID;

#include "Common/Color.glsl"
#include "Common/Hash.glsl"
#include "Common/Mapping.glsl"
#include "Common/Math.glsl"
#include "Common/Matrix.glsl"
#include "Common/Normal.glsl"
#include "Common/Quaternion.glsl"
#include "Common/Transform.glsl"

#ifdef VERTEX_SHADER

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec4 in_tangent;
layout (location = 3) in vec2 in_uv0;
layout (location = 4) in vec2 in_uv1;
layout (location = 5) in vec2 in_uv2;
layout (location = 6) in vec2 in_uv3;
layout (location = 7) in vec4 in_color0;
layout (location = 8) in vec4 in_color1;
layout (location = 9) in vec4 in_color2;
layout (location = 10) in vec4 in_color3;
layout (location = 11) in vec4 in_color4;
layout (location = 12) in vec4 in_color5;
layout (location = 13) in vec4 in_color6;
layout (location = 14) in vec4 in_color7;
layout (location = 15) in vec4 in_color8;

void VERTEX_SETUP_OUTPUT()
{
    gl_Position = PROJECTION * CAMERA * vec4(POSITION, 1);

    IO_POSITION = POSITION;
    IO_NORMAL = NORMAL;
    IO_TANGENT = TANGENT;
    IO_BITANGENT = BITANGENT;
    IO_UV = UV;
    IO_COLOR = COLOR;
    IO_ID = ID;
}

void DEFAULT_VERTEX_SHADER()
{
    MODEL = BATCH_MODEL[gl_InstanceID];
    ID = uvec4(BATCH_ID(gl_InstanceID),0,0,0);

    POSITION = transform_point(MODEL, in_position);
    NORMAL = transform_normal(MODEL, in_normal);

    if(PRECOMPUTED_TANGENTS)
    {
        TANGENT = transform_normal(MODEL, in_tangent.xyz);
        float mirror_scale = MIRROR_SCALE ? -1 : 1;
        BITANGENT = normalize(cross(NORMAL, TANGENT) * in_tangent.w) * mirror_scale;
    }

    UV[0]=in_uv0;
    UV[1]=in_uv1;
    UV[2]=in_uv2;
    UV[3]=in_uv3;

    COLOR[0]=in_color0;
    COLOR[1]=in_color1;
    COLOR[2]=in_color2;
    COLOR[3]=in_color3;
    COLOR[4]=in_color4;
    COLOR[5]=in_color5;
    COLOR[6]=in_color6;
    COLOR[7]=in_color7;
    COLOR[8]=in_color8;

    for(int i = 0; i < MAX_VERTEX_COLORS; i++)
    {
        if((COLOR_IS_SRGB & (1u << i)) != 0u)
        {
            COLOR[i].rgb = srgb_to_linear(COLOR[i].rgb);
        }
    }

    VERTEX_SETUP_OUTPUT();
}

void DEFAULT_SCREEN_VERTEX_SHADER()
{
    IO_POSITION = in_position;
    IO_UV[0] = in_position.xy * 0.5 + 0.5;
    gl_Position = vec4(in_position, 1);
}

#endif //VERTEX_SHADER

#ifdef PIXEL_SHADER

void PIXEL_SETUP_INPUT()
{
    POSITION = IO_POSITION;
    NORMAL = normalize(IO_NORMAL) * (gl_FrontFacing ? 1.0 : -1.0);
    TANGENT = IO_TANGENT;
    BITANGENT = IO_BITANGENT;
    UV = IO_UV;
    COLOR = IO_COLOR;
    ID = IO_ID;
}

#endif //PIXEL_SHADER

#endif //COMMON_GLSL
