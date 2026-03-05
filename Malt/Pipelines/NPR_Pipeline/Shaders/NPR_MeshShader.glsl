#include "NPR_Intellisense.glsl"
#include "Common.glsl"

/* META GLOBAL
    @meta: internal=true; 
*/
struct NPR_Settings
{
    // Global material settings. Can be modified in the material panel UI
    bool Receive_Shadow;
    bool Self_Shadow;
    bool Transparency;
    bool Transparency_Single_Layer;
    float Vertex_Displacement_Offset;
};

uniform NPR_Settings Settings = NPR_Settings(true, true, false, false, 0.001);

uniform ivec4 MATERIAL_LIGHT_GROUPS;

struct Vertex
{
    vec3 position;
    vec3 normal;
    uvec4 id;
    vec3 tangent;
    vec3 bitangent;
    vec2 uv[4];
    vec4 color[MAX_VERTEX_COLORS];
};

/*  META
    @id: label=ID;
    @opacity: label=Opacity Mask;
    @transparent_shadowmap_color: label=Transparent Shadow Color;
*/
struct PrePassOutput
{
    vec3 normal;
    uvec4 id;
    float opacity;
    vec3 transparent_shadowmap_color;
};

#ifdef VERTEX_SHADER

void COMMON_VERTEX_SHADER(inout Vertex V);

#ifndef CUSTOM_VERTEX_SHADER
void COMMON_VERTEX_SHADER(inout Vertex V){}
#endif

void TESSELLATION_SETTINGS(out int screen_space, out float dice_rate, out float density, inout vec3 displace_normal, out float displace_strength);

#ifndef CUSTOM_TESSELLATION
/* META
    @screen_space: subtype=ENUM(Off,On); default=0; doc=Use screen-space edge length for tessellation levels. Off uses density * max level instead.;
    @dice_rate: default=64.0; min=0.1; max=128.0; doc=Target edge length in pixels. Lower values produce more triangles and sharper silhouettes.;
    @density: default=1.0; subtype=Slider; min=0.0; max=1.0; doc=Subdivision density multiplier. Each 1 density = 1 subdivision level.;
    @displace_normal: default_initialization=NORMAL; doc=Normal used for displacement projection. Override with a smooth normal to fix sharp edge gaps.;
    @displace_strength: default=1.0; subtype=Slider; min=0.0; max=1.0; doc=Amount of Phong displacement (0=flat, 1=full curvature);
*/
void TESSELLATION_SETTINGS(out int screen_space, out float dice_rate, out float density, inout vec3 displace_normal, out float displace_strength){
    screen_space = 0;
    dice_rate = 64.0;
    density = 1.0;
    displace_strength = 1.0;
}
#endif

vec3 VERTEX_DISPLACEMENT_SHADER();

vec3 VERTEX_DISPLACEMENT_WRAPPER(Vertex V)
{
    Vertex real;
    real.position = POSITION;
    real.tangent = TANGENT;
    real.bitangent = BITANGENT;

    POSITION = V.position;
    TANGENT = V.tangent;
    BITANGENT = V.bitangent;

    vec3 result = VERTEX_DISPLACEMENT_SHADER();

    POSITION = real.position;
    TANGENT = real.tangent;
    BITANGENT = real.bitangent;

    return result;
}

#ifndef CUSTOM_VERTEX_DISPLACEMENT
vec3 VERTEX_DISPLACEMENT_SHADER(){ return vec3(0); }
#endif

#ifndef VERTEX_DISPLACEMENT_OFFSET
    #define VERTEX_DISPLACEMENT_OFFSET 0.1
#endif

#ifndef CUSTOM_MAIN
void main()
{
    DEFAULT_VERTEX_SHADER();

    Vertex V;
    V.position = POSITION;
    V.normal = NORMAL;
    V.tangent = TANGENT;
    V.bitangent = BITANGENT;
    V.uv = UV;
    V.color = COLOR;
    V.id = ID;
    
    COMMON_VERTEX_SHADER(V);

    POSITION = V.position;
    NORMAL = V.normal;
    TANGENT = V.tangent;
    BITANGENT = V.bitangent;
    UV = V.uv;
    COLOR = V.color;
    ID = V.id;

    #ifdef CUSTOM_VERTEX_DISPLACEMENT
    {
        vec3 displaced_position = POSITION + VERTEX_DISPLACEMENT_WRAPPER(V);
        
        if(!PRECOMPUTED_TANGENTS)
        {
            vec3 axis = vec3(0,0,1);
            axis = abs(dot(axis, NORMAL)) < 0.99 ? axis : vec3(1,0,0);
            vec3 tangent = normalize(cross(axis, NORMAL));
            TANGENT = tangent;
            BITANGENT = normalize(cross(NORMAL, tangent));
        }
        
        Vertex v = V;

        v.position = POSITION + TANGENT * Settings.Vertex_Displacement_Offset;
        vec3 displaced_tangent = v.position + VERTEX_DISPLACEMENT_WRAPPER(v);
        TANGENT = normalize(displaced_tangent - displaced_position);

        v.position = POSITION + BITANGENT * Settings.Vertex_Displacement_Offset;
        vec3 displaced_bitangent = v.position + VERTEX_DISPLACEMENT_WRAPPER(v);
        BITANGENT = normalize(displaced_bitangent - displaced_position);
        
        POSITION = displaced_position;
        NORMAL = normalize(cross(TANGENT, BITANGENT));
        
        if(!PRECOMPUTED_TANGENTS)
        {
            TANGENT = vec3(0);
            BITANGENT = vec3(0);
        }
    }
    #endif

    TESS_SCREEN_SPACE = 0;
    TESS_DICE_RATE = 64.0;
    TESS_DENSITY = 1.0;
    TESS_NORMAL = NORMAL;
    TESS_STRENGTH = 1.0;
    TESSELLATION_SETTINGS(TESS_SCREEN_SPACE, TESS_DICE_RATE, TESS_DENSITY, TESS_NORMAL, TESS_STRENGTH);

    VERTEX_SETUP_OUTPUT();
}
#endif //NDEF CUSTOM_MAIN

#endif //VERTEX_SHADER

#ifdef TESS_CONTROL_SHADER
#ifdef CUSTOM_TESSELLATION

uniform float TESS_MAX_LEVEL = 64.0;
uniform int TESS_DISABLED = 0;

vec2 _world_to_screen_px(vec3 world_pos) {
    vec4 clip = PROJECTION * CAMERA * vec4(world_pos, 1.0);
    vec2 ndc = clip.xy / clip.w;
    return (ndc * 0.5 + 0.5) * vec2(RESOLUTION);
}

void main()
{
    TCS_PASSTHROUGH();

    if (gl_InvocationID == 0)
    {
        if (TESS_DISABLED != 0)
        {
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelOuter[1] = 1.0;
            gl_TessLevelOuter[2] = 1.0;
            gl_TessLevelInner[0] = 1.0;
        }
        else if (IO_TESS_SCREEN_SPACE[0] != 0)
        {
            vec2 s0 = _world_to_screen_px(IO_POSITION[0]);
            vec2 s1 = _world_to_screen_px(IO_POSITION[1]);
            vec2 s2 = _world_to_screen_px(IO_POSITION[2]);

            float dice = max(0.1, (IO_TESS_DICE_RATE[0] + IO_TESS_DICE_RATE[1] + IO_TESS_DICE_RATE[2]) / 3.0);

            float px_01 = length(s0 - s1);
            float px_12 = length(s1 - s2);
            float px_20 = length(s2 - s0);

            float d0 = IO_TESS_DENSITY[0];
            float d1 = IO_TESS_DENSITY[1];
            float d2 = IO_TESS_DENSITY[2];

            gl_TessLevelOuter[0] = clamp((px_12 / dice) * mix(d1, d2, 0.5), 1.0, TESS_MAX_LEVEL);
            gl_TessLevelOuter[1] = clamp((px_20 / dice) * mix(d2, d0, 0.5), 1.0, TESS_MAX_LEVEL);
            gl_TessLevelOuter[2] = clamp((px_01 / dice) * mix(d0, d1, 0.5), 1.0, TESS_MAX_LEVEL);
            gl_TessLevelInner[0] = clamp(
                ((px_01 + px_12 + px_20) / (3.0 * dice)) * ((d0 + d1 + d2) / 3.0),
                1.0, TESS_MAX_LEVEL);
        }
        else
        {
            float avg_density = (IO_TESS_DENSITY[0] + IO_TESS_DENSITY[1] + IO_TESS_DENSITY[2]) / 3.0;
            float level = clamp(avg_density * 3.0, 1.0, TESS_MAX_LEVEL);
            gl_TessLevelOuter[0] = level;
            gl_TessLevelOuter[1] = level;
            gl_TessLevelOuter[2] = level;
            gl_TessLevelInner[0] = level;
        }
    }
}

#endif // CUSTOM_TESSELLATION
#endif // TESS_CONTROL_SHADER

#ifdef TESS_EVAL_SHADER
#ifdef CUSTOM_TESSELLATION

void main()
{
    TES_INTERPOLATE_ALL();

    // Phong tessellation displacement using per-vertex strength and normal.
    IO_POSITION = phong_tessellate(
        IO_POSITION,
        TCS_POSITION[0], TCS_POSITION[1], TCS_POSITION[2],
        TCS_TESS_NORMAL[0], TCS_TESS_NORMAL[1], TCS_TESS_NORMAL[2],
        IO_TESS_STRENGTH
    );

    gl_Position = PROJECTION * CAMERA * vec4(IO_POSITION, 1.0);
}

#endif // CUSTOM_TESSELLATION
#endif // TESS_EVAL_SHADER

#ifdef PIXEL_SHADER

#ifdef SHADOW_PASS
layout (location = 0) out uint OUT_ID;
layout (location = 1) out vec3 OUT_SHADOW_MULTIPLY_COLOR;
#endif //PRE_PASS

#ifdef PRE_PASS
uniform sampler2D IN_OPAQUE_DEPTH;
uniform sampler2D IN_TRANSPARENT_DEPTH;
uniform usampler2D IN_LAST_ID;

layout (location = 0) out vec4 OUT_NORMAL_DEPTH;
layout (location = 1) out uvec4 OUT_ID;
#endif //PRE_PASS

#ifdef MAIN_PASS
uniform sampler2D IN_NORMAL_DEPTH;
uniform usampler2D IN_ID;
#endif //MAIN_PASS

#ifndef CUSTOM_MAIN

#ifdef CUSTOM_PRE_PASS
void PRE_PASS_PIXEL_SHADER(inout PrePassOutput PPO);
#endif
#ifdef CUSTOM_DEPTH_OFFSET
void DEPTH_OFFSET(inout float depth_offset, inout bool offset_position);
#endif
#ifdef MAIN_PASS
void MAIN_PASS_PIXEL_SHADER();
#endif

void main()
{
    PIXEL_SETUP_INPUT();

    PrePassOutput PPO;
    PPO.normal = NORMAL;
    PPO.id = ID;
    PPO.opacity = 1;
    PPO.transparent_shadowmap_color = vec3(0);

    float depth = gl_FragCoord.z;
    vec3 offset_position = POSITION;

    // Discard pixel at the end of the shader, to avoid derivative glitches.
    bool discard_pixel = false;
    
    #ifdef CUSTOM_PRE_PASS
    {
        PRE_PASS_PIXEL_SHADER(PPO);

        if(PPO.opacity <= 0)
        {
            discard_pixel = true;
        }
        else if(!Settings.Transparency)
        {
            PPO.opacity = 1.0;
        }
    }
    #endif

    #ifdef CUSTOM_DEPTH_OFFSET
    {
        float depth_offset = 0;
        bool offset_position = false;
        DEPTH_OFFSET(depth_offset, offset_position);
        
        #ifdef SHADOW_PASS
        {
            if(!offset_position) depth_offset = 0;
        }
        #endif
        
        vec3 position = POSITION + view_direction() * depth_offset;

        depth = project_point_to_screen_coordinates(PROJECTION * CAMERA, position).z;
        gl_FragDepth = depth;

        if(offset_position) POSITION = position;
    }
    #endif

    #ifdef SHADOW_PASS
    {
        OUT_ID = PPO.id.r;

        if(Settings.Transparency)
        {
            float pass_through = hash(vec2(ID.x, SAMPLE_COUNT)).x;
            if(pass_through > PPO.opacity)
            {
                discard_pixel = true;
            }
            OUT_SHADOW_MULTIPLY_COLOR = PPO.transparent_shadowmap_color;
        }
    }
    #endif
    
    #ifdef PRE_PASS
    {
        if(Settings.Transparency)
        {
            float opaque_depth = texelFetch(IN_OPAQUE_DEPTH, ivec2(gl_FragCoord.xy), 0).x;
            float transparent_depth = texelFetch(IN_TRANSPARENT_DEPTH, ivec2(gl_FragCoord.xy), 0).x;
            
            if(depth >= opaque_depth || depth <= transparent_depth)
            {
                discard_pixel = true;
            }

            if(Settings.Transparency_Single_Layer)
            {
                if(PPO.id.r == texelFetch(IN_LAST_ID, ivec2(gl_FragCoord.xy), 0).x)
                {
                    discard_pixel = true;
                }
            }
        }

        OUT_NORMAL_DEPTH.xyz = PPO.normal;
        OUT_NORMAL_DEPTH.w = depth;
        OUT_ID = PPO.id;
    }
    #endif

    #ifdef MAIN_PASS
    {
        NORMAL = texelFetch(IN_NORMAL_DEPTH, ivec2(gl_FragCoord.xy), 0).xyz;
        ID = texelFetch(IN_ID, ivec2(gl_FragCoord.xy), 0);
        MAIN_PASS_PIXEL_SHADER();
    }
    #endif

    if(discard_pixel)
    {
        discard;
    }
}

#endif //NDEF CUSTOM_MAIN

#endif //PIXEL_SHADER

#if !defined(TESS_CONTROL_SHADER) && !defined(TESS_EVAL_SHADER)
#include "NPR_Pipeline/NPR_Mesh.glsl"
#include "NPR_Pipeline/NPR_Shading2.glsl"
#endif
