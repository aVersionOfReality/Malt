// Standalone Linear Blend Skinning (LBS) compute kernel.
// Dispatched once by Pipeline.py after the pre-skin graph segment.
//
// Reads positions/normals from the deformed buffers (written by the
// preceding graph segment) and applies weighted bone transforms,
// writing the skinned result back to the same buffers.
//
// SKIN_POSITION / SKIN_NORMAL uniforms control which buffers are written.
// Set by Pipeline._run_skin_step() based on which outputs the user connected.
//
// Bone indices and weights are per-vertex (indexed via corner_vert).
// Each vertex has up to 4 bone influences (ivec4 + vec4).
//
// Normal transform uses mat3(bone_matrix), which is correct for
// rotation + translation + uniform scale — matching Blender's default
// LBS armature modifier behavior.

#ifdef COMPUTE_STAGE

layout(local_size_x = 64) in;

// Position and normal buffers — read the previous segment's output,
// write the skinned result back for the next segment to consume.
layout(std430, binding = 9)  buffer DEFORMED_POSITIONS { vec4 deformed_positions[]; };
layout(std430, binding = 11) buffer NORMALS            { vec4 normals_buf[]; };

// Loop-to-vertex mapping.
layout(std430, binding = 10) readonly buffer CORNER_VERT { int corner_vert[]; };

// Bone data (matrices updated per-frame, indices/weights static per mesh load).
layout(std430, binding = 19) readonly buffer BONE_MATRICES { mat4 bone_matrices[]; };
layout(std430, binding = 20) readonly buffer BONE_INDICES  { ivec4 bone_indices[]; };
layout(std430, binding = 21) readonly buffer BONE_WEIGHTS  { vec4  bone_weights[]; };

uniform uint LOOP_COUNT = 0u;
// Per-dispatch flags: which outputs to write (set by Pipeline from dispatch plan).
uniform bool SKIN_POSITION = true;
uniform bool SKIN_NORMAL = true;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= LOOP_COUNT) return;

    int vert = corner_vert[idx];

    ivec4 bi = bone_indices[vert];
    vec4  bw = bone_weights[vert];

    if (SKIN_POSITION) {
        vec3 in_pos = deformed_positions[idx].xyz;
        vec3 skinned_pos = vec3(0.0);
        for (int i = 0; i < 4; i++) {
            float w = bw[i];
            if (w <= 0.0) continue;
            skinned_pos += w * (bone_matrices[bi[i]] * vec4(in_pos, 1.0)).xyz;
        }
        deformed_positions[idx] = vec4(skinned_pos, 0.0);
    }

    if (SKIN_NORMAL) {
        vec3 in_nrm = normals_buf[idx].xyz;
        vec3 skinned_nrm = vec3(0.0);
        for (int i = 0; i < 4; i++) {
            float w = bw[i];
            if (w <= 0.0) continue;
            skinned_nrm += w * (mat3(bone_matrices[bi[i]]) * in_nrm);
        }
        normals_buf[idx] = vec4(normalize(skinned_nrm), 0.0);
    }
}

#endif // COMPUTE_STAGE
