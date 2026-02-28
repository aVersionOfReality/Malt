#include "stdio.h"
#include "mikktspace.h"

#ifdef _WIN32
#define EXPORT extern "C" __declspec( dllexport )
#else
#define EXPORT extern "C" __attribute__ ((visibility ("default")))
#endif

EXPORT void retrieve_mesh_data(
  float* in_positions,
  int* in_loop_verts, int loop_count,
  int* in_loop_tris,
  int* in_loop_tri_polys, int loop_tri_count,
  int* in_mat_indices,
  float* out_positions, unsigned int** out_indices, unsigned int* out_index_lengths)
{
  for(int i = 0; i < loop_count; i++)
  {
    out_positions[i*3+0] = in_positions[in_loop_verts[i]*3+0];
    out_positions[i*3+1] = in_positions[in_loop_verts[i]*3+1];
    out_positions[i*3+2] = in_positions[in_loop_verts[i]*3+2];
  }

  unsigned int* mat_i = out_index_lengths;

  for(int i = 0; i < loop_tri_count; i++)
  {
    int mat = in_mat_indices ? in_mat_indices[in_loop_tri_polys[i]] : 0;
    out_indices[mat][mat_i[mat]++] = in_loop_tris[i*3+0];
    out_indices[mat][mat_i[mat]++] = in_loop_tris[i*3+1];
    out_indices[mat][mat_i[mat]++] = in_loop_tris[i*3+2];
  }
}

EXPORT bool mesh_tangents(
  int* in_indices, int index_len,
  float* in_positions, float* in_normals, float* in_uvs,
  float* out_tangents)
{
  struct MData
  {
    int* indices;
    int index_len;
    float* positions;
    float* normals;
    float* uvs;
    float* tangents;
  };

  MData data = {
    in_indices,
    index_len,
    in_positions,
    in_normals,
    in_uvs,
    out_tangents,
  };

  SMikkTSpaceInterface mti = {0};
  mti.m_getNumFaces = [](const SMikkTSpaceContext * pContext){
    return ((MData*)pContext->m_pUserData)->index_len / 3;
  };

	mti.m_getNumVerticesOfFace = [](const SMikkTSpaceContext * pContext, const int iFace){
    return 3;
  };

	mti.m_getPosition = [](const SMikkTSpaceContext * pContext, float fvPosOut[], const int iFace, const int iVert){
    MData* m = (MData*)pContext->m_pUserData;
    int i = iFace * 3 + iVert;
    fvPosOut[0] = m->positions[m->indices[i] * 3 + 0];
    fvPosOut[1] = m->positions[m->indices[i] * 3 + 1];
    fvPosOut[2] = m->positions[m->indices[i] * 3 + 2];
  };

	mti.m_getNormal = [](const SMikkTSpaceContext * pContext, float fvNormOut[], const int iFace, const int iVert){
    MData* m = (MData*)pContext->m_pUserData;
    int i = iFace * 3 + iVert;
    fvNormOut[0] = m->normals[m->indices[i] * 3 + 0];
    fvNormOut[1] = m->normals[m->indices[i] * 3 + 1];
    fvNormOut[2] = m->normals[m->indices[i] * 3 + 2];
  };

	mti.m_getTexCoord = [](const SMikkTSpaceContext * pContext, float fvTexcOut[], const int iFace, const int iVert){
    MData* m = (MData*)pContext->m_pUserData;
    int i = iFace * 3 + iVert;
    fvTexcOut[0] = m->uvs[m->indices[i] * 2 + 0];
    fvTexcOut[1] = m->uvs[m->indices[i] * 2 + 1];
  };

	mti.m_setTSpaceBasic = [](const SMikkTSpaceContext * pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert){
    MData* m = (MData*)pContext->m_pUserData;
    int i = iFace * 3 + iVert;
    m->tangents[m->indices[i] * 4 + 0] = fvTangent[0];
    m->tangents[m->indices[i] * 4 + 1] = fvTangent[1];
    m->tangents[m->indices[i] * 4 + 2] = fvTangent[2];
    m->tangents[m->indices[i] * 4 + 3] = fSign;
  };
  
  SMikkTSpaceContext mtc;
  mtc.m_pInterface = &mti;
  mtc.m_pUserData = &data;

  return genTangSpaceDefault(&mtc);
}

// Pack per-vertex bone influences into fixed-size (max 4) arrays.
//
// Input: CSR-style flat arrays of all bone influences across all vertices.
//   in_offsets[vertex_count+1] — start/end indices into in_bones/in_weights per vertex
//   in_bones[total_influences]  — bone index for each influence
//   in_weights[total_influences] — weight for each influence
//
// Output: Packed arrays with exactly 4 slots per vertex.
//   out_indices[vertex_count*4] — bone indices (unused slots = 0)
//   out_weights[vertex_count*4] — normalized weights (unused slots = 0.0)
//
// For vertices with >4 influences: sorts by weight descending, takes top 4.
// Weights are normalized to sum to 1.0.
EXPORT void pack_bone_data(
  int* in_offsets, int* in_bones, float* in_weights,
  int vertex_count,
  int* out_indices, float* out_weights)
{
  // Temp buffer for sorting influences (max reasonable per vertex).
  // Vertices with more than 32 influences are truncated before sorting.
  struct Influence { int bone; float weight; };
  const int MAX_TEMP = 32;
  Influence temp[MAX_TEMP];

  for (int v = 0; v < vertex_count; v++)
  {
    int start = in_offsets[v];
    int end   = in_offsets[v + 1];
    int count = end - start;

    // Clamp to temp buffer size.
    if (count > MAX_TEMP) count = MAX_TEMP;

    // Copy influences into temp buffer.
    for (int i = 0; i < count; i++)
    {
      temp[i].bone   = in_bones[start + i];
      temp[i].weight = in_weights[start + i];
    }

    // If more than 4, partial sort to find top 4 by weight (descending).
    if (count > 4)
    {
      // Simple selection of top 4 — sufficient for small N.
      for (int i = 0; i < 4; i++)
      {
        int best = i;
        for (int j = i + 1; j < count; j++)
        {
          if (temp[j].weight > temp[best].weight)
            best = j;
        }
        if (best != i)
        {
          Influence swap = temp[i];
          temp[i] = temp[best];
          temp[best] = swap;
        }
      }
      count = 4;
    }

    // Normalize weights.
    float total = 0.0f;
    for (int i = 0; i < count; i++)
      total += temp[i].weight;

    float inv = (total > 0.0f) ? (1.0f / total) : 0.0f;

    // Write to output (4 slots per vertex).
    int base = v * 4;
    for (int i = 0; i < 4; i++)
    {
      if (i < count)
      {
        out_indices[base + i] = temp[i].bone;
        out_weights[base + i] = temp[i].weight * inv;
      }
      else
      {
        out_indices[base + i] = 0;
        out_weights[base + i] = 0.0f;
      }
    }
  }
}

EXPORT void pad_vec3_to_vec4(float* src, float* dst, int count)
{
  for (int i = 0; i < count; i++)
  {
    dst[i*4+0] = src[i*3+0];
    dst[i*4+1] = src[i*3+1];
    dst[i*4+2] = src[i*3+2];
    dst[i*4+3] = 0.0f;
  }
}
