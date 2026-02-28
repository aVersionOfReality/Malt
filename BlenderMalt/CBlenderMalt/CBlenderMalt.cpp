#include "stdio.h"
#include "mikktspace.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

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

// Build all smooth-related data structures in a single call:
//   - Adjacency CSR (vertex → neighbor vertices, including quad 0-2 diagonals)
//   - Vertex-to-corner CSR (vertex → loop indices, inverse of corner_vert)
//   - Fan groups (per-corner fan IDs derived from normal similarity)
//   - Cotangent weights (per directed adjacency entry)
//   - Edge metadata (per directed adjacency entry: [owning_fan, gather_corner, flags])
//
// Inputs (all pre-extracted from Blender Python API):
//   edges[edge_count*2]        — flat pairs (v0, v1) per mesh edge
//   corner_vert[loop_count]    — loop → vertex mapping
//   poly_loop_start[poly_count] — first loop index per polygon
//   poly_loop_total[poly_count] — loop count per polygon
//   positions[loop_count*3]    — per-loop positions (vec3, tightly packed)
//   tri_loops[tri_count*3]     — loop indices per loop-triangle
//   normals_vec4[loop_count*4] — per-corner normals (vec4-padded, w=0)
//   vertex_count, edge_count, loop_count, poly_count, tri_count
//
// Outputs (caller allocates):
//   out_adjacency[*out_adj_total]     — packed CSR: [offsets V+1 | indices]
//   out_vert_corner[*out_vc_total]    — packed CSR: [offsets V+1 | indices]
//   out_fan_groups[loop_count]        — per-corner fan group ID (int)
//   out_cotangent[*out_adj_idx_count] — float per adjacency index
//   out_edge_meta[*out_adj_idx_count*3] — int triplets per adjacency index
//
// The caller must pre-allocate output buffers to worst-case sizes.
// Actual used sizes are written to the out_*_count / out_*_total params.
EXPORT void build_smooth_data(
  int* edges, int edge_count,
  int* corner_vert, int loop_count,
  int* poly_loop_start, int* poly_loop_total, int poly_count,
  float* positions,
  int* tri_loops, int tri_count,
  float* normals_vec4,  // per-corner normals (vec4-padded)
  int vertex_count,
  // Output buffers (pre-allocated by caller to worst-case sizes):
  int* out_adjacency,   int* out_adj_total,
  int* out_vert_corner, int* out_vc_total,
  int* out_fan_groups,
  float* out_cotangent, int* out_adj_idx_count,
  int* out_edge_meta)
{
  // ========== Phase 1: Build adjacency (neighbor sets) ==========
  // Use flat sorted adjacency lists. First count neighbors per vertex,
  // then fill them in.

  // Temporary neighbor storage: vector of vectors.
  // For large meshes this is still much faster than Python dict/set.
  std::vector<std::vector<int>> neighbors(vertex_count);

  // Reserve approximate capacity (average valence ~6 for triangulated meshes).
  for (int v = 0; v < vertex_count; v++)
    neighbors[v].reserve(8);

  // Helper to add a directed neighbor if not already present.
  auto add_neighbor = [&](int va, int vb) {
    auto& n = neighbors[va];
    // Linear scan — fine for small valence.
    for (int k = 0; k < (int)n.size(); k++)
      if (n[k] == vb) return;
    n.push_back(vb);
  };

  // Add edges.
  for (int i = 0; i < edge_count; i++)
  {
    int v0 = edges[i * 2];
    int v1 = edges[i * 2 + 1];
    add_neighbor(v0, v1);
    add_neighbor(v1, v0);
  }

  // Track quad diagonals: for each quad, add diagonal between corner 0 and 2.
  // Use a flat boolean array indexed by (min(va,vb) * vertex_count + max(va,vb))
  // — but that's too large. Instead, tag diagonals after building adjacency.
  // We'll record diagonal pairs in a temporary vector.
  struct DiagPair { int va, vb; };
  std::vector<DiagPair> diagonal_pairs;

  for (int p = 0; p < poly_count; p++)
  {
    if (poly_loop_total[p] == 4)
    {
      int ls = poly_loop_start[p];
      int v0 = corner_vert[ls];
      int v2 = corner_vert[ls + 2];
      // Only add if not already a neighbor (i.e., not a real edge).
      bool found = false;
      for (int k = 0; k < (int)neighbors[v0].size(); k++)
        if (neighbors[v0][k] == v2) { found = true; break; }
      if (!found)
      {
        add_neighbor(v0, v2);
        add_neighbor(v2, v0);
        int a = v0 < v2 ? v0 : v2;
        int b = v0 < v2 ? v2 : v0;
        diagonal_pairs.push_back({a, b});
      }
    }
  }

  // Sort each neighbor list and build CSR offsets + edge_to_adj_idx.
  for (int v = 0; v < vertex_count; v++)
    std::sort(neighbors[v].begin(), neighbors[v].end());

  // Build adjacency CSR offsets and flat index array.
  // Also build a lookup: for directed edge (va, vb), what is the index
  // into the flat adjacency_indices array?
  // We store this as adj_offsets[va] + binary_search(vb in neighbors[va]).
  int* adj_offsets = out_adjacency;  // First V+1 ints are offsets.
  adj_offsets[0] = 0;
  int total_adj = 0;
  for (int v = 0; v < vertex_count; v++)
  {
    total_adj += (int)neighbors[v].size();
    adj_offsets[v + 1] = total_adj;
  }

  // Write adjacency indices after the offsets.
  int* adj_indices = out_adjacency + (vertex_count + 1);
  for (int v = 0; v < vertex_count; v++)
  {
    int off = adj_offsets[v];
    for (int j = 0; j < (int)neighbors[v].size(); j++)
      adj_indices[off + j] = neighbors[v][j];
  }

  *out_adj_total = (vertex_count + 1) + total_adj;
  *out_adj_idx_count = total_adj;

  // Helper: find adjacency index for directed edge (va, vb).
  // Binary search in the sorted neighbor list of va.
  auto adj_idx_of = [&](int va, int vb) -> int {
    int lo = adj_offsets[va];
    int hi = adj_offsets[va + 1];
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (adj_indices[mid] < vb) lo = mid + 1;
      else if (adj_indices[mid] > vb) hi = mid;
      else return mid;
    }
    return -1;  // not found (shouldn't happen)
  };

  // ========== Phase 2: Vertex-to-corner CSR ==========
  // Count corners per vertex.
  int* vc_offsets = out_vert_corner;
  std::memset(vc_offsets, 0, (vertex_count + 1) * sizeof(int));
  for (int l = 0; l < loop_count; l++)
    vc_offsets[corner_vert[l] + 1]++;
  // Prefix sum.
  for (int v = 0; v < vertex_count; v++)
    vc_offsets[v + 1] += vc_offsets[v];

  int* vc_indices = out_vert_corner + (vertex_count + 1);
  // Temp write cursors (reuse a small vector).
  std::vector<int> vc_cursor(vertex_count);
  for (int v = 0; v < vertex_count; v++)
    vc_cursor[v] = vc_offsets[v];

  for (int l = 0; l < loop_count; l++)
  {
    int v = corner_vert[l];
    vc_indices[vc_cursor[v]++] = l;
  }

  // Sort each vertex's corner list (matches Python's sorted()).
  for (int v = 0; v < vertex_count; v++)
    std::sort(vc_indices + vc_offsets[v], vc_indices + vc_offsets[v + 1]);

  *out_vc_total = (vertex_count + 1) + loop_count;

  // ========== Phase 2.5: Fan groups from normals ==========
  // For each vertex, group corners by normal similarity.
  // Corners with dot(n1, n2) > threshold belong to the same fan.
  // Fan IDs are local per-vertex (0, 1, 2, ...).
  const float FAN_DOT_THRESHOLD = 0.9999f;

  for (int v = 0; v < vertex_count; v++)
  {
    int cs = vc_offsets[v];
    int ce = vc_offsets[v + 1];
    int nc = ce - cs;  // number of corners at this vertex

    if (nc <= 1)
    {
      // Single corner — trivially fan 0.
      if (nc == 1)
        out_fan_groups[vc_indices[cs]] = 0;
      continue;
    }

    // Assign fan group IDs via greedy flood-fill.
    // For typical valence (4-8), this O(nc^2) approach is negligible.
    int group_id = 0;
    // Use -1 as "unvisited" sentinel.
    for (int i = cs; i < ce; i++)
      out_fan_groups[vc_indices[i]] = -1;

    for (int i = cs; i < ce; i++)
    {
      int ci = vc_indices[i];
      if (out_fan_groups[ci] >= 0) continue;  // already assigned

      out_fan_groups[ci] = group_id;
      float nx_i = normals_vec4[ci * 4];
      float ny_i = normals_vec4[ci * 4 + 1];
      float nz_i = normals_vec4[ci * 4 + 2];

      // Find all other unvisited corners with similar normal.
      for (int j = i + 1; j < ce; j++)
      {
        int cj = vc_indices[j];
        if (out_fan_groups[cj] >= 0) continue;

        float nx_j = normals_vec4[cj * 4];
        float ny_j = normals_vec4[cj * 4 + 1];
        float nz_j = normals_vec4[cj * 4 + 2];

        float dot = nx_i * nx_j + ny_i * ny_j + nz_i * nz_j;
        if (dot > FAN_DOT_THRESHOLD)
          out_fan_groups[cj] = group_id;
      }
      group_id++;
    }
  }

  // ========== Phase 3: Cotangent weights ==========
  // For each triangle, for each of its 3 edges, accumulate cot(opposite angle) / 2.
  std::memset(out_cotangent, 0, total_adj * sizeof(float));

  // Build one representative corner per vertex for position lookup.
  // (Use first corner from vert_corner CSR.)
  auto vert_pos = [&](int v, float& px, float& py, float& pz) {
    int c = vc_indices[vc_offsets[v]];  // first corner
    px = positions[c * 3];
    py = positions[c * 3 + 1];
    pz = positions[c * 3 + 2];
  };

  for (int t = 0; t < tri_count; t++)
  {
    int l0 = tri_loops[t * 3];
    int l1 = tri_loops[t * 3 + 1];
    int l2 = tri_loops[t * 3 + 2];
    int v0 = corner_vert[l0];
    int v1 = corner_vert[l1];
    int v2 = corner_vert[l2];

    float p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z;
    vert_pos(v0, p0x, p0y, p0z);
    vert_pos(v1, p1x, p1y, p1z);
    vert_pos(v2, p2x, p2y, p2z);

    // For each edge, compute cot of opposite angle, clamped >= 0, * 0.5.
    // Edge (va, vb), opposite vertex vc at position pc.
    struct TriEdge { int va, vb; float pcx, pcy, pcz; float pax, pay, paz; float pbx, pby, pbz; };
    TriEdge te[3] = {
      {v0, v1, p2x, p2y, p2z, p0x, p0y, p0z, p1x, p1y, p1z},
      {v1, v2, p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z},
      {v0, v2, p1x, p1y, p1z, p0x, p0y, p0z, p2x, p2y, p2z},
    };

    for (int e = 0; e < 3; e++)
    {
      float e1x = te[e].pax - te[e].pcx;
      float e1y = te[e].pay - te[e].pcy;
      float e1z = te[e].paz - te[e].pcz;
      float e2x = te[e].pbx - te[e].pcx;
      float e2y = te[e].pby - te[e].pcy;
      float e2z = te[e].pbz - te[e].pcz;

      float dot = e1x*e2x + e1y*e2y + e1z*e2z;
      float cx = e1y*e2z - e1z*e2y;
      float cy = e1z*e2x - e1x*e2z;
      float cz = e1x*e2y - e1y*e2x;
      float cross_len = std::sqrt(cx*cx + cy*cy + cz*cz);

      if (cross_len < 1e-10f) continue;

      float cot_val = (dot / cross_len);
      if (cot_val < 0.0f) cot_val = 0.0f;
      cot_val *= 0.5f;

      int idx_ab = adj_idx_of(te[e].va, te[e].vb);
      if (idx_ab >= 0) out_cotangent[idx_ab] += cot_val;
      int idx_ba = adj_idx_of(te[e].vb, te[e].va);
      if (idx_ba >= 0) out_cotangent[idx_ba] += cot_val;
    }
  }

  // ========== Phase 4: Edge metadata ==========
  // For each directed adjacency entry: [owning_fan, gather_corner, flags]
  // flags bit 0 = IS_DIAGONAL.
  // Initialize to [-1, -1, 0].
  for (int i = 0; i < total_adj; i++)
  {
    out_edge_meta[i * 3]     = -1;
    out_edge_meta[i * 3 + 1] = -1;
    out_edge_meta[i * 3 + 2] = 0;
  }

  // Set IS_DIAGONAL flag for quad diagonal edges.
  for (int d = 0; d < (int)diagonal_pairs.size(); d++)
  {
    int va = diagonal_pairs[d].va;
    int vb = diagonal_pairs[d].vb;
    int idx_ab = adj_idx_of(va, vb);
    int idx_ba = adj_idx_of(vb, va);
    if (idx_ab >= 0) out_edge_meta[idx_ab * 3 + 2] = 1;
    if (idx_ba >= 0) out_edge_meta[idx_ba * 3 + 2] = 1;
  }

  // Compute owning_fan and gather_corner per edge using the computed fan groups.
  // For each directed adjacency entry (va→vb), find all faces containing that
  // edge and determine the fan group at va's corner(s).
  {
    struct FaceEntry {
      int loop_va;    // loop index at va
      int loop_vb;    // loop index at vb (gather corner candidate)
      int next;       // index of next entry in chain, or -1
    };

    // Pool of face entries. Each polygon edge generates 2 directed entries
    // (one per direction). Total polygon edges = loop_count (each corner
    // connects to the next). Plus each quad adds 2 diagonal entries.
    // So total = 2 * loop_count + 2 * poly_count (generous upper bound).
    int pool_cap = 2 * loop_count + 2 * poly_count + 1;
    std::vector<FaceEntry> pool(pool_cap);
    int pool_used = 0;

    // Head array: one per adjacency index, -1 = empty.
    std::vector<int> head(total_adj, -1);

    auto add_face_entry = [&](int adj_idx, int loop_va, int loop_vb) {
      if (adj_idx < 0 || pool_used >= pool_cap) return;
      pool[pool_used] = {loop_va, loop_vb, head[adj_idx]};
      head[adj_idx] = pool_used++;
    };

    for (int p = 0; p < poly_count; p++)
    {
      int ls = poly_loop_start[p];
      int lt = poly_loop_total[p];
      for (int i = 0; i < lt; i++)
      {
        int loop_i = ls + i;
        int loop_j = ls + ((i + 1) % lt);
        int vi = corner_vert[loop_i];
        int vj = corner_vert[loop_j];
        int idx_ij = adj_idx_of(vi, vj);
        int idx_ji = adj_idx_of(vj, vi);
        add_face_entry(idx_ij, loop_i, loop_j);
        add_face_entry(idx_ji, loop_j, loop_i);
      }
      if (lt == 4)
      {
        int l0 = ls;
        int l2 = ls + 2;
        int vi = corner_vert[l0];
        int vj = corner_vert[l2];
        int idx_ij = adj_idx_of(vi, vj);
        int idx_ji = adj_idx_of(vj, vi);
        add_face_entry(idx_ij, l0, l2);
        add_face_entry(idx_ji, l2, l0);
      }
    }

    // Compute owning_fan and gather_corner for each adjacency entry.
    for (int a = 0; a < total_adj; a++)
    {
      if (head[a] == -1) continue;

      int fan_id = -2;  // sentinel: not yet set
      bool all_same = true;
      int gather = -1;

      for (int e = head[a]; e != -1; e = pool[e].next)
      {
        int this_fan = out_fan_groups[pool[e].loop_va];
        if (fan_id == -2)
          fan_id = this_fan;
        else if (fan_id != this_fan)
          all_same = false;
        gather = pool[e].loop_vb;
      }

      if (all_same && fan_id != -2)
      {
        out_edge_meta[a * 3]     = fan_id;
        out_edge_meta[a * 3 + 1] = gather;
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
