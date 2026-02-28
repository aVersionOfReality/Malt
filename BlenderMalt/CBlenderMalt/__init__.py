import subprocess
import os
import ctypes

import platform

src_dir = os.path.abspath(os.path.dirname(__file__))

library = 'libCBlenderMalt.so'
if platform.system() == 'Windows': library = 'CBlenderMalt.dll'
if platform.system() == 'Darwin': library = 'libCBlenderMalt.dylib'

CBlenderMalt = ctypes.CDLL(os.path.join(src_dir, library))

retrieve_mesh_data = CBlenderMalt['retrieve_mesh_data']
retrieve_mesh_data.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)
]
retrieve_mesh_data.restype = None

mesh_tangents = CBlenderMalt['mesh_tangents']
mesh_tangents.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)
]
mesh_tangents.restype = ctypes.c_bool

pack_bone_data = CBlenderMalt['pack_bone_data']
pack_bone_data.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
]
pack_bone_data.restype = None

build_smooth_data = CBlenderMalt['build_smooth_data']
build_smooth_data.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,                   # edges, edge_count
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,                   # corner_vert, loop_count
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,  # poly_loop_start, poly_loop_total, poly_count
    ctypes.POINTER(ctypes.c_float),                                # positions
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,                   # tri_loops, tri_count
    ctypes.POINTER(ctypes.c_float),                                # normals_vec4 (per-corner, vec4-padded)
    ctypes.c_int,                                                  # vertex_count
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),   # out_adjacency, out_adj_total
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),   # out_vert_corner, out_vc_total
    ctypes.POINTER(ctypes.c_int),                                  # out_fan_groups
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), # out_cotangent, out_adj_idx_count
    ctypes.POINTER(ctypes.c_int),                                  # out_edge_meta
]
build_smooth_data.restype = None

pad_vec3_to_vec4 = CBlenderMalt['pad_vec3_to_vec4']
pad_vec3_to_vec4.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
pad_vec3_to_vec4.restype = None
