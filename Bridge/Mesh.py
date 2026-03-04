MESHES = {}

def load_mesh(pipeline, msg):
    name = msg['name']
    data = msg['data']

    MESHES[name] = pipeline.load_mesh(
        position = data['positions'],
        indices = data['indices'],
        normal = data['normals'],
        tangent = data['tangents'],
        uvs = data['uvs'],
        colors = data['colors'],
        ssbo_colors = data.get('ssbo_colors', [None]*8),
        vertex_count = data.get('vertex_count', 0),
        loop_count = data.get('loop_count', 0),
        rest_positions = data.get('rest_positions', None),
        rest_normals = data.get('rest_normals', None),
        corner_vert = data.get('corner_vert', None),
        adjacency_data = data.get('adjacency_data', None),
        vert_corner_data = data.get('vert_corner_data', None),
        fan_groups = data.get('fan_groups', None),
        cotangent_weights = data.get('cotangent_weights', None),
        edge_metadata = data.get('edge_metadata', None),
        laplacian1 = data.get('laplacian1', None),
        laplacian2 = data.get('laplacian2', None),
        bone_indices = data.get('bone_indices', None),
        bone_weights = data.get('bone_weights', None),
        bone_count = data.get('bone_count', 0),
    )
