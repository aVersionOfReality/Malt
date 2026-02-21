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
    )
