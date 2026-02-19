from Malt.PipelineNode import PipelineNode
from Malt.PipelineParameters import Parameter, Type


class ComputePass(PipelineNode):
    """
    Dispatches any compute shaders assigned to scene meshes before the geometry
    is drawn.  Place this node before the *Pre Pass* in the Render Layer graph.

    Inputs
    ------
    Scene : the scene object passed through the render layer graph.

    Outputs
    -------
    Scene : the same scene object, unchanged.  Mesh deformed-position buffers
    have been updated in GPU memory by the dispatched compute shaders.
    """

    def __init__(self, pipeline):
        PipelineNode.__init__(self, pipeline)

    @classmethod
    def reflect_inputs(cls):
        return {'Scene': Parameter('Scene', Type.OTHER)}

    @classmethod
    def reflect_outputs(cls):
        return {'Scene': Parameter('Scene', Type.OTHER)}

    def execute(self, parameters):
        inputs = parameters['IN']
        outputs = parameters['OUT']
        scene = inputs['Scene']

        # Dispatch compute shaders for all meshes that have one assigned.
        # scene.batches contains the full (unsplit) material→mesh batch dict.
        self.pipeline.run_compute_pass(scene.batches)

        outputs['Scene'] = scene


NODE = ComputePass
