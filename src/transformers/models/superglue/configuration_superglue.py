from typing import List

from transformers import PretrainedConfig


class SuperGlueConfig(PretrainedConfig):

    def __init__(
            self,
            descriptor_dim: int = 256,
            keypoint_encoder_sizes: List[int] = [32, 64, 128, 256],
            gnn_layers_types: List[str] = ['self', 'cross'] * 9,
            num_heads: int = 4,
            sinkhorn_iterations: int = 100,
            matching_threshold: float = 0.2,
            model_version: str = "indoor",
            **kwargs,
    ):
        # Check whether all gnn_layers_types are either 'self' or 'cross'
        if not all([layer_type in ['self', 'cross'] for layer_type in gnn_layers_types]):
            raise ValueError("All gnn_layers_types must be either 'self' or 'cross'")

        if model_version != "indoor" and model_version != "outdoor":
            raise ValueError("model_version must be either 'indoor' or 'outdoor'")

        self.descriptor_dim = descriptor_dim
        self.keypoint_encoder_sizes = keypoint_encoder_sizes
        self.gnn_layers_types = gnn_layers_types
        self.num_heads = num_heads
        self.sinkhorn_iterations = sinkhorn_iterations
        self.matching_threshold = matching_threshold
        self.model_version = model_version

        super().__init__(**kwargs)
