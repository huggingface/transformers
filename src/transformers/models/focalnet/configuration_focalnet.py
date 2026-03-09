# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FocalNet model configuration"""

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/focalnet-tiny")
class FocalNetConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    use_conv_embed (`bool`, *optional*, defaults to `False`):
        Whether to use convolutional embedding. The authors noted that using convolutional embedding usually
        improve the performance, but it's not used by default.
    focal_levels (`list(int)`, *optional*, defaults to `[2, 2, 2, 2]`):
        Number of focal levels in each layer of the respective stages in the encoder.
    focal_windows (`list(int)`, *optional*, defaults to `[3, 3, 3, 3]`):
        Focal window size in each layer of the respective stages in the encoder.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
        The dropout probability for all fully connected layers in the embeddings and encoder.
    use_layerscale (`bool`, *optional*, defaults to `False`):
        Whether to use layer scale in the encoder.
    layerscale_value (`float`, *optional*, defaults to 0.0001):
        The initial value of the layer scale.
    use_post_layernorm (`bool`, *optional*, defaults to `False`):
        Whether to use post layer normalization in the encoder.
    use_post_layernorm_in_modulation (`bool`, *optional*, defaults to `False`):
        Whether to use post layer normalization in the modulation layer.
    normalize_modulator (`bool`, *optional*, defaults to `False`):
        Whether to normalize the modulator.
    encoder_stride (`int`, *optional*, defaults to 32):
        Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    Example:

    ```python
    >>> from transformers import FocalNetConfig, FocalNetModel

    >>> # Initializing a FocalNet microsoft/focalnet-tiny style configuration
    >>> configuration = FocalNetConfig()

    >>> # Initializing a model (with random weights) from the microsoft/focalnet-tiny style configuration
    >>> model = FocalNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "focalnet"

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        use_conv_embed=False,
        hidden_sizes=[192, 384, 768, 768],
        depths=[2, 2, 6, 2],
        focal_levels=[2, 2, 2, 2],
        focal_windows=[3, 3, 3, 3],
        hidden_act="gelu",
        mlp_ratio=4.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.1,
        use_layerscale=False,
        layerscale_value=1e-4,
        use_post_layernorm=False,
        use_post_layernorm_in_modulation=False,
        normalize_modulator=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        encoder_stride=32,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.use_conv_embed = use_conv_embed
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.focal_levels = focal_levels
        self.focal_windows = focal_windows
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.use_layerscale = use_layerscale
        self.layerscale_value = layerscale_value
        self.use_post_layernorm = use_post_layernorm
        self.use_post_layernorm_in_modulation = use_post_layernorm_in_modulation
        self.normalize_modulator = normalize_modulator
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_stride = encoder_stride
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(out_indices=out_indices, out_features=out_features)


__all__ = ["FocalNetConfig"]
