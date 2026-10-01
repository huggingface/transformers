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

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/focalnet-tiny")
@strict
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

    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 4
    num_channels: int = 3
    embed_dim: int = 96
    use_conv_embed: bool = False
    hidden_sizes: list[int] | tuple[int, ...] = (192, 384, 768, 768)
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    focal_levels: list[int] | tuple[int, ...] = (2, 2, 2, 2)
    focal_windows: list[int] | tuple[int, ...] = (3, 3, 3, 3)
    hidden_act: str = "gelu"
    mlp_ratio: float = 4.0
    hidden_dropout_prob: float | int = 0.0
    drop_path_rate: float | int = 0.1
    use_layerscale: bool = False
    layerscale_value: float = 1e-4
    use_post_layernorm: bool = False
    use_post_layernorm_in_modulation: bool = False
    normalize_modulator: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    encoder_stride: int = 32
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["FocalNetConfig"]
