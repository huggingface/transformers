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
"""DINOv2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/dinov2-base")
@strict
class Dinov2Config(BackboneConfigMixin, PreTrainedConfig):
    r"""
    layerscale_value (`float`, *optional*, defaults to 1.0):
        Initial value to use for layer scale.
    use_swiglu_ffn (`bool`, *optional*, defaults to `False`):
        Whether to use the SwiGLU feedforward neural network.
    apply_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization to the feature maps in case the model is used as backbone.
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether to reshape the feature maps to 4D tensors of shape `(batch_size, hidden_size, height, width)` in
        case the model is used as backbone. If `False`, the feature maps will be 3D tensors of shape `(batch_size,
        seq_len, hidden_size)`.
    use_mask_token (`bool`, *optional*, defaults to `True`):
        Whether to use mask_token in embeddings.

    Example:

    ```python
    >>> from transformers import Dinov2Config, Dinov2Model

    >>> # Initializing a Dinov2 dinov2-base style configuration
    >>> configuration = Dinov2Config()

    >>> # Initializing a model (with random weights) from the dinov2-base style configuration
    >>> model = Dinov2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "dinov2"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 14
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float | int = 0.0
    use_swiglu_ffn: bool = False
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True
    use_mask_token: bool = True

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["Dinov2Config"]
