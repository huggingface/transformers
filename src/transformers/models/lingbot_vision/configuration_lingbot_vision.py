# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""LingBot-Vision model configuration."""

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="IMvision12/lingbot-vision-vit-giant-hf")
@strict
class LingbotVisionConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    patch_size (`int`, *optional*, defaults to 16):
        Patch size used by the ViT patch embedding.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the feed-forward layer. With `use_gated_mlp=True` this is the size of the gate and
        up projections, i.e. the aligned SwiGLU hidden size of the original implementation.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer.
    hidden_act (`str`, *optional*, defaults to `"gelu"`):
        Activation of the feed-forward layer. The released checkpoints use `"gelu"` with the plain MLP and
        `"silu"` with the gated MLP (SwiGLU).
    attention_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability of the attention weights.
    initializer_range (`float`, *optional*, defaults to 0.02):
        Standard deviation of the truncated normal initializer for all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of the layer normalization layers.
    rope_parameters (`dict`, *optional*):
        Parameters for the axial rotary position embeddings. `rope_theta` sets the base period and defaults
        to 100.0.
    image_size (`int`, *optional*, defaults to 224):
        The nominal image size the model is configured for. Images of other sizes are supported, the rotary
        position embeddings are computed from the actual patch grid.
    num_channels (`int`, *optional*, defaults to 3):
        Number of input image channels.
    query_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the query projection.
    key_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the key projection. The original implementation masks the key slice of the fused
        QKV bias to zero, so the released checkpoints carry an all-zero key bias; it is kept (rather than dropped)
        so that the fused projection splits and re-fuses without loss.
    value_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the value projection.
    proj_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the attention output projection.
    mlp_bias (`bool`, *optional*, defaults to `True`):
        Whether to add biases to the feed-forward projections.
    layerscale_value (`float`, *optional*, defaults to 1e-05):
        Initial value to use for layer scale.
    drop_path_rate (`float`, *optional*, defaults to 0.0):
        Stochastic depth rate.
    use_gated_mlp (`bool`, *optional*, defaults to `False`):
        Whether to use the SwiGLU feed-forward network instead of the plain MLP.
    num_register_tokens (`int`, *optional*, defaults to 4):
        Number of register tokens inserted after the class token. These are called storage tokens in the
        original implementation.
    pos_embed_shift (`float`, *optional*):
        Amount to randomly shift position embedding coordinates in [-shift, shift], applied only in training
        mode if not `None`.
    pos_embed_jitter (`float`, *optional*):
        Amount to randomly jitter position embedding coordinates by a log-uniform value in [1/jitter, jitter],
        applied only in training mode if not `None`.
    pos_embed_rescale (`float`, *optional*, defaults to 2.0):
        Amount to randomly rescale position embedding coordinates by a log-uniform value in
        [1/rescale, rescale], applied only in training mode if not `None`.
    apply_layernorm (`bool`, *optional*, defaults to `True`):
        Whether `LingbotVisionBackbone` should apply the final layer norm to selected feature maps.
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether `LingbotVisionBackbone` should reshape feature maps to `(batch, channels, height, width)`.
    return_class_token (`bool`, *optional*, defaults to `False`):
        Whether `LingbotVisionBackbone` should additionally return the class token of each selected stage.

    Example:

    ```python
    >>> from transformers import LingbotVisionConfig, LingbotVisionModel

    >>> # Initializing a LingBot-Vision ViT-B/16 style configuration
    >>> config = LingbotVisionConfig()

    >>> # Initializing a model (with random weights) from the config
    >>> model = LingbotVisionModel(config)

    >>> # Accessing the model config
    >>> config = model.config
    ```"""

    model_type = "lingbot_vision"
    default_theta = 100.0

    patch_size: int = 16
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    hidden_act: str = "gelu"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    rope_parameters: RopeParameters | dict | None = None
    image_size: int = 224
    num_channels: int = 3
    query_bias: bool = True
    key_bias: bool = True
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    layerscale_value: float = 1e-5
    drop_path_rate: float = 0.0
    use_gated_mlp: bool = False
    num_register_tokens: int = 4
    pos_embed_shift: float | None = None
    pos_embed_jitter: float | None = None
    pos_embed_rescale: float | None = 2.0
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True
    return_class_token: bool = False
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["LingbotVisionConfig"]
