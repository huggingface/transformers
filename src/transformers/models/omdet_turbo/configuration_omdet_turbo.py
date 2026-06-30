# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""OmDet-Turbo model configuration"""

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="omlab/omdet-turbo-swin-tiny-hf")
@strict
class OmDetTurboConfig(PreTrainedConfig):
    r"""
    apply_layernorm_after_vision_backbone (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization on the feature maps of the vision backbone output.
    disable_custom_kernels (`bool`, *optional*, defaults to `False`):
        Whether to disable custom kernels.
    text_projection_in_dim (`int`, *optional*, defaults to 512):
        The input dimension for the text projection.
    text_projection_out_dim (`int`, *optional*, defaults to 512):
        The output dimension for the text projection.
    task_encoder_hidden_dim (`int`, *optional*, defaults to 1024):
        The feedforward dimension for the task encoder.
    class_embed_dim (`int`, *optional*, defaults to 512):
        The dimension of the classes embeddings.
    class_distance_type (`str`, *optional*, defaults to `"cosine"`):
        The type of distance to compare predicted classes to projected classes embeddings.
        Can be `"cosine"` or `"dot"`.
    num_queries (`int`, *optional*, defaults to 900):
        The number of queries.
    csp_activation (`str`, *optional*, defaults to `"silu"`):
        The activation function of the Cross Stage Partial (CSP) networks of the encoder.
    conv_norm_activation (`str`, *optional*, defaults to `"gelu"`):
        The activation function of the ConvNormLayer layers of the encoder.
    encoder_feedforward_activation (`str`, *optional*, defaults to `"relu"`):
        The activation function for the feedforward network of the encoder.
    encoder_feedforward_dropout (`float`, *optional*, defaults to 0.0):
        The dropout rate following the activation of the encoder feedforward network.
    hidden_expansion (`int`, *optional*, defaults to 1):
        The hidden expansion of the CSP networks in the encoder.
    vision_features_channels (`tuple(int)`, *optional*, defaults to `[256, 256, 256]`):
        The projected vision features channels used as inputs for the decoder.
    encoder_in_channels (`List(int)`, *optional*, defaults to `[192, 384, 768]`):
        The input channels for the encoder.
    encoder_projection_indices (`List(int)`, *optional*, defaults to `[2]`):
        The indices of the input features projected by each layers.
    encoder_dim_feedforward (`int`, *optional*, defaults to 2048):
        The feedforward dimension for the encoder.
    positional_encoding_temperature (`int`, *optional*, defaults to 10000):
        The positional encoding temperature in the encoder.
    num_feature_levels (`int`, *optional*, defaults to 3):
        The number of feature levels for the multi-scale deformable attention module of the decoder.
    decoder_activation (`str`, *optional*, defaults to `"relu"`):
        The activation function for the decoder.
    decoder_dim_feedforward (`int`, *optional*, defaults to 2048):
        The feedforward dimension for the decoder.
    decoder_num_points (`int`, *optional*, defaults to 4):
        The number of points sampled in the decoder multi-scale deformable attention module.
    decoder_dropout (`float`, *optional*, defaults to 0.0):
        The dropout rate for the decoder.
    eval_size (`tuple[int, int]`, *optional*):
        Height and width used to computes the effective height and width of the position embeddings after taking
        into account the stride (see RTDetr).
    learn_initial_query (`bool`, *optional*, defaults to `False`):
        Whether to learn the initial query.
    cache_size (`int`, *optional*, defaults to 100):
        The cache size for the classes and prompts caches.

    Examples:

    ```python
    >>> from transformers import OmDetTurboConfig, OmDetTurboForObjectDetection

    >>> # Initializing a OmDet-Turbo omlab/omdet-turbo-swin-tiny-hf style configuration
    >>> configuration = OmDetTurboConfig()

    >>> # Initializing a model (with random weights) from the omlab/omdet-turbo-swin-tiny-hf style configuration
    >>> model = OmDetTurboForObjectDetection(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "omdet-turbo"
    sub_configs = {"backbone_config": AutoConfig, "text_config": AutoConfig}
    attribute_map = {
        "encoder_hidden_dim": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    text_config: dict | PreTrainedConfig | None = None
    backbone_config: dict | PreTrainedConfig | None = None
    apply_layernorm_after_vision_backbone: bool = True
    image_size: int | list[int] | tuple[int, int] = 640
    disable_custom_kernels: bool = False
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    init_std: float = 0.02
    text_projection_in_dim: int = 512
    text_projection_out_dim: int = 512
    task_encoder_hidden_dim: int = 1024
    class_embed_dim: int = 512
    class_distance_type: Literal["cosine", "dot"] = "cosine"
    num_queries: int = 900
    csp_activation: str = "silu"
    conv_norm_activation: str = "gelu"
    encoder_feedforward_activation: str = "relu"
    encoder_feedforward_dropout: float | int = 0.0
    encoder_dropout: float | int = 0.0
    hidden_expansion: int = 1
    encoder_hidden_dim: int = 256
    vision_features_channels: list[int] | tuple[int, ...] = (256, 256, 256)
    encoder_in_channels: list[int] | tuple[int, ...] = (192, 384, 768)
    encoder_projection_indices: list[int] | tuple[int, ...] = (2,)
    encoder_attention_heads: int = 8
    encoder_dim_feedforward: int = 2048
    encoder_layers: int = 1
    positional_encoding_temperature: int = 10000
    num_feature_levels: int = 3
    decoder_hidden_dim: int = 256
    decoder_num_heads: int = 8
    decoder_num_layers: int = 6
    decoder_activation: str = "relu"
    decoder_dim_feedforward: int = 2048
    decoder_num_points: int = 4
    decoder_dropout: float | int = 0.0
    eval_size: int | None = None
    learn_initial_query: bool = False
    cache_size: int = 100
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        # Init timm backbone with hardcoded values for BC
        timm_default_kwargs = {
            "out_indices": [1, 2, 3],
            "img_size": self.image_size,
            "always_partition": True,
        }
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_backbone="swin_tiny_patch4_window7_224",
            default_config_type="swin",
            default_config_kwargs={"image_size": self.image_size, "out_indices": [2, 3, 4]},
            timm_default_kwargs=timm_default_kwargs,
            **kwargs,
        )

        # Extract timm.create_model kwargs; TimmBackbone doesn't forward arbitrary config attrs to timm
        self.timm_kwargs = {}
        if getattr(self.backbone_config, "model_type", None) == "timm_backbone":
            for attr in ("img_size", "always_partition"):
                if hasattr(self.backbone_config, attr):
                    self.timm_kwargs[attr] = getattr(self.backbone_config, attr)

        if self.text_config is None:
            logger.info("`text_config` is `None`. Initializing the config with the default `clip_text_model`")
            self.text_config = CONFIG_MAPPING["clip_text_model"]()
        elif isinstance(self.text_config, dict):
            text_model_type = self.text_config.get("model_type")
            self.text_config = CONFIG_MAPPING[text_model_type](**self.text_config)

        super().__post_init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output.pop("timm_kwargs", None)
        return output


__all__ = ["OmDetTurboConfig"]
