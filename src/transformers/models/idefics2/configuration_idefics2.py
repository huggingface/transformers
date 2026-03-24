# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Idefics2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="HuggingFaceM4/idefics2-8b")
@strict
class Idefics2VisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer
    >>> from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig

    >>> # Initializing a Idefics2VisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = Idefics2VisionConfig()

    >>> # Initializing a Idefics2VisionTransformer (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = Idefics2VisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics2_vision"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 32
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="HuggingFaceM4/idefics2-8b")
@strict
class Idefics2PerceiverConfig(PreTrainedConfig):
    r"""
    resampler_n_latents (`int`, *optional*, defaults to 64):
        Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
    resampler_depth (`int`, *optional*, defaults to 3):
        Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (<= 3).
    resampler_n_heads (`int`, *optional*, defaults to 16):
        Number of heads in each Transformer block (for multi-headed self-attention).
    resampler_head_dim (`int`, *optional*, defaults to 96):
        Dimensionality of each head projection in the Transformer block.
    """

    model_type = "idefics2_perceiver"

    hidden_act: str = "silu"
    hidden_size: int = 4096
    rms_norm_eps: float = 1e-06
    resampler_n_latents: int = 64
    resampler_depth: int = 3
    resampler_n_heads: int = 16
    resampler_head_dim: int = 96
    num_key_value_heads: int = 4
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_key_value_heads > self.resampler_n_heads:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} must be less than or equal to"
                f" resampler_n_heads={self.resampler_n_heads}"
            )


@auto_docstring(checkpoint="HuggingFaceM4/idefics2-8b")
@strict
class Idefics2Config(PreTrainedConfig):
    r"""
    perceiver_config (`IdeficsPerceiverConfig` or `dict`, *optional*):
        Custom perceiver config or dict

    Example:
    ```python
    >>> from transformers import Idefics2Model, Idefics2Config
    >>> # Initializing configuration
    >>> configuration = Idefics2Config()
    >>> # Initializing a model from the configuration
    >>> model = Idefics2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics2"
    sub_configs = {
        "text_config": AutoConfig,
        "perceiver_config": Idefics2PerceiverConfig,
        "vision_config": Idefics2VisionConfig,
    }

    use_cache: bool = True
    image_token_id: int = 32_001
    tie_word_embeddings: bool = False
    vision_config: dict | PreTrainedConfig | None = None
    perceiver_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.perceiver_config is None:
            self.perceiver_config = Idefics2PerceiverConfig()
            logger.info("perciver_config is None, using default perceiver config")
        elif isinstance(self.perceiver_config, dict):
            self.perceiver_config = Idefics2PerceiverConfig(**self.perceiver_config)

        if self.vision_config is None:
            self.vision_config = Idefics2VisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Idefics2VisionConfig(**self.vision_config)

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "mistral")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            logger.info("text_config is None, using default text config")
            self.text_config = CONFIG_MAPPING["mistral"](
                max_position_embeddings=4096 * 8,
                rms_norm_eps=1e-5,
                # None in the original configuration_mistral, we set it to the unk_token_id
                pad_token_id=0,
            )

        if self.text_config.hidden_size != self.perceiver_config.hidden_size:
            self.perceiver_config.hidden_size = self.text_config.hidden_size
            self.perceiver_config.rms_norm_eps = self.text_config.rms_norm_eps
            logger.warning_once(
                "Perceiver config has a different `hidden_size` than text config, which means default values were used. "
                "In your model's config on the hub, add `hidden_size` and `rms_norm_eps` keys under the `perceiver_config` dict. "
            )

        super().__post_init__(**kwargs)


__all__ = ["Idefics2Config"]
