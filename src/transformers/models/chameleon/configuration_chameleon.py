# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""chameleon model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/chameleon-7b")
@strict
class ChameleonVQVAEConfig(PreTrainedConfig):
    r"""
    resolution (`int`, *optional*, defaults to 512):
        Resolution of the input images.
    base_channels (`int`, *optional*, defaults to 128):
        Base channel count.
    channel_multiplier (`list[int]`, *optional*, defaults to `[1, 1, 2, 2, 4]`):
        Channel multipliers for each resolution.
    num_res_blocks (`int`, *optional*, defaults to 2):
        Number of residual blocks.
    attn_resolutions (`list[int]`, *optional*):
        Resolutions to apply attention.
    dropout (`float`, *optional*, defaults to 0.0):
        Dropout rate.
    attn_type (`str`, *optional*, defaults to `"vanilla"`):
        Attention type used in VQ-GAN encoder. Can be "vanilla" or None
    """

    model_type = "chameleon_vqgan"
    base_config_key = "vq_config"

    embed_dim: int = 256
    num_embeddings: int = 8192
    double_latent: bool = False
    latent_channels: int = 256
    resolution: int = 512
    in_channels: int = 3
    base_channels: int = 128
    channel_multiplier: list[int] | tuple[int, ...] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: list[int] | None = None
    dropout: float | int = 0.0
    attn_type: str = "vanilla"
    initializer_range = 0.02


@auto_docstring(checkpoint="facebook/chameleon-7b")
@strict
class ChameleonConfig(PreTrainedConfig):
    r"""
    model_parallel_size (`int`, *optional*, defaults to 1):
        Number of shards used when training the model. This will be used in qk layernorm because the original Chameleon inference
        doesn't do reduction in those layers and each rank has its own biases.
    swin_norm (`bool`, *optional*, defaults to `False`):
        Use Swin Transformer normalization.
    vocabulary_map (`dict`, *optional*):
        A dictionary containing the vocabulary map from the tokenizer. Used to obtain tokens from the image inputs.

    ```python
    >>> from transformers import ChameleonModel, ChameleonConfig

    >>> # Initializing a chameleon chameleon-7b style configuration
    >>> configuration = ChameleonConfig()

    >>> # Initializing a model from the chameleon-7b style configuration
    >>> model = ChameleonModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "chameleon"
    sub_configs = {"vq_config": ChameleonVQVAEConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 65536
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: int | None = False
    attention_dropout: float | int | None = 0.0
    model_parallel_size: int | None = 1
    swin_norm: bool | None = False
    vq_config: dict | PreTrainedConfig | None = None
    vocabulary_map: dict | None = None
    mlp_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.vq_config is None:
            logger.info("vq_config is None. initializing the ChameleonVQConfig with default values.")
            self.vq_config = ChameleonVQVAEConfig()
        elif isinstance(self.vq_config, dict):
            self.vq_config = ChameleonVQVAEConfig(**self.vq_config)

        self.image_token_id = self.vocabulary_map.get("<image>") if self.vocabulary_map is not None else None

        super().__post_init__(**kwargs)


__all__ = ["ChameleonConfig", "ChameleonVQVAEConfig"]
