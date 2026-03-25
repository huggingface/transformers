# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="sesame/csm-1b")
@strict
class CsmDepthDecoderConfig(PreTrainedConfig):
    r"""
    backbone_hidden_size (`int`, *optional*, defaults to 2048):
        Dimension of the hidden representations of the backbone model used with this depth decoder.

    Example:

    ```python
    >>> from transformers import CsmDepthDecoder, CsmDepthDecoderConfig

    >>> # Initializing a CsmDepthDecoder
    >>> configuration = CsmDepthDecoderConfig()
    >>> model = CsmDepthDecoderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "csm_depth_decoder_model"
    base_config_key = "depth_decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "codebook_size": "vocab_size",
    }
    default_theta = 500000.0

    num_codebooks: int | None = 32
    backbone_hidden_size: int = 2048
    vocab_size: int = 2051
    hidden_size: int = 1024
    intermediate_size: int = 8192
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int | None = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 33
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None

    def __post_init__(self, **kwargs):
        if kwargs.pop("tie_word_embeddings", False):
            raise ValueError("`tie_word_embeddings=True` is not supported for CsmDepthDecoderConfig")

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="sesame/csm-1b")
@strict
class CsmConfig(PreTrainedConfig):
    r"""
    codebook_pad_token_id (`int`, *optional*, defaults to 2050):
        Padding token id for codebook tokens.
    codebook_eos_token_id (`int`, *optional*, defaults to 0):
        End of stream token id for codebook tokens.
    audio_token_id (`int`, *optional*, defaults to 128002):
        Audio token id in the text input.
    audio_eos_token_id (`int`, *optional*, defaults to 128003):
        End of stream token id for audio in the text input.
    tie_codebooks_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie the codebook tokens embeddings of the backbone model to the codebook tokens embeddings of the depth decoder.
    depth_decoder_config (`CsmDepthDecoderConfig`, *optional*):
        Configuration for the depth decoder.
    codec_config (`PreTrainedConfig`, *optional*):
        Configuration for the codec.

    ```python
    >>> from transformers import CsmForConditionalGeneration, CsmConfig

    >>> # Initializing a CsmConfig
    >>> configuration = CsmConfig()

    >>> # Initializing a model
    >>> model = CsmForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "csm"
    base_config_key = "csm_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    sub_configs = {
        "codec_config": AutoConfig,
        "depth_decoder_config": CsmDepthDecoderConfig,
    }
    attribute_map = {
        "codebook_size": "vocab_size",
    }

    num_codebooks: int | None = 32
    vocab_size: int = 2051
    text_vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 128002
    codebook_pad_token_id: int | None = 2050
    codebook_eos_token_id: int | list[int] | None = 0
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = None
    audio_token_id: int | None = 128002
    audio_eos_token_id: int | list[int] | None = 128003
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None
    tie_codebooks_embeddings: bool | None = True
    depth_decoder_config: dict | PreTrainedConfig | None = None
    codec_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if kwargs.pop("tie_word_embeddings", False):
            raise ValueError("`tie_word_embeddings=True` is not supported for CsmConfig")

        if self.depth_decoder_config is None:
            self.depth_decoder_config = CsmDepthDecoderConfig()
            logger.info("depth_decoder_config is None, using default depth decoder config.")
        elif isinstance(self.depth_decoder_config, dict):
            self.depth_decoder_config = CsmDepthDecoderConfig(**self.depth_decoder_config)

        if self.codec_config is None:
            self.codec_config = AutoConfig.for_model("mimi")
            logger.info("codec_config is None, using default audio encoder config.")
        elif isinstance(self.codec_config, dict):
            self.codec_config = AutoConfig.for_model(**self.codec_config)

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        self.tie_word_embeddings = False
        super().__post_init__(**kwargs)


__all__ = [
    "CsmDepthDecoderConfig",
    "CsmConfig",
]
