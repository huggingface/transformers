# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.s


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="kyutai/stt-2.6b-en-trfs")
@strict
class KyutaiSpeechToTextConfig(PreTrainedConfig):
    r"""
    codebook_vocab_size (`int`, *optional*, defaults to 2049):
        Vocabulary size of the codebook. Defines the number of different audio tokens that can be represented by each codebook.
    audio_bos_token_id (`int`, *optional*, defaults to 2048):
        Beginning of stream token id for codebook tokens.
    audio_pad_token_id (`int`, *optional*, defaults to 69569):
        Padding token id for codebook tokens.
    codec_config (`PreTrainedConfig`, *optional*):
        Configuration for the codec.

    Example:
    ```python
    >>> from transformers import KyutaiSpeechToTextConfig, KyutaiSpeechToTextForConditionalGeneration

    >>> # Initializing a KyutaiSpeechToTextConfig
    >>> configuration = KyutaiSpeechToTextConfig()

    >>> # Initializing a model
    >>> model = KyutaiSpeechToTextForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kyutai_speech_to_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"codec_config": AutoConfig}

    codebook_vocab_size: int = 2049
    vocab_size: int = 4001
    hidden_size: int = 2048
    num_hidden_layers: int = 48
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 750
    rope_parameters: RopeParameters | dict | None = None
    hidden_act: str = "silu"
    head_dim: int | None = None
    initializer_range: float = 0.02
    use_cache: bool = True
    sliding_window: int = 375
    attention_dropout: float | int = 0.0
    ffn_dim: int = 11264
    rms_norm_eps: float = 1e-8
    num_codebooks: int = 32
    audio_bos_token_id: int | None = 2048
    audio_pad_token_id: int | None = 69569
    tie_word_embeddings: bool = False
    pad_token_id: int | None = 3
    bos_token_id: int | None = 48000
    eos_token_id: int | list[int] | None = None
    codec_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.codec_config is None:
            self.codec_config = AutoConfig.for_model("mimi")
            logger.info("codec_config is None, using default audio encoder config.")
        elif isinstance(self.codec_config, dict):
            self.codec_config = AutoConfig.for_model(**self.codec_config)

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.frame_size = self.codec_config.frame_size
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.ffn_dim % 2 == 1:
            raise ValueError(f"`ffn_dim={self.ffn_dim}` must be even.")


__all__ = ["KyutaiSpeechToTextConfig"]
