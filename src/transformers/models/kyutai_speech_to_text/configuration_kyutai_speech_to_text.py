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


from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="kyutai/stt-2.6b-en-trfs")
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
    kwargs (*optional*):
        Dictionary of keyword arguments. Notably:
            - **audio_encoder_config** ([`PreTrainedConfig`], *optional*) -- An instance of a configuration object that
              defines the audio encoder config.
            - **depth__config** ([`PreTrainedConfig`], *optional*) -- An instance of a configuration object that
              defines the depth decoder config.

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

    def __init__(
        self,
        codebook_vocab_size: int | None = 2049,
        vocab_size: int | None = 4001,
        hidden_size: int | None = 2048,
        num_hidden_layers: int | None = 48,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int | None = 750,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        hidden_act: str | None = "silu",
        head_dim: int | None = None,
        initializer_range: float | None = 0.02,
        use_cache: bool | None = True,
        sliding_window: int | None = 375,
        attention_dropout: float | None = 0.0,
        ffn_dim: int | None = 11264,
        rms_norm_eps: int | None = 1e-8,
        num_codebooks: int | None = 32,
        audio_bos_token_id: int | None = 2048,
        audio_pad_token_id: int | None = 69569,
        tie_word_embeddings: bool | None = False,
        pad_token_id: int | None = 3,
        bos_token_id: int | None = 48000,
        eos_token_id: int | None = None,
        codec_config: dict | None = None,
        **kwargs,
    ):
        if codec_config is None:
            self.codec_config = AutoConfig.for_model("mimi")
            logger.info("codec_config is None, using default audio encoder config.")
        elif isinstance(codec_config, dict):
            self.codec_config = AutoConfig.for_model(**codec_config)
        elif isinstance(codec_config, PreTrainedConfig):
            self.codec_config = codec_config

        self.num_codebooks = num_codebooks
        self.frame_size = self.codec_config.frame_size

        self.audio_bos_token_id = audio_bos_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.codebook_vocab_size = codebook_vocab_size

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        if ffn_dim % 2 == 1:
            raise ValueError(f"`ffn_dim={ffn_dim}` must be even.")
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.sliding_window = sliding_window
        self.rope_parameters = rope_parameters

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["KyutaiSpeechToTextConfig"]
