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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ...utils.type_validators import interval, positive_int_field


@auto_docstring(checkpoint="bezzam/yue2-3b")
@strict
class YuE2Config(PreTrainedConfig):
    r"""
    latent_dim (`int`, *optional*, defaults to 64):
        Dimension of the acoustic latents the model generates with flow matching (the latent dimension of the acoustic
        tokenizer).
    max_latent_frames (`int`, *optional*, defaults to 24576):
        Number of positions of the sinusoidal position embedding of the acoustic latents.
    frequency_embedding_size (`int`, *optional*, defaults to 256):
        Size of the sinusoidal embedding of the flow-matching time.
    max_period (`float`, *optional*, defaults to 10000.0):
        Maximum period of the sinusoidal embeddings of the flow-matching time and of the latent positions.
    timestep_shift (`float`, *optional*, defaults to 1.0):
        Shift of the flow-matching time schedule: a time `t` is mapped to `shift * t / (1 + (shift - 1) * t)`; 1 leaves
        it unchanged.
    text_vocab_size (`int`, *optional*, defaults to 151643):
        Number of ordinary text tokens, the ids below the special tokens: the symbolic score (ABC notation) is
        generated among them.
    abc_start_token_id (`int`, *optional*, defaults to 151847):
        Token id opening the symbolic score (ABC notation) of the song.
    abc_end_token_id (`int`, *optional*, defaults to 151848):
        Token id closing the symbolic score; the end of generation of the score.
    music_start_token_id (`int`, *optional*, defaults to 151851):
        Token id opening the semantic (codec) tokens of the song.
    music_end_token_id (`int`, *optional*, defaults to 151852):
        Token id closing the semantic tokens; the end of generation of the song.
    codec_token_offset (`int`, *optional*, defaults to 151853):
        Id of the first semantic token: semantic code `i` is token `codec_token_offset + i`.
    codec_vocab_size (`int`, *optional*, defaults to 32768):
        Number of semantic codes. Each semantic token stands for one acoustic latent frame.

    Example:

    ```python
    >>> from transformers import YuE2Config, YuE2ForConditionalGeneration

    >>> # Initializing a YuE2-3B style configuration
    >>> configuration = YuE2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = YuE2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "yue2"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    vocab_size: int = positive_int_field(default=184704)
    hidden_size: int = positive_int_field(default=2048)
    intermediate_size: int = positive_int_field(default=6144)
    num_hidden_layers: int = positive_int_field(default=28)
    num_attention_heads: int = positive_int_field(default=16)
    num_key_value_heads: int | None = 8
    head_dim: int = positive_int_field(default=128)
    hidden_act: str = "silu"
    max_position_embeddings: int = positive_int_field(default=24576)
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    rms_norm_eps: float = interval(min=0.0)(default=1e-6)
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    latent_dim: int = positive_int_field(default=64)
    max_latent_frames: int = positive_int_field(default=24576)
    frequency_embedding_size: int = positive_int_field(default=256)
    max_period: float = interval(min=0.0)(default=10000.0)
    timestep_shift: float = interval(min=0.0)(default=1.0)
    text_vocab_size: int = positive_int_field(default=151643)
    abc_start_token_id: int = 151847
    abc_end_token_id: int = 151848
    music_start_token_id: int = 151851
    music_end_token_id: int = 151852
    codec_token_offset: int = 151853
    codec_vocab_size: int = positive_int_field(default=32768)
    pad_token_id: int | None = 151643
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151852

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.codec_token_offset + self.codec_vocab_size > self.vocab_size:
            raise ValueError(
                f"The semantic tokens (`codec_token_offset` {self.codec_token_offset} + `codec_vocab_size` "
                f"{self.codec_vocab_size}) must fit in the vocabulary (`vocab_size` {self.vocab_size})."
            )


__all__ = ["YuE2Config"]
