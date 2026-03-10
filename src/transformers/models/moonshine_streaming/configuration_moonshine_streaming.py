# Copyright 2026 the HuggingFace Team. All rights reserved.
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


from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING


@auto_docstring(checkpoint="UsefulSensors/moonshine-streaming-tiny")
class MoonshineStreamingEncoderConfig(PreTrainedConfig):
    r"""
    sample_rate (`int`, *optional*, defaults to 16000):
        The sample rate of the audio input in Hz.
    frame_ms (`float`, *optional*, defaults to 5.0):
        The frame duration in milliseconds for audio processing.
    sliding_windows (`list[tuple[int, int]]`, *optional*, defaults to `[(16, 4), (16, 4), (16, 0), (16, 0), (16, 4), (16, 4)]`):
        List of sliding window configurations for each encoder layer. Each tuple contains (window_size, shift).


    ```python
    >>> from transformers import MoonshineStreamingEncoder, MoonshineStreamingEncoderConfig

    >>> # Initializing a Moonshine Streaming encoder configuration
    >>> configuration = MoonshineStreamingEncoderConfig()

    >>> # Initializing a model from the configuration
    >>> model = MoonshineStreamingEncoder(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moonshine_streaming_encoder"

    def __init__(
        self,
        hidden_size: int | None = 320,
        intermediate_size: int | None = 1280,
        hidden_act: str | None = "gelu",
        num_hidden_layers: int | None = 6,
        num_attention_heads: int | None = 8,
        num_key_value_heads: int | None = 8,
        max_position_embeddings: int | None = 4096,
        attention_dropout: float | None = 0.0,
        attention_bias: bool | None = False,
        sample_rate: int = 16000,
        frame_ms: float = 5.0,
        sliding_windows: list[tuple[int, int]] = [(16, 4), (16, 4), (16, 0), (16, 0), (16, 4), (16, 4)],
        head_dim: int | None = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.sliding_windows = [list(window) for window in sliding_windows]

        super().__init__(**kwargs)


@auto_docstring(checkpoint="UsefulSensors/moonshine-streaming-tiny")
class MoonshineStreamingConfig(PreTrainedConfig):
    r"""
    pad_head_dim_to_multiple_of (`int`, *optional*):
        If set, the head dimension will be padded to a multiple of this value.

    ```python
    >>> from transformers import MoonshineStreamingModel, MoonshineStreamingConfig

    >>> # Initializing a Moonshine Streaming configuration
    >>> configuration = MoonshineStreamingConfig()

    >>> # Initializing a model from the configuration
    >>> model = MoonshineStreamingModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moonshine_streaming"
    sub_configs = {"encoder_config": MoonshineStreamingEncoderConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        encoder_config: MoonshineStreamingEncoderConfig = None,
        vocab_size: int = 32768,
        hidden_size: int | None = 320,
        intermediate_size: int | None = 1280,
        num_hidden_layers: int | None = 6,
        num_attention_heads: int | None = 8,
        hidden_act: str | None = "silu",
        max_position_embeddings: int = 4096,
        use_cache: bool | None = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.8,
        },
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        decoder_start_token_id: int | None = None,
        head_dim: int | None = None,
        pad_head_dim_to_multiple_of: int | None = None,
        tie_word_embeddings: bool = False,
        is_encoder_decoder: bool = True,
        **kwargs,
    ):
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = encoder_config.get("model_type", "moonshine_streaming_encoder")
            encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        elif encoder_config is None:
            encoder_config = CONFIG_MAPPING["moonshine_streaming_encoder"]()

        self.encoder_config = encoder_config

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.rope_parameters = rope_parameters
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


__all__ = ["MoonshineStreamingConfig", "MoonshineStreamingEncoderConfig"]
