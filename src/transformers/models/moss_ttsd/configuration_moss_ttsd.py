# coding=utf-8
# Copyright 2025 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...modeling_rope_utils import rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class MossTTSDConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MossTTSDModel`]. It is used to instantiate a
    MOSS-TTSD model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MOSS-TTSD
    [fnlp/MOSS-TTSD-v0.5](https://huggingface.co/fnlp/MOSS-TTSD-v0.5) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import MossTTSDConfig, MossTTSDModel

    >>> # Initializing a MOSS-TTSD configuration
    >>> configuration = MossTTSDConfig()

    >>> # Initializing a model from the configuration
    >>> model = MossTTSDModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    Args:
            vocab_size (`int`, *optional*, defaults to 152697):
                Vocabulary size of the MOSS-TTSD model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`MossTTSDModel`]
            hidden_size (`int`, *optional*, defaults to 2048):
                Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 6144):
                Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 28):
                Number of hidden layers in the Transformer encoder.
            num_attention_heads (`int`, *optional*, defaults to 16):
                Number of attention heads for each attention layer in the Transformer encoder.
            num_key_value_heads (`int`, *optional*, defaults to 8):
                This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
                converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
                by meanpooling all the original heads within that group. For more details, check out [this
                paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
            head_dim (`int`, *optional*, defaults to 128):
                The attention head dimension.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to 32768):
                The maximum sequence length that this model might ever be used with.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-06):
                The epsilon used by the rms normalization layers.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models). Only
                relevant if `config.is_decoder=True`.
            tie_word_embeddings (`bool`, *optional*, defaults to `True`):
                Whether the model's input and output word embeddings should be tied.
            rope_theta (`float`, *optional*, defaults to 1000000.0):
                The base period of the RoPE embeddings.
            rope_scaling (`Dict`, *optional*):
                Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
                and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
                accordingly.
                Expected contents:
                    `rope_type` (`str`):
                        The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                        'llama3'], with 'default' being the original RoPE implementation.
                    `factor` (`float`, *optional*):
                        Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                        most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                        original maximum pre-trained length.
                    `original_max_position_embeddings` (`int`, *optional*):
                        Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                        pretraining.
                    `attention_factor` (`float`, *optional*):
                        Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                        computation. If unspecified, it defaults to value recommended by the implementation, using the
                        `factor` field to infer the suggested value.
                    `beta_fast` (`float`, *optional*):
                        Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                        ramp function. If unspecified, it defaults to 32.
                    `beta_slow` (`float`, *optional*):
                        Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                        ramp function. If unspecified, it defaults to 1.
                    `short_factor` (`list[float]`, *optional*):
                        Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                        `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                        size divided by the number of attention heads divided by 2
                    `long_factor` (`list[float]`, *optional*):
                        Only used with 'longrope'. The scaling factor to be applied to long contexts (>
                        `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                        size divided by the number of attention heads divided by 2
                    `low_freq_factor` (`float`, *optional*):
                        Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                    `high_freq_factor` (`float`, *optional*):
                        Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
            attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
                Whether to use a bias in the query, key, value and output projection layers during self-attention.
            use_sliding_window (`bool`, *optional*, defaults to `False`):
                Whether to use sliding window attention.
            sliding_window (`int`, *optional*):
                Sliding window attention (SWA) window size. If not specified, will default to `None`.
            max_window_layers (`int`, *optional*, defaults to 28):
                The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
                additional layer afterwards will use SWA (Sliding Window Attention).
            layer_types (`list`, *optional*):
                Attention pattern for each layer. Each element should be 'full_attention' or 'sliding_attention'.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            channels (`int`, *optional*, defaults to 8):
                Number of quantization channels (codebooks) used in the audio tokenization. This determines how many
                parallel token streams are used to represent the audio signal.
            speech_vocab_size (`int`, *optional*, defaults to 1025):
                Vocabulary size of the speech tokens in each quantization channel. Defines the number of different
                discrete speech tokens that can be used to represent audio.
            speech_pad_token (`int`, *optional*, defaults to 1024):
                The token ID used for padding in non-primary speech channels. This is used to maintain consistent
                sequence lengths across all quantization channels.
            speech_token_range (`tuple`, *optional*, defaults to `(151665, 152689)`):
                A tuple of (start, end) token IDs that define the range of vocabulary tokens allocated for speech
                representation in the model's vocabulary. Tokens in this range are mapped to speech tokens.
            speech_eos_token (`int`, *optional*, defaults to 152694):
                The end-of-speech token ID used to mark the end of a speech sequence. This token signals the
                completion of audio generation.

    ```python
    >>> from transformers import MossTTSDModel, MossTTSDConfig

    >>> # Initializing a Qwen3 style configuration
    >>> configuration = MossTTSDConfig()

    >>> # Initializing a model from the Qwen3-8B style configuration
    >>> model = MossTTSDModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moss_ttsd"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `MossTTSD`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=152697,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0.0,
        channels=8,
        speech_vocab_size=1025,
        speech_pad_token=1024,
        speech_token_range=(151665, 152689),
        speech_eos_token=152694,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        self.channels = channels
        self.speech_vocab_size = speech_vocab_size
        self.speech_pad_token = speech_pad_token
        # Ensure speech_token_range is always a tuple
        self.speech_token_range = tuple(speech_token_range) if speech_token_range is not None else None
        self.speech_eos_token = speech_eos_token

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MossTTSDConfig"]
