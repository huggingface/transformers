# coding=utf-8
# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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
"""HiggsAudioConfig."""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation


class HiggsAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class for the HiggsAudioModel. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [bosonai/higgs-audio-v2-generation-3B-base](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) architecture.

    This class inherits from [`LlamaConfig`]. All text backbone fields are available directly
    on this config. For backward compatibility, a `text_config` argument can still be provided
    (as a dict or some config class) and will be merged into the root config.

    Args:
            audio_adapter_type (`str`, *optional*, defaults to `"dual_ffn_fast_forward"`):
                The type of audio adapter to use. We support three types of adapter:
                - stack:
                    We stack additional Transformer layers after the main LLM backbone for audio generation.
                - dual_ffn_fast_forward:
                    We pick a few layers in the LLM backbone to plug-in the audio FFN. For the remaining layers,
                    the audio hidden states will be directly fast-forward to the next layer.
            audio_dual_ffn_layers (`list[int]`, *optional*, defaults to `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`):
                The layers in the LLM backbone to plug-in the dual FFN layer (mixture of audio FFN and text FFN).
            encode_audio_in_tokens (`bool`, *optional*, defaults to `True`):
                Whether to encode the input audio directly as discrete audio tokens.
                When True, the model uses `audio_in_token`
                positions filled with audio tokens extracted via the audio tokenizer.
                Note that `encode_audio_in_tokens` can be combined with `encode_whisper_embed`.
            use_delay_pattern (`bool`, *optional*, defaults to `True`):
                Whether to use delay pattern in the audio decoder.
            use_audio_out_embed_projector (`bool`, *optional*, defaults to `False`):
                Whether to use an embedding projector to map audio out embeddings.
            audio_num_codebooks (`int`, *optional*, defaults to 8):
                The number of codebooks in RVQGAN.
            audio_codebook_size (`int`, *optional*, defaults to 1024):
                The size of each codebook in RVQGAN.
                The id of the bos in the audio stream
                The id of the eos in the audio stream
            audio_stream_bos_id (`int`, *optional*, defaults to 1024):
                The token ID in the audio codebook representing the beginning of a streaming audio sequence.
                Used to signal the start of an audio stream input when generating
                audio tokens in the model.
            audio_stream_eos_id (`int`, *optional*, defaults to 1025):
                The token ID in the audio codebook representing the end of a streaming audio sequence.
                Used to signal the end of an audio stream input when generating
                audio tokens in the model.
            audio_bos_token (`str`, *optional*, defaults to `"<|audio_bos|>"`):
                The special `<|audio_bos|>` token. In Higgs-Audio, it is mapped to 128011,
                which is the index of `<|reserved_special_token_3|>` in Llama-3.1-8B-Instruct's tokenizer.
            audio_eos_token (`str`, *optional*, defaults to `"<|audio_eos|>"`):
                The special `<|audio_eos|>` token. We use 128012 as the default value,
                which is the index of `<|reserved_special_token_4|>` in Llama-3.1-8B-Instruct's tokenizer.
            audio_out_bos_token (`str`, *optional*, defaults to `"<|audio_out_bos|>"`):
                The special `<|audio_out_bos|>` token. We use 128013 as the default value,
                which is the index of `<|reserved_special_token_5|>` in Llama-3.1-8B-Instruct's tokenizer.
            audio_in_token (`str`, *optional*, defaults to `"<|AUDIO|>"`):
                The special `<|AUDIO|>` token. We use 128015 as the default value,
                which is the index of `<|reserved_special_token_7|>` in Llama-3.1-8B-Instruct's tokenizer.
                This token indicates that the location should be filled in with whisper features.
            audio_out_token (`str`, *optional*, defaults to `"<|AUDIO_OUT|>"`):
                The special `<|AUDIO_OUT|>` token. We use 128016 as the default value,
                which is the index of `<|reserved_special_token_8|>` in Llama-3.1-8B-Instruct's tokenizer.
                This token indicates that the location should be filled in with audio tokens extracted via audio tokenizer.
            audio_in_token_idx (`int`, *optional*, defaults to 128015):
                The token ID corresponding to `audio_in_token` ("<|AUDIO|>").
                Used to indicate positions in the input sequence where audio features
                (e.g., whisper features) should be inserted.
            audio_out_token_idx (`int`, *optional*, defaults to 128016):
                The token ID corresponding to `audio_out_token` ("<|AUDIO_OUT|>").
                Used to indicate positions in the output sequence where audio tokens
                generated by the audio tokenizer should appear.
            audio_out_bos_token_id (`int`, *optional*, defaults to 128013):
                The token ID corresponding to `audio_out_bos_token` ("<|audio_out_bos|>").
                Marks the beginning of an audio output segment.
            audio_eos_token_id (`int`, *optional*, defaults to 128012):
                The token ID corresponding to `audio_eos_token` ("<|audio_eos|>").
                Marks the end of an audio segment.
            vocab_size (`int`, *optional*, defaults to 32000):
                Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`LlamaModel`]
            hidden_size (`int`, *optional*, defaults to 4096):
                Dimension of the hidden representations.
            intermediate_size (`int`, *optional*, defaults to 11008):
                Dimension of the MLP representations.
            num_hidden_layers (`int`, *optional*, defaults to 32):
                Number of hidden layers in the Transformer decoder.
            num_attention_heads (`int`, *optional*, defaults to 32):
                Number of attention heads for each attention layer in the Transformer decoder.
            num_key_value_heads (`int`, *optional*):
                This is the number of key_value heads that should be used to implement Grouped Query Attention. If
                `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
                `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
                converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
                by meanpooling all the original heads within that group. For more details, check out [this
                paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
                `num_attention_heads`.
            hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
                The non-linear activation function (function or string) in the decoder.
            max_position_embeddings (`int`, *optional*, defaults to 2048):
                The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
                Llama 2 up to 4096, CodeLlama up to 16384.
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            rms_norm_eps (`float`, *optional*, defaults to 1e-06):
                The epsilon used by the rms normalization layers.
            use_cache (`bool`, *optional*, defaults to `True`):
                Whether or not the model should return the last key/values attentions (not used by all models). Only
                relevant if `config.is_decoder=True`.
            pad_token_id (`int`, *optional*, defaults to 128001):
                Padding token id.
            bos_token_id (`int`, *optional*, defaults to 1):
                Beginning of stream token id.
            eos_token_id (`int`, *optional*, defaults to 2):
                End of stream token id.
            pretraining_tp (`int`, *optional*, defaults to 1):
                Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
                document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
                understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
                results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
            tie_word_embeddings (`bool`, *optional*, defaults to `False`):
                Whether to tie weight embeddings
            rope_theta (`float`, *optional*, defaults to 10000.0):
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
                        Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                        `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                        size divided by the number of attention heads divided by 2
                    `low_freq_factor` (`float`, *optional*):
                        Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                    `high_freq_factor` (`float`, *optional*):
                        Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
            attention_bias (`bool`, *optional*, defaults to `False`):
                Whether to use a bias in the query, key, value and output projection layers during self-attention.
            attention_dropout (`float`, *optional*, defaults to 0.0):
                The dropout ratio for the attention probabilities.
            mlp_bias (`bool`, *optional*, defaults to `False`):
                Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
            head_dim (`int`, *optional*):
                The attention head dimension. If None, it will default to hidden_size // num_attention_heads
    """

    model_type = "higgs_audio"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `HiggsAudioModel`
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
        # Higgs-Audio specific
        audio_adapter_type="dual_ffn_fast_forward",
        audio_dual_ffn_layers=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ],
        encode_audio_in_tokens=True,
        use_delay_pattern=True,
        use_audio_out_embed_projector=False,
        audio_num_codebooks=8,
        audio_codebook_size=1024,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_out_bos_token="<|audio_out_bos|>",
        audio_in_token="<|AUDIO|>",
        audio_out_token="<|AUDIO_OUT|>",
        audio_in_token_idx=128015,
        audio_out_token_idx=128016,
        audio_out_bos_token_id=128013,
        audio_eos_token_id=128012,
        # Llama specific
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=128001,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        if audio_adapter_type not in [
            "stack",
            "dual_ffn_fast_forward",
        ]:
            raise ValueError("Invalid audio adapter type: {audio_adapter_type}")
        if audio_adapter_type.startswith("dual_ffn"):
            if audio_dual_ffn_layers is None:
                raise ValueError("audio_dual_ffn_layers must be specified when using dual_ffn adapter.")

        self.audio_adapter_type = audio_adapter_type
        self.audio_dual_ffn_layers = audio_dual_ffn_layers
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.use_delay_pattern = use_delay_pattern
        self.use_audio_out_embed_projector = use_audio_out_embed_projector
        self.audio_num_codebooks = audio_num_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_out_bos_token = audio_out_bos_token
        self.audio_in_token = audio_in_token
        self.audio_out_token = audio_out_token
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.pad_token_id = pad_token_id


__all__ = ["HiggsAudioConfig"]
