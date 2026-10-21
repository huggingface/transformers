# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
"""Configurations for the MiMo V2.6 multimodal architecture."""

import inspect

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLTextConfig(PreTrainedConfig):
    r"""
    Configure the hybrid global/sliding-window MoE decoder of the MiMo V2 VL series.

    `hybrid_layer_pattern` uses 0 for global and 1 for sliding attention. `moe_layer_freq` specifies which
    layers use routed experts. Sliding layers have independent query/KV head counts and dimensions.
    `sliding_window_size` counts preceding tokens; the HF cache window additionally includes the query token.

    swa_num_attention_heads (`int`, *optional*):
        Number of query heads in sliding-attention layers.
    swa_num_key_value_heads (`int`, *optional*):
        Number of key/value heads in sliding-attention layers.
    swa_head_dim (`int`, *optional*):
        Query/key head dimension in sliding-attention layers.
    swa_v_head_dim (`int`, *optional*):
        Value head dimension in sliding-attention layers.
    rope_theta (`float` or `int`, *optional*):
        Rotary embedding base for global-attention layers.
    swa_rope_theta (`float` or `int`, *optional*):
        Rotary embedding base for sliding-attention layers.
    partial_rotary_factor (`float`, *optional*):
        Fraction of each query/key head rotated by RoPE.
    attention_value_scale (`float`, *optional*):
        Multiplier applied to value projections before attention. `None` disables scaling.
    add_full_attention_sink_bias (`bool`, *optional*):
        Whether global-attention layers include learned softmax sinks.
    add_swa_attention_sink_bias (`bool`, *optional*):
        Whether sliding-attention layers include learned softmax sinks.
    sliding_window_size (`int`, *optional*):
        Number of preceding tokens visible to sliding attention.
    hybrid_layer_pattern (`list[int]`, *optional*):
        One value per layer: 0 for global attention and 1 for sliding attention.
    moe_layer_freq (`list[int]` or `int`, *optional*):
        Per-layer expert flags, or a frequency selecting every nth layer as an expert layer.
    n_group (`int`, *optional*):
        Number of groups used when selecting routed experts.
    moe_router_dtype (`str`, *optional*):
        Dtype of the router projection, with routing probabilities accumulated in float32.
    """

    model_type = "mimo_v2_vl_text"
    base_config_key = "text_config"
    attribute_map = {"num_local_experts": "n_routed_experts", "layernorm_epsilon": "rms_norm_eps"}
    vocab_size: int = 152576
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 192
    v_head_dim: int = 128
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 192
    swa_v_head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    max_position_embeddings: int = 1048576
    rope_theta: float | int = 10000000.0
    swa_rope_theta: float | int = 10000.0
    partial_rotary_factor: float = 0.334
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attention_value_scale: float | None = 0.707
    add_full_attention_sink_bias: bool = False
    add_swa_attention_sink_bias: bool = True
    sliding_window_size: int = 128
    sliding_window: int | None = None
    hybrid_layer_pattern: list[int] | None = None
    layer_types: list[str] | None = None
    moe_layer_freq: list[int] | int | None = None
    n_routed_experts: int = 256
    moe_intermediate_size: int = 2048
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float | None = 1.0
    moe_router_dtype: str = "float32"
    use_cache: bool = True
    tie_word_embeddings: bool = False
    pad_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151645

    def __post_init__(self, **kwargs):
        if "layernorm_epsilon" in kwargs:
            self.rms_norm_eps = kwargs.pop("layernorm_epsilon")
        self.sliding_window = self.sliding_window_size + 1
        if self.hybrid_layer_pattern is None:
            self.hybrid_layer_pattern = [int(i != 0 and (i + 1) % 6 != 0) for i in range(self.num_hidden_layers)]
        if self.moe_layer_freq is None:
            self.moe_layer_freq = [0] + [1] * (self.num_hidden_layers - 1)
        elif isinstance(self.moe_layer_freq, int):
            frequency = self.moe_layer_freq
            self.moe_layer_freq = [int(frequency > 0 and i % frequency == 0) for i in range(self.num_hidden_layers)]
        if (
            len(self.hybrid_layer_pattern) != self.num_hidden_layers
            or len(self.moe_layer_freq) != self.num_hidden_layers
        ):
            raise ValueError("Attention and MoE layer patterns must match num_hidden_layers.")
        self.layer_types = ["sliding_attention" if kind else "full_attention" for kind in self.hybrid_layer_pattern]
        if self.routed_scaling_factor is None:
            self.routed_scaling_factor = 1.0
        for num_heads, num_kv_heads, head_dim in (
            (self.num_attention_heads, self.num_key_value_heads, self.head_dim),
            (self.swa_num_attention_heads, self.swa_num_key_value_heads, self.swa_head_dim),
        ):
            rotary_dim = int(head_dim * self.partial_rotary_factor)
            if num_heads % num_kv_heads or rotary_dim <= 0 or rotary_dim % 2:
                raise ValueError(
                    "Query heads must be divisible by KV heads and rotary dimensions must be positive and even."
                )
        if self.n_routed_experts % self.n_group or not 1 <= self.topk_group <= self.n_group:
            raise ValueError("Expert groups must partition the experts and topk_group must select valid groups.")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be between one and n_routed_experts.")
        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        return kwargs


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLVisionConfig(PreTrainedConfig):
    r"""
    Configure the spatial/temporal patch encoder and merger.

    `qk_channels` is the attention head dimension, independently of `hidden_size // num_heads`.
    `vit_window_attn_types` selects row order (0), column order (1), or global layers (-1).
    `fullatt_block_indexes` disables sliding masks and sinks for the listed layers.

    qk_channels (`int`, *optional*):
        Query/key head dimension, independent of hidden size divided by head count.
    out_hidden_size (`int`, *optional*):
        Output dimension of the patch merger; must match the language model's hidden size.
    merger_bias (`bool`, *optional*):
        Whether the patch merger's linear projections include biases.
    use_sink (`bool`, *optional*):
        Whether windowed vision attention includes learned softmax sinks.
    visual_token_window_size (`int`, *optional*):
        Number of visible patches on each side of a query in windowed attention.
    fullatt_block_indexes (`list[int]`, *optional*):
        Indices of vision blocks that use full attention.
    vit_window_attn_types (`list[int]`, *optional*):
        Per-block attention traversal: row order (0), column order (1), or global (-1).
    """

    model_type = "mimo_v2_vl_vision"
    base_config_key = "vision_config"
    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    hidden_act: str = "silu"
    num_heads: int = 32
    num_key_value_heads: int = 8
    qk_channels: int = 64
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096
    rms_norm_eps: float = 1e-6
    merger_bias: bool = False
    initializer_range: float = 0.02
    use_sink: bool = True
    visual_token_window_size: int = 64
    fullatt_block_indexes: list[int] | None = None
    vit_window_attn_types: list[int] | None = None

    def __post_init__(self, **kwargs):
        if "in_chans" in kwargs:
            self.in_channels = kwargs.pop("in_chans")
        if self.fullatt_block_indexes is None:
            self.fullatt_block_indexes = [i for i in (0, 9, 18, 27) if i < self.depth]
        if self.vit_window_attn_types is None:
            self.vit_window_attn_types = [
                -1 if i in self.fullatt_block_indexes else int(i % 9 >= 5) for i in range(self.depth)
            ]
        if len(self.vit_window_attn_types) != self.depth or any(
            kind not in (-1, 0, 1) for kind in self.vit_window_attn_types
        ):
            raise ValueError("vit_window_attn_types must contain one of -1, 0, or 1 for every vision layer.")
        if self.num_heads % self.num_key_value_heads or self.qk_channels % 4:
            raise ValueError("Vision heads must be divisible by KV heads and qk_channels must be divisible by four.")
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLAudioConfig(PreTrainedConfig):
    r"""
    Configure codec-channel embeddings, the local audio transformer, and its language-space projection.

    Each group of `group_size` codec frames is a separate local-attention sequence. Channel embeddings are summed
    before the local transformer. `speech_vocab_size` and `speech_zeroemb_idx` accept a scalar or per-channel list;
    the original checkpoint also stores these as strings, optionally separated by hyphens.

    group_size (`int`, *optional*):
        Number of consecutive codec frames represented by one language-model token.
    input_local_dim (`int`, *optional*):
        Hidden size of the local audio transformer and codec embeddings.
    input_local_layers (`int`, *optional*):
        Number of local audio transformer layers.
    input_local_attn_heads (`int`, *optional*):
        Number of attention heads in the local audio transformer.
    input_local_head_dim (`int`, *optional*):
        Dimension of each local audio attention head.
    input_local_intermediate_size (`int`, *optional*):
        Intermediate dimension of the local audio feed-forward layers.
    input_local_hidden_dropout (`float`, *optional*):
        Dropout probability for local audio attention.
    input_full_attention (`bool`, *optional*):
        Whether codec frames within a group attend bidirectionally instead of causally.
    add_post_norm (`bool`, *optional*):
        Whether to apply RMS normalization after the local audio transformer.
    out_hidden_size (`int`, *optional*):
        Audio projection output dimension; must match the language model's hidden size.
    projection_layers (`int`, *optional*):
        Number of audio projection layers, either one or two.
    rope_theta (`float` or `int`, *optional*):
        Rotary embedding base of the local audio transformer.
    speech_vocab_size (`str`, `int`, or `list[int]`, *optional*):
        Codec vocabulary size, shared or specified per channel.
    speech_zeroemb_idx (`str`, `int`, or `list[int]`, *optional*):
        Padding code ID, shared or specified per channel.
    audio_segment_size (`int`, *optional*):
        Maximum number of mel frames in each learned audio-tokenizer segment.
    """

    model_type = "mimo_v2_vl_audio"
    base_config_key = "audio_config"
    audio_channels: int = 20
    group_size: int = 4
    input_local_dim: int = 1024
    input_local_layers: int = 6
    input_local_attn_heads: int = 16
    input_local_head_dim: int = 64
    input_local_intermediate_size: int = 4096
    input_local_hidden_dropout: float = 0.0
    input_full_attention: bool = True
    add_post_norm: bool = True
    out_hidden_size: int = 4096
    projection_layers: int = 2
    rope_theta: float | int = 640000.0
    rms_norm_eps: float = 1e-6
    speech_vocab_size: str | int | list[int] = "1280"
    speech_zeroemb_idx: str | int | list[int] = "1024"
    audio_segment_size: int = 6000

    def __post_init__(self, **kwargs):
        for name in ("speech_vocab_size", "speech_zeroemb_idx"):
            values = getattr(self, name)
            if isinstance(values, str):
                values = [int(value) for value in values.split("-")]
            elif isinstance(values, int):
                values = [values]
            if len(values) == 1:
                values = values * self.audio_channels
            if len(values) != self.audio_channels:
                raise ValueError(f"{name} must contain one value per audio channel.")
            setattr(self, name, values)
        if self.projection_layers not in (1, 2):
            raise ValueError("projection_layers must be one or two.")
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLAudioTokenizerConfig(PreTrainedConfig):
    r"""
    Configure the learned mel-to-codec encoder distributed in the checkpoint's `audio_tokenizer` directory.

    Only the encoder and residual vector quantizer are used for multimodal understanding. Speech synthesis decoder
    and vocoder weights are not required for text generation from audio inputs.

    n_mels (`int`, *optional*):
        Number of mel bins expected by the first audio convolution.
    stride_size (`int`, *optional*):
        Stride of the second audio convolution.
    avg_pooler (`int`, *optional*):
        Temporal downsampling factor after the tokenizer transformer.
    encoder_skip_layer_id (`int`, *optional*):
        One-based layer index whose output is added after the final encoder layer.
    encoder_causal (`bool`, *optional*):
        Whether the tokenizer encoder uses causal attention.
    encoder_attn_window_size (`list[int]`, *optional*):
        Left and right attention-window sizes; -1 denotes an unbounded side.
    hybrid_attention (`bool`, *optional*):
        Whether to alternate windowed and global audio attention.
    swa_per_block (`int`, *optional*):
        Number of layers per attention cycle; the last layer of each cycle has no window restriction.
    rope_theta (`float` or `int`, *optional*):
        Rotary embedding base for the learned audio tokenizer.
    num_quantizers (`int`, *optional*):
        Number of residual vector quantization stages and output codec channels.
    """

    model_type = "mimo_v2_vl_audio_tokenizer"
    base_config_key = "audio_tokenizer_config"
    attribute_map = {"d_model": "hidden_size"}
    hidden_size: int = 1024
    n_mels: int = 128
    kernel_size: int = 3
    stride_size: int = 2
    avg_pooler: int = 2
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    encoder_skip_layer_id: int | None = 3
    encoder_causal: bool = True
    encoder_attn_window_size: list[int] | None = None
    hybrid_attention: bool = True
    swa_per_block: int = 2
    activation_function: str = "gelu"
    rope_theta: float | int = 10000.0
    num_quantizers: int = 20
    codebook_size: list[int] | None = None
    layer_norm_eps: float = 1e-5

    def __post_init__(self, **kwargs):
        if "d_model" in kwargs:
            self.hidden_size = kwargs.pop("d_model")
        if self.encoder_attn_window_size is None:
            self.encoder_attn_window_size = [128, 0]
        if self.codebook_size is None:
            self.codebook_size = [1024, 1024, 256] + [128] * max(0, self.num_quantizers - 3)
        if len(self.codebook_size) < self.num_quantizers:
            self.codebook_size = self.codebook_size + [self.codebook_size[-1]] * (
                self.num_quantizers - len(self.codebook_size)
            )
        if self.d_model % self.encoder_attention_heads or (self.d_model // self.encoder_attention_heads) % 2:
            raise ValueError("Audio tokenizer head dimensions must be integral and even.")
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLConfig(PreTrainedConfig):
    r"""
    Configuration for the MiMo V2 VL multimodal series.

    Native saves use nested text, vision, audio, and audio-tokenizer configurations. Original flat text settings are
    accepted when constructing this configuration from a released checkpoint. Original quantized weights require
    the conversion/loading utility shipped with this model to undo their TP-interleaved QKV and MXFP4 layout.

    audio_tokenizer_config (`dict` or [`MiMoV2VLAudioTokenizerConfig`], *optional*):
        Configuration of the learned mel-to-codec encoder.
    processor_config (`dict`, *optional*):
        Checkpoint preprocessing settings used when a standalone processor configuration is unavailable.
    audio_start_token_id (`int`, *optional*):
        Token ID marking the start of an audio span.
    audio_end_token_id (`int`, *optional*):
        Token ID marking the end of an audio span.
    """

    model_type = "mimo_v2_vl"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=MiMoV2VLTextConfig),
        "vision_config": SubConfigSpec(config_class=MiMoV2VLVisionConfig),
        "audio_config": SubConfigSpec(config_class=MiMoV2VLAudioConfig),
        "audio_tokenizer_config": SubConfigSpec(config_class=MiMoV2VLAudioTokenizerConfig),
    }
    keys_to_ignore_at_inference = ["past_key_values"]
    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None
    audio_tokenizer_config: dict | PreTrainedConfig | None = None
    processor_config: dict | None = None
    image_token_id: int | None = None
    video_token_id: int | None = None
    audio_token_id: int | None = None
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    audio_start_token_id: int = 151673
    audio_end_token_id: int = 151674
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_fields = set(inspect.signature(MiMoV2VLTextConfig).parameters) | {"layernorm_epsilon"}
            self.text_config = {key: kwargs.pop(key) for key in text_fields if key in kwargs}
            for name in ("pad_token_id", "eos_token_id"):
                if getattr(self, name) is not None:
                    self.text_config[name] = getattr(self, name)
        self.processor_config = self.processor_config or {}
        for name, default in (("image_token_id", 151655), ("video_token_id", 151656), ("audio_token_id", 151669)):
            if getattr(self, name) is None:
                setattr(self, name, self.processor_config.get(name, default))
        super().__post_init__(**kwargs)
        if self.pad_token_id is None:
            self.pad_token_id = self.text_config.pad_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id
        if (
            self.vision_config.out_hidden_size != self.text_config.hidden_size
            or self.audio_config.out_hidden_size != self.text_config.hidden_size
        ):
            raise ValueError("Vision and audio output dimensions must match text_config.hidden_size.")


__all__ = [
    "MiMoV2VLConfig",
    "MiMoV2VLTextConfig",
    "MiMoV2VLVisionConfig",
    "MiMoV2VLAudioConfig",
    "MiMoV2VLAudioTokenizerConfig",
]
