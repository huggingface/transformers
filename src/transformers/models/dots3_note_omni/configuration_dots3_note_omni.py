# Copyright 2026 The rednote-hilab team and the HuggingFace Inc. team. All rights reserved.
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
from ...utils import auto_docstring


@auto_docstring(checkpoint="rednote-hilab/dots3.note.omni")
@strict
class Dots3NoteOmniVisionConfig(PreTrainedConfig):
    r"""
    Configuration of the Dots3-Note vision encoder.

    embed_dim (`int`, *optional*, defaults to 1536):
        Width of the patch embeddings and transformer blocks.
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether linear projections in the vision encoder use a bias.
    is_causal (`bool`, *optional*, defaults to `False`):
        Whether vision self-attention uses a causal mask.
    post_norm (`bool`, *optional*, defaults to `True`):
        Whether the released vision tower applies its final normalization.
    pre_pixel_shuffle (`bool`, *optional*, defaults to `True`):
        Whether spatial tokens are rearranged before the vision adapter.
    pyramid_num_routed (`list[int]` or `tuple[int, ...]`, *optional*):
        Number of routed tokens kept by each vision transformer layer. A value of `-1` disables routing for that
        layer.
    capacity_factor (`int`, *optional*, defaults to 2):
        Capacity multiplier used by the vision token router.
    router_scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
        Scoring function used by the vision token router. Supported values are `"sigmoid"` and `"softmax"`.
    router_scale (`float`, *optional*, defaults to 1.0):
        Scale applied to the vision router scores.
    adapter_type (`str`, *optional*, defaults to `"patch_merger"`):
        Type of adapter that projects vision tokens into the language-model width.
    adapter_in_dim (`int`, *optional*, defaults to 1536):
        Input width of the vision adapter. It must match `embed_dim`.
    adapter_out_dim (`int`, *optional*, defaults to 5120):
        Output width of the vision adapter. It must match the text hidden size.
    adapter_merge_size (`int`, *optional*, defaults to 2):
        Spatial merge factor used by the vision adapter.
    """

    model_type = "dots3_note_omni_vision_encoder"
    base_config_key = "vision_config"

    embed_dim: int = 1536
    hidden_size: int = 5120
    intermediate_size: int = 4224
    moe_intermediate_size: int = 2112
    num_hidden_layers: int = 42
    num_attention_heads: int = 24
    num_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 1
    rms_norm_eps: float = 1e-5
    use_bias: bool = False
    use_qk_norm: bool = True
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    is_causal: bool = False
    post_norm: bool = True
    pre_pixel_shuffle: bool = True
    pyramid_num_routed: list[int] | tuple[int, ...] | None = None
    capacity_factor: int = 2
    router_scoring_func: str = "sigmoid"
    router_scale: float = 1.0
    adapter_type: str = "patch_merger"
    adapter_in_dim: int = 1536
    adapter_out_dim: int = 5120
    adapter_merge_size: int = 2

    def __post_init__(self, **kwargs):
        if self.pyramid_num_routed is None:
            self.pyramid_num_routed = [-1] * 25 + list(range(4, 65, 4)) + [64]
        else:
            self.pyramid_num_routed = list(self.pyramid_num_routed)

        if len(self.pyramid_num_routed) != self.num_hidden_layers:
            raise ValueError("pyramid_num_routed must contain one entry per vision layer")
        if self.embed_dim % self.num_attention_heads != 0:
            raise ValueError("embed_dim must be divisible by num_attention_heads")
        if self.adapter_in_dim != self.embed_dim:
            raise ValueError("adapter_in_dim must match embed_dim")
        if self.adapter_out_dim != self.hidden_size:
            raise ValueError("adapter_out_dim must match hidden_size")
        if self.adapter_merge_size != self.spatial_merge_size:
            raise ValueError("adapter_merge_size must match spatial_merge_size")
        if self.router_scoring_func not in {"sigmoid", "softmax"}:
            raise ValueError("router_scoring_func must be 'sigmoid' or 'softmax'")
        if self.temporal_patch_size != 1:
            raise ValueError("Dots3-Note vision preprocessing requires temporal_patch_size=1")
        if self.is_causal:
            raise ValueError("Dots3-Note vision attention requires is_causal=False")
        if not self.pre_pixel_shuffle or self.adapter_type != "patch_merger":
            raise ValueError("Dots3-Note requires pre_pixel_shuffle=True and adapter_type='patch_merger'")
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="rednote-hilab/dots3.note.omni")
@strict
class Dots3NoteOmniAudioConfig(PreTrainedConfig):
    r"""
    Configuration of the Dots3-Note audio encoder and adapter.

    encoder_type (`str`, *optional*, defaults to `"dots"`):
        Identifier of the audio encoder architecture stored in the released checkpoint.
    whisper_config (`dict`, *optional*):
        Configuration of the Whisper-style audio transformer. When omitted, the released 32-layer configuration is
        used.
    feature_size (`int`, *optional*, defaults to 128):
        Number of log-mel filter-bank channels.
    n_fft (`int`, *optional*, defaults to 400):
        FFT window size used by the audio feature extractor.
    hop_length (`int`, *optional*, defaults to 160):
        Number of waveform samples between adjacent log-mel frames.
    chunk_seconds (`int`, *optional*, defaults to 60):
        Maximum duration, in seconds, of each audio encoder chunk.
    use_conv2d_stem (`bool`, *optional*, defaults to `True`):
        Whether to use the released two-dimensional convolutional subsampling stem.
    use_rope (`bool`, *optional*, defaults to `True`):
        Whether the audio transformer uses rotary position embeddings.
    use_rms_norm (`bool`, *optional*, defaults to `True`):
        Whether the audio transformer uses RMS normalization.
    use_causal (`bool`, *optional*, defaults to `False`):
        Whether audio self-attention uses a causal mask.
    downsample_hidden_size (`int`, *optional*, defaults to 480):
        Hidden width of the temporal downsampling projection.
    conv_chunksize (`int`, *optional*, defaults to 500):
        Number of mel frames processed per convolutional stem chunk.
    conv_stem_gradient_checkpointing (`bool`, *optional*, defaults to `False`):
        Whether to checkpoint the convolutional stem during training.
    conv_bucket_step (`int`, *optional*, defaults to 10):
        Length step used to bucket variable-length audio chunks. Set to `None` to disable bucketing.
    conv_bucket_max_elements (`int`, *optional*, defaults to 20000):
        Maximum number of elements assigned to one convolution bucket. Set to `None` to disable the limit.
    attention_backend (`str`, *optional*, defaults to `"sdpa"`):
        Attention backend used by the audio encoder.
    merge_factor (`int`, *optional*, defaults to 1):
        Additional temporal merge factor applied after convolutional subsampling.
    adapter_input_size (`int`, *optional*, defaults to 1280):
        Input width of the audio-to-text adapter.
    adapter_output_size (`int`, *optional*, defaults to 5120):
        Output width of the audio-to-text adapter. It must match the text hidden size.
    audio_start_token (`str`, *optional*, defaults to `"<|audio_comp_start|>"`):
        Token delimiting the start of an audio span.
    audio_pad_token (`str`, *optional*, defaults to `"<|audio_comp_pad|>"`):
        Placeholder token replaced with encoded audio features.
    audio_end_token (`str`, *optional*, defaults to `"<|audio_comp_end|>"`):
        Token delimiting the end of an audio span.
    """

    model_type = "dots3_note_omni_audio_encoder"
    base_config_key = "audio_config"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
        "intermediate_size": "encoder_ffn_dim",
    }

    encoder_type: str = "dots"
    whisper_config: dict | None = None
    sampling_rate: int = 16_000
    feature_size: int = 128
    n_fft: int = 400
    hop_length: int = 160
    chunk_seconds: int = 60
    use_conv2d_stem: bool = True
    use_rope: bool = True
    use_rms_norm: bool = True
    use_causal: bool = False
    downsample_hidden_size: int = 480
    conv_chunksize: int = 500
    conv_stem_gradient_checkpointing: bool = False
    conv_bucket_step: int | None = 10
    conv_bucket_max_elements: int | None = 20_000
    rope_parameters: dict | None = None
    attention_backend: str = "sdpa"
    merge_factor: int = 1
    adapter_input_size: int = 1280
    adapter_output_size: int = 5120
    audio_start_token: str = "<|audio_comp_start|>"
    audio_pad_token: str = "<|audio_comp_pad|>"
    audio_end_token: str = "<|audio_comp_end|>"

    def __post_init__(self, **kwargs):
        # Aliases used by the released configuration.
        self.adapter_input_size = kwargs.pop("whisper_adapter_in_dim", self.adapter_input_size)
        self.adapter_output_size = kwargs.pop("whisper_adapter_out_dim", self.adapter_output_size)
        self.audio_start_token = kwargs.pop("audio_comp_start", self.audio_start_token)
        self.audio_pad_token = kwargs.pop("audio_comp_span", self.audio_pad_token)
        self.audio_end_token = kwargs.pop("audio_comp_end", self.audio_end_token)

        if self.whisper_config is None:
            self.whisper_config = {
                "d_model": 1280,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "encoder_layers": 32,
                "num_mel_bins": 128,
                "max_source_positions": 6000,
                "activation_function": "swiglu",
            }
        if self.rope_parameters is None:
            self.rope_parameters = {"partial_rotary_factor": 0.5, "rope_theta": 10_000.0}

        if min(self.sampling_rate, self.feature_size, self.n_fft, self.hop_length) <= 0:
            raise ValueError("audio sampling and feature dimensions must be positive")
        if self.chunk_seconds <= 0 or self.merge_factor <= 0:
            raise ValueError("chunk_seconds and merge_factor must be positive")
        if self.encoder_type != "dots":
            raise ValueError("Dots3-Note only supports encoder_type='dots'")
        if not self.use_conv2d_stem:
            raise ValueError("the released Dots3-Note audio encoder requires use_conv2d_stem=True")
        if not self.use_rope or self.use_causal:
            raise ValueError("Dots3-Note audio inference requires use_rope=True and use_causal=False")
        if self.whisper_config.get("activation_function") != "swiglu":
            raise ValueError("the released Dots3-Note audio encoder requires activation_function='swiglu'")
        if self.adapter_input_size != self.whisper_config["d_model"]:
            raise ValueError("adapter_input_size must match whisper_config.d_model")
        if self.attention_backend not in {"sdpa", "flash_attention_2", "flash_attention_3"}:
            raise ValueError("unsupported audio attention backend")
        super().__post_init__(**kwargs)

    @property
    def conv_temporal_stride(self) -> int:
        return 8 if self.use_conv2d_stem else 2

    @property
    def token_stride(self) -> int:
        return self.hop_length * self.conv_temporal_stride * self.merge_factor

    @property
    def chunk_samples(self) -> int:
        return self.chunk_seconds * self.sampling_rate

    @property
    def chunk_mel_frames(self) -> int:
        return self.chunk_seconds * 100


@auto_docstring(checkpoint="rednote-hilab/dots3.note.omni")
@strict
class Dots3NoteOmniConfig(PreTrainedConfig):
    r"""
    Configuration for the Dots3-Note Omni multimodal causal language model.

    seq_length (`int`, *optional*, defaults to 393216):
        Sequence length used during pretraining. This is retained for checkpoint compatibility.
    rope_theta (`float`, *optional*, defaults to 80000000.0):
        Base period of the rotary position embeddings used by full-attention layers.
    rope_scaling (`dict`, *optional*):
        Legacy rotary scaling parameters retained for compatibility with the released configuration.
    normalization (`str`, *optional*, defaults to `"RMSNorm"`):
        Normalization type used inside decoder layers.
    final_norm (`str`, *optional*, defaults to `"RMSNorm"`):
        Normalization type used after the final decoder layer.
    multi_latent_attention (`bool`, *optional*, defaults to `True`):
        Whether full-attention layers use multi-head latent attention.
    k_rope_only_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to normalize the rotary portion of compressed keys independently.
    apply_mla_qkv_lora_rescale (`bool`, *optional*, defaults to `True`):
        Whether to apply the checkpoint-compatible scale to low-rank MLA query, key, and value projections.
    attention_gate_type (`str`, *optional*, defaults to `"headwise"`):
        Granularity of output gates in full-attention layers.
    softmax_type (`str`, *optional*, defaults to `"vanilla"`):
        Softmax variant used by full-attention layers.
    sliding_window_size (`int`, *optional*, defaults to 512):
        Window length used by sliding-attention layers.
    swa_num_attention_heads (`int`, *optional*, defaults to 64):
        Number of query heads in sliding-window attention.
    swa_num_key_value_heads (`int`, *optional*, defaults to 64):
        Number of key/value heads in sliding-window attention.
    swa_q_lora_rank (`int`, *optional*, defaults to 1024):
        Rank of the sliding-attention query projection.
    swa_kv_lora_rank (`int`, *optional*, defaults to 1024):
        Rank of the sliding-attention key/value projection.
    swa_head_dim (`int`, *optional*, defaults to 256):
        Total per-head width in sliding-window attention.
    swa_qk_nope_head_dim (`int`, *optional*, defaults to 192):
        Non-rotary query/key width per sliding-attention head.
    swa_qk_rope_head_dim (`int`, *optional*, defaults to 64):
        Rotary query/key width per sliding-attention head.
    swa_v_head_dim (`int`, *optional*, defaults to 128):
        Value width per sliding-attention head.
    swa_rope_theta (`float`, *optional*, defaults to 50000.0):
        Base period of the rotary position embeddings in sliding-attention layers.
    swa_attention_gate_type (`str`, *optional*, defaults to `"headwise"`):
        Granularity of output gates in sliding-attention layers.
    index_n_heads (`int`, *optional*, defaults to 64):
        Number of heads in the dynamic sparse attention indexer.
    index_head_dim (`int`, *optional*, defaults to 128):
        Per-head width of the dynamic sparse attention indexer.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of key positions selected by dynamic sparse attention.
    use_dsa (`bool`, *optional*, defaults to `True`):
        Whether full-attention layers use the dynamic sparse attention indexer.
    shared_experts_intermediate_size (`int`, *optional*, defaults to 1536):
        Intermediate width of each shared expert.
    moe_shared_expert_intermediate_size (`int`, *optional*, defaults to 1536):
        Compatibility alias for the shared expert intermediate width in the released checkpoint.
    moe_layer_freq (`int` or `list[int]`, *optional*, defaults to 1):
        Frequency or explicit pattern of sparse MoE decoder layers.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of initial decoder layers that use a dense MLP instead of routed experts.
    scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
        Scoring function used by the expert router.
    topk_method (`str`, *optional*, defaults to `"noaux_tc"`):
        Expert selection method used by the router.
    moe_topk (`int`, *optional*, defaults to 8):
        Compatibility alias for the number of experts selected per token.
    moe_gating_fp32 (`bool`, *optional*, defaults to `False`):
        Whether to compute expert router logits in float32.
    n_group (`int`, *optional*, defaults to 1):
        Number of groups into which routed experts are partitioned during selection.
    use_dynamic_rsf (`bool`, *optional*, defaults to `False`):
        Whether to derive the routed scaling factor dynamically from selected expert scores.
    image_start_token_id (`int`, *optional*, defaults to 151661):
        Token id delimiting the start of an image span.
    image_end_token_id (`int`, *optional*, defaults to 151662):
        Token id delimiting the end of an image span.
    audio_start_token_id (`int`, *optional*, defaults to 151718):
        Token id delimiting the start of an audio span.
    audio_end_token_id (`int`, *optional*, defaults to 151719):
        Token id delimiting the end of an audio span.
    """

    model_type = "dots3_note_omni"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    sub_configs = {
        "vision_config": Dots3NoteOmniVisionConfig,
        "audio_config": Dots3NoteOmniAudioConfig,
    }
    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 152064
    hidden_size: int = 5120
    intermediate_size: int = 13824
    num_hidden_layers: int = 46
    num_attention_heads: int = 128
    num_key_value_heads: int | None = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 393216
    seq_length: int = 393216
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 151659
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151668
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 80_000_000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 256

    normalization: str = "RMSNorm"
    final_norm: str = "RMSNorm"
    multi_latent_attention: bool = True
    q_lora_rank: int | None = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    k_rope_only_layernorm: bool | None = True
    apply_mla_qkv_lora_rescale: bool = True
    qk_layernorm: bool = True
    attention_gate_type: str = "headwise"
    softmax_type: str = "vanilla"

    use_sliding_window: bool = True
    sliding_window_size: int = 512
    sliding_window: int | None = None
    layer_types: list[str] | tuple[str, ...] | None = None
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 64
    swa_q_lora_rank: int = 1024
    swa_kv_lora_rank: int = 1024
    swa_head_dim: int = 256
    swa_qk_nope_head_dim: int = 192
    swa_qk_rope_head_dim: int = 64
    swa_v_head_dim: int = 128
    swa_rope_theta: float = 50_000.0
    swa_attention_gate_type: str = "headwise"

    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    use_dsa: bool = True

    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536
    shared_experts_intermediate_size: int = 1536
    moe_shared_expert_intermediate_size: int = 1536
    moe_layer_freq: int | list[int] = 1
    first_k_dense_replace: int = 1
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    routed_scaling_factor: float = 1.0
    moe_topk: int = 8
    moe_gating_fp32: bool = False
    n_group: int = 1
    topk_group: int = 1
    use_dynamic_rsf: bool = False

    vision_config: dict | Dots3NoteOmniVisionConfig | None = None
    audio_config: dict | Dots3NoteOmniAudioConfig | None = None
    image_token_id: int = 151660
    image_start_token_id: int = 151661
    image_end_token_id: int = 151662
    video_token_id: int = 151680
    audio_start_token_id: int = 151718
    audio_end_token_id: int = 151719
    audio_token_id: int = 151720

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("Dots3-Note requires num_key_value_heads to match num_attention_heads")
        if self.swa_num_key_value_heads != self.swa_num_attention_heads:
            raise ValueError("Dots3-Note requires swa_num_key_value_heads to match swa_num_attention_heads")
        if self.attention_gate_type not in {"headwise", "elementwise"}:
            raise ValueError(f"Unsupported attention_gate_type: {self.attention_gate_type!r}")
        if self.swa_attention_gate_type not in {"headwise", "elementwise"}:
            raise ValueError(f"Unsupported swa_attention_gate_type: {self.swa_attention_gate_type!r}")
        if self.sliding_window is None:
            self.sliding_window = self.sliding_window_size
        if self.k_rope_only_layernorm is None:
            # The released JSON contains null for this historical field, while the
            # checkpoint includes the corresponding RMSNorm weights.
            self.k_rope_only_layernorm = True

        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if not self.use_sliding_window or i < 2 or i % 4 == 1 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = list(self.layer_types)
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per hidden layer")
        self.layer_types = [
            "deepseek_sparse_attention"
            if self.use_dsa and layer_type == "full_attention"
            else "full_attention"
            if not self.use_dsa and layer_type == "deepseek_sparse_attention"
            else layer_type
            for layer_type in self.layer_types
        ]
        unsupported = set(self.layer_types) - {"full_attention", "sliding_attention", "deepseek_sparse_attention"}
        if unsupported:
            raise ValueError(f"Unsupported layer types: {sorted(unsupported)}")
        if self.index_head_dim != 128:
            raise ValueError(f"Dots3 Note Omni requires index_head_dim=128, got {self.index_head_dim}")
        if self.qk_rope_head_dim > self.index_head_dim:
            raise ValueError("qk_rope_head_dim must not exceed index_head_dim")
        if self.n_group < 1 or self.n_routed_experts % self.n_group != 0:
            raise ValueError("n_group must evenly divide n_routed_experts")
        if self.topk_group < 1 or self.topk_group > self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")

        if self.vision_config is None:
            self.vision_config = Dots3NoteOmniVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = Dots3NoteOmniVisionConfig(**self.vision_config)
        if self.audio_config is None:
            self.audio_config = Dots3NoteOmniAudioConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = Dots3NoteOmniAudioConfig(**self.audio_config)
        if self.vision_config.hidden_size != self.hidden_size:
            raise ValueError("vision adapter output width must match the text hidden size")
        if self.audio_config.adapter_output_size != self.hidden_size:
            raise ValueError("audio adapter output width must match the text hidden size")

        quantization_config = kwargs.get("quantization_config")
        if isinstance(quantization_config, dict) and quantization_config.get("quant_method") == "fp8":
            for name, expected in (("activation_scheme", "dynamic"), ("fmt", "e4m3")):
                actual = quantization_config.get(name, expected if name == "fmt" else None)
                if actual != expected:
                    raise ValueError(f"Dots3 Note Omni FP8 requires {name}={expected!r}, got {actual!r}")
            block_size = quantization_config.get("weight_block_size")
            if not isinstance(block_size, (list, tuple)) or tuple(block_size) != (128, 128):
                raise ValueError(f"Dots3 Note Omni FP8 requires weight_block_size=(128, 128), got {block_size!r}")
            scale_fmt = quantization_config.get("scale_fmt", "float")
            if scale_fmt != "float":
                raise ValueError(f"Dots3 Note Omni FP8 requires scale_fmt='float', got {scale_fmt!r}")
            # Distinguish a pre-quantized checkpoint before the generic quantizer runs after model construction.
            self._is_quantized = True
            quantization_config.setdefault("modules_to_not_convert", ["vision_encoder", "audio_encoder", "lm_head"])
        super().__post_init__(**kwargs)


__all__ = [
    "Dots3NoteOmniAudioConfig",
    "Dots3NoteOmniConfig",
    "Dots3NoteOmniVisionConfig",
]
