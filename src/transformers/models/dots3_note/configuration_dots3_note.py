# Copyright 2026 The Dots Studio team and the HuggingFace Inc. team. All rights reserved.
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


@auto_docstring(checkpoint="dots-studio/dots3-note-prev")
@strict
class Dots3NoteVisionConfig(PreTrainedConfig):
    r"""
    Configuration of the Dots 3 Note Preview vision encoder.

    embed_dim (`int`, *optional*, defaults to 1536):
        Width of the patch embeddings and transformer blocks.
    pyramid_num_routed (`list[int]` or `tuple[int, ...]`, *optional*):
        Number of routed experts in each vision transformer layer. A non-positive value selects a dense MLP.
    capacity_factor (`int`, *optional*, defaults to 2):
        Number of experts selected per token, capped by the layer's expert count.
    router_scale (`float`, *optional*, defaults to 1.0):
        Scale applied to the vision router scores.
    adapter_in_dim (`int`, *optional*, defaults to 1536):
        Input width of the vision adapter. It must match `embed_dim`.
    adapter_out_dim (`int`, *optional*, defaults to 5120):
        Output width of the vision adapter. It must match the text hidden size.
    adapter_merge_size (`int`, *optional*, defaults to 2):
        Spatial merge factor used by the vision adapter.
    """

    model_type = "dots3_note_vision_encoder"
    base_config_key = "vision_config"
    attribute_map = {"num_heads": "num_attention_heads", "use_bias": "attention_bias"}

    embed_dim: int = 1536
    hidden_size: int = 5120
    intermediate_size: int = 4224
    moe_intermediate_size: int = 2112
    num_hidden_layers: int = 42
    num_attention_heads: int = 24
    num_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    rope_parameters: dict | None = None
    initializer_range: float = 0.02
    pyramid_num_routed: list[int] | tuple[int, ...] | None = None
    capacity_factor: int = 2
    router_scale: float = 1.0
    adapter_in_dim: int = 1536
    adapter_out_dim: int = 5120
    adapter_merge_size: int = 2

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "axial", "rope_theta": 10_000.0}
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
        if self.hidden_size not in (self.embed_dim, self.adapter_out_dim):
            raise ValueError("legacy hidden_size must match adapter_out_dim")
        self.hidden_size = self.embed_dim
        self.attention_bias = kwargs.pop("use_bias", self.attention_bias)
        self.mlp_bias = self.attention_bias
        if self.mlp_bias and any(num_experts > 0 for num_experts in self.pyramid_num_routed):
            raise ValueError("Dots3 vision stacked experts require mlp_bias=False")
        self.hidden_act = "silu"
        if self.adapter_merge_size != self.spatial_merge_size:
            raise ValueError("adapter_merge_size must match spatial_merge_size")
        if kwargs.pop("router_scoring_func", "sigmoid") != "sigmoid":
            raise ValueError("Dots 3 Note Preview vision routing requires router_scoring_func='sigmoid'")
        if kwargs.pop("temporal_patch_size", 1) != 1:
            raise ValueError("Dots 3 Note Preview vision preprocessing requires temporal_patch_size=1")
        if not kwargs.pop("use_qk_norm", True) or not kwargs.pop("post_norm", True):
            raise ValueError("Dots 3 Note Preview vision requires Q/K and post-trunk normalization")
        if kwargs.pop("is_causal", False):
            raise ValueError("Dots 3 Note Preview vision attention requires is_causal=False")
        if not kwargs.pop("pre_pixel_shuffle", True) or kwargs.pop("adapter_type", "patch_merger") != "patch_merger":
            raise ValueError("Dots 3 Note Preview requires pre_pixel_shuffle=True and adapter_type='patch_merger'")
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="dots-studio/dots3-note-prev")
@strict
class Dots3NoteAudioConfig(PreTrainedConfig):
    r"""
    Configuration of the Dots 3 Note Preview audio encoder and adapter.

    feature_size (`int`, *optional*, defaults to 128):
        Number of log-mel filter-bank channels.
    hop_length (`int`, *optional*, defaults to 160):
        Number of waveform samples between adjacent log-mel frames.
    downsample_hidden_size (`int`, *optional*, defaults to 480):
        Hidden width of the temporal downsampling projection.
    adapter_input_size (`int`, *optional*, defaults to 1280):
        Input width of the audio-to-text adapter.
    adapter_output_size (`int`, *optional*, defaults to 5120):
        Output width of the audio-to-text adapter. It must match the text hidden size.
    """

    model_type = "dots3_note_audio_encoder"
    base_config_key = "audio_config"
    attribute_map = {
        "d_model": "hidden_size",
        "encoder_attention_heads": "num_attention_heads",
        "encoder_layers": "num_hidden_layers",
        "encoder_ffn_dim": "intermediate_size",
    }

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_attention_heads: int = 20
    num_hidden_layers: int = 32
    max_position_embeddings: int = 6000
    dropout: float = 0.0
    attention_dropout: float = 0.0
    sampling_rate: int = 16_000
    feature_size: int = 128
    hop_length: int = 160
    downsample_hidden_size: int = 480
    rope_parameters: dict | None = None
    head_dim: int | None = None
    num_key_value_heads: int | None = None
    attention_bias: bool = True
    hidden_act: str = "silu"
    adapter_input_size: int = 1280
    adapter_output_size: int = 5120

    def __post_init__(self, **kwargs):
        self.adapter_input_size = kwargs.pop("whisper_adapter_in_dim", self.adapter_input_size)
        self.adapter_output_size = kwargs.pop("whisper_adapter_out_dim", self.adapter_output_size)
        # Released checkpoints store encoder dimensions in a Whisper-style dictionary.
        whisper_config = kwargs.pop("whisper_config", None) or {}
        for legacy_name, name in {
            **self.attribute_map,
            "num_mel_bins": "feature_size",
            "max_source_positions": "max_position_embeddings",
        }.items():
            if legacy_name in whisper_config:
                setattr(self, name, whisper_config[legacy_name])
        self.dropout = whisper_config.get("dropout", self.dropout)
        self.attention_dropout = whisper_config.get("attention_dropout", self.attention_dropout)
        if whisper_config.get("activation_function", "swiglu") != "swiglu":
            raise ValueError("Dots 3 Note Preview audio requires activation_function='swiglu'")
        if whisper_config.get("activation_dropout", kwargs.pop("activation_dropout", 0.0)) != 0:
            raise ValueError("Dots 3 Note Preview audio requires activation_dropout=0")
        for name, expected in (
            ("encoder_type", "dots"),
            ("use_conv2d_stem", True),
            ("use_rope", True),
            ("use_rms_norm", True),
            ("use_causal", False),
            ("merge_factor", 1),
        ):
            if kwargs.pop(name, expected) != expected:
                raise ValueError(f"Dots 3 Note Preview audio requires {name}={expected!r}")
        if (attention_backend := kwargs.pop("attention_backend", None)) is not None:
            kwargs.setdefault("attn_implementation", attention_backend)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.hidden_act = "silu"
        self.attention_bias = True
        self.rope_parameters = {
            "rope_type": "default",
            "partial_rotary_factor": 0.5,
            "rope_theta": 10_000.0,
            **(self.rope_parameters or {}),
        }
        rotary_dim = int(self.head_dim * self.rope_parameters["partial_rotary_factor"]) // 2 * 2
        self.rope_parameters["partial_rotary_factor"] = rotary_dim / self.head_dim
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("audio hidden_size must be divisible by num_attention_heads")
        if self.adapter_input_size != self.hidden_size:
            raise ValueError("adapter_input_size must match hidden_size")
        if min(self.sampling_rate, self.feature_size, self.hop_length) <= 0:
            raise ValueError("audio sampling and feature dimensions must be positive")


@auto_docstring(checkpoint="dots-studio/dots3-note-prev")
@strict
class Dots3NoteConfig(PreTrainedConfig):
    r"""
    Configuration for the Dots 3 Note Preview multimodal causal language model.

    rope_theta (`float`, *optional*, defaults to 80000000.0):
        Base period of the rotary position embeddings used by full-attention layers.
    index_n_heads (`int`, *optional*, defaults to 64):
        Number of heads in the dynamic sparse attention indexer.
    index_head_dim (`int`, *optional*, defaults to 128):
        Per-head width of the dynamic sparse attention indexer.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of key positions selected by dynamic sparse attention.
    shared_experts_intermediate_size (`int`, *optional*, defaults to 1536):
        Intermediate width of each shared expert.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer `"dense"` or `"sparse"` MLP types, derived from the released MoE schedule when omitted.
    n_group (`int`, *optional*, defaults to 1):
        Number of groups into which routed experts are partitioned during selection.
    """

    model_type = "dots3_note"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    sub_configs = {
        "vision_config": Dots3NoteVisionConfig,
        "audio_config": Dots3NoteAudioConfig,
    }
    attribute_map = {"num_local_experts": "n_routed_experts"}

    vocab_size: int = 152064
    hidden_size: int = 5120
    intermediate_size: int = 13824
    num_hidden_layers: int = 46
    num_attention_heads: int = 128
    num_key_value_heads: int | None = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 524288
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 151659
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151668
    tie_word_embeddings: bool = False
    rope_theta: float = 80_000_000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 256

    q_lora_rank: int | None = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    sliding_window: int | None = None
    layer_types: list[str] | tuple[str, ...] | None = None

    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048

    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536
    shared_experts_intermediate_size: int = 1536
    mlp_layer_types: list[str] | None = None
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1

    vision_config: dict | Dots3NoteVisionConfig | None = None
    audio_config: dict | Dots3NoteAudioConfig | None = None
    image_token_id: int = 151660
    video_token_id: int = 151680
    audio_token_id: int = 151720

    def __post_init__(self, **kwargs):
        first_k_dense_replace = kwargs.pop("first_k_dense_replace", 1)
        moe_layer_freq = kwargs.pop("moe_layer_freq", 1)
        # Legacy checkpoints call DSA layers "full_attention"; canonical configs also store per_layer_config.
        use_dsa = kwargs.pop("use_dsa", None if "per_layer_config" in kwargs else True)
        use_sliding_window = kwargs.pop("use_sliding_window", True)
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "sparse"
                if i >= first_k_dense_replace
                and (bool(moe_layer_freq[i]) if isinstance(moe_layer_freq, (list, tuple)) else i % moe_layer_freq == 0)
                else "dense"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.mlp_layer_types) != self.num_hidden_layers or set(self.mlp_layer_types) - {"dense", "sparse"}:
            raise ValueError("mlp_layer_types must contain one 'dense' or 'sparse' entry per hidden layer")
        if self.n_shared_experts is None or self.n_shared_experts < 1:
            raise ValueError(
                f"Dots 3 Note Preview requires shared experts, got n_shared_experts={self.n_shared_experts!r}"
            )
        # Validate legacy architecture flags once; the model follows the released architecture.
        for name, expected in (
            ("normalization", "RMSNorm"),
            ("final_norm", "RMSNorm"),
            ("multi_latent_attention", True),
            ("apply_mla_qkv_lora_rescale", True),
            ("attention_gate_type", "headwise"),
            ("swa_attention_gate_type", "headwise"),
            ("softmax_type", "vanilla"),
            ("scoring_func", "sigmoid"),
            ("topk_method", "noaux_tc"),
            ("moe_gating_fp32", False),
            ("use_dynamic_rsf", False),
        ):
            if kwargs.pop(name, expected) != expected:
                raise ValueError(f"Dots 3 Note Preview requires {name}={expected!r}")
        self.shared_experts_intermediate_size = kwargs.pop(
            "moe_shared_expert_intermediate_size", self.shared_experts_intermediate_size
        )
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("Dots 3 Note Preview requires num_key_value_heads to match num_attention_heads")
        if self.attention_bias:
            raise ValueError("Dots 3 Note Preview text attention requires attention_bias=False")
        if self.sliding_window is None:
            self.sliding_window = kwargs.pop("sliding_window_size", 512)
        # Older JSONs store null, but the released checkpoint contains the K-RoPE norm weights.
        if kwargs.pop("k_rope_only_layernorm", True) not in (True, None):
            raise ValueError("Dots 3 Note Preview requires K-RoPE normalization")

        if self.layer_types is None:
            full_attention_type = "full_attention" if use_dsa is False else "deepseek_sparse_attention"
            self.layer_types = [
                full_attention_type if not use_sliding_window or i < 2 or i % 4 == 1 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = list(self.layer_types)
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per hidden layer")
        self.layer_types = [
            "deepseek_sparse_attention"
            if use_dsa is True and layer_type == "full_attention"
            else "full_attention"
            if use_dsa is False and layer_type == "deepseek_sparse_attention"
            else layer_type
            for layer_type in self.layer_types
        ]
        unsupported = set(self.layer_types) - {"full_attention", "sliding_attention", "deepseek_sparse_attention"}
        if unsupported:
            raise ValueError(f"Unsupported layer types: {sorted(unsupported)}")
        if self.qk_rope_head_dim > self.index_head_dim:
            raise ValueError("qk_rope_head_dim must not exceed index_head_dim")
        if self.n_group < 1 or self.n_routed_experts % self.n_group != 0:
            raise ValueError("n_group must evenly divide n_routed_experts")
        if self.n_routed_experts // self.n_group < 2:
            raise ValueError("Grouped routing requires at least two experts per group")
        if self.topk_group < 1 or self.topk_group > self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")

        if self.vision_config is None:
            self.vision_config = Dots3NoteVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = Dots3NoteVisionConfig(**self.vision_config)
        if self.audio_config is None:
            self.audio_config = Dots3NoteAudioConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = Dots3NoteAudioConfig(**self.audio_config)
        if self.vision_config.adapter_out_dim != self.hidden_size:
            raise ValueError("vision adapter output width must match the text hidden size")
        if self.audio_config.adapter_output_size != self.hidden_size:
            raise ValueError("audio adapter output width must match the text hidden size")

        swa_rope_theta = kwargs.pop("swa_rope_theta", 50_000.0)
        self.rope_parameters = {
            layer_type: {
                "rope_type": "default",
                "rope_theta": swa_rope_theta if layer_type == "sliding_attention" else self.rope_theta,
            }
            for layer_type in set(self.layer_types)
        }
        # Normalize the released checkpoint's SWA aliases into the shared per-layer configuration.
        sliding_config = {
            name: kwargs.pop(f"swa_{name}", default)
            for name, default in {
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "q_lora_rank": 1024,
                "kv_lora_rank": 1024,
                "head_dim": 256,
                "qk_nope_head_dim": 192,
                "v_head_dim": 128,
            }.items()
        }
        sliding_config["qk_rope_head_dim"] = kwargs.pop(
            "swa_qk_rope_head_dim", sliding_config["head_dim"] - sliding_config["qk_nope_head_dim"]
        )
        kwargs.setdefault(
            "per_layer_config",
            {i: sliding_config for i, layer_type in enumerate(self.layer_types) if layer_type == "sliding_attention"},
        )
        super().__post_init__(**kwargs)
        for layer_type in set(self.layer_types):
            layer_config = self.per_layer_config[layer_type]
            self.rope_parameters[layer_type]["partial_rotary_factor"] = (
                layer_config.qk_rope_head_dim / layer_config.head_dim
            )

    def convert_rope_params_to_dict(self, **kwargs):
        # Legacy checkpoints contain null; its compatibility setter would erase the per-layer RoPE parameters.
        kwargs.pop("rope_scaling", None)
        return kwargs

    def validate_architecture(self):
        if any(self.per_layer_config[layer_type].q_lora_rank is None for layer_type in set(self.layer_types)):
            raise ValueError("Dots 3 Note Preview requires q_lora_rank for MLA queries")
        if "sliding_attention" in self.layer_types:
            config = self.per_layer_config["sliding_attention"]
            if config.num_key_value_heads != config.num_attention_heads:
                raise ValueError("SWA num_key_value_heads must match num_attention_heads")
            if config.head_dim != config.qk_nope_head_dim + config.qk_rope_head_dim:
                raise ValueError("SWA head_dim must equal qk_nope_head_dim + qk_rope_head_dim")


__all__ = [
    "Dots3NoteAudioConfig",
    "Dots3NoteConfig",
    "Dots3NoteVisionConfig",
]
