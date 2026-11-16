# Copyright 2026 Baidu and The HuggingFace Inc. team. All rights reserved.
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


ERNIE5_5_MOE_LAYER_TYPES = ("heavily_compressed_attention", "indexed_attention")


def _compress_ratio_to_layer_type(ratio: int) -> str:
    # The released config marks the MLA + DSA layers with a non-positive ratio (`-2`)
    # and the Heavily Compressed Attention layers with their compression ratio (`128`).
    return "indexed_attention" if ratio <= 0 else "heavily_compressed_attention"


@auto_docstring(checkpoint="baidu/ERNIE-5.5-Flash-Base-BF16")
@strict
class Ernie5_5_MoeConfig(PreTrainedConfig):
    r"""
    q_lora_rank (`int`):
        Rank of the low-rank query projection, shared by both attention types.
    qk_rope_head_dim (`int`):
        Number of rotary channels per head in the HCA layers.
    o_groups (`int`):
        Number of head groups in the grouped output projection of the HCA layers.
    o_lora_rank (`int`):
        Per-group output rank of the grouped output projection of the HCA layers.
    sliding_window (`int`):
        Local window that every attention layer attends to at token level.
    layer_types (`list[str]`):
        Per-layer attention type, `"heavily_compressed_attention"` (HCA) or `"indexed_attention"` (MLA with DSA).
        Derived from the checkpoint's `compress_ratios` when not given.
    compress_rates (`dict[str, int]`):
        Number of tokens pooled into one compressed KV entry, per layer type. Only HCA layers compress.
    hybrid_mla_num_attention_heads (`int`):
        Number of heads of the MLA layers.
    hybrid_mla_q_lora_rank (`int`):
        Query LoRA rank of the MLA layers.
    hybrid_mla_kv_lora_rank (`int`):
        Latent KV rank of the MLA layers.
    hybrid_mla_qk_nope_head_dim (`int`):
        Non-rotary query/key channels per head of the MLA layers.
    hybrid_mla_qk_rope_head_dim (`int`):
        Rotary query/key channels per head of the MLA layers.
    hybrid_mla_v_head_dim (`int`):
        Value channels per head of the MLA layers.
    index_n_heads (`int`):
        Number of lightning-indexer heads in the MLA layers.
    index_head_dim (`int`):
        Head dimension of the lightning indexer.
    index_topk (`int`):
        Number of tokens older than the sliding window that the indexer keeps per query.
    rope_theta (`float`):
        RoPE base of the MLA layers and their indexer (plain RoPE, `rope_parameters["main"]`).
    compress_rope_theta (`float`):
        RoPE base of the HCA layers (paired with `rope_parameters["compress"]`, YaRN in the released model).
    hc_mult (`int`):
        Number of residual streams of the manifold-constrained Hyper-Connections (mHC).
    hc_sinkhorn_iters (`int`):
        Sinkhorn-Knopp iterations of the mHC residual mixing matrix.
    hc_eps (`float`):
        Numerical floor of the mHC weights.
    first_k_dense_replace (`int`):
        Number of leading layers with a dense feed-forward block instead of a MoE block.
    moe_latent_size (`int`):
        Width of the latent space the routed experts operate in.
    n_routed_experts (`int`):
        Number of routed experts per MoE layer.
    n_shared_experts (`int`):
        Number of shared experts per MoE layer.
    routed_scaling_factor (`float`):
        Initial value of the learned per-expert routing scale `routed_scaling_factor_param`.
    activation_situ_beta (`float`):
        Saturation of the SiTU gate activation `beta * tanh(x / beta) * sigmoid(x)`.
    activation_situ_linear_beta (`float`):
        Saturation of the SiTU-GLU up branch `beta * tanh(x / beta)`.
    multimax (`str`, *optional*):
        `"lm_head"` to apply the learned MultiMax transform to the logits, `None` to disable it.
    num_nextn_predict_layers (`int`):
        Number of ECHO multi-token-prediction layers in the checkpoint (not instantiated here).
    vha_rank (`int`):
        Rank of the Virtual Head Attention post-mix across heads.
    """

    model_type = "ernie5_5_moe"
    keys_to_ignore_at_inference = ["cache"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }

    vocab_size: int = 201216
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 43
    max_position_embeddings: int = 1048576

    # HCA attention
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    o_groups: int = 8
    o_lora_rank: int = 1024
    sliding_window: int = 128
    layer_types: list[str] | None = None
    compress_rates: dict | None = None

    # MLA + DSA attention
    hybrid_mla_num_attention_heads: int = 64
    hybrid_mla_q_lora_rank: int = 1024
    hybrid_mla_kv_lora_rank: int = 512
    hybrid_mla_qk_nope_head_dim: int = 192
    hybrid_mla_qk_rope_head_dim: int = 64
    hybrid_mla_v_head_dim: int = 256
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048

    rope_theta: float | int = 8000000.0
    compress_rope_theta: float | int = 160000.0
    rope_parameters: RopeParameters | dict | None = None

    # mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6

    # MoE
    first_k_dense_replace: int = 1
    moe_intermediate_size: int = 2048
    moe_latent_size: int = 2048
    n_routed_experts: int = 512
    n_shared_experts: int = 1
    num_experts_per_tok: int = 10
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 3.0
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    hidden_act: str = "situ"
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    multimax: str | None = "lm_head"
    num_nextn_predict_layers: int = 1
    vha_rank: int = 16

    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # `rope_parameters` is keyed by rope-type labels: `main` (MLA layers and their indexer, plain RoPE at
    # `rope_theta`) and `compress` (HCA layers, at `compress_rope_theta`). See `__post_init__`.
    _rope_type_labels = ("main", "compress")

    def validate_rope(self):
        rope_parameters_dict = getattr(self, "rope_parameters", None) or {}
        ignore_keys = self.ignore_keys_at_rope_validation
        for rope_type_label in self._rope_type_labels:
            rope_parameters = rope_parameters_dict.get(rope_type_label)
            if not isinstance(rope_parameters, dict):
                continue
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            rope_parameters["rope_type"] = rope_type
            validation_fn = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            if validation_fn is None:
                continue
            # The validators read `self.rope_parameters[<key>]` directly, so point it at the sub-dict
            self.rope_parameters = rope_parameters
            try:
                validation_fn(rope_parameters, ignore_keys=ignore_keys)
            finally:
                self.rope_parameters = rope_parameters_dict

    def validate_layer_type(self):
        if self.num_hidden_layers is None or self.layer_types is None:
            return
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`num_hidden_layers` ({self.num_hidden_layers}) must equal `len(layer_types)` ({len(self.layer_types)})."
            )
        bad = [t for t in self.layer_types if t not in ERNIE5_5_MOE_LAYER_TYPES]
        if bad:
            raise ValueError(f"`layer_types` entries must be one of {ERNIE5_5_MOE_LAYER_TYPES}; got {bad}.")

    def __post_init__(self, **kwargs):
        # The released `config.json` describes the layer schedule with per-layer `compress_ratios` (one extra
        # entry for the ECHO layer) and the HCA YaRN settings with a legacy `rope_scaling` dict.
        compress_ratios = kwargs.pop("compress_ratios", None)
        hca_rope_type = kwargs.pop("hca_rope_type", None)
        PreTrainedConfig.__post_init__(self, **kwargs)
        n = self.num_hidden_layers

        if self.layer_types is None:
            if compress_ratios is not None:
                self.layer_types = [_compress_ratio_to_layer_type(r) for r in compress_ratios[:n]]
            else:
                # Released schedule: layers 7, 14, ..., 42 are MLA + DSA, the others HCA
                self.layer_types = [
                    "indexed_attention" if i > 0 and i % 7 == 0 else "heavily_compressed_attention" for i in range(n)
                ]
        self.layer_types = list(self.layer_types[:n])

        if self.compress_rates is None:
            hca_ratios = {r for r in (compress_ratios or [])[:n] if r > 0}
            if len(hca_ratios) > 1:
                raise ValueError(f"All HCA layers must share one compression ratio, got {sorted(hca_ratios)}.")
            self.compress_rates = {"heavily_compressed_attention": hca_ratios.pop() if hca_ratios else 128}

        rp = self.rope_parameters or {}
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            self.rope_parameters = {"main": rp["main"], "compress": rp["compress"]}
        else:
            # Both rope types rotate `qk_rope_head_dim` channels; the rotary helpers size their tables from
            # `head_dim * partial_rotary_factor`.
            partial_rotary_factor = self.qk_rope_head_dim / self.head_dim
            main = {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
            }
            # The base class has already folded the legacy `rope_scaling` dict into `rope_parameters`
            scaling = {k: v for k, v in rp.items() if k not in ("main", "compress", "rope_theta", "type")}
            # `hca_rope_type` selects the HCA RoPE; the reference falls back to plain RoPE when it is missing
            if hca_rope_type is not None and hca_rope_type != "yarn":
                scaling = {}
            compress = {
                **scaling,
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
            }
            compress.setdefault("rope_type", "default")
            if compress["rope_type"] == "yarn":
                # The HCA layers rotate with YaRN frequencies but without YaRN's attention scaling (mscale=1.0)
                compress.setdefault("attention_factor", 1.0)
            self.rope_parameters = {"main": main, "compress": compress}


__all__ = ["Ernie5_5_MoeConfig"]
