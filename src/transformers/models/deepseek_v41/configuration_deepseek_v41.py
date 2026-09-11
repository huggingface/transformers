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

import dataclasses

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


# V4.1 attention layer types. `layer_types` (and therefore the cache dispatch) only
# distinguishes *KV-source* layers from plain sliding-window layers: every layer runs
# sliding-window attention, and the compressed sparse KV is produced once per
# `kv_source_layer_ids` group and shared by the layers in between (CSA2 "Reuse").
DEEPSEEK_V41_LAYER_TYPES = ("sliding_attention", "shared_compressed_attention")


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4.1-Flash")
@strict
class DeepseekV41TextConfig(PreTrainedConfig):
    r"""
    Configuration for the text backbone of DeepSeek-V4.1 models ("DeepSeek-V4.1-Flash:
    Pushing the Limits of KV Cache Compression", DeepSeek-AI 2026). Field names mirror
    the `text_config` section of the released
    [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
    checkpoint config.

    Architecture summary (layer indices below refer to the released 40-layer Flash
    checkpoint; every mechanism is per-layer configurable through this config):

    * **Causal Encoder–Decoder (CED)**: the per-layer `compress_ratios` schedule splits
      the stack into a causal "encoder" (ratios > 1) and a "decoder" (ratio 1) whose
      global KV is *projected once* by the encoder-side source layers, instead of being
      derived per decoder layer. Layers with ratio 0 keep only a sliding window.
    * **CSA2**: `kv_source_layer_ids` lists the layers that own a compressor and publish
      a shared compressed-KV cache; `index_source_layer_ids` lists the layers that run
      the sparse indexer and publish top-k indices (the layers in between reuse them);
      `candidate_source_layer_id` (< 0 disables) enables the hierarchical two-level
      top-k: its candidate-block mask constrains the top-k of all later index sources.
    * **Engram conditional memory**: n-gram hash tables sparsely looked up and gated
      into the residual stream at `engram_layer_ids` (~196B of the released
      checkpoint's 763B parameters live here).
    * **DSpark**: MTP-style draft layers (`num_nextn_predict_layers`) with Markov and
      confidence heads, semi-autoregressive block drafting.

    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the text model.
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimensionality of the residual stream (one per hyper-connection copy).
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of backbone (decoder) layers. The DSpark draft layers of
            `num_nextn_predict_layers` are appended after these.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of query heads per attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 1):
            V4.1 uses shared-KV latent attention: a single KV latent broadcast to all
            heads.
        head_dim (`int`, *optional*, defaults to 512):
            Dimensionality of each attention head (and of the KV latent).
        q_lora_rank (`int`, *optional*, defaults to 1280):
            Rank of the low-rank factorization of the query projection.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Trailing channels of each head that receive RoPE (interleaved pairs).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE base for pure sliding-window layers (no YaRN scaling).
        compress_rope_theta (`float`, *optional*, defaults to 160000.0):
            RoPE base for the compressed branch: one latent stands for
            `compress_ratio` tokens, so its positions are further apart.
        rope_parameters (`dict`, *optional*):
            Per-rope-type parameters (`main` / `compress`), following V4. Derived from
            `rope_scaling` when not given: yarn applies ONLY to the `compress` rope.
        rope_scaling (`dict`, *optional*):
            YaRN parameters for the compressed branch, e.g. the released
            `{"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
            "original_max_position_embeddings": 65536}`.
        max_position_embeddings (`int`, *optional*, defaults to 1048576):
            Maximum supported context (1M for the released model).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether attention projections carry a bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout on attention weights.
        sliding_window (`int`, *optional*, defaults to 128):
            Size of the sliding-window KV ring kept by every layer.
        compress_ratios (`list[int]`, *optional*):
            Per-layer compressed-KV pooling ratio: 0 = sliding window only, 1 =
            full-resolution shared KV (CED decoder), r > 1 = r tokens pooled into one
            latent (CED encoder). Length `num_hidden_layers` (+ optionally
            `num_nextn_predict_layers` trailing entries, always 0). Defaults to the
            released schedule `[0, 0] + [2]*18 + [1]*20`.
        kv_source_layer_ids (`list[int]`, *optional*):
            Layers that own a compressor and publish the shared compressed-KV and
            indexer-key caches for their group (layers up to the next source). Defaults
            to the released `[2, 8, 14, 20]`.
        index_source_layer_ids (`list[int]`, *optional*):
            Layers that run the sparse indexer and publish top-k indices reused by the
            layers in between (CSA2 "Reindex" sources still score with their own
            weights, inside the candidate mask). Defaults to the released
            `[2, 8, 14, 20, 24, 28, 32, 36]`.
        candidate_source_layer_id (`int`, *optional*):
            The index source whose top-`candidate_topk_blocks` blocks form the
            candidate pool constraining all later index sources (hierarchical
            two-level top-k). Defaults to the last KV-source layer (20 with the
            default `compress_ratios`); negative disables the second level.
        candidate_topk_blocks (`int`, *optional*, defaults to 2048):
            Number of candidate blocks kept by the candidate source layer.
        candidate_block_size (`int`, *optional*, defaults to 8):
            Number of compressed positions per candidate block.
        index_n_heads (`int`, *optional*, defaults to 32):
            Number of query heads of the sparse indexer.
        index_head_dim (`int`, *optional*, defaults to 128):
            Head dimension of the sparse indexer.
        index_topk (`int`, *optional*, defaults to 512):
            Number of compressed positions each query attends to.
        o_groups (`int`, *optional*, defaults to 8):
            Number of head groups in the grouped low-rank output projection.
        o_lora_rank (`int`, *optional*, defaults to 1024):
            Per-group intermediate dimension of the output projection.
        moe_intermediate_size (`int`, *optional*, defaults to 2304):
            Intermediate size of each (routed and shared) expert.
        n_routed_experts (`int`, *optional*, defaults to 384):
            Number of routed experts per MoE layer.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts every token always goes through.
        num_experts_per_tok (`int`, *optional*, defaults to 6):
            Number of routed experts activated per token.
        scoring_func (`str`, *optional*, defaults to `"sqrtsoftplus"`):
            Router activation: `sqrtsoftplus`, `softmax`, or `sigmoid`.
        gate_temp (`float`, *optional*, defaults to 1.0):
            Temperature dividing the router logits before the score function.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Normalize the routing weights of the selected experts (with a `+1e-20`
            floor on the sum, matching training).
        routed_scaling_factor (`float`, *optional*, defaults to 1.5):
            Multiplier applied to the routed experts' output.
        topk_method (`str`, *optional*, defaults to `"noaux_tc"`):
            Top-k method: the correction bias steers expert *selection* only; routing
            weights come from the unbiased scores.
        swiglu_limit (`float`, *optional*, defaults to 10.0):
            Clamp on the experts' SwiGLU pre-activations (up branch clamped on both
            sides, gate branch from above), keeping quantized activations in range.
        hc_mult (`int`, *optional*, defaults to 4):
            Number of parallel residual streams (hyper-connections).
        hc_sinkhorn_iters (`int`, *optional*, defaults to 20):
            Sinkhorn-Knopp iterations projecting the residual-combine matrix onto the
            doubly-stochastic manifold.
        hc_eps (`float`, *optional*, defaults to 1e-6):
            Numerical floor of the Sinkhorn normalization and of the `pre` gates.
        engram_layer_ids (`list[int]`, *optional*):
            Layers with an engram conditional-memory module (released: `[1, 14]`).
            Empty disables the engram entirely.
        engram_vocab_size (`int`, *optional*, defaults to 16000000):
            Bucket size each (n-gram size, head) pair starts searching primes from.
        engram_num_embeddings (`list[int]`, *optional*):
            Total table rows per engram layer (released: `[384006168, 384016682]`).
        engram_max_ngram_size (`int`, *optional*, defaults to 4):
            Largest n-gram hashed per position (all sizes 2..N are looked up).
        engram_n_heads (`int`, *optional*, defaults to 8):
            Number of hash heads per n-gram size.
        engram_head_dim (`int`, *optional*, defaults to 256):
            Row dimension of the engram tables.
        engram_pad_id (`int`, *optional*, defaults to 2):
            Token id filling n-gram slots with no history (matches training).
        engram_compressed_vocab_size (`int`, *optional*, defaults to 99092):
            Size of the tokenizer-normalized compressed vocabulary the hashes operate
            on. Asserted at model build against the value derived from the tokenizer.
        num_nextn_predict_layers (`int`, *optional*, defaults to 3):
            Number of DSpark draft layers after the backbone.
        dspark_block_size (`int`, *optional*, defaults to 5):
            Tokens drafted per DSpark block (semi-autoregressive drafting).
        dspark_noise_token_id (`int`, *optional*, defaults to 128799):
            Token id filling the draft block's non-anchor positions.
        dspark_target_layer_ids (`list[int]`, *optional*):
            Backbone layers whose *attention input* (mean over the hc streams) feeds
            the draft model (released: `[37, 38, 39]`).
        dspark_markov_rank (`int`, *optional*, defaults to 256):
            Rank of the Markov head's embedding.
        dspark_n_routed_experts (`int`, *optional*, defaults to 128):
            Number of routed experts in the draft layers' MoE.
        dspark_num_experts_per_tok (`int`, *optional*, defaults to 3):
            Routed experts activated per token in the draft layers.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation of the MLP / experts.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-20):
            Epsilon of every RMSNorm (unusually small — matches training).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the past key values.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the LM head is tied to the embedding.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to return router logits from the MoE layers.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Coefficient of the load-balancing auxiliary loss.
        router_jitter_noise (`float`, *optional*, defaults to 0.0):
            Noise added to router logits during training.
    """

    model_type = "deepseek_v41_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # `num_local_experts` is the standard MoE attr name (read by FP8 / TP integrations).
    # `intermediate_size` is what the shared-expert MLP base class reads.
    attribute_map = {"num_local_experts": "n_routed_experts", "intermediate_size": "moe_intermediate_size"}

    # --- attention / rope --------------------------------------------------------------
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1280
    qk_rope_head_dim: int = 64
    rope_theta: float | int = 10000.0
    compress_rope_theta: float | int = 160000.0
    rope_parameters: RopeParameters | dict | None = None
    rope_scaling: dict | None = None
    max_position_embeddings: int = 1048576
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # --- sliding window + compressed sparse attention (CSA2) ---------------------------
    sliding_window: int = 128
    compress_ratios: list[int] | None = None
    kv_source_layer_ids: list[int] | None = None
    index_source_layer_ids: list[int] | None = None
    candidate_source_layer_id: int | None = None
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512

    # --- grouped low-rank output projection ---------------------------------------------
    o_groups: int = 8
    o_lora_rank: int = 1024

    # --- MoE ------------------------------------------------------------------------------
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    topk_method: str = "noaux_tc"
    swiglu_limit: float = 10.0

    # --- hyper-connections (mHC) -----------------------------------------------------------
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6

    # --- engram conditional memory ------------------------------------------------------------
    engram_layer_ids: list[int] | None = None
    engram_vocab_size: int = 16000000
    engram_num_embeddings: list[int] | None = None
    engram_max_ngram_size: int = 4
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092

    # --- DSpark (MTP draft) -------------------------------------------------------------------
    num_nextn_predict_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: list[int] | None = None
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3

    # --- misc ------------------------------------------------------------------------------------
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-20
    use_cache: bool = True
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 1

    # --- training / loss ---------------------------------------------------------------------------
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    # Populated in `__post_init__` from `kv_source_layer_ids`; declared so it survives
    # `to_dict` round-trips and drives cache-layer dispatch in `DynamicCache`.
    layer_types: list[str] | None = None

    # Like V4: rope validation iterates the rope-type-keyed sub-dicts, not layer types.
    _rope_type_labels = ("main", "compress")

    # The released layer schedule: 2 pure sliding-window layers, then ratio-2 groups
    # (18 layers; sources at 2/8/14), then ratio-1 "decoder" layers (20; source at 20).
    # Only used when `compress_ratios` is not given (i.e. tiny test configs); the
    # released checkpoint always ships the explicit 40(+3 nextn) list.
    default_compress_ratios = [0, 0] + [2] * 18 + [1] * 20
    default_kv_source_layer_ids = [2, 8, 14, 20]
    default_index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]

    def __post_init__(self, **kwargs):
        PreTrainedConfig.__post_init__(self, **kwargs)
        n = self.num_hidden_layers
        n_nextn = self.num_nextn_predict_layers

        # --- compress_ratios: backbone (+ nextn draft layers, always ratio 0) ----------
        if self.compress_ratios is None:
            if n == len(self.default_compress_ratios):
                self.compress_ratios = list(self.default_compress_ratios)
            else:
                # Scale the released pattern (2 sliding-only layers, a ratio-2
                # "encoder" block, a ratio-1 "decoder" block) to `n` layers.
                n_slide = min(2, n)
                n_enc = max(0, (n - n_slide + 1) // 2)
                self.compress_ratios = [0] * n_slide + [2] * n_enc + [1] * max(0, n - n_slide - n_enc)
        ratios = list(self.compress_ratios)
        if len(ratios) < n:
            raise ValueError(f"`compress_ratios` needs at least `num_hidden_layers` ({n}) entries, got {len(ratios)}.")
        # DSpark draft layers are sliding-window-only; accept and normalize schedules
        # that include them (the released config ships 40 + 3 entries).
        if len(ratios) > n + n_nextn:
            raise ValueError(
                f"`compress_ratios` has {len(ratios)} entries but the model has "
                f"`num_hidden_layers + num_nextn_predict_layers` = {n + n_nextn} layers."
            )
        self.compress_ratios = ratios[:n] + ratios[n : n + n_nextn] + [0] * (n_nextn - max(0, len(ratios) - n))
        backbone_ratios = self.compress_ratios[:n]

        # --- kv / index source schedules --------------------------------------------------
        if self.kv_source_layer_ids is None:
            if self.compress_ratios[:n] == self.default_compress_ratios[:n]:
                self.kv_source_layer_ids = list(self.default_kv_source_layer_ids)
            else:
                # Custom schedule without explicit sources: the first layer of each
                # same-ratio run publishes for its group.
                self.kv_source_layer_ids = [
                    i
                    for i in range(n)
                    if backbone_ratios[i] > 0 and (i == 0 or backbone_ratios[i - 1] != backbone_ratios[i])
                ]
        if self.index_source_layer_ids is None:
            self.index_source_layer_ids = (
                list(self.default_index_source_layer_ids)
                if self.kv_source_layer_ids == self.default_kv_source_layer_ids
                else list(self.kv_source_layer_ids)
            )
        if self.dspark_target_layer_ids is None:
            self.dspark_target_layer_ids = sorted([i for i in range(n) if i not in self.kv_source_layer_ids][-3:])

        kv_sources = list(self.kv_source_layer_ids)
        index_sources = list(self.index_source_layer_ids)
        if any(i < 0 or i >= n for i in kv_sources):
            raise ValueError(f"`kv_source_layer_ids` out of range [0, {n}): {kv_sources}")
        if any(backbone_ratios[i] == 0 for i in kv_sources):
            raise ValueError(f"`kv_source_layer_ids` {kv_sources} must have `compress_ratios[i] > 0`.")
        # Index sources are not necessarily KV sources: a "Reindex" source scores with
        # its own weights against the shared indexer keys of the preceding KV source,
        # so it only needs a compressed branch itself.
        if any(backbone_ratios[i] == 0 for i in index_sources):
            raise ValueError(f"`index_source_layer_ids` {index_sources} must have `compress_ratios[i] > 0`.")
        if any(i not in kv_sources and not any(s < i for s in kv_sources) for i in index_sources):
            raise ValueError(
                f"`index_source_layer_ids` {index_sources} each need a `kv_source_layer_ids` entry "
                f"before them to publish the shared indexer keys."
            )
        # `None` (or a negative id) disables the two-level top-k; the released
        # schedule uses the last KV source (layer 20).
        if self.candidate_source_layer_id is None:
            self.candidate_source_layer_id = kv_sources[-1] if kv_sources else -1
        if self.candidate_source_layer_id >= 0 and self.candidate_source_layer_id not in index_sources:
            raise ValueError(
                f"`candidate_source_layer_id` ({self.candidate_source_layer_id}) must be in "
                f"`index_source_layer_ids` {index_sources}."
            )
        # Every layer with ratio > 0 reads the cache of the closest source at or before it.
        if any(backbone_ratios[i] > 0 and not any(s <= i for s in kv_sources) for i in range(n)):
            raise ValueError(
                "Every layer with `compress_ratios[i] > 0` needs a `kv_source_layer_ids` entry at or before it."
            )

        # --- `layer_types`: only KV-source layers get a compressor cache layer -----------
        self.layer_types = [
            "shared_compressed_attention" if i in kv_sources else "sliding_attention" for i in range(n)
        ]

        # --- engram defaults ---------------------------------------------------------------
        if self.engram_layer_ids is None:
            self.engram_layer_ids = []
        if self.engram_num_embeddings is None:
            self.engram_num_embeddings = [0] * len(self.engram_layer_ids)
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError(
                f"`engram_num_embeddings` (len {len(self.engram_num_embeddings)}) must match "
                f"`engram_layer_ids` (len {len(self.engram_layer_ids)})."
            )

        # --- rope: yarn ONLY on compressed branches ----------------------------------------
        # Same scheme as V4: layers with a compressed branch rotate with
        # `compress_rope_theta` (160000) + YaRN (`rope_scaling`); pure sliding-window
        # layers use plain `rope_theta` (10000) with no scaling. YaRN's mscale is NOT
        # applied (`attention_factor=1.0`), matching the reference implementation.
        rp = self.rope_parameters or {}
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            self.rope_parameters = {"main": rp["main"], "compress": rp["compress"]}
        else:
            partial = self.qk_rope_head_dim / self.head_dim
            yarn = {k: v for k, v in (self.rope_scaling or {}).items() if k not in ("main", "compress")}
            main = {"rope_type": "default", "rope_theta": self.rope_theta, "partial_rotary_factor": partial}
            compress = {**yarn, "rope_theta": self.compress_rope_theta, "partial_rotary_factor": partial}
            compress.setdefault("rope_type", "default")
            if compress["rope_type"] == "yarn":
                compress.setdefault("attention_factor", 1.0)
            self.rope_parameters = {"main": main, "compress": compress}

    def validate_rope(self):
        # Same as V4: the yarn validators read `self.rope_parameters[<label>]` directly,
        # so point `self.rope_parameters` at each rope-type sub-dict for the duration of
        # the validation call, then restore it.
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
            self.rope_parameters = rope_parameters
            try:
                validation_fn(rope_parameters, ignore_keys=ignore_keys)
            finally:
                self.rope_parameters = rope_parameters_dict

    def validate_layer_type(self):
        """Narrow the global `ALLOWED_LAYER_TYPES` to V4.1's two attention-block types,
        on top of the standard length / type-membership checks."""
        if self.num_hidden_layers is None or self.layer_types is None:
            return
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`num_hidden_layers` ({self.num_hidden_layers}) must equal "
                f"`len(layer_types)` ({len(self.layer_types)})."
            )
        bad = [t for t in self.layer_types if t not in DEEPSEEK_V41_LAYER_TYPES]
        if bad:
            raise ValueError(f"`layer_types` entries must be one of {DEEPSEEK_V41_LAYER_TYPES}; got {bad}.")


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4.1-Flash")
@strict
class DeepseekV41VisionConfig(PreTrainedConfig):
    r"""
    Configuration for the DeepSeek-V4.1 vision tower (DeepSeek-ViT with 2D-RoPE and
    pixel-unshuffle downsampling). Field names mirror the `vision_config` section of
    the released checkpoint. The vision tower / aligner modules ship in a follow-up;
    this config exists so the composite checkpoint config parses.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the vision tower.
        intermediate_size (`int`, *optional*, defaults to 2816):
            Intermediate size of the vision MLPs.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of ViT layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads per ViT layer.
        patch_size (`int`, *optional*, defaults to 14):
            Patch size of the ViT patch embedding.
        downsample_ratio (`int`, *optional*, defaults to 3):
            Pixel-unshuffle downsampling ratio (effective patch = `patch_size` ×
            `downsample_ratio`).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base of the vision tower's 2D RoPE.
        max_image_tokens (`int`, *optional*, defaults to 1024):
            Maximum number of visual tokens per image.
        min_pixels (`int`, *optional*, defaults to 295936):
            Minimum number of pixels per image (544²).
        max_wh_ratio (`int`, *optional*):
            Maximum width/height aspect ratio; `None` disables the cap.
    """

    model_type = "deepseek_v41_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    patch_size: int = 14
    downsample_ratio: int = 3
    rope_theta: float | int = 10000.0
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: int | None = None


@strict
@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4.1-Flash")
class DeepseekV41Config(PreTrainedConfig):
    r"""
    Composite configuration for [`DeepseekV41ForCausalLM`]: the released
    DeepSeek-V4.1-Flash checkpoints are image-text-to-text models, so the top-level
    config holds a [`DeepseekV41TextConfig`] (`text_config`) and a
    [`DeepseekV41VisionConfig`] (`vision_config`).

    Examples:

    ```python
    >>> from transformers import DeepseekV41TextConfig, DeepseekV41TextModel

    >>> # Initializing a text backbone with random weights
    >>> config = DeepseekV41TextConfig()
    >>> model = DeepseekV41TextModel(config)
    ```
    """

    model_type = "deepseek_v41"
    sub_configs = {"vision_config": DeepseekV41VisionConfig, "text_config": DeepseekV41TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 129264
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            # A flat `deepseek_v41_text` config.json — e.g. saved from a model built
            # directly with a DeepseekV41TextConfig — carries the text params at the
            # top level (they arrive here as leftover kwargs). Adopt them instead of
            # falling back to the release-sized defaults (which would build a
            # multi-billion-parameter model and "hang" loading a tiny checkpoint).
            # `tie_word_embeddings` is a declared field of BOTH classes, so the
            # composite consumed it above: forward our value explicitly.
            text_cls = self.sub_configs["text_config"]
            text_fields = {field.name for field in dataclasses.fields(text_cls)}
            if text_fields.intersection(kwargs):
                text_kwargs = dict(kwargs)
                # declared field of BOTH classes: prefer the checkpoint's own value if
                # it reached us, else forward the one consumed above
                text_kwargs.setdefault("tie_word_embeddings", self.tie_word_embeddings)
                self.text_config = text_cls(**text_kwargs)
            else:
                self.text_config = text_cls()
        super().__post_init__(**kwargs)


__all__ = ["DeepseekV41TextConfig", "DeepseekV41VisionConfig", "DeepseekV41Config"]
