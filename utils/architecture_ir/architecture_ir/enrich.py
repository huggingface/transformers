"""Semantic-depth enrichment: normalized architecture facts derived from config + modules.

These recognizers turn raw config fields and module class names into the *semantic* facts a
consumer actually wants — attention variant (MHA/GQA/MQA/MLA), positional scheme, norm family,
mixture-of-experts parameters, and the model's task family / view. Everything here is generic
(no per-model special-casing) and config-parametric where it can be. Facts that don't apply are
left ``None`` so a consumer can tell "not present" from "unknown".
"""

from __future__ import annotations

from typing import Any


def _cfg(config: Any, *names: str, default: Any = None) -> Any:
    """First present, non-None attribute among ``names`` on ``config``."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def attention_variant(config: Any) -> str | None:
    """MHA | GQA | MQA | MLA from head counts / low-rank KV, or None if undeterminable."""
    if _cfg(config, "kv_lora_rank", "q_lora_rank") is not None:
        return "MLA"
    n_heads = _cfg(config, "num_attention_heads", "num_heads", "n_heads", "encoder_attention_heads")
    if not isinstance(n_heads, int):
        return None
    n_kv = _cfg(config, "num_key_value_heads", "num_kv_heads", "n_kv_heads", default=n_heads)
    if not isinstance(n_kv, int) or n_kv == n_heads:
        return "MHA"
    if n_kv == 1:
        return "MQA"
    return "GQA"


def attention_spec(config: Any) -> dict[str, Any]:
    """Config-derived attention facts attached to attention components."""
    n_heads = _cfg(config, "num_attention_heads", "num_heads", "n_heads", "encoder_attention_heads")
    n_kv = _cfg(config, "num_key_value_heads", "num_kv_heads", "n_kv_heads", default=n_heads)
    hidden = _cfg(config, "hidden_size", "d_model", "n_embd")
    head_dim = _cfg(config, "head_dim", "d_kv")
    if head_dim is None and isinstance(hidden, int) and isinstance(n_heads, int) and n_heads:
        head_dim = hidden // n_heads
    spec = {
        "variant": attention_variant(config),
        "n_heads": n_heads if isinstance(n_heads, int) else None,
        "n_kv_heads": n_kv if isinstance(n_kv, int) else None,
        "head_dim": head_dim if isinstance(head_dim, int) else None,
        "rope": positional_scheme(config) == "rope",
        "sliding_window": _cfg(config, "sliding_window"),
    }
    return {key: value for key, value in spec.items() if value is not None}


def positional_scheme(config: Any) -> str | None:
    """rope | relative | alibi | learned | sinusoidal | None."""
    if _cfg(config, "rope_parameters", "rope_theta", "rope_scaling") is not None:
        return "rope"
    if _cfg(config, "relative_attention_num_buckets") is not None:
        return "relative"
    if _cfg(config, "alibi", "use_alibi") is True:
        return "alibi"
    pos_type = _cfg(config, "position_embedding_type")
    if isinstance(pos_type, str):
        if pos_type == "absolute":
            return "learned"
        if pos_type.startswith("relative"):
            return "relative"
        if pos_type == "rotary":
            return "rope"
        return pos_type
    if _cfg(config, "max_position_embeddings") is not None:
        return "learned"
    return None


def norm_type(class_name: str) -> str:
    """Normalized norm family from a norm module's class name."""
    lower = class_name.lower()
    if "rmsnorm" in lower:
        return "rms"
    if "layernorm" in lower:
        return "layer"
    if "groupnorm" in lower:
        return "group"
    if "batchnorm" in lower:
        return "batch"
    return lower.replace("norm", "") or "norm"


def moe_spec(config: Any) -> dict[str, Any] | None:
    """Mixture-of-experts facts, or None if the config describes a dense model."""
    n_experts = _cfg(config, "num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
    if not isinstance(n_experts, int) or n_experts <= 1:
        return None
    spec = {
        "num_experts": n_experts,
        "experts_per_token": _cfg(config, "num_experts_per_tok", "top_k", "moe_top_k", "num_experts_per_token"),
        "num_shared_experts": _cfg(config, "n_shared_experts", "num_shared_experts"),
    }
    return {key: value for key, value in spec.items() if value is not None}


# Auto-mapping module attribute -> task family. Ordered by priority so the model's *canonical*
# family wins when a model_type appears in several mappings. masked_lm is checked before causal_lm
# to disambiguate encoder MLM models (e.g. BERT, which is registered in both) toward "encoder";
# decoder-only models are never in the masked_lm mapping, so they are unaffected.
_FAMILY_MAPPINGS = (
    ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", "image_text_to_text"),
    ("MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES", "vision_to_text"),
    ("MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES", "seq2seq"),
    ("MODEL_FOR_MASKED_LM_MAPPING_NAMES", "masked_lm"),
    ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "causal_lm"),
    ("MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES", "image_classification"),
    ("MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES", "audio_classification"),
    ("MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES", "sequence_classification"),
)


def detect_family(model_type: str) -> str | None:
    """Task family for a model_type, via the Auto* model mappings (generic, no special-casing)."""
    try:
        from transformers.models.auto import modeling_auto
    except Exception:
        return None
    for attr, family in _FAMILY_MAPPINGS:
        mapping = getattr(modeling_auto, attr, None)
        if mapping and model_type in mapping:
            return family
    return None


def detect_view(config: Any, family: str | None) -> str:
    """decoder | encoder | enc_dec | multimodal — which high-level layout the model uses."""
    if _cfg(config, "vision_config", "audio_config") is not None or family in {
        "image_text_to_text",
        "vision_to_text",
    }:
        return "multimodal"
    if getattr(config, "is_encoder_decoder", False) or family == "seq2seq":
        return "enc_dec"
    if family == "causal_lm":
        return "decoder"
    if family in {"masked_lm", "sequence_classification", "image_classification", "audio_classification"}:
        return "encoder"
    return "decoder" if getattr(config, "is_decoder", False) else "encoder"


def architecture_facts(model_type: str, config: Any) -> dict[str, Any]:
    """Model-level normalized facts block."""
    family = detect_family(model_type)
    moe = moe_spec(config)
    facts = {
        "view": detect_view(config, family),
        "family": family,
        "attention_variant": attention_variant(config),
        "positional": positional_scheme(config),
        "is_moe": moe is not None,
        "sliding_window": _cfg(config, "sliding_window"),
        "tie_word_embeddings": _cfg(config, "tie_word_embeddings"),
    }
    if moe is not None:
        facts["moe"] = moe
    return {key: value for key, value in facts.items() if value is not None}
