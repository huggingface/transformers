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


def embedding_spec(config: Any, raw_attributes: dict[str, Any] | None = None) -> dict[str, Any]:
    """Config-derived token-embedding facts (vocab size, embedding dim).

    Prefers the concrete ``num_embeddings``/``embedding_dim`` observed on the module (covers
    padded/extended vocabularies), falling back to the config's vocab/hidden sizes.
    """
    raw_attributes = raw_attributes or {}
    vocab = raw_attributes.get("num_embeddings") or _cfg(config, "vocab_size")
    dim = raw_attributes.get("embedding_dim") or _cfg(config, "hidden_size", "d_model", "n_embd")
    spec = {"num_embeddings": vocab, "embedding_dim": dim}
    return {key: value for key, value in spec.items() if value is not None}


def position_spec(config: Any) -> dict[str, Any]:
    """Config-derived positional-encoding facts (scheme + its salient parameter)."""
    scheme = positional_scheme(config)
    spec: dict[str, Any] = {"scheme": scheme}
    if scheme == "rope":
        rope = _cfg(config, "rope_parameters")
        theta = rope.get("rope_theta") if isinstance(rope, dict) else None
        spec["rope_theta"] = theta if theta is not None else _cfg(config, "rope_theta")
        spec["head_dim"] = _cfg(config, "head_dim", "d_kv")
    elif scheme == "relative":
        spec["num_buckets"] = _cfg(config, "relative_attention_num_buckets")
    elif scheme in {"learned", "sinusoidal"}:
        spec["max_position_embeddings"] = _cfg(config, "max_position_embeddings", "max_target_positions")
    return {key: value for key, value in spec.items() if value is not None}


# Leaf names of the *output* projection of an attention / MLP block (writes back to the residual
# stream). Everything else under the block is treated as an input-side projection (reads it).
_OUTPUT_PROJECTION_NAMES = (
    "o_proj",
    "out_proj",
    "down_proj",
    "wo",
    "fc2",
    "c_proj",
    "proj_out",
    "w2",
    "dense_4h_to_h",
    "o",
)


def projection_role(name: str) -> str:
    """Classify a projection leaf name as ``"input"`` (fans out from the block) or ``"output"``.

    Role-based (not execution-order-based): q/k/v and gate/up read the block input in parallel,
    then o_proj / down_proj write the result back — a fan-out then fan-in, which is the true
    topology a linear call-order trace would misrepresent.
    """
    lower = name.lower()
    if lower in _OUTPUT_PROJECTION_NAMES or lower.endswith(("out_proj", "down_proj")):
        return "output"
    return "input"


def state_space_spec(config: Any) -> dict[str, Any]:
    """Config-derived state-space (Mamba/SSM) facts attached to state_space mixer components."""
    spec = {
        "state_size": _cfg(config, "state_size", "d_state", "mamba_d_state", "ssm_state_size"),
        "conv_kernel": _cfg(config, "conv_kernel", "d_conv", "mamba_d_conv", "mamba_conv_kernel"),
        "expand": _cfg(config, "expand", "mamba_expand"),
        "n_groups": _cfg(config, "n_groups", "mamba_n_groups"),
    }
    return {key: value for key, value in spec.items() if value is not None}


def mlp_spec(config: Any) -> dict[str, Any]:
    """Config-derived feed-forward facts (dims + activation) attached to MLP components."""
    spec = {
        "hidden_size": _cfg(config, "hidden_size", "d_model", "n_embd"),
        "intermediate_size": _cfg(config, "intermediate_size", "d_ff", "ffn_dim", "moe_intermediate_size"),
        "activation": _cfg(config, "hidden_act", "activation_function"),
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


def backbone_config(config: Any) -> Any:
    """The primary text backbone config for a multimodal model, else the config itself.

    Multimodal models nest the LLM under ``text_config`` (and vision/audio towers under their own
    sub-configs), so the model-level attention/positional facts live there, not on the composite.
    """
    return getattr(config, "text_config", None) or config


# ---------------------------------------------------------------------- capability recognizers


def attention_backends(model_class: type) -> list[str]:
    """Attention implementations this model *supports* (implementation-agnostic capability).

    Read from the model class's support flags — ``eager`` is always available as the reference
    implementation; the others require the model to opt in (and, at runtime, the backend to be
    installed). This says "can run with", not "is installed here"."""
    backends = ["eager"]
    if getattr(model_class, "_supports_sdpa", False):
        backends.append("sdpa")
    if getattr(model_class, "_supports_flash_attn", getattr(model_class, "_supports_flash_attn_2", False)):
        backends.append("flash_attention")
    if getattr(model_class, "_supports_flex_attn", False):
        backends.append("flex_attention")
    return backends


def task_heads(model_type: str) -> list[str]:
    """Task families available for this model_type across the Auto* model mappings."""
    try:
        from transformers.models.auto import modeling_auto
    except Exception:
        return []
    heads: set[str] = set()
    for name in dir(modeling_auto):
        if name.startswith("MODEL_FOR_") and name.endswith("_MAPPING_NAMES"):
            mapping = getattr(modeling_auto, name)
            if isinstance(mapping, dict) and model_type in mapping:
                heads.add(name[len("MODEL_FOR_") : -len("_MAPPING_NAMES")].lower())
    return sorted(heads)


def _normalize_attention_pattern(layer_type: str | None, view: str | None, sliding_window: Any) -> str:
    n = (layer_type or "").lower()
    if "sliding" in n:
        return "sliding"
    if "chunk" in n:
        return "chunked"
    if "compress" in n:
        return "compressed"
    if "bidirectional" in n:
        return "bidirectional"
    return "bidirectional" if view == "encoder" else "causal"


def attention_schedule(config: Any, view: str | None) -> dict[str, Any]:
    """Per-layer attention pattern schedule (drives the mask visualisation).

    Returns ``{schedule, patterns}``: ``schedule`` is the raw ``config.layer_types`` list (or None
    for a uniform model), ``patterns`` the distinct normalized kinds (causal / bidirectional /
    sliding / chunked / compressed)."""
    layer_types = getattr(config, "layer_types", None)
    sliding = _cfg(config, "sliding_window")
    if layer_types:
        patterns = list(dict.fromkeys(_normalize_attention_pattern(t, view, sliding) for t in layer_types))
        # Only carry the raw per-layer list when it actually varies; a uniform schedule is redundant.
        schedule = list(layer_types) if len(set(layer_types)) > 1 else None
        return {"schedule": schedule, "patterns": patterns}
    if view == "enc_dec":
        return {"schedule": None, "patterns": ["bidirectional", "causal"]}
    base = "sliding" if sliding else ("bidirectional" if view == "encoder" else "causal")
    return {"schedule": None, "patterns": [base]}


def tensor_parallel_plan(config: Any, backbone: Any = None) -> dict[str, str] | None:
    """The base-model tensor-parallel plan (module glob -> colwise/rowwise), if the model has one."""
    plan = getattr(config, "base_model_tp_plan", None)
    if not plan and backbone is not None:
        plan = getattr(backbone, "base_model_tp_plan", None)
    return dict(plan) if isinstance(plan, dict) and plan else None


def tp_style_for(path_pattern: str, plan: dict[str, str] | None) -> str | None:
    """Look up a component's TP style by matching its symbolic path against the plan's globs."""
    if not plan:
        return None
    key = path_pattern.removeprefix("model.").replace(".{i}.", ".*.")
    return plan.get(key)


def capabilities(model_type: str, config: Any, model_class: type, view: str | None) -> dict[str, Any]:
    """Implementation-agnostic capability facts: what the architecture *can* do / run with."""
    backbone = backbone_config(config)
    schedule = attention_schedule(backbone, view)
    caps = {
        "attention_backends": attention_backends(model_class),
        "attention_patterns": schedule["patterns"],
        "attention_schedule": schedule["schedule"],
        "task_heads": task_heads(model_type),
        "tensor_parallel": tensor_parallel_plan(config, backbone) is not None,
    }
    return caps


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
    """Model-level normalized facts block.

    ``view``/``family`` come from the composite config (they depend on vision/audio sub-configs and
    encoder-decoder-ness); the attention/positional/MoE facts come from the text **backbone**, so a
    multimodal model reports its LLM's attention variant rather than ``None``."""
    family = detect_family(model_type)
    backbone = backbone_config(config)
    moe = moe_spec(backbone)
    facts = {
        "view": detect_view(config, family),
        "family": family,
        "attention_variant": attention_variant(backbone),
        "positional": positional_scheme(backbone),
        "is_moe": moe is not None,
        "sliding_window": _cfg(backbone, "sliding_window"),
        "tie_word_embeddings": _cfg(config, "tie_word_embeddings"),
    }
    if moe is not None:
        facts["moe"] = moe
    return {key: value for key, value in facts.items() if value is not None}
