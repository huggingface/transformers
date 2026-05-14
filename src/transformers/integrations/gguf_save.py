"""Write a GGUF file from a transformers model.

Two flows are supported:

1. **Round-trip**: a model that was loaded via ``from_pretrained(..., gguf_file=...)``
   keeps a handle to its :class:`~transformers.quantizers.quantizer_gguf.GGUFQuantizer`,
   which holds the original ``hf_name → gguf_name`` map and the per-tensor quant
   types. :func:`save_pretrained_gguf` walks the live module tree, recovers the raw
   GGUF bytes from each :class:`GgufLinear` / :class:`GgufQwen2MoeExperts` (which
   never lost them — load uses direct byte routing), and writes them back out
   verbatim under their original GGUF names.

2. **Quantize-on-save**: tensors that don't already carry GGUF bytes (LM heads,
   embeddings, norms, or every layer when starting from a plain HF checkpoint)
   are routed through ``gguf.quants.quantize`` per the user-supplied
   ``quant_config`` policy. The default policy keeps everything at F32 / F16 —
   pass ``quant_config={"default": "Q4_K"}`` (or per-pattern overrides) to
   actually quantize.

Designed for Apple Silicon + MoE workflows: a Q4_K_M Qwen2-MoE-A2.7B that
round-trips through transformers and back to a .gguf produces bit-identical
output bytes (gate/up = Q4_K, down = Q8_0 or Q5_0 — preserved verbatim).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import torch

from .gguf_dequant import GGUFQuantizedTensor


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


# Sentinel used by quant_config to mean "use the loaded tensor's original quant
# type if it has one, otherwise fall back to F32." Round-trip default.
_KEEP = "_keep_"


def _resolve_quant(name: str, original_quant: Optional[str], policy: dict) -> str:
    """Pick the target GGUF quant for one tensor.

    Lookup order: longest matching pattern key in ``policy`` (regex match against
    the GGUF tensor name) → ``policy["default"]`` → ``_KEEP``. ``_KEEP`` means
    write under the original quant type if known, else F32.
    """
    matched = None
    for pat, qt in policy.items():
        if pat == "default":
            continue
        if re.search(pat, name) and (matched is None or len(pat) > len(matched[0])):
            matched = (pat, qt)
    target = matched[1] if matched else policy.get("default", _KEEP)
    if target == _KEEP:
        return original_quant or "F32"
    return target


def _arch_from_model(model) -> str:
    """Map ``model.config.model_type`` to a GGUF architecture string.

    We rely on the same mapping the loader uses to recognise the .gguf — the
    set of supported HF model_types in :mod:`modeling_gguf_pytorch_utils`.
    """
    model_type = getattr(model.config, "model_type", None)
    if model_type is None:
        raise ValueError("Cannot infer GGUF architecture: model.config.model_type is unset")
    # GGUF uses the same labels as transformers' model_type for the families we
    # support — same names show up on both sides of GGUF_TO_TRANSFORMERS_MAPPING.
    return model_type


def _collect_gguf_module_bytes(
    model,
) -> tuple[dict[str, tuple[bytes, str, tuple[int, ...]]], set[str]]:
    """Walk modules and pull raw GGUF bytes out of every GgufLinear /
    GgufQwen2MoeExperts.

    Returns:
        ``(hf_name → (raw_bytes, quant_type, logical_shape), absorbed_state_keys)``.

        - ``hf_name`` is the HF *weight* name that the loader's GGUF rename
          pipeline would map back to a GGUF tensor (``…q_proj.weight``,
          ``…experts.<idx>.<proj>_proj.weight``). It is *not* a state_dict key.
        - ``absorbed_state_keys`` lists every ``state_dict()`` key we've already
          covered (``…qweight``, ``…gate_proj_q``, etc.) — callers walking
          state_dict skip them.

    For ``GgufQwen2MoeExperts``, the fused expert buffer is split into
    ``num_experts`` slices, each emitted under the per-expert HF name
    ``…experts.<idx>.<proj>_proj.weight`` so the reverse rename hits the
    standard ``ffn_<proj>_exps.weight`` pattern the loader knows.
    """
    from .gguf_linear import GgufLinear, GgufQwen2MoeExperts

    out: dict[str, tuple[bytes, str, tuple[int, ...]]] = {}
    absorbed: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, GgufLinear):
            qbytes = bytes(module.qweight.detach().cpu().numpy().tobytes())
            shape = (module.out_features, module.in_features)
            out[f"{name}.weight"] = (qbytes, module.quant_type, shape)
            absorbed.add(f"{name}.qweight")
            if module.bias is not None:
                # Bias survives state_dict; we still want to emit it as F32.
                pass  # left for the state_dict pass
        elif isinstance(module, GgufQwen2MoeExperts):
            for proj, buf_name, bytes_per, quant in (
                ("gate", "gate_proj_q", module._gate_bytes_per, module.gate_quant),
                ("up",   "up_proj_q",   module._up_bytes_per,   module.up_quant),
                ("down", "down_proj_q", module._down_bytes_per, module.down_quant),
            ):
                buf = getattr(module, buf_name).detach().cpu().numpy()
                if proj == "down":
                    logical = (module.hidden_dim, module.intermediate_dim)
                else:
                    logical = (module.intermediate_dim, module.hidden_dim)
                for e in range(module.num_experts):
                    start = e * bytes_per
                    expert_bytes = bytes(buf[start : start + bytes_per].tobytes())
                    hf_name = f"{name}.{e}.{proj}_proj.weight"
                    out[hf_name] = (expert_bytes, quant, logical)
                absorbed.add(f"{name}.{buf_name}")
    return out, absorbed


def _reverse_rename_map(model, quantizer=None) -> dict[str, str]:
    """Build ``hf_name → gguf_name`` for the *current* model.

    When the model was loaded from a .gguf, ``quantizer.hf_to_gguf`` already
    captured this map (built in ``_process_model_after_weight_loading``).
    Otherwise, derive it on the fly from the architecture's GGUF converters.
    """
    if quantizer is not None and getattr(quantizer, "hf_to_gguf", None):
        return dict(quantizer.hf_to_gguf)

    # Cold path: build it by enumerating model param names and running the
    # forward rename in reverse via reverse_transform().
    from ..core_model_loading import WeightConverter, WeightRenaming, rename_source_key
    from ..modeling_gguf_pytorch_utils import get_gguf_converters

    arch = _arch_from_model(model)
    forward_table = get_gguf_converters(arch)
    if not forward_table:
        raise ValueError(
            f"No GGUF rename rules registered for model_type {arch!r}; "
            "either add them in modeling_gguf_pytorch_utils.py or set "
            "quant_config={'default': 'F32'} and supply a raw mapping."
        )
    # Strip quantization_operation before reverse_transform — it'd otherwise
    # reject the call (the GGUFDequantize on-load op has no save-side reverse).
    safe = []
    for t in forward_table:
        copied = type(t).__new__(type(t))
        copied.__dict__.update(t.__dict__)
        copied.quantization_operation = None
        safe.append(copied)
    reverses = [t.reverse_transform() for t in safe]
    inv_renamings  = [t for t in reverses if isinstance(t, WeightRenaming) and not isinstance(t, WeightConverter)]
    inv_converters = [t for t in reverses if isinstance(t, WeightConverter)]

    mapping: dict[str, str] = {}
    prefix = getattr(model, "base_model_prefix", "") or ""
    meta_state_dict = {k: None for k in model.state_dict().keys()}
    for hf_name in meta_state_dict:
        try:
            gguf_name, _ = rename_source_key(
                hf_name, inv_renamings, inv_converters,
                prefix=prefix, meta_state_dict=meta_state_dict,
            )
        except Exception:
            continue
        mapping[hf_name] = gguf_name
    return mapping


def save_pretrained_gguf(
    model: "PreTrainedModel",
    path: str,
    *,
    quant_config: Optional[dict] = None,
    quantizer=None,
) -> str:
    """Write ``model`` to ``path`` as a GGUF file.

    Args:
        model: a ``PreTrainedModel`` instance, either loaded via
            ``from_pretrained(gguf_file=...)`` (round-trip path — bytes are
            copied verbatim from the in-memory ``GgufLinear`` /
            ``GgufQwen2MoeExperts`` modules) or built any other way
            (quantize-on-save path).
        path: output GGUF file path.
        quant_config: per-tensor quantization policy. Keys are regex patterns
            matched against the GGUF tensor name; the longest match wins.
            ``"default"`` is the fallback for unmatched tensors. Tensor types
            are GGUF quant type names (``"Q4_K"``, ``"Q8_0"``, ``"F16"``, …).
            If absent, round-trip semantics (keep original quant where known,
            F32 otherwise).
        quantizer: optional explicit ``GGUFQuantizer`` to take the rename map
            from. Defaults to ``model.hf_quantizer``.

    Returns:
        ``path`` on success.
    """
    import gguf
    import numpy as np

    if quantizer is None:
        quantizer = getattr(model, "hf_quantizer", None)

    policy = dict(quant_config or {})
    policy.setdefault("default", _KEEP)

    hf_to_gguf = _reverse_rename_map(model, quantizer)
    module_bytes, absorbed_state_keys = _collect_gguf_module_bytes(model)
    gguf_to_hf = {g: h for h, g in hf_to_gguf.items()}

    # Per-tensor original quant types from the quantizer's loaded GGUF tensors.
    # Falls back to the value stashed on the live GgufLinear / GgufQwen2MoeExperts.
    gguf_original_quant: dict[str, str] = {}
    if quantizer is not None and getattr(quantizer, "gguf_tensors", None):
        for gname, t in quantizer.gguf_tensors.items():
            qt = getattr(t, "quant_type", None)
            if qt is not None:
                gguf_original_quant[gname] = qt.name if hasattr(qt, "name") else str(qt)

    arch = _arch_from_model(model)
    writer = gguf.GGUFWriter(path, arch=arch)

    # Replay the source file's kv pairs (block_count, head_count, tokenizer,
    # quantization_version, …) so reloading produces the same config. We skip
    # ``general.architecture`` (GGUFWriter sets it from the ctor) and any keys
    # the writer rejects (e.g. unknown enum types).
    if quantizer is not None and getattr(quantizer, "gguf_kv", None):
        from gguf import GGUFValueType

        # Skip auto-written magic/header keys (the writer emits these from the
        # file structure, replaying them produces a "Duplicate <key>" error on
        # load) and the architecture (already set by the writer ctor).
        for key, (val, vtype) in quantizer.gguf_kv.items():
            if key.startswith("GGUF.") or key in ("general.architecture", "general.alignment"):
                continue
            try:
                if isinstance(val, list):
                    sub = GGUFValueType(vtype) if isinstance(vtype, int) else None
                    writer.add_array(key, val)
                else:
                    writer.add_key_value(key, val, GGUFValueType(vtype))
            except Exception:
                continue

    seen: set[str] = set()

    # --- Pass 0: round-trip path. For tensors the quantizer kept verbatim
    # from the source .gguf (``quantizer.gguf_tensors``), copy the original
    # bytes through. This is the only correct path for attn_q / attn_k since
    # GGUF stores them in llama.cpp's permuted layout; the live ``GgufLinear``
    # holds the *unpermuted* re-quantization, which would deserialize wrong
    # on reload.
    if quantizer is not None and getattr(quantizer, "gguf_tensors", None):
        _float_dtype = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
        for gguf_name, t in quantizer.gguf_tensors.items():
            if gguf_name in seen:
                continue
            qt = getattr(t, "quant_type", None)
            if qt is None:
                continue
            qt_str = qt.name if hasattr(qt, "name") else str(qt)
            target = _resolve_quant(gguf_name, qt_str, policy)
            if target != qt_str:
                # Re-quantizing during round-trip is out of scope; user can
                # always load → modify state_dict → save with quant_config.
                continue
            seen.add(gguf_name)
            raw_u8 = t.detach().cpu().numpy()
            tensor_shape = tuple(t.shape)  # source GGUF logical shape

            if qt_str in _float_dtype:
                # Non-quantized: view the raw bytes as the float dtype and pass
                # the array directly; gguf-py will use its shape verbatim.
                arr = np.frombuffer(raw_u8.tobytes(), dtype=_float_dtype[qt_str]).reshape(tensor_shape)
                writer.add_tensor(gguf_name, arr, raw_dtype=getattr(gguf.GGMLQuantizationType, qt_str))
                continue

            # Quantized: reshape (rows, row_bytes) so gguf-py recovers the
            # logical (rows, cols) via quant_shape_from_byte_shape.
            M = tensor_shape[0] if tensor_shape else 1
            row_bytes = raw_u8.size // max(M, 1)
            arr = raw_u8.reshape(M, row_bytes) if row_bytes else raw_u8
            writer.add_tensor(
                gguf_name, arr,
                raw_dtype=getattr(gguf.GGMLQuantizationType, qt_str),
            )

    # --- Pass A: raw GGUF bytes pulled from live Gguf* modules (fallback for
    # tensors not in quantizer.gguf_tensors — e.g. modules created from scratch).
    for hf_name, (raw, qt_str, logical) in module_bytes.items():
        gguf_name = hf_to_gguf.get(hf_name, hf_name)
        if gguf_name in seen:
            continue
        seen.add(gguf_name)
        target = _resolve_quant(gguf_name, qt_str, policy)
        if target != qt_str:
            raise ValueError(
                f"Cannot re-quantize {gguf_name!r} from {qt_str} to {target} "
                "on save — drop the override or re-load without linear_mode."
            )
        M, _ = logical
        arr = np.frombuffer(raw, dtype=np.uint8)
        row_bytes = arr.size // M
        arr = arr.reshape(M, row_bytes)
        writer.add_tensor(
            gguf_name, arr,
            raw_dtype=getattr(gguf.GGMLQuantizationType, qt_str),
        )

    # --- Pass B: remaining state_dict entries (norms / embeddings / fp32) ---
    state_dict = model.state_dict()
    for hf_name, hf_tensor in state_dict.items():
        if hf_name in absorbed_state_keys:
            continue
        gguf_name = hf_to_gguf.get(hf_name, hf_name)
        if gguf_name in seen:
            continue
        seen.add(gguf_name)
        original_quant = gguf_original_quant.get(gguf_name)
        if not torch.is_tensor(hf_tensor):
            continue
        target = _resolve_quant(gguf_name, original_quant, policy)

        # F32 / F16 / BF16 don't go through gguf.quants.quantize — they map to
        # GGML float dtypes directly.
        if target in ("F32", "F16", "BF16"):
            np_dtype = {"F32": np.float32, "F16": np.float16, "BF16": "bfloat16"}[target]
            cpu = hf_tensor.detach().to(torch.float32 if target == "F32" else torch.float16).cpu().numpy()
            writer.add_tensor(
                gguf_name, cpu,
                raw_dtype=getattr(gguf.GGMLQuantizationType, target),
            )
            continue

        # Quantized target — let gguf-py do the work. Only available for the
        # quant types where gguf-py ships a Python quantizer (Q4_0/Q5_0/Q5_1/
        # Q8_0). K-quants raise NotImplementedError there, so flag clearly.
        fp = hf_tensor.detach().to(torch.float32).cpu().numpy()
        try:
            packed = gguf.quants.quantize(fp, getattr(gguf.GGMLQuantizationType, target))
        except NotImplementedError as e:
            raise NotImplementedError(
                f"gguf-py has no Python quantizer for {target} (needed for {gguf_name!r}); "
                "either pick a Q4_0/Q5_0/Q5_1/Q8_0/F16 target or leave it at F32."
            ) from e
        writer.add_tensor(
            gguf_name, packed,
            raw_dtype=getattr(gguf.GGMLQuantizationType, target),
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
