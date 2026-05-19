# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GGUF quantizer — mirrors the FP8 quantizer pattern.

Three load paths:

* ``from_pretrained(repo, gguf_file=...)`` on MPS: modules swap to
  :class:`GgufLinear` / :class:`GgufExperts` at meta time. Bytes flow through
  the standard rename pipeline via the target-aware :class:`GGUFDequantize`
  (which passes uint8 bytes straight to the swapped buffers) and the experts
  merge converter is rewritten in-place to per-projection renames so the
  whole load is one machinery — no post-load fix-up.
* ``from_pretrained(repo, gguf_file=..., dtype=...)`` or non-MPS:
  :class:`GGUFDequantize` dequantizes and the model loads as a standard
  ``nn.Linear`` chain in the requested dtype.
* ``from_pretrained(repo, quantization_config=GgufQuantizeConfig(...))``:
  load fp16/bf16 normally; :class:`GGUFQuantize` (the
  :class:`GGUFDequantize` reverse op) packs matching Linears on the fly and
  the meta-time swap drops in :class:`GgufLinear` modules.
"""

from __future__ import annotations

import logging

import torch

from .base import HfQuantizer


logger = logging.getLogger(__name__)


class GGUFQuantizer(HfQuantizer):
    """GGUF quantizer."""

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        from ..utils.quantization_config import GgufQuantizeConfig

        if quantization_config is None:
            quantization_config = GgufQuantizeConfig()
        # On-the-fly path = caller passed a ``GgufQuantizeConfig`` without a
        # source file (we're packing a live fp16/bf16 checkpoint). ``gguf_file=``
        # path = config carries a source file (we're loading existing GGUF bytes).
        on_the_fly = getattr(quantization_config, "gguf_file", None) is None
        kwargs.setdefault("pre_quantized", not on_the_fly)
        super().__init__(quantization_config=quantization_config, **kwargs)
        # Re-annotate with the concrete type so attribute accesses (``quant_type``,
        # ``modules_to_convert``, ``dequantize``, …) type-check without casts.
        self.quantization_config: GgufQuantizeConfig = quantization_config
        self.on_the_fly = on_the_fly
        # ``linear_mode`` is the on-the-fly default and the gguf_file default on
        # MPS without explicit dequant; ``_process_model_before_weight_loading``
        # finalizes it once it can see the live ``device_map``.
        self.linear_mode = on_the_fly
        # GGUF file state — populated by :meth:`load_checkpoint_state` when the
        # gguf_file path is active. Empty for the on-the-fly path.
        self.weight_mapping: list = []
        self.gguf_tensors: dict = {}
        self.gguf_kv: dict = {}
        # Reverse rename map (hf_target -> gguf_source), filled during
        # ``_build_quant_info_from_gguf_file``. Consumed by ``save_gguf``.
        self.hf_to_gguf: dict[str, str] = {}

    # ---- HfQuantizer hooks ----------------------------------------------------

    def update_weight_conversions(self, weight_conversions):
        """Inject :class:`GGUFDequantize` at the head of every weight converter
        + attach it as the ``quantization_operation`` on every rename — same
        shape as ``Fp8Quantizer.update_weight_conversions``. No experts-specific
        rewrite needed: :class:`GgufExperts` exposes the same ``gate_up_proj`` /
        ``down_proj`` buffer names as ``MixtralExperts`` / ``Qwen2MoeExperts``,
        so the GGUF merge converter (``ffn_gate_exps + ffn_up_exps →
        gate_up_proj``) lands its uint8 ``Concatenate(dim=1)`` output directly
        into the swapped buffer through the target-aware op."""
        from ..core_model_loading import WeightConverter, WeightRenaming
        from ..gguf_conversion_ops import GGUFDequantize

        injected = []
        for conv in self.weight_mapping:
            if isinstance(conv, WeightConverter):
                conv = WeightConverter(
                    source_patterns=conv._original_source_patterns,
                    target_patterns=conv._original_target_patterns,
                    operations=[GGUFDequantize(self), *conv.operations],
                )
            elif isinstance(conv, WeightRenaming):
                conv.quantization_operation = GGUFDequantize(self)
            injected.append(conv)
        return injected + list(weight_conversions)

    def get_quantize_ops(self):
        """On-the-fly path: the loader calls this when ``param_needs_quantization``
        returns True. Returns the :class:`GGUFQuantize` op that packs a live
        fp16/bf16 weight into GGUF block bytes."""
        from ..gguf_conversion_ops import GGUFQuantize

        return GGUFQuantize(self)

    def param_needs_quantization(self, model, param_name, **kwargs) -> bool:
        """Only the on-the-fly path needs per-param quantization (re-packing the
        incoming fp16/bf16 weight into GGUF bytes). The pre-quantized path leans
        on the rename pipeline + the post-load byte-copy."""
        if not self.on_the_fly:
            return False
        import torch.nn as nn

        from ..integrations.gguf_linear import GgufLinear

        target_path = param_name.rsplit(".", 1)[0]
        try:
            module = model.get_submodule(target_path)
        except AttributeError:
            return False
        # After ``_process_model_before_weight_loading`` swapped the matching
        # Linears, those modules are GgufLinear and need quantization. Anything
        # else (norms, lm_head, embeddings) is left alone.
        return isinstance(module, GgufLinear) and not param_name.endswith(".bias") and isinstance(module, nn.Module)

    def load_checkpoint_state(self, gguf_path: str):
        """Pre-load the ``.gguf`` file's tensors + rename map + KV metadata.

        Called by :func:`PreTrainedModel.from_pretrained` for the gguf_file path
        after :func:`_get_resolved_checkpoint_files` resolves the file. Returns
        the state-dict (``{gguf_name: GGUFQuantizedTensor}``) that the standard
        loader needs — GGUF isn't a safetensors format, so this hook gives the
        quantizer ownership of the on-disk → in-memory state load.
        """
        from ..modeling_gguf_pytorch_utils import load_gguf_checkpoint

        parsed = load_gguf_checkpoint(gguf_path, return_tensors=True)
        self.weight_mapping = list(parsed.get("weight_mapping", []) or [])
        self.gguf_tensors = parsed["tensors"]
        self.gguf_kv = parsed.get("raw_kv", {}) or {}
        return parsed["tensors"]

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap nn.Linear / fused-expert modules in place at meta time.

        For the gguf_file path, finalize ``self.linear_mode`` here: native
        kernels run on MPS today (matching CUDA kernels are a TODO), so we
        swap when MPS is available and the caller didn't explicitly request
        dequantize-on-load (``dtype=`` -> ``GgufQuantizeConfig.dequantize``).
        """
        if not self.on_the_fly:
            self.linear_mode = torch.backends.mps.is_available() and not self.quantization_config.dequantize
        if not self.linear_mode:
            return
        from ..integrations.gguf_linear import replace_with_gguf_linear

        quant_info_by_target = self._build_quant_info(model)
        replace_with_gguf_linear(model, quant_info_by_target)

    def postprocess_model(self, model, **kwargs):
        """Run the base post-process, then apply good generation defaults when
        the GgufLinear swap is in effect. Byte routing for both GgufLinear and
        GgufExperts happens through the rename pipeline (target-aware
        GGUFDequantize) — no post-load byte copy. Kernel ops are resolved
        eagerly in each module's __init__ (no post-load bind step)."""
        super().postprocess_model(model, **kwargs)
        if self.linear_mode:
            self._apply_generation_defaults(model)
        return model

    def save_gguf(self, model, path: str, *, quant_config: dict | None = None) -> str:
        """Write the model back to a .gguf file via the thin reverse-rename
        driver in :mod:`integrations.gguf_writer`."""
        from ..integrations.gguf_writer import save_pretrained_gguf

        return save_pretrained_gguf(model, path, quant_config=quant_config, quantizer=self)

    @property
    def is_trainable(self):
        return True

    def is_serializable(self):
        return False

    # ---- Internals ------------------------------------------------------------

    def _build_quant_info(self, model) -> dict[str, dict]:
        """Walk the live model and emit one entry per module that should swap.

        On-the-fly path (``GgufQuantizeConfig`` without a source file): every
        :class:`nn.Linear` matching ``modules_to_convert`` gets the config's
        single ``quant_type``.

        gguf_file path: per-tensor quant types come from :attr:`gguf_tensors`
        after renaming the GGUF source names through ``self.weight_mapping``.
        :attr:`hf_to_gguf` is filled here as a side effect (consumed by
        ``save_gguf``).
        """
        import torch.nn as nn

        from ..integrations.gguf_linear import MODEL_TYPE_TO_GGUF_EXPERTS

        if self.on_the_fly:
            import fnmatch

            cfg = self.quantization_config
            include = cfg.modules_to_convert
            exclude = set(cfg.modules_to_not_convert or [])
            return {
                name: {"quant_type": cfg.quant_type}
                for name, mod in model.named_modules()
                if isinstance(mod, nn.Linear)
                and (include is None or any(fnmatch.fnmatchcase(name, pat) for pat in include))
                and not any(skip in name for skip in exclude)
            }

        # gguf_file path: rename gguf sources → hf targets → record quant_type.
        from ..core_model_loading import WeightConverter, WeightRenaming, rename_source_key
        from ..integrations.gguf_linear import gguf_linear_supports

        renamings = [e for e in self.weight_mapping if isinstance(e, WeightRenaming)]
        converters = [e for e in self.weight_mapping if isinstance(e, WeightConverter)]
        meta_state_dict = model.state_dict()
        prefix = getattr(model, "base_model_prefix", "") or ""

        hf_quant: dict[str, str] = {}
        for gguf_name, tensor in self.gguf_tensors.items():
            qt = getattr(tensor, "quant_type", None)
            if qt is None or not gguf_linear_supports(qt):
                continue
            try:
                hf_name, _ = rename_source_key(
                    gguf_name, renamings, converters, prefix=prefix, meta_state_dict=meta_state_dict
                )
            except Exception:
                continue
            self.hf_to_gguf[hf_name] = gguf_name
            hf_quant[hf_name] = qt.name if hasattr(qt, "name") else str(qt)

        experts_cls = MODEL_TYPE_TO_GGUF_EXPERTS.get(getattr(model.config, "model_type", None))
        info: dict[str, dict] = {}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                qt = hf_quant.get(f"{name}.weight")
                if qt is not None:
                    info[name] = {"quant_type": qt}
            elif experts_cls is not None and hasattr(mod, "gate_up_proj") and hasattr(mod, "down_proj"):
                gate_up = hf_quant.get(f"{name}.gate_up_proj")
                down = hf_quant.get(f"{name}.down_proj")
                if gate_up is not None and down is not None:
                    info[name] = {"gate_up_quant": gate_up, "down_quant": down}
        return info

    def _apply_generation_defaults(self, model) -> None:
        """Set ``cache_implementation`` + ``compile_config`` defaults on
        ``model.generation_config`` so the user gets good performance out of the
        box. Doesn't overwrite values the user already set."""
        from ..generation.configuration_utils import CompileConfig

        gc = getattr(model, "generation_config", None)
        if gc is None:
            return
        changed = []
        if getattr(gc, "cache_implementation", None) is None:
            gc.cache_implementation = "static"
            changed.append('cache_implementation="static"')
        if getattr(gc, "compile_config", None) is None:
            gc.compile_config = CompileConfig(mode="reduce-overhead", dynamic=False)
            changed.append('compile_config=CompileConfig(mode="reduce-overhead", dynamic=False)')
        if changed:
            logger.info(
                "GGUFQuantizer: defaulting generation_config to %s for the GgufLinear kernel path. "
                "Pass values explicitly on `model.generation_config` to override.",
                ", ".join(changed),
            )
