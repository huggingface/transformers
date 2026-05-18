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

from .base import HfQuantizer


logger = logging.getLogger(__name__)


class GGUFQuantizer(HfQuantizer):
    """GGUF quantizer."""

    requires_calibration = False

    def __init__(
        self,
        quantization_config=None,
        weight_mapping=None,
        linear_mode=False,
        gguf_tensors=None,
        gguf_kv=None,
        **kwargs,
    ):
        on_the_fly = quantization_config is not None and getattr(quantization_config, "quant_method", None) == "gguf"
        if quantization_config is None:
            # ``gguf_file=`` path: the base class needs a config to record on the
            # model so ``model.quantization_method`` etc. can be set. Attach the
            # default so callers don't have to pass one explicitly.
            from ..utils.quantization_config import GgufQuantizeConfig

            quantization_config = GgufQuantizeConfig()
        kwargs.setdefault("pre_quantized", not on_the_fly)
        super().__init__(quantization_config=quantization_config, **kwargs)
        self.weight_mapping = list(weight_mapping or [])
        # ``linear_mode``: when True, modules are swapped to GgufLinear / GgufExperts
        # at meta time and run the metal kernels on raw bytes. Always True for the
        # on-the-fly path (the quant config implies we want the kernels).
        self.linear_mode = bool(linear_mode) or on_the_fly
        self.on_the_fly = on_the_fly
        # ``{gguf_name: GGUFQuantizedTensor}`` map populated by modeling_utils for
        # the ``gguf_file=`` path. Empty for on-the-fly.
        self.gguf_tensors: dict = dict(gguf_tensors or {})
        # Reverse rename map (hf_target -> gguf_source), filled lazily during the
        # post-load byte-copy step. Consumed by ``save_gguf``.
        self.hf_to_gguf: dict[str, str] = {}
        # Raw GGUF key-value snapshot from the source file (replayed verbatim by
        # ``save_gguf`` so the round-trip preserves tokenizer / block_count / ...).
        self.gguf_kv: dict = dict(gguf_kv or {})

    # ---- HfQuantizer hooks ----------------------------------------------------

    @property
    def renaming_dequantize_op(self):
        """Op the rename pipeline attaches to ``WeightRenaming`` entries so the
        renamed key lands as a regular floating-point tensor rather than a raw
        uint8 ``GGUFQuantizedTensor``."""
        from ..gguf_conversion_ops import GGUFDequantize

        return GGUFDequantize(self)

    def update_weight_conversions(self, weight_conversions):
        """Inject :class:`GGUFDequantize` at the head of every weight converter +
        attach it as the ``quantization_operation`` on every rename — mirrors
        ``Fp8Quantizer.update_weight_conversions``.

        Linear-mode-specific rewrite: the GGUF rename rules for MoE archs ship a
        merge ``WeightConverter`` (``ffn_gate_exps + ffn_up_exps → gate_up_proj``)
        whose target doesn't exist on the swapped :class:`GgufExperts` (which
        keeps gate/up as separate ``gate_proj_q`` / ``up_proj_q`` byte buffers).
        Replace that merge with three plain :class:`WeightRenaming` entries so
        the rename pipeline lands bytes directly into the per-projection
        buffers via the target-aware :class:`GGUFDequantize`. Down-proj's
        existing ``ffn_down_exps → mlp.experts.down_proj`` rename gets its
        target rewritten to ``...down_proj_q``.
        """
        from ..core_model_loading import WeightConverter, WeightRenaming
        from ..gguf_conversion_ops import GGUFDequantize

        rewritten_mapping = self._rewrite_experts_mapping() if self.linear_mode else self.weight_mapping

        injected = []
        for conv in rewritten_mapping:
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

    def _rewrite_experts_mapping(self):
        """Return a copy of ``self.weight_mapping`` with the experts merge
        ``WeightConverter`` replaced by per-projection ``WeightRenaming`` entries
        and ``ffn_down_exps`` retargeted at the ``GgufExperts`` byte buffer."""
        from ..core_model_loading import WeightConverter, WeightRenaming

        out: list = []
        for conv in self.weight_mapping:
            if isinstance(conv, WeightConverter):
                srcs = list(conv.source_patterns)
                has_gate = any("ffn_gate_exps" in s for s in srcs)
                has_up = any("ffn_up_exps" in s for s in srcs)
                if has_gate and has_up:
                    # Drop the merge converter; emit per-projection renames.
                    gate_src = next(s for s in srcs if "ffn_gate_exps" in s)
                    up_src = next(s for s in srcs if "ffn_up_exps" in s)
                    out.append(WeightRenaming(gate_src, ".mlp.experts.gate_proj_q"))
                    out.append(WeightRenaming(up_src, ".mlp.experts.up_proj_q"))
                    continue
                out.append(conv)
                continue
            if isinstance(conv, WeightRenaming):
                # Retarget ``ffn_down_exps → mlp.experts.down_proj`` to the
                # GgufExperts buffer name.
                if any("ffn_down_exps" in s for s in conv._original_source_patterns):
                    out.append(WeightRenaming(conv._original_source_patterns[0], ".mlp.experts.down_proj_q"))
                    continue
            out.append(conv)
        return out

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

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap ``nn.Linear`` / fused-expert modules in place at meta time.
        Mirrors :func:`replace_with_fp8_linear` in the FP8 quantizer."""
        if not self.linear_mode:
            return
        from ..integrations.gguf_linear import replace_with_gguf_linear

        quant_info_by_target = self._build_quant_info(model)
        replace_with_gguf_linear(model, quant_info_by_target)

    def postprocess_model(self, model, **kwargs):
        """Run the base post-process, then apply good generation defaults when
        the GgufLinear swap is in effect. Byte routing for both
        :class:`GgufLinear` and :class:`GgufExperts` happens through the rename
        pipeline (target-aware :class:`GGUFDequantize` +
        :meth:`_rewrite_experts_mapping`) — no post-load byte copy."""
        super().postprocess_model(model, **kwargs)
        if self.linear_mode:
            self._bind_experts_kernels(model)
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
        """Map each module-name target to its quant info, ready for
        :func:`replace_with_gguf_linear`. The on-the-fly path uses the config's
        ``modules_to_convert`` glob + the single ``quant_type``; the
        ``gguf_file=`` path reads per-tensor quant types out of
        :attr:`gguf_tensors` and renames source names to module-name targets."""
        if self.on_the_fly:
            return self._build_quant_info_on_the_fly(model)
        return self._build_quant_info_from_gguf_file(model)

    def _build_quant_info_on_the_fly(self, model) -> dict[str, dict]:
        import fnmatch

        import torch.nn as nn

        cfg = self.quantization_config
        include = getattr(cfg, "modules_to_convert", None)
        exclude = set(getattr(cfg, "modules_to_not_convert", None) or [])
        quant_type = getattr(cfg, "quant_type", "Q4_0")
        info: dict[str, dict] = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if include is not None and not any(fnmatch.fnmatchcase(name, pat) for pat in include):
                continue
            if any(skip in name for skip in exclude):
                continue
            info[name] = {"quant_type": quant_type}
        return info

    def _build_quant_info_from_gguf_file(self, model) -> dict[str, dict]:
        """For each gguf tensor name, find the corresponding hf module path
        (Linear or fused-experts) and record its quant type."""
        from ..core_model_loading import WeightConverter, WeightRenaming, rename_source_key
        from ..integrations.gguf_linear import gguf_linear_supports

        renamings = [e for e in self.weight_mapping if isinstance(e, WeightRenaming)]
        converters = [e for e in self.weight_mapping if isinstance(e, WeightConverter)]
        meta_state_dict = model.state_dict()
        prefix = getattr(model, "base_model_prefix", "") or ""

        # Map each gguf name to its hf-target param name + quant_type.
        gguf_to_hf: dict[str, tuple[str, str]] = {}
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
            gguf_to_hf[gguf_name] = (hf_name, qt.name if hasattr(qt, "name") else str(qt))

        # Linear targets: ``<module>.weight`` -> {"quant_type": "Q4_K"}
        # Experts targets: group ffn_gate_exps / ffn_up_exps / ffn_down_exps by
        # the shared parent path (``...mlp.experts``).
        info: dict[str, dict] = {}
        for gguf_name, (hf_name, quant_type) in gguf_to_hf.items():
            if ".ffn_gate_exps." in gguf_name:
                key, slot = hf_name.rsplit(".", 1)[0], "gate_quant"
            elif ".ffn_up_exps." in gguf_name:
                key, slot = hf_name.rsplit(".", 1)[0], "up_quant"
            elif ".ffn_down_exps." in gguf_name:
                key, slot = hf_name.rsplit(".", 1)[0], "down_quant"
            else:
                # Plain Linear weight -> module path (strip ``.weight`` suffix)
                if hf_name.endswith(".weight"):
                    info[hf_name[: -len(".weight")]] = {"quant_type": quant_type}
                continue
            info.setdefault(key, {})[slot] = quant_type
        # Drop incomplete experts triples (missing gate/up/down).
        info = {
            k: v
            for k, v in info.items()
            if "quant_type" in v or all(s in v for s in ("gate_quant", "up_quant", "down_quant"))
        }
        return info

    def _bind_experts_kernels(self, model) -> None:
        """Resolve ``mul_mat_id_<fmt>_f32`` kernel refs + pre-allocate decode
        scratch buffers on every swapped :class:`GgufExperts` module. Called
        after weight loading (the rename pipeline already populated the byte
        buffers via the target-aware :class:`GGUFDequantize`)."""
        from ..integrations.gguf_kernels import bind_id_kernel_refs
        from ..integrations.gguf_linear import GgufExperts

        for _, module in model.named_modules():
            if isinstance(module, GgufExperts):
                bind_id_kernel_refs(module)

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
