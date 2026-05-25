# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GGUF quantizer — mirrors the FP8 quantizer pattern.

Three load paths:

* `from_pretrained(repo, gguf_file=...)` on MPS: modules swap to
  :class:`GgufLinear` / :class:`GgufExperts` at meta time. Bytes flow through
  the standard rename pipeline via the target-aware :class:`GGUFDequantize`
  (which passes uint8 bytes straight to the swapped buffers) and the experts
  merge converter is rewritten in-place to per-projection renames so the
  whole load is one machinery — no post-load fix-up.
* `from_pretrained(repo, gguf_file=..., dtype=...)` or non-MPS:
  :class:`GGUFDequantize` dequantizes and the model loads as a standard
  `nn.Linear` chain in the requested dtype.
* `from_pretrained(repo, quantization_config=GgufQuantizeConfig(...))`:
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
        # Three load paths:
        #   - safetensors reload: `module_quant_types` carries the swap plan
        #     from a prior `save_pretrained` — bytes live in the model's own
        #     safetensors, no .gguf file needed. pre_quantized=True.
        #   - gguf_file=: source bytes come from an external .gguf — quantizer
        #     loads them via `load_checkpoint_state`. pre_quantized=True.
        #   - on-the-fly: live fp16/bf16 checkpoint, GGUFQuantize op packs at
        #     load time. pre_quantized=False.
        has_module_map = bool(getattr(quantization_config, "module_quant_types", None))
        has_gguf_file = getattr(quantization_config, "gguf_file", None) is not None
        on_the_fly = not (has_module_map or has_gguf_file)
        kwargs.setdefault("pre_quantized", not on_the_fly)
        super().__init__(quantization_config=quantization_config, **kwargs)
        # Re-annotate with the concrete type so attribute accesses type-check.
        self.quantization_config: GgufQuantizeConfig = quantization_config
        self.on_the_fly = on_the_fly
        # `linear_mode` is the on-the-fly default and the gguf_file default on
        # MPS without explicit dequant; `_process_model_before_weight_loading`
        # finalizes it once it can see the live `device_map`.
        self.linear_mode = on_the_fly
        # GGUF file state — populated by :meth:`load_checkpoint_state` when the
        # gguf_file path is active. Empty for the on-the-fly path.
        self.weight_mapping: list = []
        self.gguf_tensors: dict = {}
        self.gguf_kv: dict = {}
        # Reverse rename map (hf_target -> gguf_source), filled during
        # `_build_quant_info_from_gguf_file`. Consumed by `save_gguf`.
        self.hf_to_gguf: dict[str, str] = {}

    # ---- HfQuantizer hooks ----------------------------------------------------

    def update_weight_conversions(self, weight_conversions):
        """Prepend the GGUF rename / merge / reverse-permute converters
        produced from the source `.gguf` (`load_checkpoint_state` populates
        `self.weight_mapping`) onto the model's existing conversion list.

        FP8's `update_weight_conversions` exists to *fuse per-block scales*
        into the weight tensors before any merge/concat ops. GGUF has no
        separate scale tensor — each block carries its own scale inline — so
        there's nothing to fuse here. The dequant decision is binary and
        made up-front in `load_checkpoint_state` (every quantized source is
        either kept as bytes for the GgufLinear/GgufExperts swap, or already
        dequantized at load time when `dequantize=True`). By the time this
        runs, the value side is settled; we just splice the GGUF rename map
        in front of the model-side conversions.
        """
        return list(self.weight_mapping) + list(weight_conversions)

    def get_quantize_ops(self):
        """On-the-fly path: the loader calls this when `param_needs_quantization`
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
        # After `_process_model_before_weight_loading` swapped the matching
        # Linears, those modules are GgufLinear and need quantization. Anything
        # else (norms, lm_head, embeddings) is left alone.
        return isinstance(module, GgufLinear) and not param_name.endswith(".bias") and isinstance(module, nn.Module)

    def load_checkpoint_state(self, gguf_path: str):
        """Pre-load the `.gguf` file's tensors + rename map + KV metadata.

        Called by :func:`PreTrainedModel.from_pretrained` for the gguf_file path
        after :func:`_get_resolved_checkpoint_files` resolves the file. Returns
        the state-dict that the standard loader needs — GGUF isn't a
        safetensors format, so this hook gives the quantizer ownership of the
        on-disk → in-memory state load.

        Dequant decision lives here, not in a conversion op:

          * `dequantize=True` (or non-MPS): dequant every quantized tensor
            up-front. Modules stay as plain `nn.Linear` / `nn.Embedding`
            with fp weights. No op in the conversion chain.
          * Default (MPS swap path): keep quantized tensors as
            `GGUFQuantizedTensor` raw bytes when their quant_type has a
            Metal kernel (`_QUANT_INFO`) — they flow into `GgufLinear` /
            `GgufExperts` uint8 buffers. Anything else still has to be
            pre-dequanted: embedding-bound tensors (their HF target is
            `nn.Embedding`, not a swappable Linear), and any quant_type
            without a Metal kernel (IQ1_S, IQ2_*, IQ3_*, Q2_K, Q3_K, Q4_1)
            — for those the swap step skips the layer and the bytes would
            otherwise be assigned raw into a plain `nn.Linear.weight`.
        """
        from ..integrations.gguf_dequant import GGUFQuantizedTensor, dequantize_gguf_tensor
        from ..integrations.gguf_kernels import metal_kernels_available
        from ..integrations.gguf_linear import gguf_linear_supports
        from ..modeling_gguf_pytorch_utils import load_gguf_checkpoint

        parsed = load_gguf_checkpoint(gguf_path, return_tensors=True)
        self.weight_mapping = list(parsed.get("weight_mapping", []) or [])
        tensors = parsed["tensors"]
        # Gemma2 / Gemma3 / Nemotron store every RMSNorm weight as `w + 1`
        # in the GGUF; the matching `SubtractOne` op in the converter chain
        # runs *after* the loader's `.to(dtype)` cast, which on fp16/bf16
        # loses 1 ULP near `w = 1` (the steady-state norm scale) and breaks
        # the weights-conversion tests. Pre-apply the subtraction here on
        # the fp32 source so the loader's cast is the only rounding step,
        # matching the legacy `NemotronTensorProcessor.process` / `Gemma2TensorProcessor.process`.
        arch = (parsed.get("config", {}) or {}).get("model_type")
        if arch in ("gemma2", "gemma3", "gemma3_text", "nemotron"):
            for name, t in tensors.items():
                # Only norm *weights* are stored as `w + 1`; biases pass through
                # unchanged.
                if name.endswith("_norm.weight") and isinstance(t, torch.Tensor) and t.is_floating_point():
                    # Detach to break any GGUFQuantizedTensor wrapping; clone to
                    # own the storage the loader will cast.
                    tensors[name] = t.detach().clone() - 1.0
        # Linear-mode (byte passthrough → GgufLinear) requires MPS + the metal
        # kernels package; without either, fall back to dequant-at-load so the
        # model can still run as plain `nn.Linear` weights.
        dequant_all = (
            self.quantization_config.dequantize
            or not torch.backends.mps.is_available()
            or not metal_kernels_available()
        )
        # GGUF tensor names whose HF target is an `nn.Embedding` need fp values
        # at assignment time. `token_embd` is universal; extend as more archs land.
        embedding_prefixes = ("token_embd",)
        for name, t in list(tensors.items()):
            if not isinstance(t, GGUFQuantizedTensor):
                continue
            # Already-float sources (F32 / F16 / BF16) need no dequant — just
            # drop the `GGUFQuantizedTensor` subclass so downstream ops see a
            # plain tensor (the subclass's `__torch_function__` re-wraps on
            # `.to`, which on MPS corrupts mmap-backed storage during the
            # device cast).
            if t.is_floating_point():
                tensors[name] = t.as_subclass(torch.Tensor)
                continue
            needs_dequant = (
                dequant_all or name.split(".")[0] in embedding_prefixes or not gguf_linear_supports(t.quant_type)
            )
            if needs_dequant:
                tensors[name] = dequantize_gguf_tensor(t, t.quant_type, device=t.device)
        self.gguf_tensors = tensors
        self.gguf_kv = parsed.get("raw_kv", {}) or {}
        return tensors

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap nn.Linear / fused-expert modules in place at meta time.

        Three paths converge here:
          * safetensors reload (`module_quant_types` non-empty): swap from
            the saved plan unconditionally — the bytes are already uint8 in
            safetensors and only GgufLinear/GgufExperts have buffers shaped
            to receive them.
          * gguf_file= load: finalize `linear_mode` from MPS availability
            and the caller's `dtype=` choice. The plan comes from
            `_build_quant_info_from_gguf_file` walking `gguf_tensors`.
          * on-the-fly: swap unconditionally (the config asked for it).

        Records the swap plan on `quantization_config.module_quant_types`
        so it survives `save_pretrained` → `from_pretrained`.
        """
        from ..integrations.gguf_kernels import metal_kernels_available

        has_module_map = bool(getattr(self.quantization_config, "module_quant_types", None))
        if has_module_map:
            self.linear_mode = True
        elif not self.on_the_fly:
            # Byte-passthrough swap requires MPS + the metal kernels package.
            self.linear_mode = (
                torch.backends.mps.is_available()
                and not self.quantization_config.dequantize
                and metal_kernels_available()
            )
        if not self.linear_mode:
            return
        from ..integrations.gguf_linear import replace_with_gguf_linear

        quant_info_by_target = self._build_quant_info(model)
        self.quantization_config.module_quant_types = dict(quant_info_by_target)
        replace_with_gguf_linear(model, quant_info_by_target)

    def postprocess_model(self, model, **kwargs):
        """Run the base post-process, then apply good generation defaults when
        the GgufLinear swap is in effect. On the swap path raw uint8 source
        bytes flow straight into the GgufLinear / GgufExperts buffers via the
        standard loader — `update_weight_conversions` returns the rename /
        merge pipeline unchanged (no dequant op injected). On the dequant path
        `GGUFDequantize` heads each converter and produces fp tensors before
        any merge / permute. Kernel ops are resolved in each module's
        `__init__` (no post-load bind step)."""
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

    def is_serializable(self, safe_serialization: bool = True) -> bool:
        # save_pretrained → safetensors works: GgufLinear / GgufExperts buffers
        # are uint8, biases / norms are fp32, all native safetensors dtypes.
        # The swap layout is replayed on reload via
        # `GgufQuantizeConfig.module_quant_types` (filled in
        # `_process_model_before_weight_loading`).
        return True

    # ---- Internals ------------------------------------------------------------

    def _build_quant_info(self, model) -> dict[str, dict]:
        """Walk the live model and emit one entry per module that should swap.

        Safetensors reload path: `module_quant_types` already carries the
        full plan; return it verbatim.

        On-the-fly path (`GgufQuantizeConfig` without a source file): every
        :class:`nn.Linear` matching `modules_to_convert` gets the config's
        single `quant_type`.

        gguf_file path: per-tensor quant types come from :attr:`gguf_tensors`
        after renaming the GGUF source names through `self.weight_mapping`.
        :attr:`hf_to_gguf` is filled here as a side effect (consumed by
        `save_gguf`).
        """
        import torch.nn as nn

        from ..integrations.gguf_linear import MODEL_TYPE_TO_GGUF_EXPERTS

        if self.quantization_config.module_quant_types:
            return dict(self.quantization_config.module_quant_types)

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
        """Set `cache_implementation` + `compile_config` defaults on
        `model.generation_config` to leverage the best perf when running GGUF Models"""
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
