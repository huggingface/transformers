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
from typing import TYPE_CHECKING

from .base import HfQuantizer


# `quantizers/__init__.py` is imported even from torch-less installs (PIL-only
# CI does this). Keep the top-level import set torch-free; the methods that
# actually need torch import it lazily.
if TYPE_CHECKING:
    import torch  # noqa: F401


logger = logging.getLogger(__name__)


class GGUFQuantizer(HfQuantizer):
    """GGUF quantizer."""

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        """Distinguish the three sources a GGUF-quantized model can come from and
        set ``pre_quantized`` accordingly. The source is inferred from the config,
        not passed explicitly:

        * **safetensors reload** — the config carries ``module_quant_types``, the
          per-module swap plan recorded by a prior ``save_pretrained``. The
          quantized bytes already live in the model's own safetensors shards, so
          no ``.gguf`` file is involved; we only need to replay the swap. This is a
          pre-quantized load (``pre_quantized=True``).
        * **external gguf file** — the config has ``gguf_file=``. The source bytes
          come from that ``.gguf`` and are read by :meth:`load_checkpoint_state`.
          Also pre-quantized (``pre_quantized=True``).
        * **on-the-fly quantization** — neither of the above. The checkpoint is a
          regular fp16/bf16 model and the :class:`GGUFQuantize` conversion op packs
          each matching weight into GGUF bytes at load time, so the model is *not*
          pre-quantized (``pre_quantized=False``).

        The first two are mutually exclusive with the third; ``on_the_fly`` is
        simply "neither a saved swap plan nor a gguf file was given".
        """
        from ..utils.quantization_config import GgufQuantizeConfig

        if quantization_config is None:
            quantization_config = GgufQuantizeConfig()
        has_module_map = bool(getattr(quantization_config, "module_quant_types", None))
        has_gguf_file = getattr(quantization_config, "gguf_file", None) is not None
        on_the_fly = not (has_module_map or has_gguf_file)
        kwargs.setdefault("pre_quantized", not on_the_fly)
        super().__init__(quantization_config=quantization_config, **kwargs)
        # Re-annotate with the concrete type so attribute accesses type-check.
        self.quantization_config: GgufQuantizeConfig = quantization_config
        self.on_the_fly = on_the_fly
        # `keep_quantized`: True ⇒ the model keeps GGUF byte modules (GgufLinear /
        # GgufExperts, the Metal-kernel path); False ⇒ weights are dequantized to
        # plain float `nn.Linear`. Resolved for real in
        # `_process_model_before_weight_loading` via `_resolve_keep_quantized`; this
        # is just the default for the on-the-fly case.
        self.keep_quantized = on_the_fly
        # GGUF file state — populated by :meth:`load_checkpoint_state` when the
        # gguf_file path is active. Empty for the on-the-fly path.
        self.weight_mapping: list = []
        self.gguf_tensors: dict = {}
        # `{gguf_name: ggml_quant_type}` metadata read off the GGUF header (no
        # tensor data). Drives the coarse all-supported decision below.
        self.gguf_tensor_types: dict = {}
        # Whether every quant type in the source file has a metal kernel (set in
        # `load_checkpoint_state`). Gates the byte/swap path vs full dequant.
        self._all_supported: bool = True

    # ---- HfQuantizer hooks ----------------------------------------------------

    def validate_environment(self, *args, **kwargs):
        """Disk offload writes dequantized fp shards to disk and reloads them as
        plain tensors — incompatible with the GGUF byte-passthrough / kernel path.
        Reject a `device_map` that sends any module to ``"disk"``."""
        device_map = kwargs.get("device_map")
        if device_map is not None and (
            (isinstance(device_map, dict) and "disk" in device_map.values()) or "disk" in device_map
        ):
            raise RuntimeError(
                "One or more modules is configured to be mapped to disk. Disk offload is not supported for models "
                "loaded from GGUF files."
            )

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
        import torch

        from ..integrations.gguf_dequant import GGUFQuantizedTensor, dequantize_gguf_tensor
        from ..integrations.gguf_linear import _quant_type_name, gguf_linear_supports
        from ..modeling_gguf_pytorch_utils import load_gguf_checkpoint

        parsed = load_gguf_checkpoint(gguf_path, return_tensors=True)
        self.weight_mapping = list(parsed.get("weight_mapping", []) or [])
        self.gguf_tensor_types = dict(parsed.get("tensor_quant_types", {}) or {})
        tensors = parsed["tensors"]
        # GGUF tensor names whose HF target is an `nn.Embedding` need fp values at
        # assignment time. `token_embd` is universal; extend as more archs land.
        embedding_prefixes = ("token_embd",)
        _FLOAT_TYPES = {"F32", "F16", "BF16", "F64"}
        # Coarse, metadata-only decision: can the whole model take the byte/kernel
        # path? Only if every quantized (non-float, non-embedding) tensor has a
        # metal kernel — checked off the header types, no rename, no tensor load.
        # If any unsupported quant is present (e.g. Q2_K, IQ1_S), the whole model
        # dequantizes; otherwise we keep bytes and swap uniformly. (Per-module
        # support is unnecessary: a uniform swap reverts float-source Linears and
        # the kernels cover everything else.)
        self._all_supported = all(
            gguf_linear_supports(qt)
            for name, qt in self.gguf_tensor_types.items()
            if name.split(".")[0] not in embedding_prefixes and _quant_type_name(qt) not in _FLOAT_TYPES
        )
        # The Gemma/Nemotron `w + 1` norm de-offset is handled by the `SubtractOne`
        # converter in fp32 — `_mark_offset_norms_fp32` keeps those norms fp32.
        dequant_all = (
            self.quantization_config.dequantize or not torch.backends.mps.is_available() or not self._all_supported
        )
        for name, t in list(tensors.items()):
            if not isinstance(t, GGUFQuantizedTensor):
                continue
            # Already-float sources (F32 / F16 / BF16): drop the `GGUFQuantizedTensor`
            # subclass so downstream ops see a plain tensor (its `__torch_function__`
            # re-wraps on `.to`, which on MPS corrupts mmap-backed storage during a
            # device cast). These flow into the swap as-is and get reverted to a
            # plain `nn.Linear` post-load if their target was swapped.
            if t.is_floating_point():
                tensors[name] = t.as_subclass(torch.Tensor)
                continue
            # Quantized: dequantize embeddings (never swapped) and everything on the
            # dequant path; otherwise keep the raw bytes for the GgufLinear swap.
            if dequant_all or name.split(".")[0] in embedding_prefixes:
                tensors[name] = dequantize_gguf_tensor(t, t.quant_type, device=t.device)
        self.gguf_tensors = tensors
        return tensors

    def _resolve_keep_quantized(self) -> bool:
        """Decide whether the model keeps GGUF byte modules (GgufLinear /
        GgufExperts, the Metal-kernel path) or is dequantized to plain float
        `nn.Linear`. Single source of truth for the three load paths:

          * safetensors reload (`module_quant_types` present) → always keep:
            the saved uint8 bytes only fit the swapped buffers.
          * on-the-fly (`GgufQuantizeConfig` without a file) → always keep:
            the config explicitly asked to quantize.
          * gguf_file= → keep only on MPS, without an explicit `dequantize`, and
            only if every quant type in the file has a kernel (`_all_supported`).
        """
        import torch

        if bool(getattr(self.quantization_config, "module_quant_types", None)):
            return True
        if self.on_the_fly:
            return True
        return (
            torch.backends.mps.is_available() and not self.quantization_config.dequantize and self._all_supported
        )

    def _mark_offset_norms_fp32(self, model) -> None:
        """Gemma/Nemotron store RMSNorm weights as `w + 1`; the `SubtractOne`
        converter de-offsets them, but only correctly in fp32 — the loader casts
        to the target dtype *before* the converter chain runs. Force exactly those
        norms to stay fp32 for the load through `_keep_in_fp32_modules_strict`
        (which covers both fp16 and bf16). The set of norms is the targets of the
        `SubtractOne` converters in `weight_mapping` — no hardcoded arch list, so a
        new arch works as soon as its converter graph declares the offset. Runs on
        every load path (the dequant path needs it too). Must run before
        `_get_dtype_plan` reads the flag — `preprocess_model` is called first.
        """
        from ..core_model_loading import WeightConverter
        from ..gguf_conversion_ops import SubtractOne

        offset_targets = {
            pat
            for entry in self.weight_mapping
            if isinstance(entry, WeightConverter) and any(isinstance(op, SubtractOne) for op in entry.operations)
            for pat in entry.target_patterns
        }
        if offset_targets:
            model._keep_in_fp32_modules_strict = set(model._keep_in_fp32_modules_strict or []) | offset_targets

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap nn.Linear / fused-expert modules to GgufLinear / GgufExperts at
        meta time, when `_resolve_keep_quantized` selects the byte path.

        The swap is **uniform** — every Linear becomes a placeholder, with no
        source→target rename and no per-module quant type (those are bound in
        `postprocess_model` from the bytes that arrive). The on-the-fly path is the
        one exception: its quant type is a single config value, known up front.
        """
        self._mark_offset_norms_fp32(model)
        self.keep_quantized = self._resolve_keep_quantized()
        if not self.keep_quantized:
            return
        from ..integrations.gguf_linear import replace_with_gguf_linear

        cfg = self.quantization_config
        skip = set(cfg.modules_to_not_convert or [])
        # Tied output embeddings (e.g. lm_head on tie_word_embeddings models) have
        # no standalone quantized tensor in the GGUF — they reuse `token_embd`, which
        # is dequantized. Keep them as plain modules so the tie lands a float weight,
        # rather than swapping them to an unfilled GgufLinear.
        tied = getattr(model, "all_tied_weights_keys", None) or {}
        skip |= {k.removesuffix(".weight") for k in (set(tied) | set(tied.values()))}
        if self.on_the_fly:
            # Live fp model: uniform config quant type, filtered to modules_to_convert;
            # `GGUFQuantize` packs the weights at load, so bind kernels at meta.
            replace_with_gguf_linear(
                model, modules_to_not_convert=skip, modules_to_convert=cfg.modules_to_convert, quant_type=cfg.quant_type
            )
        else:
            # gguf_file / safetensors reload: load the raw bytes/floats without dtype
            # coercion (so a uint8 buffer doesn't truncate a float router), then bind
            # or revert per module in `postprocess_model`.
            self.pre_quantized = True
            replace_with_gguf_linear(model, modules_to_not_convert=skip)

    def postprocess_model(self, model, **kwargs):
        """Run the base post-process, then — on the swap path — reconcile each
        swapped module against the bytes that loaded (bind kernels / revert float
        sources) and apply the generation defaults."""
        super().postprocess_model(model, **kwargs)
        if self.keep_quantized:
            self._reconcile_swapped_modules(model)
            self._apply_generation_defaults(model)
        return model

    @property
    def is_trainable(self):
        return True

    def is_serializable(self, safe_serialization: bool = True) -> bool:
        # save_pretrained → safetensors works: GgufLinear / GgufExperts buffers
        # are uint8, biases / norms are fp32, all native safetensors dtypes.
        # The swap layout is replayed on reload via
        # `GgufQuantizeConfig.module_quant_types`, recorded in `postprocess_model`.
        return True

    # ---- Internals ------------------------------------------------------------

    def _reconcile_swapped_modules(self, model) -> None:
        """Post-load reconciliation for the uniform swap. For every placeholder:

        * bind the metal kernels from the quant type that arrived with the bytes
          (a `GGUFQuantizedTensor` carries it; the gate/up merge keeps it via
          `torch.cat`), or from the saved `module_quant_types` on a safetensors
          reload (where the bytes are plain uint8 with no carried type);
        * revert any `GgufLinear` whose GGUF source was actually float (e.g. MoE
          routers, stored F32) back to a plain `nn.Linear`.

        Records the resolved per-module quant types on the config so a
        `save_pretrained` round-trip can rebind on reload. This replaces the old
        rename-based `_build_quant_info` — the swap plan now falls out of the
        bytes, not a source→target name map.
        """
        from ..integrations.gguf_linear import GgufExperts, GgufLinear, _quant_type_name, gguf_linear_supports

        saved = dict(getattr(self.quantization_config, "module_quant_types", None) or {})
        recorded: dict = {}
        for name, mod in list(model.named_modules()):
            if isinstance(mod, GgufLinear):
                quant_type = mod.quant_type or saved.get(name) or _quant_type_name(getattr(mod.weight, "quant_type", None))
                if gguf_linear_supports(quant_type):
                    if mod._mv_op is None:
                        mod.bind_after_load(quant_type)
                    recorded[name] = mod.quant_type
                else:
                    # Float source (router) or unsupported → plain nn.Linear.
                    self._revert_to_linear(model, name, mod)
            elif isinstance(mod, GgufExperts):
                if mod._id_op_gate_up is None:
                    info = saved.get(name) or {}
                    mod.bind_after_load(info.get("gate_up_quant"), info.get("down_quant"))
                recorded[name] = {"gate_up_quant": mod.gate_up_quant, "down_quant": mod.down_quant}
        self.quantization_config.module_quant_types = recorded
        # The swapped model is now a standard HF module tree holding uint8 buffers;
        # its "original format" is GGUF, which can't round-trip through safetensors
        # (a plain reload has no rename map). Drop the GGUF load-renames so
        # `save_pretrained` writes HF names — reload re-swaps uniformly and rebinds
        # from `module_quant_types`.
        model._weight_conversions = None

    @staticmethod
    def _revert_to_linear(model, name: str, mod) -> None:
        """Swap a uniformly-created `GgufLinear` whose source turned out to be float
        back to a plain `nn.Linear` holding the loaded float weight."""
        import torch
        import torch.nn as nn

        weight = mod.weight
        new = nn.Linear(
            mod.in_features, mod.out_features, bias=mod.bias is not None, device=weight.device, dtype=weight.dtype
        )
        with torch.no_grad():
            new.weight.copy_(weight.reshape(new.weight.shape))
            if mod.bias is not None:
                new.bias.copy_(mod.bias)
        new.weight.requires_grad_(False)
        if new.bias is not None:
            new.bias.requires_grad_(False)
        model.set_submodule(name, new)

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
