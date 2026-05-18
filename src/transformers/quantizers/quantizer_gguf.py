# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""GGUF quantizer.

The state-dict produced by ``load_gguf_checkpoint`` contains
:class:`GGUFQuantizedTensor` instances — ``torch.Tensor`` subclasses that
carry raw uint8 bytes plus the per-tensor ``quant_type`` metadata. They flow
through the standard loader unchanged (``.to(device)`` preserves the
subclass via ``__torch_function__``), and the :class:`GGUFDequantize` op
injected at the head of every ``WeightConverter`` produces the final
floating-point tensor — same pattern as :class:`Fp8Dequantize` for FP8.
"""

from __future__ import annotations

import torch

from .base import HfQuantizer


class GGUFQuantizer(HfQuantizer):
    """GGUF quantizer with two construction modes:

    * ``from_pretrained(..., gguf_file=...)`` (or auto-detected ``.gguf`` in the
      repo): ``modeling_utils`` builds the quantizer directly from a parsed
      ``gguf`` file and passes ``weight_mapping``/``gguf_tensors`` here. Default
      keeps weights at native quant (``linear_mode=True``) and runs Metal
      kernels via :class:`GgufLinear`. If the caller passed ``dtype=...``
      explicitly, dequantize-on-load instead.
    * ``from_pretrained(..., quantization_config=GgufQuantizeConfig(...))``:
      load an fp16/bf16 model normally, then quantize the matching
      :class:`nn.Linear` modules on the fly via ``gguf-py`` and swap them for
      :class:`GgufLinear`.
    """

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
        # ``pre_quantized`` controls whether the loader expects raw bytes from
        # disk. The ``gguf_file=`` path is pre-quantized; the on-the-fly path
        # (driven by ``GgufQuantizeConfig``) loads a normal fp16/bf16 checkpoint
        # and quantizes after weight loading.
        on_the_fly = quantization_config is not None and getattr(
            quantization_config, "quant_method", None
        ) == "gguf"
        kwargs.setdefault("pre_quantized", not on_the_fly)
        super().__init__(quantization_config=quantization_config, **kwargs)
        self.weight_mapping = list(weight_mapping or [])
        # ``linear_mode``: when True, the post-load step swaps each
        # ``nn.Linear`` that has a supported quant type for a :class:`GgufLinear`
        # running matmul/matvec on the raw bytes (kernels-community Metal kernels,
        # ~llama.cpp parity on Apple Silicon decode). On-the-fly mode is always
        # linear_mode.
        self.linear_mode = bool(linear_mode) or on_the_fly
        self.on_the_fly = on_the_fly
        # ``gguf_tensors``: ``{gguf_name: GGUFQuantizedTensor}`` map from
        # :func:`load_gguf_checkpoint`. Empty for on-the-fly mode.
        self.gguf_tensors: dict = dict(gguf_tensors or {})
        # Populated during ``_process_model_after_weight_loading`` so
        # ``integrations.gguf_save.save_pretrained_gguf`` can run the rename in
        # reverse without re-deriving the forward map.
        self.hf_to_gguf: dict[str, str] = {}
        # Raw GGUF key-value snapshot from the source file ({key: (value, gguf_value_type)}).
        # Replayed verbatim by ``save_pretrained_gguf`` so the round-trip preserves
        # block_count / head_count / tokenizer / quantization_version etc.
        self.gguf_kv: dict = dict(gguf_kv or {})

    @property
    def renaming_dequantize_op(self):
        """Op the loader attaches to freshly-constructed ``WeightRenaming``s so the
        rename path produces dequantized fp32 (not raw GGUF bytes cast to fp32)."""
        from ..gguf_conversion_ops import GGUFDequantize

        return GGUFDequantize()

    def update_weight_conversions(self, weight_conversions):
        """Prepend the GGUF→HF rename table and inject ``GGUFDequantize`` at the head of every
        ``WeightConverter`` — same pattern as ``Fp8Quantizer.update_weight_conversions``.

        ``WeightRenaming`` entries also need dequant on the rename path: ``WeightRenaming.convert``
        returns the source ``GGUFQuantizedTensor`` directly under the renamed key, and
        ``set_param_for_module`` skips the shape check when a quantizer is active, so otherwise
        ``v_proj`` / ``o_proj`` / FFN projs / RMSNorms / embeddings would land in the model as
        uint8-bytes-cast-to-float32 with the wrong (byte-) shape. Attaching ``GGUFDequantize``
        as the ``quantization_operation`` on each renaming routes them through the dequant op
        (``WeightRenaming.convert`` runs ``self.quantization_operation`` when ``hf_quantizer`` is
        active), producing the correctly-expanded fp32 tensor.
        """
        from ..core_model_loading import WeightConverter, WeightRenaming
        from ..gguf_conversion_ops import GGUFDequantize

        injected = []
        for conv in self.weight_mapping:
            if isinstance(conv, WeightConverter):
                conv = WeightConverter(
                    source_patterns=conv._original_source_patterns,
                    target_patterns=conv._original_target_patterns,
                    operations=[GGUFDequantize(), *conv.operations],
                )
            elif isinstance(conv, WeightRenaming):
                conv.quantization_operation = GGUFDequantize()
            injected.append(conv)
        return injected + list(weight_conversions)

    def param_needs_quantization(self, model, param_name, **kwargs):
        # TODO: for on the fly quantization :)
        return False

    def preprocess_model(self, model, **kwargs):
        # On-the-fly mode runs the base hook so ``is_quantized`` /
        # ``quantization_method`` get set. The ``gguf_file=`` path has no
        # ``quantization_config`` and skips the base hook (the swap happens
        # later in ``_process_model_after_weight_loading``).
        if self.on_the_fly:
            return super().preprocess_model(model, **kwargs)
        return None

    def _process_model_before_weight_loading(self, model, **kwargs):
        pass

    def postprocess_model(self, model, **kwargs):
        # On-the-fly mode goes through the base postprocess (which sets the
        # quantization config on ``model.config``). The ``gguf_file=`` path
        # has no config to record, so we just fire the after-load hook to
        # perform the ``nn.Linear`` → ``GgufLinear`` swap.
        if self.on_the_fly:
            return super().postprocess_model(model, **kwargs)
        return self._process_model_after_weight_loading(model, **kwargs)

    def _process_model_after_weight_loading(self, model, **kwargs):
        from ..core_model_loading import WeightConverter, WeightRenaming, rename_source_key

        # On-the-fly mode: model loaded in fp16/bf16; quantize matching Linear
        # weights to GGUF bytes and swap for GgufLinear.
        if self.on_the_fly:
            self._quantize_on_the_fly(model)
            return

        if not self.gguf_tensors:
            return
        renamings = [e for e in self.weight_mapping if isinstance(e, WeightRenaming)]
        converters = [e for e in self.weight_mapping if isinstance(e, WeightConverter)]
        meta_state_dict = model.state_dict()
        prefix = getattr(model, "base_model_prefix", "") or ""

        # Always compute the hf→gguf map (needed by save_pretrained_gguf even
        # when ``linear_mode=False``).
        for gguf_name in self.gguf_tensors:
            try:
                hf_name, _ = rename_source_key(
                    gguf_name,
                    renamings,
                    converters,
                    prefix=prefix,
                    meta_state_dict=meta_state_dict,
                )
            except Exception:
                continue
            self.hf_to_gguf[hf_name] = gguf_name

        if not self.linear_mode:
            return

        from ..integrations.gguf_linear import replace_with_gguf_linear

        weight_info_by_name: dict[str, dict] = {}
        for gguf_name, tensor in self.gguf_tensors.items():
            qt = getattr(tensor, "quant_type", None)
            if qt is None:
                continue
            try:
                hf_name, _ = rename_source_key(
                    gguf_name,
                    renamings,
                    converters,
                    prefix=prefix,
                    meta_state_dict=meta_state_dict,
                )
            except Exception:
                continue
            # Older GGUFs (llama-1 / llama-2 era) store attn_q / attn_k in
            # llama.cpp's head-permuted layout — applying the permute reversal
            # here is correct.  Modern exporters for Qwen2/3-MoE (and a few
            # others) instead ship Q/K already in HF layout; reverse-permuting
            # those corrupts the matmul ~140% relative.  Skip whenever the
            # rename pipeline didn't go through ``_ROPE_ATTN_CONVERTERS`` (the
            # only path that signals "this arch was permuted on export").
            #
            # NB: ``target_patterns`` may be a str or a list (multi-input fused
            # converters carry a list of targets) — normalize before scanning.
            arch_uses_rope_permute = False
            for c in converters:
                if not hasattr(c, "operations"):
                    continue
                if not any(type(op).__name__.startswith("ReversePermuteAttn") for op in c.operations):
                    continue
                tp = c.target_patterns
                targets = tp if isinstance(tp, (list, tuple)) else [tp]
                if any(t == ".self_attn.q_proj.weight" or t == ".self_attn.k_proj.weight" for t in targets):
                    arch_uses_rope_permute = True
                    break
            permute = None
            if arch_uses_rope_permute:
                if ".attn_q." in gguf_name:
                    permute = "q"
                elif ".attn_k." in gguf_name:
                    permute = "k"
            weight_info_by_name[hf_name] = {
                "quant_type": qt.name if hasattr(qt, "name") else str(qt),
                "bytes": tensor,
                "permute": permute,
            }

        replace_with_gguf_linear(model, weight_info_by_name)
        self._swap_moe_experts(model, renamings, converters, meta_state_dict, prefix)

    def _quantize_on_the_fly(self, model):
        """On-the-fly quantize path driven by :class:`GgufQuantizeConfig`.

        Walk the model, quantize each :class:`nn.Linear` whose name matches
        ``modules_to_convert`` (glob), and swap for :class:`GgufLinear`. No
        rename table / no GGUF file involved — bytes come straight out of
        ``gguf.quants.quantize`` applied to the live fp16/bf16 weight.
        """
        import fnmatch

        import gguf
        import torch.nn as nn

        from ..integrations.gguf_linear import replace_with_gguf_linear

        cfg = self.quantization_config
        quant_type = cfg.quant_type  # validated to Q4_0 / Q8_0 in GgufQuantizeConfig.post_init
        ggml_type = getattr(gguf.GGMLQuantizationType, quant_type)
        include = cfg.modules_to_convert
        exclude = set(cfg.modules_to_not_convert or [])

        def _matches(name: str) -> bool:
            if include is None:
                return True
            return any(fnmatch.fnmatchcase(name, pat) for pat in include)

        weight_info_by_name: dict[str, dict] = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not _matches(name) or any(skip in name for skip in exclude):
                continue
            fp = mod.weight.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
            packed = gguf.quants.quantize(fp, ggml_type)
            weight_info_by_name[f"{name}.weight"] = {
                "quant_type": quant_type,
                "bytes": torch.from_numpy(packed.view("uint8").reshape(-1)),
                "permute": None,
            }

        replace_with_gguf_linear(model, weight_info_by_name)

    def save_gguf(self, model, path: str, *, quant_config: dict | None = None) -> str:
        """Write the model back to a .gguf file.

        Delegate to :func:`transformers.integrations.gguf_save.save_pretrained_gguf`,
        which uses ``self.hf_to_gguf`` (populated at load time) for the reverse
        rename. Pass ``quant_config`` to override per-tensor quant types — see
        the function docstring for the policy DSL.
        """
        from ..integrations.gguf_save import save_pretrained_gguf

        return save_pretrained_gguf(model, path, quant_config=quant_config, quantizer=self)

    def _swap_moe_experts(self, model, renamings, converters, meta_state_dict, prefix):
        """Detect ``Qwen2MoeExperts``-style fused-expert modules and swap them
        for the quantized :class:`GgufQwen2MoeExperts` so the expert weights
        (typically >90% of an MoE model's params) stay native-quantized.

        Matches gate / up / down expert tensors by their GGUF source names
        (``blk.{N}.ffn_{gate,up,down}_exps.weight``) and groups them by the HF
        parent path the dequant rename pipeline would have produced for *any*
        of them (the three share a single ``...mlp.experts`` parent).
        """
        from ..core_model_loading import rename_source_key
        from ..integrations.gguf_linear import replace_qwen2_moe_experts

        # Group the three expert source tensors per layer. Each projection gets
        # its own quant_type because Q4_K_M (and friends) mix quant types per
        # tensor — e.g. gate/up = Q4_K but down = Q8_0.
        kind_to_keys = {
            "ffn_gate_exps": ("gate_bytes", "gate_quant"),
            "ffn_up_exps": ("up_bytes", "up_quant"),
            "ffn_down_exps": ("down_bytes", "down_quant"),
        }
        groups: dict[str, dict] = {}
        for gguf_name, tensor in self.gguf_tensors.items():
            for kind, (bytes_key, quant_key) in kind_to_keys.items():
                if f".{kind}." not in gguf_name:
                    continue
                try:
                    hf_name, _ = rename_source_key(
                        gguf_name,
                        renamings,
                        converters,
                        prefix=prefix,
                        meta_state_dict=meta_state_dict,
                    )
                except Exception:
                    break
                parent_path = hf_name.rsplit(".", 1)[0]  # drop ``.weight`` suffix
                groups.setdefault(parent_path, {})
                groups[parent_path][bytes_key] = tensor
                groups[parent_path][quant_key] = (
                    tensor.quant_type.name if hasattr(tensor.quant_type, "name") else str(tensor.quant_type)
                )
                break

        complete = {
            p: info
            for p, info in groups.items()
            if all(k in info for k in ("gate_bytes", "up_bytes", "down_bytes", "gate_quant", "up_quant", "down_quant"))
        }
        if complete:
            replace_qwen2_moe_experts(model, complete)

    @property
    def is_trainable(self):
        return True

    def is_serializable(self):
        return False
