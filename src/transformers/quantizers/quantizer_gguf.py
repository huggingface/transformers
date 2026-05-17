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

from .base import HfQuantizer


class GGUFQuantizer(HfQuantizer):
    """Quantizer for GGUF checkpoints — carries the rename table and injects ``GGUFDequantize``."""

    requires_calibration = False

    def __init__(self, weight_mapping=None, linear_mode=False, gguf_tensors=None, gguf_kv=None, **kwargs):
        # ``pre_quantized=True`` so the loader keeps `_dtype=None` (no uint8→float
        # cast in ``spawn_materialize`` — the GGUFDequantize op handles dtype).
        kwargs.setdefault("pre_quantized", True)
        super().__init__(quantization_config=None, **kwargs)
        self.weight_mapping = list(weight_mapping or [])
        # When ``linear_mode=True``, ``_process_model_after_weight_loading`` walks
        # the model and swaps each ``nn.Linear`` whose source GGUF tensor has a
        # supported quant type for a :class:`GgufLinear` that runs matmul/matvec
        # directly on the quantized bytes via the kernels-community Metal kernels
        # (same kernels as llama.cpp; ~llama.cpp parity at decode on Apple Silicon).
        #
        # ``gguf_tensors`` is the ``{gguf_name: GGUFQuantizedTensor}`` map from
        # :func:`load_gguf_checkpoint`. We hold it here so we can recover each
        # original tensor's quant type after the model has been loaded with
        # dequantized weights, then re-quantize per-Linear in Phase 2.
        self.linear_mode = linear_mode
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
        # No module swapping needed; skip the base class logic that tries to
        # set is_quantized / quantization_method on the model.
        pass

    def _process_model_before_weight_loading(self, model, **kwargs):
        pass

    def postprocess_model(self, model, **kwargs):
        # GGUF loading does not set any quantization config on the model — but
        # we still need to fire ``_process_model_after_weight_loading`` so the
        # opt-in ``linear_mode`` swap (nn.Linear → GgufLinear) happens.
        return self._process_model_after_weight_loading(model, **kwargs)

    def _process_model_after_weight_loading(self, model, **kwargs):
        from ..core_model_loading import WeightConverter, WeightRenaming, rename_source_key

        if not self.gguf_tensors:
            return
        renamings = [e for e in self.weight_mapping if isinstance(e, WeightRenaming)]
        converters = [e for e in self.weight_mapping if isinstance(e, WeightConverter)]
        meta_state_dict = model.state_dict()
        prefix = getattr(model, "base_model_prefix", "") or ""

        # Always compute the hf→gguf map (needed by save_pretrained_gguf even
        # when ``linear_mode=False``).
        for gguf_name, _tensor in self.gguf_tensors.items():
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
