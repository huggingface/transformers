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
"""The `HfQuantizer` for the fine-grained family: converter chains, the parallel plan, and the
environment checks. The modules are in `integrations/finegrained`.

Two axes, deliberately separate. A per-scheme arm subclasses this one for what a PRODUCER's key
layout needs — modelopt's two-level scales, GPT-OSS's packed `_blocks` — and the registry picks
one by `quant_method`. What FORMAT each module is quantized in comes from the config's groups
instead (`group_for`), because a checkpoint can be several: DeepSeek-V4 is mxfp4 experts over
block-FP8 linears, and one arm per checkpoint could never express that.
"""

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from ...utils import (
    is_accelerate_available,
    is_torch_available,
    is_torch_distributed_available,
    is_torch_xpu_available,
    logging,
)
from ...utils.quantization_config import groups_with_expert_dtype
from ..base import HfQuantizer
from ..quantizers_utils import get_module_from_name, should_convert_module


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ...modeling_utils import PreTrainedModel
    from ...utils.quantization_config import FineGrainedConfig

logger = logging.get_logger(__name__)


class FineGrainedHfQuantizer(HfQuantizer):
    """Quantizer for the fine-grained family the `kernels-community/finegrained-kernels` package
    serves: block-FP8 (fp32 or UE8M0 scales), MXFP8, MXFP4 and NVFP4, for dense linears, embeddings
    and MoE experts alike.

    It never declares the weight format — that is resolved from the checkpoint tensors themselves,
    the way the kernels resolve it. What it does own is the vocabulary each producer ships its
    scales under (modelopt's `weight_scale_2` / `input_scale`, GPT-OSS's `_blocks` / `_scales`) and
    the conversion ops that bring them to the layout `FineGrainedExperts` holds.
    """

    requires_calibration = False
    quantization_config: "FineGrainedConfig"
    default_activation_format: str | None = None
    quantization_param_suffixes = ("_scale_inv",)

    def param_keeps_checkpoint_dtype(self, param_name: str) -> bool:
        # a scale loads as the checkpoint ships it (Qwen3 and Mistral ship BF16, DeepSeek-V3 fp32): the
        # kernels read either, and a save then writes it back unchanged
        return param_name.endswith(("_scale_inv", "activation_scale", "_global_scale"))

    @property
    def supports_dequantize(self) -> bool:
        """Whether `dequantize=True` can fold this format's scales into a full-precision weight.
        Every one-level format can; the NVFP4 arm says no."""
        return True

    def _assert_dequantize_supported(self) -> None:
        if not self.supports_dequantize:
            raise NotImplementedError(
                f"`dequantize=True` is not supported for {self.quant_method} checkpoints. Load them "
                "quantized on a GPU that serves the format, or start from a bf16 checkpoint."
            )

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading a fine-grained quantized model requires accelerate (`pip install accelerate`)")

        if self.quantization_config.dequantize:
            self._assert_dequantize_supported()
            return

        if not torch.cuda.is_available() and not is_torch_xpu_available():
            if self.pre_quantized:
                self._assert_dequantize_supported()
                logger.warning_once(
                    "Fine-grained quantized models require a GPU or XPU; dequantizing to bf16 since neither is "
                    "available."
                )
                self.quantization_config.dequantize = True
                return
            else:
                raise RuntimeError("No GPU or XPU found. A GPU or XPU is needed for fine-grained quantization.")

        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            major, minor = compute_capability
            if (major < 8) or (major == 8 and minor < 9):
                self._assert_dequantize_supported()
                logger.warning_once(
                    "Fine-grained quantized models are only supported on GPUs with compute capability >= 8.9 "
                    f"(e.g 4090/H100), actual = `{major}.{minor}`. We will default to dequantizing the model to "
                    "bf16. Feel free to use a different quantization method like bitsandbytes or torchao"
                )
                self.quantization_config.dequantize = True
                return

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded a fine-grained quantized model on CPU and have a CUDA or XPU device available, "
                "make sure to set your model on a GPU or XPU device in order to run your model. To remove this "
                "warning, pass device_map = 'cuda' or 'xpu'."
            )
        elif isinstance(device_map, dict):
            if (
                not self.pre_quantized
                and len(device_map) > 1
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                raise ValueError(
                    "You are attempting to load a fine-grained quantized model with a device_map that contains "
                    "a cpu/disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the cpu/disk device from the device_map."
                )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        # `FineGrainedGroupedLinear` subclasses `FineGrainedLinear`, so the tuple covers it implicitly.
        from ...integrations.finegrained import FineGrainedExperts, FineGrainedLinear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, (FineGrainedLinear, FineGrainedExperts)):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        return False

    def param_element_size(self, model: "PreTrainedModel", param_name: str, param: "torch.Tensor") -> float:
        "Return the element size (in bytes) for `param_name`."
        if self.param_needs_quantization(model, param_name):
            # 8 bit, this is needed as when `pre_quantized`` is False, we don't set the dtype of the FineGrainedLinear in order to correctly load the weights
            return 1
        return super().param_element_size(model, param_name, param)

    def _normalize_modules_to_not_convert(self, model: "PreTrainedModel"):
        """Extend the skip-list, which names modules in the checkpoint's namespace, with the model's
        modules it covers there. A released model renames its modules on load, so each one is named back
        through the reversed renames (written on keys, so on the module's key prefix) and taken with its
        sub-tree when the list covers it: a VLM's bare `vision_tower` matches no rename on its own."""
        skip = self.quantization_config.modules_to_not_convert
        if not skip:
            return

        from ...conversion_mapping import get_model_conversion_mapping
        from ...core_model_loading import WeightRenaming

        renamings = get_model_conversion_mapping(model)
        reverse = [r.reverse_transform() for r in renamings[::-1] if isinstance(r, WeightRenaming)]
        covered = []
        for module_name, _ in model.named_modules():
            if any(module_name.startswith(f"{parent}.") for parent in covered):
                continue
            checkpoint_name = f"{module_name}."
            for rename in reverse:
                checkpoint_name, _ = rename.rename_source_key(checkpoint_name)
            checkpoint_name = checkpoint_name.removesuffix(".")
            if checkpoint_name != module_name and not should_convert_module(checkpoint_name, skip):
                covered.append(module_name)
        self.quantization_config.modules_to_not_convert = [*skip, *covered]

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from ...integrations.finegrained import replace_with_finegrained_embedding, replace_with_finegrained_layer

        self._normalize_modules_to_not_convert(model)
        # the one place both configs are in hand: a legacy checkpoint declares a second format
        # for its experts on the MODEL config, and it becomes a group here
        expert_dtype = getattr(model.config.get_text_config(), "expert_dtype", None)
        self.quantization_config.groups = groups_with_expert_dtype(self.quantization_config.groups, expert_dtype)
        if self.quantization_config.activation_format is None:
            self.quantization_config.activation_format = self.default_activation_format
        # the modules read their group's format, so the default lands on this arm's groups too
        self.quantization_config.groups = {
            name: replace(group, activation_format=self.default_activation_format)
            if group.activation_format is None and group.quant_method == self.quant_method
            else group
            for name, group in self.quantization_config.groups.items()
        }
        if self.quantization_config.activation_format == "bf16":
            # Weight-only holds no activation global, so a calibrated checkpoint's `input_scale`
            # has no slot to load into. Not an unexpected key — one this run has no use for, which
            # a later W4A4 run of the same checkpoint reads.
            model._keys_to_ignore_on_load_unexpected = set(model._keys_to_ignore_on_load_unexpected or []) | {
                r"input_scale$"
            }
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        if self.pre_quantized and self.quantization_config.modules_to_convert:
            if self.quant_method != "fp8":
                logger.warning_once(
                    f"Embedding tables are quantized to FP8; {self.quant_method} has no embedding path. "
                    f"{self.quantization_config.modules_to_convert} will hold FP8 rows with one per-tensor scale."
                )
            replace_with_finegrained_embedding(
                model, self.quantization_config.modules_to_convert, self.modules_to_not_convert
            )
        replace_with_finegrained_layer(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )

    def _process_model_after_weight_loading(self, model, **kwargs):
        from ...integrations.finegrained import assert_modules_are_quantized, disable_deepgemm_on_multi_device

        assert_modules_are_quantized(model)
        disable_deepgemm_on_multi_device(model)

        return model

    def update_tp_plan(self, config):
        # Per-impl rewrite of the experts parallel-layer kind. Applied LAST so it composes on
        # top of any plan written above (e.g. the Qwen3 dense plan). Models carry the experts
        # mapping under `base_model_tp_plan` and/or `base_model_ep_plan` — rewrite both.
        from ...integrations.finegrained import FineGrainedExperts

        impl = getattr(config, "_experts_implementation", None)
        layer_overrides = FineGrainedExperts._impl_tp_layer_overrides.get(impl, {})
        sub_configs = [c for name in getattr(type(config), "sub_configs", {}) if (c := getattr(config, name, None))]
        for plan_owner, plan_attr in (
            (owner, attr) for owner in (config, *sub_configs) for attr in ("base_model_tp_plan", "base_model_ep_plan")
        ):
            base_plan = getattr(plan_owner, plan_attr, None) or {}
            updated_plan = {k: layer_overrides.get(v, v) for k, v in base_plan.items()}

            # every companion beside a projection weight is expert-indexed too, so each needs its
            # own entry: the matcher keys on the parameter name and the module's entry shards
            # nothing, which would leave this rank's weight slice beside every rank's scales
            for key, style in list(updated_plan.items()):
                # only the experts' own projections: a dense `self_attn.q_proj` is `colwise` too
                projection = key.rpartition(".")[2]
                if projection not in ("gate_up_proj", "up_proj", "down_proj"):
                    continue
                if style == "grouped_gemm":
                    companions = ["_scale_inv", "_bias", "_weight_global_scale", "_activation_scale"]
                    if projection == "down_proj":
                        companions.append("_input_global_scale")
                    for suffix in companions:
                        updated_plan.setdefault(f"{key}{suffix}", style)
                    continue

                # ...and only a STACKED experts module: a dense `mlp.shared_experts.up_proj` ends
                # in the same word but colwise/rowwise already shards it correctly
                if updated_plan.get(key.rpartition(".")[0]) not in ("moe_tp_experts", "megamoe_experts"):
                    continue
                if not is_torch_distributed_available():
                    continue  # the styles below need a distributed build; without one, no TP

                # Intra-expert TP: the experts stay whole and the projection's own axis splits,
                # so only the scale GRID follows its weight — the per-expert globals, activation
                # scales and the down bias stay replicated, as does a per-tensor `(E, 1, 1)`.
                blocked = self.quantization_config.weight_block_size is not None
                if projection == "down_proj" and style in ("rowwise", "packed_rowwise"):
                    if blocked:
                        updated_plan.setdefault(f"{key}_scale_inv", "moe_experts_rowwise")
                elif projection != "down_proj" and style in ("packed_colwise", "colwise"):
                    # the CHECKPOINT's split, since the interleave into the kernels' row order
                    # runs on each rank's shard afterwards
                    rows = "moe_experts_packed_colwise" if style == "packed_colwise" else "moe_experts_colwise"
                    if blocked:
                        updated_plan.setdefault(f"{key}_scale_inv", rows)
                    updated_plan.setdefault(f"{key}_bias", rows)

            if updated_plan != base_plan:
                setattr(plan_owner, plan_attr, updated_plan)

        return config

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def get_quantize_ops(self):
        from ...integrations.finegrained.conversions import FineGrainedQuantize

        return FineGrainedQuantize(self)

    @property
    def quant_method(self) -> str:
        method = self.quantization_config.quant_method
        return getattr(method, "value", method)  # a QuantizationMethod enum, or already its str

    def get_weight_conversions(self):
        """The converters this CHECKPOINT's own key layout needs, on top of the model's plan."""
        if not self.pre_quantized:
            return []
        if self.quantization_config.dequantize:
            return self._dequantize_conversions()
        return []

    def _dequantize_conversions(self):
        """Every quantized key folded back to a full-precision weight, from the
        `weight` + `weight_scale_inv` pair. The MXFP4 arm overrides this for GPT-OSS's
        `{proj}_blocks` + `{proj}_scales`."""
        from ...core_model_loading import WeightConverter
        from ...integrations.finegrained.conversions import FineGrainedDequantize

        # anchored `weight$` so the scale keys land in the scale slots; the activation
        # scales are collected only to be dropped
        return [
            WeightConverter(
                source_patterns=["weight$", "weight_scale_inv", "activation_scale"],
                target_patterns="weight",
                operations=[FineGrainedDequantize(self)],
            )
        ]

    def update_weight_conversions(self, weight_conversions):
        """Rewrite the model's conversion plan for this checkpoint:

        - every mode: ``*.scale`` -> ``*.weight_scale_inv`` (DeepSeek-V4-Flash ships per-block
          scales under ``.scale``; kept here rather than in each model's mapping so non-FP8
          round-trips never see the rule);
        - ``dequantize=True``: a :class:`FineGrainedDequantize` runs first in every converter with a
          ``.weight`` source (anchored, with the sibling ``.weight_scale_inv`` collected alongside),
          so per-expert (weight, scale) pairs fold into full-precision tensors before merge/concat
          ops collapse the per-expert structure;
        - every mode: sharded ``.weight`` converter targets are anchored, since they become source
          patterns on save and would otherwise match their scale keys;
        - modelopt: the arch plan's ``.weight`` sources are anchored so they don't swallow the
          ``weight_scale`` / ``weight_scale_2`` keys, and the packed weights get the uint8 -> int8 view;
        - otherwise the expert converters get the module layout ops (``_with_expert_layout_ops``).

        :meth:`get_weight_conversions` is appended in every mode."""
        from ...core_model_loading import WeightConverter, WeightRenaming
        from ...integrations.finegrained.conversions import FineGrainedDequantize

        scale_rename = WeightRenaming(source_patterns=r"^(.+)\.scale$", target_patterns=r"\1.weight_scale_inv")
        weight_conversions = [scale_rename, *weight_conversions]

        def anchor(conversions):
            """A converter's targets become its source patterns on SAVE, and a parameter name is a
            prefix of its own companions' — `experts.gate_up_proj` matches
            `experts.gate_up_proj_scale_inv` and `..._input_global_scale` too. Unanchored, the
            weight converter claims them on the way out and tries to split a one-value global into
            gate|up halves. Anchor every target that names a parameter outright: a `.weight`
            (Qwen4-Exp's `ngram_embedding`) or an expert projection. Runs LAST, since
            `_with_expert_layout_ops` rebuilds converters and would drop it."""
            for conv in conversions:
                if isinstance(conv, WeightConverter):
                    conv._original_target_patterns = [
                        f"{p}$" if p.endswith((".weight", "gate_up_proj", "up_proj", "down_proj")) else p
                        for p in conv._original_target_patterns
                    ]
            return conversions

        if self.pre_quantized and self.quantization_config.dequantize:
            updated = []
            for conv in weight_conversions:
                # collect the sibling `.weight_scale_inv` alongside each `.weight` so the pair folds
                # to full precision BEFORE merge/concat ops collapse the per-expert structure
                weights = [p for p in conv.source_patterns if p.endswith(".weight")]
                if isinstance(conv, WeightConverter) and weights:
                    conv = WeightConverter(
                        source_patterns=[p + "$" for p in weights]
                        + [p.removesuffix(".weight") + ".weight_scale_inv$" for p in weights]
                        + [p for p in conv.source_patterns if not p.endswith(".weight")],
                        target_patterns=conv._original_target_patterns,
                        operations=[FineGrainedDequantize(self), *conv.operations],
                    )
                updated.append(conv)
            return anchor(updated + self.get_weight_conversions())

        return anchor(self._with_expert_layout_ops(weight_conversions + self.get_weight_conversions()))

    def _with_expert_layout_ops(self, weight_conversions):
        """Give every converter that produces expert tensors the layout ``FineGrainedExperts`` holds
        (see its ``__init__``), in chain order: gate|up rows interleaved (weight, scale grid and bias
        share the row axis), the scale in the held container dtype, then the swizzled scale layout.
        Each op decides from the model it runs against and each has a reverse, so saving restores
        the checkpoint layout. Catch-all converters at the end take keys that arrive already under
        the fused names (a plain rename, a transformers-format checkpoint) and dense linears' scale
        keys, so they get the same treatment."""
        from ...core_model_loading import WeightConverter
        from ...integrations.finegrained.conversions import (
            FineGrainedInterleaveGateUp,
            FineGrainedScaleContainer,
            FineGrainedSwizzleScales,
        )

        def layout_ops(target: str) -> list:
            ops = []
            if "gate_up_proj" in target:
                ops.append(FineGrainedInterleaveGateUp(self))
            if "_bias" not in target:
                ops += [
                    FineGrainedScaleContainer(self),
                    FineGrainedSwizzleScales(self),
                ]  # the swizzle reads 1-byte scales
            return ops

        def sources_the_fused_key(conv, fused: str) -> bool:
            """Whether `conv` can SOURCE the fused transformers-format key, not merely target it.

            modelopt-style sources (`...gate_proj.weight_scale`) never match an already-fused
            `experts.gate_up_proj_scale_inv`, so keying on the target would let them suppress the
            catch-all below: the scale then arrives as a plain rename with none of the layout ops
            — an affine tensor in the swizzled 5-D slot the module allocated, which is a malformed
            DTensor under sharding.
            """
            probe = f"model.layers.0.mlp.experts.{fused}"
            return any(re.search(pattern, probe) for pattern in (conv.source_patterns or []))

        expert_target = re.compile(r"experts\.(gate_up_proj|down_proj)(_scale_inv|_bias)?\$?$")
        updated = []
        covered: set[str] = set()
        for conv in weight_conversions:
            targets = conv.target_patterns if isinstance(conv, WeightConverter) else []
            expert_targets = [t for t in targets if expert_target.search(t)]
            if expert_targets:
                # Append to the EXISTING converter rather than rebuild it: `scope_prefix`,
                # `base_model_prefix` and `force_cpu` are set on it before this hook runs and
                # are not `__init__` arguments, so a fresh one silently loses them.
                conv.operations = list(conv.operations) + layout_ops(expert_targets[0])
                for target in expert_targets:
                    fused = target.rpartition("experts.")[2].removesuffix("$")
                    if sources_the_fused_key(conv, fused):
                        covered.add(fused)
            updated.append(conv)
        # dense linears have no converter: their scale keys take the container cast alone
        updated.append(
            WeightConverter(
                source_patterns=r"weight_scale_inv$",
                target_patterns="weight_scale_inv",
                operations=[FineGrainedScaleContainer(self)],
            )
        )
        for name in ("gate_up_proj", "gate_up_proj_scale_inv", "gate_up_proj_bias", "down_proj_scale_inv"):
            # only where no converter already produces that target: a second converter for the
            # same key is appended LATER and shadows the first, dropping whatever ops it carried
            # that these do not (the packed uint8 -> int8 view, on a fused modelopt checkpoint)
            ops = layout_ops(name)
            if name not in covered and ops:
                # the target is replacement text (op outputs are keyed by it), so no regex escapes
                updated.append(
                    WeightConverter(
                        source_patterns=rf"experts\.{name}$",
                        target_patterns=f"experts.{name}",
                        operations=ops,
                    )
                )
        return updated
