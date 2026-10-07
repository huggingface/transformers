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
"""What the block-scaled weight formats' quantizers share: block-FP8, MXFP8, MXFP4 and NVFP4 are each a quantizer with
its own config, served by the modules in `integrations/finegrained`, and each quantizes a whole model alone;
`FineGrainedHfQuantizer` hosts several of them, each on the modules its `dtype_plan` gives it.
"""

import re
from typing import TYPE_CHECKING

from ..utils import (
    is_accelerate_available,
    is_torch_available,
    is_torch_distributed_available,
    is_torch_xpu_available,
    logging,
)
from .base import HfQuantizer
from .quantizers_utils import (
    FUSED_STACKS,
    STACK_COMPANIONS,
    embedding_pattern,
    get_module_from_name,
    plan_entry_for,
    replace_with_quantized_modules,
    should_convert_module,
)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import BlockScaledConfig

logger = logging.get_logger(__name__)


class BlockScaledQuantizer(HfQuantizer):
    """The quantizer of a block-scaled weight format (block-FP8, MXFP8, MXFP4, NVFP4), written over `plan()`: the
    modules it covers and the format config each loads in. Alone, a format covers every module with its own config;
    `FineGrainedHfQuantizer` covers each part of a model with the format quantizer its plan entry names."""

    requires_calibration = False
    # a scale loads as the checkpoint ships it (Qwen3 and Mistral ship BF16, DeepSeek-V3 fp32): the kernels
    # read either, and a save then writes it back unchanged
    dtype_plan = dict.fromkeys(("_scale_inv", "activation_scale", "_global_scale"))
    # checkpoint keys this format's producers ship beside the weights that no module holds
    unexpected_keys: tuple[str, ...] = ()
    # narrowed once the modules are built: whether this format holds experts, and the up projection they stack
    # (`gate_up_proj` gated, `up_proj` not)
    holds_experts = True
    up_projections: tuple[str, ...] = ("gate_up_proj",)

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        # the embedding tables the plan quantizes, by name suffix: each holds its own per-tensor `weight_scale`
        self.embeddings = list(getattr(quantization_config, "modules_to_convert", None) or [])

    def plan(self) -> dict[str, "BlockScaledConfig"]:
        """`{module regex: format config}` for the modules this quantizer covers."""
        return {".*": self.quantization_config}

    def members(self) -> list["BlockScaledQuantizer"]:
        """The quantizer of each format config in `plan()`."""
        return [self]

    def member_of(self, module) -> "BlockScaledQuantizer | None":
        """The quantizer whose plan entry `module` was built from; `None` for a module left as it was."""
        config = getattr(module, "quantization_config", None)
        return next((member for member in self.members() if member.quantization_config is config), None)

    # what a format brings

    def format_conversions(self, dequantize: bool) -> list:
        """The converters this format's producers' key layouts need on top of the model's plan. Under `dequantize`,
        what folding this format's keys needs beyond the shared `weight` + `weight_scale_inv` chain."""
        return []

    def get_param_ops(self, model: "PreTrainedModel", param_name: str) -> list:
        """The ops a tensor loading into `param_name` runs, for the layout this format's modules hold it in:
        an expert's stacked gate|up rows interleaved (weight, scale grid and bias share the row axis), a scale
        in the container dtype its module holds, then in the swizzled layout. Each op has a reverse, so a save
        restores the checkpoint layout."""
        from ..integrations.finegrained import FineGrainedExperts, FineGrainedLinear
        from ..integrations.finegrained.conversions import (
            FineGrainedInterleaveGateUp,
            FineGrainedScaleContainer,
            FineGrainedSwizzleScales,
        )

        module, name = get_module_from_name(model, param_name)
        if isinstance(module, FineGrainedExperts) and re.fullmatch(
            r"(gate_up_proj|up_proj|down_proj)(_scale_inv|_bias)?", name
        ):
            ops = [FineGrainedInterleaveGateUp(self)] if name.startswith("gate_up_proj") else []
            if not name.endswith("_bias"):
                ops += [FineGrainedScaleContainer(self, module=module), FineGrainedSwizzleScales(self)]
            return ops
        if isinstance(module, FineGrainedLinear) and name == "weight_scale_inv":
            return [FineGrainedScaleContainer(self, module=module)]
        return []

    def finish_modules(self, model):
        """What this format does to its loaded modules beyond what every format does."""

    def _process_model_after_weight_loading(self, model, **kwargs):
        from ..integrations.finegrained import assert_modules_are_quantized, disable_deepgemm_on_multi_device

        for member in self.members():
            member.finish_modules(model)
        disable_deepgemm_on_multi_device(model)
        assert_modules_are_quantized(model)
        return model

    @staticmethod
    def _per_expert_sources(proj: str, slot: str) -> list[str]:
        """The per-expert checkpoint keys of `slot` that stack into the experts' `proj`, under any parent
        (`mlp.experts`, `mixer.experts`, `feed_forward.experts`): the gate|up stack reads both halves."""
        halves = ("gate_proj", "up_proj") if proj == "gate_up_proj" else (proj,)
        return [f"experts.*.{half}.{slot}" for half in halves]

    def _block_scale_rename(self):
        """A producer's `weight_scale` block scale onto the `weight_scale_inv` slot every module holds it in; the
        embedding tables the plan quantizes hold their per-tensor scale under that name, and keep it."""
        from ..core_model_loading import WeightRenaming

        kept = "".join(rf"(?<!{re.escape(name)})" for name in self.embeddings)
        return WeightRenaming(source_patterns=rf"^(.+){kept}\.weight_scale$", target_patterns=r"\1.weight_scale_inv")

    def _activation_scale_conversions(self, projections=FUSED_STACKS, ops=()) -> list:
        """A calibrated checkpoint's `input_scale` onto the `activation_scale` its module holds: renamed for every
        module first, then one value per expert of each of `projections` the experts hold, the gate|up pair reduced
        to the one both halves read; `ops` run on each value before that reduction. Both expert layouts are covered, a
        checkpoint matching one while the other never fires; neither when this format holds no experts."""
        from ..core_model_loading import MergeModulelist, WeightConverter, WeightRenaming
        from ..integrations.finegrained.conversions import FineGrainedInputScales

        rename = WeightRenaming(source_patterns=r"^(.+)\.input_scale$", target_patterns=r"\1.activation_scale")
        if not self.holds_experts:
            return [rename]
        held = [proj for proj in projections if proj in (*self.up_projections, "down_proj")]
        return [
            rename,
            *(
                WeightConverter(
                    source_patterns=self._per_expert_sources(proj, "activation_scale"),
                    target_patterns=f"experts.{proj}_activation_scale",
                    operations=[MergeModulelist(dim=0), *ops, FineGrainedInputScales(self)],
                )
                for proj in held
            ),
            *(
                WeightConverter(
                    source_patterns=rf"experts\.{proj}_input_scale$",
                    target_patterns=f"experts.{proj}_activation_scale",
                    operations=[*ops, FineGrainedInputScales(self)],
                )
                for proj in held
            ),
        ]

    # what every format shares

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading a fine-grained quantized model requires accelerate (`pip install accelerate`)")

        if self.quantization_config.dequantize:
            return

        if not torch.cuda.is_available() and not is_torch_xpu_available():
            if self.pre_quantized:
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
        from ..integrations.finegrained import FineGrainedExperts, FineGrainedLinear

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
            # one byte per held element: an FP8 value, or two FP4 values packed into a weight already half as wide
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

        from ..conversion_mapping import get_model_conversion_mapping
        from ..core_model_loading import WeightRenaming

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

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        self._normalize_modules_to_not_convert(model)
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        # an embedding table worth quantizing is too big to expand, so it stays FP8 under `dequantize=True`
        # too; quantizing one on the fly has no path
        plan = self.plan()
        embeddings = {
            embedding_pattern(name) for name in getattr(self.quantization_config, "modules_to_convert", None) or []
        }
        if not self.pre_quantized:
            plan = {pattern: config for pattern, config in plan.items() if pattern not in embeddings}
        elif self.quantization_config.dequantize:
            plan = {pattern: config for pattern, config in plan.items() if pattern in embeddings}
        replaced = self.replace_modules(model, plan, self.modules_to_not_convert)
        from ..integrations.finegrained import FineGrainedExperts

        experts = [module for module in model.modules() if isinstance(module, FineGrainedExperts)]
        ignored = set(model._keys_to_ignore_on_load_unexpected or [])
        for member in self.members():
            # a format's expert converters exist only where its experts do, for the stacks they hold
            held = [module for module in experts if module.quantization_config is member.quantization_config]
            member.holds_experts = bool(held)
            member.up_projections = tuple(sorted({module.gate_up_name for module in held}))
            ignored.update(member.unexpected_keys)
        # with no module holding an activation scale (weight-only, dynamic activations), a calibrated
        # `input_scale` is one this run has no use for, not an unexpected key: a later run may read it
        if not any(name.endswith("activation_scale") for name, _ in model.named_parameters()):
            ignored.add(r"input(_global)?_scale$")
        model._keys_to_ignore_on_load_unexpected = ignored
        if not replaced and not self.quantization_config.dequantize:
            logger.warning(
                "You are loading a fine-grained quantized model but no linear or experts modules were "
                "found to convert. Please double check your model architecture."
            )

    def replace_modules(self, model, plan=None, modules_to_not_convert=None) -> bool:
        """Swap the modules `plan` (this quantizer's own by default) covers for their quantized classes."""
        from ..integrations.finegrained import (
            FineGrainedEmbedding,
            FineGrainedExperts,
            FineGrainedGroupedLinear,
            FineGrainedLinear,
        )

        module_classes = {
            "linear": FineGrainedLinear,
            "grouped_linear": FineGrainedGroupedLinear,
            "experts": FineGrainedExperts,
            "embedding": FineGrainedEmbedding,
        }
        plan = self.plan() if plan is None else plan
        return replace_with_quantized_modules(model, plan, modules_to_not_convert, module_classes)

    def _experts_hold_scale_grid(self, glob: str, config) -> bool:
        """Whether the experts a TP glob (`layers.*.mlp.experts`) names hold a scale grid, which splits with its
        weight. The glob's one style serves every layer, so its layers must agree."""
        layers = range(getattr(config, "num_hidden_layers", None) or 1) if "*" in glob else (0,)
        entries = (plan_entry_for(self.plan(), glob.replace("*", str(layer))) for layer in layers)
        grids = {_has_scale_grid(entry) for entry in entries if entry is not None}
        if len(grids) > 1:
            raise ValueError(
                f"Tensor parallelism shards every `{glob}` alike, but this plan gives some of them a block-scale "
                "grid and others a per-tensor scale. Give those experts one format."
            )
        return True in grids

    def update_tp_plan(self, config):
        # Per-impl rewrite of the experts parallel-layer kind, over the plan the model declares. Models carry the
        # experts mapping under `base_model_tp_plan` and/or `base_model_ep_plan` — rewrite both.
        from ..integrations.finegrained import FineGrainedExperts

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
                if projection not in FUSED_STACKS:
                    continue
                if style == "grouped_gemm":
                    for suffix in ("_bias", *STACK_COMPANIONS):
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
                blocked = self._experts_hold_scale_grid(key.rpartition(".")[0], plan_owner)
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
        from ..integrations.finegrained.conversions import FineGrainedQuantize

        return FineGrainedQuantize(self)

    def get_weight_conversions(self):
        """The converters every format's producer key layout needs, on top of the model's plan."""
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedDequantize

        if not self.pre_quantized:
            return []
        dequantize = self.quantization_config.dequantize
        conversions, claimed = [], {}
        if dequantize:
            # every quantized key folded back to a full-precision weight from the `weight` + `weight_scale_inv` pair,
            # times NVFP4's `weight_global_scale` where there is one, and the activation scales collected only to be
            # dropped; the companions start a name (`(?<!\w)`), so a fused stack's `gate_up_proj_weight_global_scale`
            # stays with its own converter below
            conversions.append(
                WeightConverter(
                    source_patterns=["weight$", *_DENSE_COMPANIONS],
                    target_patterns="weight",
                    operations=[FineGrainedDequantize(self)],
                )
            )
            # an experts module's fused stacks, as a checkpoint written in the model's own layout ships them
            for proj in FUSED_STACKS:
                conversions.append(
                    WeightConverter(
                        source_patterns=[rf"(?<!\w){proj}{suffix}$" for suffix in ("", *STACK_COMPANIONS)],
                        target_patterns=proj,
                        operations=[FineGrainedDequantize(self)],
                    )
                )
        # once each: formats reading the same producer keys emit the same converter; two that read a key
        # differently cannot both load it, so the plan is refused rather than one picked
        for member in self.members():
            for conversion in member.format_conversions(dequantize):
                owner = next((claimed[p] for p in conversion.source_patterns if p in claimed), None)
                if owner is None:
                    conversions.append(conversion)
                    claimed.update(dict.fromkeys(conversion.source_patterns, conversion))
                elif _signature(owner) != _signature(conversion):
                    raise ValueError(
                        f"The plan's formats read {conversion.source_patterns} differently, so they cannot be "
                        "loaded together."
                    )
        return conversions

    def update_weight_conversions(self, weight_conversions):
        """Rewrite the model's conversion plan for this checkpoint, then append :meth:`get_weight_conversions`.

        Each of the model's converters carries a weight's block-scale grid along with it, both ways: the merge
        that stacks the experts' weights stacks their grids too, and nothing else rides it (a global or an
        activation scale has a converter of its own). Under ``dequantize=True`` a :class:`FineGrainedDequantize`
        runs first in every converter with a ``.weight`` source, with the scales collected alongside, so each
        per-expert pair folds to full precision before the merge collapses the per-expert structure. The
        layout each module holds its tensors in comes per module instead, from :meth:`get_param_ops`."""
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedDequantize

        dequantize = self.pre_quantized and self.quantization_config.dequantize
        updated = []
        for conv in weight_conversions:
            weights = [p for p in conv.source_patterns if p.endswith(".weight")]
            if isinstance(conv, WeightConverter) and weights and dequantize:
                conv = _rebuilt(
                    conv,
                    [p + "$" for p in weights]
                    + [p.removesuffix(".weight") + ".weight_scale_inv$" for p in weights]
                    + [p + "_global_scale$" for p in weights]
                    + [p for p in conv.source_patterns if not p.endswith(".weight")],
                    operations=[FineGrainedDequantize(self), *conv.operations],
                )
            elif isinstance(conv, WeightConverter):
                conv = _rebuilt(
                    conv, [_with_scale_grid(p) if p.endswith(".weight") else p for p in conv.source_patterns]
                )
            updated.append(conv)
        return updated + self.get_weight_conversions()


# a parameter and its block-scale grid, and none of its other companions (`_weight_global_scale`, `_activation_scale`)
_WITH_SCALE_GRID = r"(?=$|_scale_inv$)"
# what a dense quantized weight carries beside it
_DENSE_COMPANIONS = (r"(?<!\w)weight_scale_inv", r"(?<!\w)weight_global_scale", r"(?<!\w)activation_scale")


def _with_scale_grid(pattern: str) -> str:
    return pattern if pattern.endswith((_WITH_SCALE_GRID, "$")) else pattern + _WITH_SCALE_GRID


def _rebuilt(conv, source_patterns: list[str], operations: list | None = None):
    """`conv` over `source_patterns`, its targets taking only a parameter and its scale grid on save (a target is
    a source pattern there, and a parameter's name prefixes all its companions'), and its scope and placement kept."""
    from ..core_model_loading import WeightConverter

    params = (".weight", *FUSED_STACKS)
    rebuilt = WeightConverter(
        source_patterns=source_patterns,
        target_patterns=[_with_scale_grid(p) if p.endswith(params) else p for p in conv._original_target_patterns],
        operations=conv.operations if operations is None else operations,
        force_cpu=conv.force_cpu,
    )
    rebuilt.scope_prefix, rebuilt.base_model_prefix = conv.scope_prefix, conv.base_model_prefix
    return rebuilt


def _has_scale_grid(config) -> bool:
    """Whether a weight in `config`'s format carries a scale grid, which shards along with it under tensor parallelism,
    rather than one scale for the whole tensor."""
    from ..integrations.finegrained.layers import resolve_weight_format

    return resolve_weight_format(config.method.value, config.scale_fmt, config.weight_block_size)[2] is not None


def _signature(conversion) -> tuple:
    """What a converter does: which keys it reads, where they go and through which ops."""
    operations = tuple(type(op).__name__ for op in getattr(conversion, "operations", ()))
    return type(conversion), tuple(conversion.source_patterns), tuple(conversion.target_patterns), operations
