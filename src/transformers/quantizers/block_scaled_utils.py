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
from copy import deepcopy
from typing import TYPE_CHECKING

from ..utils import (
    is_accelerate_available,
    is_torch_available,
    is_torch_xpu_available,
    logging,
)
from .base import HfQuantizer
from .quantizers_utils import (
    FUSED_STACKS,
    STACK_COMPANIONS,
    get_module_from_name,
    module_kind,
    plan_entry_for,
    replace_with_quantized_modules,
    should_convert_module,
)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import BlockScaledConfig, FineGrainedConfig

logger = logging.get_logger(__name__)


class BlockScaledQuantizer(HfQuantizer):
    """The quantizer of a block-scaled weight format (block-FP8, MXFP8, MXFP4, NVFP4), written over `plan()`: the
    modules it covers and the format config each loads in. Alone, a format covers every module with its own config;
    `FineGrainedHfQuantizer` covers each part of a model with the format quantizer its plan entry names."""

    requires_calibration = False
    quantization_config: "BlockScaledConfig | FineGrainedConfig"
    # checkpoint keys this format's producers ship beside the weights that no module holds
    unexpected_keys: tuple[str, ...] = ()
    # narrowed once the modules are built: whether this format holds experts, and the up projection they stack
    # (`gate_up_proj` gated, `up_proj` not)
    holds_experts = True
    holds_embeddings = False
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
        if config is None:
            return None
        return next(
            (
                member
                for member in self.members()
                if member.quantization_config is config or member.quantization_config.to_dict() == config.to_dict()
            ),
            None,
        )

    # what a format brings

    def format_conversions(self, dequantize: bool) -> list:
        """The converters this format's producers' key layouts need on top of the model's plan. Under `dequantize`,
        what folding this format's keys needs beyond the shared `weight` + `weight_scale_inv` chain."""
        return []

    def format_weight_conversions(self, converters, model):
        """Adjust the model's companion converters after the base quantizer has cloned them."""

    def format_parallel_plan(self, model, plan):
        """Set this format's companion placements after the shared parallel plan is built."""

    def get_param_ops(self, model: "PreTrainedModel", param_name: str) -> list:
        """The ops a tensor loading into `param_name` runs, for the layout this format's modules hold it in:
        an expert's stacked gate|up rows interleaved (weight, scale grid and bias share the row axis), a scale
        in the swizzled layout. Each op has a reverse, so a save
        restores the checkpoint layout."""
        from ..integrations.finegrained import FineGrainedExperts
        from ..integrations.finegrained.conversions import (
            FineGrainedInterleaveGateUp,
            FineGrainedSwizzleScales,
        )

        module, name = get_module_from_name(model, param_name)
        if isinstance(module, FineGrainedExperts) and re.fullmatch(
            r"(gate_up_proj|up_proj|down_proj)(_scale_inv|_bias)?", name
        ):
            ops = [FineGrainedInterleaveGateUp(self)] if name.startswith("gate_up_proj") else []
            if not name.endswith("_bias"):
                ops += [FineGrainedSwizzleScales(self)]
            return ops
        return []

    def finish_modules(self, model):
        """What this format does to its loaded modules beyond what every format does."""

    def _process_model_after_weight_loading(self, model, **kwargs):
        from ..integrations.finegrained import assert_modules_are_quantized, disable_deepgemm_on_multi_device

        for member in self.members():
            member.finish_modules(model)
        disable_deepgemm_on_multi_device(model)
        assert_modules_are_quantized(model)
        if self.pre_quantized and self.quantization_config.dequantize:
            # The loading plan also consumed scales. Saving full-precision weights must reverse only
            # the original model transforms, preserving expert split/merge and user key mappings.
            model._weight_conversions = [
                original for original, loaded in self._dequantized_save_conversions if loaded.was_used()
            ]
        if not self.pre_quantized and getattr(model, "_weight_conversions", None) is not None:
            # Quantization emitted scales that were absent on disk. Give them the same
            # reverse conversions as their weights when saving.
            model._weight_conversions = self.update_weight_conversions(model._weight_conversions, model, for_save=True)
        return model

    @staticmethod
    def _per_expert_sources(proj: str, slot: str) -> list[str]:
        """The per-expert checkpoint keys of `slot` that stack into the experts' `proj`, under any parent
        (`mlp.experts`, `mixer.experts`, `feed_forward.experts`): the gate|up stack reads both halves."""
        halves = ("gate_proj", "up_proj") if proj == "gate_up_proj" else (proj,)
        return [f"experts.*.{half}.{slot}" for half in halves]

    def _block_scale_rename(self):
        """Rename block scales while leaving the host's scalar embedding scales untouched."""
        from ..core_model_loading import WeightRenaming

        kept = "".join(rf"(?<!{re.escape(name)})" for name in self.embeddings)
        return WeightRenaming(source_patterns=rf"^(.+){kept}\.weight_scale", target_patterns=r"\1.weight_scale_inv")

    def _activation_scale_conversions(self, projections=FUSED_STACKS, ops=()) -> list:
        """A calibrated checkpoint's `input_scale` onto the `activation_scale` its module holds: renamed for every
        module first, then one value per expert of each of `projections` the experts hold, the gate|up pair reduced
        to the one both halves read; `ops` run on each value before that reduction. Both expert layouts are covered, a
        checkpoint matching one while the other never fires; neither when this format holds no experts."""
        from ..core_model_loading import MergeModulelist, WeightConverter, WeightRenaming
        from ..integrations.finegrained.conversions import FineGrainedInputScales

        rename = WeightRenaming(source_patterns=r"^(.+)\.input_scale", target_patterns=r"\1.activation_scale")
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
                    source_patterns=rf"experts\.{proj}_input_scale",
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
            # W4A16 decodes packed FP4 weights before a regular dot product; it does not
            # need the FP8 instructions used by the other activation/weight formats.
            weight_only_mxfp4 = all(
                config.quant_method == "mxfp4" and config.activation_format == "bf16"
                for config in self.plan().values()
            )
            minimum_capability = (7, 5) if weight_only_mxfp4 else (8, 9)
            if compute_capability < minimum_capability:
                logger.warning_once(
                    "This fine-grained quantization plan requires a GPU with compute capability >= "
                    f"{minimum_capability[0]}.{minimum_capability[1]}, actual = `{major}.{minor}`. "
                    "We will default to dequantizing the model to bf16."
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
        if self.pre_quantized:
            return False
        if isinstance(module, FineGrainedExperts):
            return tensor_name in FUSED_STACKS
        return isinstance(module, FineGrainedLinear) and tensor_name == "weight"

    @property
    def quantized_element_size(self) -> float:
        # FP4 modules already hold two values per byte in a half-width weight.
        return 1

    def get_new_dtype(self, converter, param, dtype):
        if any(name.endswith(("activation_scale", "_global_scale")) for name in converter.target_patterns):
            return None
        # The initialized scale dtype is authoritative for UE8M0, including float32 containers
        # (DeepSeek-V4-Base). Integer exponent containers retain their checkpoint dtype.
        is_float = (
            param.get_dtype().startswith(("F", "BF")) if hasattr(param, "get_dtype") else param.is_floating_point()
        )
        if dtype == getattr(torch, "float8_e8m0fnu", None) and is_float:
            return dtype
        return super().get_new_dtype(converter, param, dtype)

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
            rf"^{re.escape(name)}$": config
            for name, module in model.named_modules()
            if module_kind(name, module) == "embedding"
            and (config := plan_entry_for(plan, name, catch_all=False)) is not None
        }
        if not self.pre_quantized:
            self.modules_to_not_convert = [*self.modules_to_not_convert, *embeddings]
        elif self.quantization_config.dequantize:
            plan = embeddings
        replaced = self.replace_modules(model, plan, self.modules_to_not_convert)
        from ..integrations.finegrained import FineGrainedEmbedding, FineGrainedExperts

        if excluded := getattr(model, "_no_placement_params", None):
            excluded = set(excluded)
            for name, module in model.named_modules():
                if isinstance(module, FineGrainedEmbedding):
                    for prefix in list(excluded):
                        if prefix.endswith(".weight") and (name + ".weight").endswith(prefix):
                            excluded.update(
                                prefix.removesuffix(".weight") + "." + param
                                for param, _ in module.named_parameters(recurse=False)
                            )
            setattr(model, "_no_placement_params", sorted(excluded))

        experts = [module for module in model.modules() if isinstance(module, FineGrainedExperts)]
        ignored = set(model._keys_to_ignore_on_load_unexpected or [])
        for member in self.members():
            # a format's expert converters exist only where its experts do, for the stacks they hold
            held = [module for module in experts if module.quantization_config is member.quantization_config]
            member.holds_embeddings = any(
                isinstance(module, FineGrainedEmbedding) and self.member_of(module) is member
                for module in model.modules()
            )
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

    def update_parallel_plan(self, model, plan):
        from ..distributed.tensor_parallel import _get_parameter_plan, replace_layer_number_by_wildcard
        from ..integrations.finegrained import FineGrainedExperts

        original = dict(plan)
        super().update_parallel_plan(model, plan)
        overrides = FineGrainedExperts._impl_tp_layer_overrides.get(
            getattr(model.config, "_experts_implementation", None), {}
        )
        plan.update({key: overrides.get(style, style) for key, style in plan.items()})
        # Temporary until the EP implementation PR is merged: use replicated routing,
        # local expert masking, and all-reduce to keep token shapes fixed across ranks.
        if "ep_dispatch_experts" in plan.values():
            modules = {replace_layer_number_by_wildcard(name) for name, _ in model.named_modules()}
            for name, style in list(plan.items()):
                if style != "ep_dispatch_experts":
                    continue
                parent = name.rpartition(".")[0]
                router = next((f"{parent}.{key}" for key in ("gate", "router") if f"{parent}.{key}" in modules), None)
                if router is None:
                    raise ValueError(f"Cannot find a gate or router for masked expert parallelism at `{name}`.")
                plan[name] = "moe_tp_experts"
                plan[router] = "ep_router"
        grids = {}
        for name, module in model.named_modules():
            if not isinstance(module, FineGrainedExperts):
                continue
            glob = replace_layer_number_by_wildcard(name)
            if _get_parameter_plan(name, plan, is_weight=False) not in ("moe_tp_experts", "megamoe_experts"):
                continue
            blocked = _has_scale_grid(module.quantization_config)
            if glob in grids and grids[glob] != blocked:
                raise ValueError(f"Tensor parallelism requires every `{glob}` to use the same scale grid layout.")
            grids[glob] = blocked
            for projection in (module.gate_up_name, "down_proj"):
                key = f"{glob}.{projection}"
                style = _get_parameter_plan(f"{name}.{projection}", plan)
                rows = {
                    "packed_colwise": "moe_experts_packed_colwise",
                    "colwise": "moe_experts_colwise",
                    "rowwise": "moe_experts_rowwise",
                    "packed_rowwise": "moe_experts_rowwise",
                }.get(style)
                if rows is None:
                    continue
                for suffix in ("_bias", *STACK_COMPANIONS):
                    companion = key + suffix
                    if companion in original or not hasattr(module, projection + suffix):
                        continue
                    if (suffix == "_scale_inv" and blocked) or (suffix == "_bias" and projection != "down_proj"):
                        plan[companion] = rows
                    else:
                        # Per-expert globals and down biases stay whole under intra-expert TP.
                        plan.pop(companion, None)
        plan.update(
            {
                replace_layer_number_by_wildcard(name): "replicate"
                for name, param in model.named_parameters()
                if param.ndim == 0 and _get_parameter_plan(name, plan) is not None
            }
        )

        for member in self.members():
            member.format_parallel_plan(model, plan)

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

    @property
    def serialized_names(self):
        dequantize = self.quantization_config.dequantize
        dense = ["weight", "weight_scale_inv"]
        if dequantize:
            dense += ["weight_global_scale", "activation_scale"]
        return {
            "weight": dense,
            **{
                proj: [
                    proj,
                    *(proj + suffix for suffix in STACK_COMPANIONS),
                    *([] if dequantize else [proj + "_bias"]),
                ]
                for proj in FUSED_STACKS
            },
        }

    def get_load_operations(self):
        from ..integrations.finegrained.conversions import FineGrainedDequantize

        return {"dequantize": FineGrainedDequantize}

    def get_weight_conversions(self):
        conversions, claimed = [], {}
        if not self.pre_quantized:
            return conversions
        # Fused stacks precede the dense fallback, whose companion names are suffixes of theirs.
        general = super().get_weight_conversions()
        for converter in [*general[1:], *general[:1]]:
            conversions.append(
                converter.copy(
                    [rf"(?<!\w){name}(?=$)" for name in converter._original_source_patterns],
                    converter._original_target_patterns,
                    operations=converter.operations,
                )
            )
        for member in self.members():
            for conversion in member.format_conversions(self.quantization_config.dequantize):
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

    def update_weight_conversions(self, weight_conversions, model, *, for_save=False):
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedDeserialize

        dequantize = self.pre_quantized and self.quantization_config.dequantize
        save_conversions = deepcopy(weight_conversions) if dequantize else []
        conversions = super().update_weight_conversions(weight_conversions, model)
        if dequantize:
            self._dequantized_save_conversions = []
            for original in save_conversions:
                loaded = next(
                    conversion
                    for conversion in conversions
                    if conversion._original_target_patterns == original._original_target_patterns
                    and all(
                        source in conversion._original_source_patterns for source in original._original_source_patterns
                    )
                )
                self._dequantized_save_conversions.append((original, loaded))
        converters = [conversion for conversion in conversions if isinstance(conversion, WeightConverter)]
        for member in self.members():
            member.format_weight_conversions(converters, model)
        for converter in converters:
            if not any(isinstance(op, FineGrainedDeserialize) for op in converter.operations):
                converter.operations = [*converter.operations, FineGrainedDeserialize(self)]
        reverse = [converter.reverse_transform() for converter in converters] if for_save else []
        for name, _ in model.named_parameters():
            if self.get_param_ops(model, name) and not any(c.rename_source_key(name)[1] is not None for c in reverse):
                conversions.append(WeightConverter(re.escape(name), name, [FineGrainedDeserialize(self)]))
        return conversions


def _has_scale_grid(config) -> bool:
    """Whether a weight in `config`'s format carries a scale grid, which shards along with it under tensor parallelism,
    rather than one scale for the whole tensor."""
    from ..integrations.finegrained.layers import resolve_weight_format

    return resolve_weight_format(config.method.value, config.scale_fmt, config.weight_block_size)[2] is not None


def _signature(conversion) -> tuple:
    """What a converter does: which keys it reads, where they go and through which ops."""
    operations = tuple(type(op).__name__ for op in getattr(conversion, "operations", ()))
    return type(conversion), tuple(conversion.source_patterns), tuple(conversion.target_patterns), operations
