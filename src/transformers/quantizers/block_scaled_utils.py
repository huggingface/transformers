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
"""What the quantizers of the block-scaled formats (block-FP8, MXFP8, MXFP4, NVFP4) share. Each quantizes a whole
model alone; `FineGrainedHfQuantizer` hosts several, each on the modules its `dtype_plan` gives it."""

import re
from dataclasses import dataclass
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
    embedding_pattern,
    get_module_from_name,
    replace_with_quantized_modules,
)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import BlockScaledConfig, FineGrainedConfig

logger = logging.get_logger(__name__)

# the up projection a model's experts stack, assumed until the built modules say otherwise
GATED_UP_PROJECTIONS = ("gate_up_proj",)


@dataclass(frozen=True)
class ProducerLayout:
    """How one producer's checkpoints ship a format's tensors, where that differs from the slots its modules hold."""

    # the name the block scale ships under where it isn't its slot's, `weight_scale_inv`: e.g. `weight_scale`
    block_scale_name: str | None = None
    # packed FP4 values ship as uint8: the modules hold them as int8, which a copy would convert rather than view
    packed_as_uint8: bool = False


class BlockScaledQuantizer(HfQuantizer):
    """The quantizer of a block-scaled format, written over `plan()`: the modules it covers and the format config each
    loads in."""

    requires_calibration = False
    dequantized_saves_unquantized = True
    quantization_config: "BlockScaledConfig | FineGrainedConfig"
    # checkpoint keys this format's producers ship beside the weights that no module holds
    unexpected_keys: tuple[str, ...] = ()
    # by `checkpoint_format`: how each producer ships this format, the slots' own layout where none is listed
    producer_layouts: dict[str | None, ProducerLayout] = {}

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

    @property
    def producer_layout(self) -> ProducerLayout:
        """How this format's checkpoint ships its tensors."""
        return self.producer_layouts.get(
            getattr(self.quantization_config, "checkpoint_format", None), ProducerLayout()
        )

    def _up_projections(self, model) -> tuple[str, ...]:
        """The up projections of the experts this format holds in `model`, none when it holds no experts; without a
        model (conversions built before one), the gated stack."""
        from ..integrations.finegrained import FineGrainedExperts

        if model is None:
            return GATED_UP_PROJECTIONS
        held = {
            module.gate_up_name
            for module in model.modules()
            if isinstance(module, FineGrainedExperts) and module.quantization_config is self.quantization_config
        }
        return tuple(sorted(held))

    def format_conversions(self, dequantize: bool, up_projections: tuple[str, ...] = GATED_UP_PROJECTIONS) -> list:
        """The converters this format's producers' key layouts need on top of the model's plan. Under `dequantize`,
        what folding this format's keys needs beyond the shared `weight` + `weight_scale_inv` chain.
        `up_projections` are the up projections of the experts this format holds, none when it holds no experts."""
        return []

    def format_weight_conversions(self, converters, model):
        """Adjust the model's companion converters after the base quantizer has cloned them."""

    def format_parallel_plan(self, model, plan):
        """Set this format's companion placements after the shared parallel plan is built."""

    def get_param_ops(self, model: "PreTrainedModel", param_name: str) -> list:
        """The layout ops a tensor loading into `param_name` runs (gate|up interleave, scale swizzle, packed-FP4 view),
        each reversed on save."""
        from ..integrations.finegrained import FineGrainedExperts
        from ..integrations.finegrained.conversions import (
            FineGrainedInterleaveGateUp,
            FineGrainedSwizzleScales,
            FineGrainedViewPackedInt8,
        )
        from ..integrations.finegrained.layers import _FineGrainedModule

        module, name = get_module_from_name(model, param_name)
        ops = []
        if isinstance(module, FineGrainedExperts) and re.fullmatch(
            r"(gate_up_proj|up_proj|down_proj)(_scale_inv|_bias)?", name
        ):
            ops = [FineGrainedInterleaveGateUp(self)] if name.startswith("gate_up_proj") else []
            if not name.endswith("_bias"):
                ops += [FineGrainedSwizzleScales(self)]
        if (
            self.producer_layout.packed_as_uint8
            and isinstance(module, _FineGrainedModule)
            and name in ("weight", *FUSED_STACKS)
        ):
            ops = [FineGrainedViewPackedInt8(self), *ops]
        return ops

    def finish_modules(self, model):
        """What this format does to its loaded modules beyond what every format does."""

    def _process_model_after_weight_loading(self, model, **kwargs):
        from ..integrations.finegrained import assert_modules_are_quantized, disable_deepgemm_on_multi_device

        for member in self.members():
            member.finish_modules(model)
        disable_deepgemm_on_multi_device(model)
        assert_modules_are_quantized(model)
        if not self.pre_quantized and getattr(model, "_weight_conversions", None) is not None:
            # the scales quantization emitted save through their weights' conversions
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
        shipped = self.producer_layout.block_scale_name
        return WeightRenaming(source_patterns=rf"^(.+){kept}\.{shipped}", target_patterns=r"\1.weight_scale_inv")

    def _activation_scale_conversions(self, up_projections, projections=FUSED_STACKS, ops=()) -> list:
        """A calibrated `input_scale` onto the module's `activation_scale`: renamed, then reduced to one per expert
        of `projections` from either expert layout, `ops` running on each value first."""
        from ..core_model_loading import MergeModulelist, WeightConverter, WeightRenaming
        from ..integrations.finegrained.conversions import FineGrainedInputScales

        rename = WeightRenaming(source_patterns=r"^(.+)\.input_scale", target_patterns=r"\1.activation_scale")
        if not up_projections:
            return [rename]
        held = [proj for proj in projections if proj in (*up_projections, "down_proj")]
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
            # weight-only MXFP4 decodes to a regular dot product, without FP8 instructions
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
        from ..integrations.finegrained import FineGrainedExperts, FineGrainedLinear

        module, tensor_name = get_module_from_name(model, param_name)
        if self.pre_quantized:
            return False
        if isinstance(module, FineGrainedExperts):
            return tensor_name in FUSED_STACKS
        return isinstance(module, FineGrainedLinear) and tensor_name == "weight"

    @property
    def quantized_element_size(self) -> float:
        # FP4 modules hold two values per byte in a half-width weight
        return 1

    def get_new_dtype(self, converter, param, dtype):
        if any(name.endswith(("activation_scale", "_global_scale")) for name in converter.target_patterns):
            return None
        # the module's UE8M0 scale dtype wins over a float container (DeepSeek-V4-Base ships fp32); an integer one
        # keeps the checkpoint's
        is_float = (
            param.get_dtype().startswith(("F", "BF")) if hasattr(param, "get_dtype") else param.is_floating_point()
        )
        if dtype == getattr(torch, "float8_e8m0fnu", None) and is_float:
            return dtype
        return super().get_new_dtype(converter, param, dtype)

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations.finegrained import FineGrainedEmbedding

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        # an embedding table worth quantizing is too big to expand, so it stays FP8 under `dequantize=True`
        # too; quantizing one on the fly has no path
        plan = self.plan()
        embeddings = set(map(embedding_pattern, self.embeddings))
        if not self.pre_quantized:
            plan = {pattern: config for pattern, config in plan.items() if pattern not in embeddings}
        elif self.quantization_config.dequantize:
            plan = {pattern: config for pattern, config in plan.items() if pattern in embeddings}
        replaced = self.replace_modules(model, plan, self.modules_to_not_convert)

        # an FP8 table kept off the device map keeps its scale with it
        if excluded := getattr(model, "_no_placement_params", None):
            excluded = set(excluded)
            for name, module in model.named_modules():
                if isinstance(module, FineGrainedEmbedding):
                    excluded.update(
                        entry.removesuffix("weight") + "weight_scale"
                        for entry in list(excluded)
                        if entry.endswith(".weight") and f"{name}.weight".endswith(entry)
                    )
            model._no_placement_params = sorted(excluded)

        ignored = set(model._keys_to_ignore_on_load_unexpected or [])
        for member in self.members():
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
        from ..distributed.tensor_parallel import apply_experts_parallel_styles

        original = dict(plan)
        super().update_parallel_plan(model, plan)
        # the experts implementation's own styles go in ahead of the masked-EP rewrite, which leaves them alone
        apply_experts_parallel_styles(model, plan)
        _mask_expert_parallelism(model, plan)
        _shard_expert_companions(model, plan, original)
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

    def get_weight_conversions(self, model=None):
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedDequantize

        if not self.pre_quantized:
            return []
        conversions = []
        if self.quantization_config.dequantize:
            # one chain per slot folds its scales in, each name matched whole; the fused stacks come first, as the
            # dense names are suffixes of their companions'
            for slot, names in sorted(self.serialized_names.items(), key=lambda item: item[0] == "weight"):
                conversions.append(
                    WeightConverter([rf"(?<!\w){name}(?=$)" for name in names], slot, [FineGrainedDequantize(self)])
                )
        dequantize = self.quantization_config.dequantize
        formats = (
            conversion
            for member in self.members()
            for conversion in member.format_conversions(dequantize, member._up_projections(model))
        )
        return conversions + _claimed(formats)

    def get_checkpoint_renamings(self, model=None):
        # a block scale shipped under its producer's name (`weight_scale`, DeepSeek's `scale`) is renamed onto its
        # slot, and then rides its weight's converter
        if not self.pre_quantized:
            return []
        return _claimed(
            member._block_scale_rename() for member in self.members() if member.producer_layout.block_scale_name
        )

    def update_weight_conversions(self, weight_conversions, model, *, for_save=False):
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained.conversions import FineGrainedDeserialize

        conversions = super().update_weight_conversions(weight_conversions, model)
        converters = [conversion for conversion in conversions if isinstance(conversion, WeightConverter)]
        for member in self.members():
            member.format_weight_conversions(converters, model)
        for converter in converters:
            if not any(isinstance(op, FineGrainedDeserialize) for op in converter.operations):
                converter.operations = [*converter.operations, FineGrainedDeserialize(self)]
        # a param's own converter loads a stack the checkpoint ships fused; on save its full-name pattern would match
        # first, so a param a model converter writes back keeps only that one
        reverse = [converter.reverse_transform() for converter in converters] if for_save else []
        for name, _ in model.named_parameters():
            if self.get_param_ops(model, name) and not any(c.rename_source_key(name)[1] is not None for c in reverse):
                conversions.append(WeightConverter(re.escape(name), name, [FineGrainedDeserialize(self)]))
        return conversions


def _claimed(conversions) -> list:
    """`conversions` with each source pattern read once: the plan's formats may emit the same conversion, but not two
    that read one pattern differently."""
    claimed, kept = {}, []
    for conversion in conversions:
        owner = next((claimed[p] for p in conversion.source_patterns if p in claimed), None)
        if owner is None:
            kept.append(conversion)
            claimed.update(dict.fromkeys(conversion.source_patterns, conversion))
        elif _signature(owner) != _signature(conversion):
            raise ValueError(
                f"The plan's formats read {conversion.source_patterns} differently, so they cannot be loaded together."
            )
    return kept


# an intra-expert style of a projection -> the style its row-aligned companions (bias, scale grid) take
_EXPERT_ROW_STYLES = {
    "packed_colwise": "moe_experts_packed_colwise",
    "colwise": "moe_experts_colwise",
    "rowwise": "moe_experts_rowwise",
    "packed_rowwise": "moe_experts_rowwise",
}


def _mask_expert_parallelism(model, plan):
    """Run `ep_dispatch_experts` as masked expert parallelism (each rank's experts masked to its own, an all-reduce
    keeping token shapes fixed) until the expert-parallel strategies dispatch the quantized experts themselves."""
    from ..distributed.tensor_parallel import replace_layer_number_by_wildcard

    if "ep_dispatch_experts" not in plan.values():
        return
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


def _shard_expert_companions(model, plan, original):
    """Under intra-expert TP, shard each projection's row-aligned companions (a scale grid, a gate|up bias) with its
    rows and keep the rest whole; entries the `original` plan set stay."""
    from ..distributed.tensor_parallel import _get_parameter_plan, replace_layer_number_by_wildcard
    from ..integrations.finegrained import FineGrainedExperts

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
            rows = _EXPERT_ROW_STYLES.get(_get_parameter_plan(f"{name}.{projection}", plan))
            if rows is None:
                continue
            for suffix in ("_bias", *STACK_COMPANIONS):
                companion = key + suffix
                if companion in original or not hasattr(module, projection + suffix):
                    continue
                if (suffix == "_scale_inv" and blocked) or (suffix == "_bias" and projection != "down_proj"):
                    plan[companion] = rows
                else:
                    plan.pop(companion, None)


def _has_scale_grid(config) -> bool:
    """Whether a weight in `config`'s format carries a scale grid rather than one per-tensor scale."""
    from ..integrations.finegrained.layers import resolve_weight_format

    return resolve_weight_format(config.method.value, config.scale_fmt, config.weight_block_size)[2] is not None


def _signature(conversion) -> tuple:
    """What a converter does: which keys it reads, where they go and through which ops."""
    operations = tuple(type(op).__name__ for op in getattr(conversion, "operations", ()))
    return type(conversion), tuple(conversion.source_patterns), tuple(conversion.target_patterns), operations
