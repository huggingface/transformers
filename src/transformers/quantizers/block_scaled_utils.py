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
"""Shared base of the block-scaled quantizers: block FP8, MXFP8, MXFP4 and NVFP4.

Each one quantizes a whole model on its own. `FineGrainedHfQuantizer` hosts several of them. Each runs on the modules
its `dtype_plan` gives it.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..utils import (
    is_accelerate_available,
    is_kernels_available,
    is_torch_available,
    is_torch_xpu_available,
    logging,
)
from ..utils.import_utils import KERNELS_MAX_VERSION, KERNELS_MIN_VERSION
from ..utils.quantization_config import Fp8Config
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

# the default up projection of an experts stack, used until the built modules are known
GATED_UP_PROJECTIONS = ("gate_up_proj",)


@dataclass(frozen=True)
class ProducerLayout:
    """Checkpoint layout of one producer, where it differs from the slots the modules hold."""

    # checkpoint name of the block scale when it is not `weight_scale_inv`, e.g. `weight_scale`
    block_scale_name: str | None = None
    # packed FP4 ships as uint8 but the modules hold int8, so the bytes need a view and not a copy
    packed_as_uint8: bool = False


class BlockScaledQuantizer(HfQuantizer):
    """Base quantizer of a block-scaled format.

    It is written over `plan()`, which maps the modules it covers to their format config.
    """

    requires_calibration = False
    dequantized_saves_unquantized = True
    quantization_config: "BlockScaledConfig | FineGrainedConfig"
    # checkpoint keys that producers ship and no module holds
    unexpected_keys: tuple[str, ...] = ()
    # producer layout per `checkpoint_format`. Unlisted formats use the module slots as is.
    producer_layouts: dict[str | None, ProducerLayout] = {}

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        # embedding tables to quantize, by name suffix. Each holds FP8 rows with one per-tensor scale.
        self.embeddings = list(getattr(quantization_config, "modules_to_convert", None) or [])
        self._embedding_config = Fp8Config(weight_block_size=None)

    def _embedding_plan(self) -> dict[str, "BlockScaledConfig"]:
        return dict.fromkeys(map(embedding_pattern, self.embeddings), self._embedding_config)

    def plan(self) -> dict[str, "BlockScaledConfig"]:
        """Return `{module regex: format config}` for the modules this quantizer covers."""
        return {**self._embedding_plan(), ".*": self.quantization_config}

    def members(self) -> list["BlockScaledQuantizer"]:
        """Return the quantizer of each format config in `plan()`."""
        return [self]

    def member_of(self, module) -> "BlockScaledQuantizer | None":
        """Return the quantizer whose plan entry `module` was built from, or `None` for an unquantized module."""
        return self if getattr(module, "quantization_config", None) is not None else None

    @property
    def producer_layout(self) -> ProducerLayout:
        """Return the checkpoint layout of this format's producer."""
        return self.producer_layouts.get(
            getattr(self.quantization_config, "checkpoint_format", None), ProducerLayout()
        )

    def _up_projections(self, model) -> tuple[str, ...]:
        """Return the up projections of the experts this format holds in `model`.

        It is empty when the format holds no experts. Without a model, it is the gated stack.
        """
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
        """Return the converters the producer key layouts need on top of the model plan.

        Under `dequantize`, return what folding the keys needs beyond the shared `weight` + `weight_scale_inv` chain.
        `up_projections` are the up projections of the experts this format holds. It is empty without experts.
        For example, static FP8 maps a checkpoint `input_scale` to each projection's `*_activation_scale`.
        """
        return []

    def format_weight_conversions(self, converters, model):
        """Adjust the model's companion converters after the base quantizer has cloned them."""

    def format_parallel_plan(self, model, plan):
        """Set this format's companion placements after the shared parallel plan is built."""

    def get_param_ops(self, model: "PreTrainedModel", param_name: str) -> list:
        """Return the layout ops a tensor runs when it loads into `param_name`. Each op is reversed on save.

        The ops are the gate|up interleave, the scale swizzle and the packed FP4 view.
        """
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

    def postprocess_model(self, model, **kwargs):
        if not self.pre_quantized and self.quantization_config.dequantize:
            # nothing was quantized, so the model loads as an unquantized one
            self.remove_quantization_config(model)
            return model
        return super().postprocess_model(model, **kwargs)

    def _process_model_after_weight_loading(self, model, **kwargs):
        from ..integrations.finegrained.layers import assert_modules_are_quantized, disable_deepgemm_on_multi_device

        disable_deepgemm_on_multi_device(model)
        assert_modules_are_quantized(model)
        if not self.pre_quantized and getattr(model, "_weight_conversions", None) is not None:
            # scales from on-the-fly quantization save through the conversions of their weights
            model._weight_conversions = self.update_weight_conversions(model._weight_conversions, model, for_save=True)
        return model

    @staticmethod
    def _per_expert_sources(proj: str, slot: str) -> list[str]:
        """Return the per-expert checkpoint keys of `slot` that stack into the experts `proj`.

        They match under any parent, such as `mlp.experts` or `mixer.experts`. The gate|up stack reads both halves.
        """
        halves = ("gate_proj", "up_proj") if proj == "gate_up_proj" else (proj,)
        return [f"experts.*.{half}.{slot}" for half in halves]

    def _block_scale_rename(self):
        """Rename the shipped block scale to `weight_scale_inv`.

        `FineGrainedEmbedding` keeps its per-tensor scale named `weight_scale`, so its tables are left out.
        """
        from ..core_model_loading import WeightRenaming

        kept = "".join(rf"(?<!{re.escape(name)})" for name in self.embeddings)
        shipped = self.producer_layout.block_scale_name
        return WeightRenaming(source_patterns=rf"^(.+){kept}\.{shipped}", target_patterns=r"\1.weight_scale_inv")

    def _activation_scale_conversions(self, up_projections, projections=FUSED_STACKS, ops=()) -> list:
        """Return the conversions that load a calibrated `input_scale` into the module `activation_scale`.

        The scale is renamed, then reduced to one per expert of `projections`, from either expert layout. `ops` run on
        each value first.
        """
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

    def _missing_requirement(self) -> str | None:
        """Return the first requirement of quantized compute that is missing, or `None`."""
        if not torch.cuda.is_available() and not is_torch_xpu_available():
            return "a GPU or XPU"
        if torch.cuda.is_available():
            # The kernels use Triton's FP8 type, which needs compute capability 8.9
            minimum = (8, 9)
            if torch.cuda.get_device_capability() < minimum:
                return f"a GPU with compute capability >= {minimum[0]}.{minimum[1]}"
        if not is_kernels_available():
            return f"the `kernels` package ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION})"
        return None

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading a fine-grained quantized model requires accelerate (`pip install accelerate`)")

        if self.quantization_config.dequantize:
            return

        if (missing := self._missing_requirement()) is not None:
            if not self.pre_quantized:
                raise RuntimeError(f"Fine-grained quantization requires {missing}.")
            logger.warning_once(f"Fine-grained quantized models require {missing}. Dequantizing the model to bf16.")
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

        if self.pre_quantized or self.quantization_config.dequantize:
            return False
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, FineGrainedExperts):
            return tensor_name in FUSED_STACKS
        return isinstance(module, FineGrainedLinear) and tensor_name == "weight"

    @property
    def quantized_element_size(self) -> float:
        # an FP4 weight is already `K // 2` bytes wide, so count one byte per held element
        return 1

    def get_new_dtype(self, converter, param, dtype):
        # keep calibrated scales in their fp32 checkpoint dtype, never the model dtype
        if any(name.endswith(("activation_scale", "_global_scale")) for name in converter.target_patterns):
            return None
        # a float scale takes the module UE8M0 dtype, e.g. the fp32 scales of DeepSeek-V4-Base
        # an integer scale keeps the checkpoint dtype
        is_float = (
            param.get_dtype().startswith(("F", "BF")) if hasattr(param, "get_dtype") else param.is_floating_point()
        )
        if dtype == getattr(torch, "float8_e8m0fnu", None) and is_float:
            return dtype
        return super().get_new_dtype(converter, param, dtype)

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        plan = self.plan()
        embeddings = set(map(embedding_pattern, self.embeddings))
        if not self.pre_quantized:
            # a dequantized on-the-fly load quantizes nothing. Embedding tables have no on-the-fly path.
            plan = (
                {} if self.quantization_config.dequantize else {p: c for p, c in plan.items() if p not in embeddings}
            )
        elif self.quantization_config.dequantize:
            # an FP8 embedding table is too big to expand, so it stays FP8
            plan = {pattern: config for pattern, config in plan.items() if pattern in embeddings}
        replaced = self.replace_modules(model, plan, self.modules_to_not_convert)

        ignored = set(model._keys_to_ignore_on_load_unexpected or [])
        for member in self.members():
            ignored.update(member.unexpected_keys)
        # without an activation scale in the model, a calibrated `input_scale` is unused but still valid
        if not any(name.endswith("activation_scale") for name, _ in model.named_parameters()):
            ignored.add(r"input(_global)?_scale$")
        model._keys_to_ignore_on_load_unexpected = ignored
        if not replaced and not self.quantization_config.dequantize:
            logger.warning(
                "You are loading a fine-grained quantized model but no linear or experts modules were "
                "found to convert. Please double check your model architecture."
            )

    def replace_modules(self, model, plan=None, modules_to_not_convert=None) -> bool:
        """Swap the modules `plan` covers for their quantized classes. `plan` defaults to this quantizer's own."""
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
        from ..integrations.finegrained import FineGrainedExperts

        original = dict(plan)
        super().update_parallel_plan(model, plan)
        # apply the experts implementation styles first, since the masked EP rewrite leaves them alone
        overwrite = FineGrainedExperts._impl_tp_layer_overrides.get(
            getattr(model.config, "_experts_implementation", None), {}
        )
        plan.update({key: overwrite.get(style, style) for key, style in plan.items()})
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
            # one chain per slot folds its scales in, each name matched whole
            # fused stacks go first because the dense names are suffixes of their companions
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

    def get_checkpoint_renamings(self):
        # rename a producer's block scale, e.g. `weight_scale`, onto its slot so it follows its weight's converter
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
        # a fused stack in the checkpoint loads through a per-param converter
        # on save, skip it for params a model converter writes back, since its full-name pattern would match first
        reverse = [converter.reverse_transform() for converter in converters] if for_save else []
        for name, _ in model.named_parameters():
            if self.get_param_ops(model, name) and not any(c.rename_source_key(name)[1] is not None for c in reverse):
                conversions.append(WeightConverter(re.escape(name), name, [FineGrainedDeserialize(self)]))
        return conversions


def _claimed(conversions) -> list:
    """Return `conversions` with each source pattern read once.

    The plan formats may emit the same conversion twice. Two conversions that read one pattern differently raise.
    """
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


# intra-expert style of a projection -> style of its row-aligned companions, such as the bias and scale grid
_EXPERT_ROW_STYLES = {
    "packed_colwise": "moe_experts_packed_colwise",
    "colwise": "moe_experts_colwise",
    "rowwise": "moe_experts_rowwise",
    # a rowwise split cuts K, not the packed gate|up rows
    "packed_rowwise": "moe_experts_rowwise",
}


def _mask_expert_parallelism(model, plan):
    """Run `ep_dispatch_experts` as masked expert parallelism.

    Each rank masks the experts it does not own. An all-reduce keeps token shapes fixed. This holds until the expert
    parallel strategies can dispatch quantized experts themselves.
    """
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
    """Shard the row-aligned companions of each projection with its rows under intra-expert TP.

    Row-aligned companions are the scale grid and the gate|up bias. Other companions stay whole. Entries set by the
    `original` plan are kept.
    """
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
        splits = {}
        for projection in (module.gate_up_name, "down_proj"):
            key = f"{glob}.{projection}"
            rows = _EXPERT_ROW_STYLES.get(_get_parameter_plan(f"{name}.{projection}", plan))
            if rows is None:
                continue
            splits[projection] = rows
            for suffix in ("_bias", *STACK_COMPANIONS):
                companion = key + suffix
                if companion in original or not hasattr(module, projection + suffix):
                    continue
                if (suffix == "_scale_inv" and blocked) or (suffix == "_bias" and projection != "down_proj"):
                    plan[companion] = rows
                else:
                    plan.pop(companion, None)
        if not _swizzled_grids_survive_split(model, name, module, splits):
            module.hold_row_major_scales()


def _swizzled_grids_survive_split(model, name, module, splits) -> bool:
    """Check whether the swizzled scale grids of `module` can still be read swizzled once intra-expert TP splits them.

    `splits` maps each projection to the style of its grid. A rowwise style splits the K columns, the others the rows.
    A grid that stops tiling loads row-major, which the kernels read at any shape.
    """
    from ..distributed.tensor_parallel import _get_parameter_plan
    from ..integrations.finegrained.layers import _tiles_swizzled

    distributed_config = getattr(model.config, "distributed_config", None)
    if distributed_config is None:
        return True
    # EP rules replace the TP ones for the experts they name, and EP keeps each expert whole
    if distributed_config.ep_size > 1 and _get_parameter_plan(name, model.ep_plan, is_weight=False) is not None:
        return True
    tp_size = distributed_config.tp_size
    for projection, style in splits.items():
        scale = getattr(module, f"{projection}_scale_inv")
        if scale.ndim != 5:
            continue
        rows, cols = scale.shape[1] * 128, scale.shape[2] * 4
        if style == "moe_experts_rowwise":
            cols, remainder = divmod(cols, tp_size)
        else:
            rows, remainder = divmod(rows, tp_size)
        if remainder or not _tiles_swizzled(rows, cols):
            return False
    return True


def _has_scale_grid(config) -> bool:
    """Return whether a weight in the `config` format has a scale grid rather than one per-tensor scale."""
    from ..integrations.finegrained.layers import resolve_weight_format

    return resolve_weight_format(config.method.value, config.scale_fmt, config.weight_block_size)[2] is not None


def _signature(conversion) -> tuple:
    """Return what a converter does: the keys it reads, where they go and the ops they run through."""
    operations = tuple(type(op).__name__ for op in getattr(conversion, "operations", ()))
    return type(conversion), tuple(conversion.source_patterns), tuple(conversion.target_patterns), operations
