# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import re
from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from ..utils import is_compressed_tensors_available, is_torch_available, is_torch_greater_or_equal, logging
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


if TYPE_CHECKING:
    from ..core_model_loading import WeightConverter, WeightRenaming
    from ..modeling_utils import PreTrainedModel


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter
    from ..integrations.compressed_tensors import CompressedTensorsDequantize

    class CompressedTensorsWeightConverter(WeightConverter):
        """Subclass of WeightConverter for compressed-tensors.

        When a model is dequantized into standard unquantized parameters in memory,
        reverting the conversion on save should reverse only the base transformations
        (e.g. un-fusing gate_up_proj -> gate_proj.weight, up_proj.weight) without
        emitting companion serialized patterns (_packed, _scale, _shape).
        """

        def __init__(self, source_patterns, target_patterns, operations, base_source_patterns=None, **kwargs):
            super().__init__(source_patterns, target_patterns, operations, **kwargs)
            self.base_source_patterns = (
                base_source_patterns if base_source_patterns is not None else self._original_source_patterns
            )

        def copy(self, source_patterns=None, target_patterns=None, **kwargs):
            kwargs.setdefault("base_source_patterns", self.base_source_patterns)
            return super().copy(source_patterns, target_patterns, **kwargs)

        def reverse_transform(self):
            from ..core_model_loading import _IdentityOp

            rev_ops = []
            for op in self.operations[::-1]:
                rop = op.reverse_op
                if not isinstance(rop, _IdentityOp):
                    rev_ops.append(rop)
            if not rev_ops:
                rev_ops = [_IdentityOp()]

            base = (
                self.base_source_patterns if self.base_source_patterns is not None else self._original_source_patterns
            )
            rev_conv = WeightConverter(
                source_patterns=self._original_target_patterns,
                target_patterns=base,
                operations=rev_ops,
                force_cpu=self.force_cpu,
            )
            rev_conv.scope_prefix = self.scope_prefix
            rev_conv.base_model_prefix = self.base_model_prefix
            return rev_conv
else:
    CompressedTensorsWeightConverter = None


logger = logging.get_logger(__name__)


def _is_fp8_scheme(scheme) -> bool:
    """Whether a compressed-tensors quantization scheme quantizes weights to FP8."""
    weights = scheme.weights
    return weights is not None and weights.type == "float" and weights.num_bits == 8


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package. Loads and restores models to
    quantized state with compressed_tensors.

    With `use_optimized_inference=True`, FP8 checkpoints are kept in FP8 and their matmuls run through
    row-wise FP8 kernels (`torch.nn.functional.scaled_mm`) via `CompressedTensorsFP8Linear`, when FP8 matmul
    hardware is available (CUDA SM89+, XPU, or MPS with torch >= 2.15). This is opt-in and inference only.

    Otherwise the model goes through the regular compressed-tensors route: `dequantize=True`
    dequantizes the weights at load time, while `dequantize=False` leaves them compressed and lets
    compressed-tensors decompress them on the first forward pass.
    """

    requires_calibration = True
    quantization_config: CompressedTensorsConfig

    def __init__(self, quantization_config: CompressedTensorsConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.use_fp8_kernel = False
        # Call post_init here to ensure proper config setup when `dequantize` /
        # `use_optimized_inference` are provided directly via CompressedTensorsConfig, and to avoid
        # duplicate logging.

        quantization_config.post_init()
        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires compressed-tensors>=0.15.0: "
                "`pip install compressed-tensors>=0.15.0`"
            )

        # `use_optimized_inference` has been resolved against the checkpoint by
        # `CompressedTensorsConfig.post_init`; what is left to check is whether the hardware can run
        # the kernels. When it cannot, the model goes through the regular compressed-tensors route.
        self.use_fp8_kernel = self.quantization_config.use_optimized_inference
        # MPS has FP8 tensors and a `torch._scaled_mm_v2` kernel from torch 2.15 on.
        has_fp8_kernel = (
            torch.xpu.is_available()
            or (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9))
            or (torch.backends.mps.is_available() and is_torch_greater_or_equal("2.15", accept_dev=True))
        )
        if self.use_fp8_kernel and not has_fp8_kernel:
            logger.warning_once(
                "Ignoring `use_optimized_inference=True`: FP8 matmul kernels need a CUDA GPU with compute capability "
                ">= 8.9 (e.g. 4090/H100), an Intel XPU, or Apple silicon (MPS) with torch >= 2.15, and none was found."
            )
            self.use_fp8_kernel = False

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self.quantization_config.dequantize:
            # The model stays dense: the `dequantize` load operation runs on every checkpoint weight
            return
        from compressed_tensors.quantization import apply_quantization_config

        ct_config = deepcopy(self.compressor.quantization_config)
        if self.use_fp8_kernel:
            from ..integrations.compressed_tensors import replace_with_compressed_tensors_fp8_linear

            # Quantization may only target a subset of the layers: each config group scopes
            # its scheme with `targets` (class names or `re:` regexes) minus `ignore`.
            fp8_groups = {name: group for name, group in ct_config.config_groups.items() if _is_fp8_scheme(group)}
            ct_config.config_groups = {
                name: group for name, group in ct_config.config_groups.items() if name not in fp8_groups
            }
            self.modules_to_not_convert = self.get_modules_to_not_convert(model, None, model._keep_in_fp32_modules)
            replace_with_compressed_tensors_fp8_linear(
                model,
                targets=[target for group in fp8_groups.values() for target in group.targets],
                ignore=ct_config.ignore,
                modules_to_not_convert=self.modules_to_not_convert,
            )

        apply_quantization_config(model, ct_config, run_compressed=False)
        # Packed formats (e.g. int4 `weight_packed`) need the compressed module layout to
        # receive the checkpoint tensors.
        if self.quantization_config.is_quantization_compressed:
            self.compressor.compress_model(model=model)

    @property
    def is_trainable(self):
        # The FP8 kernel path is inference-only; load with `dequantize=True` to fine-tune.
        return not self.use_fp8_kernel

    @property
    def is_compileable(self) -> bool:
        return True

    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        if self.use_fp8_kernel:
            return False
        # models need to be decompressed carry out qat
        return self.quantization_config.dequantize or not self.quantization_config.is_quantization_compressed

    def is_serializable(self) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True

    def get_weight_conversions(self):
        """On the FP8 kernel path, a generic converter reshapes the checkpoint ``weight_scale``
        tensors into the row-wise kernel layout once at load time — see
        :class:`ConvertFP8LinearScale`."""
        if not self.use_fp8_kernel:
            return super().get_weight_conversions()

        from ..integrations.compressed_tensors import ConvertFP8LinearScale

        return [
            WeightConverter(
                source_patterns=["weight_scale"],
                target_patterns=["weight_scale"],
                operations=[ConvertFP8LinearScale()],
            )
        ]

    # Every name a compressed-tensors format stores for a weight; a checkpoint only has its own format's
    serialized_names = {
        "weight": [
            "weight",
            "weight_scale",
            "weight_zero_point",
            "weight_global_scale",
            "weight_packed",
            "weight_shape",
        ]
    }

    def is_quantized_converter(
        self,
        converter: "WeightConverter",
        model: Optional["PreTrainedModel"] = None,
        checkpoint_keys: Iterable[str] | None = None,
        renamings: list["WeightRenaming"] | None = None,
    ) -> bool:
        if checkpoint_keys is not None:
            return super().is_quantized_converter(
                converter, model=model, checkpoint_keys=checkpoint_keys, renamings=renamings
            )

        compressor = getattr(self, "compressor", None)
        ct_quant_config = getattr(compressor, "quantization_config", None)
        if ct_quant_config is None:
            return True
        config_groups = getattr(ct_quant_config, "config_groups", None)
        if not config_groups:
            return getattr(self.quantization_config, "is_quantization_compressed", False)

        patterns_to_check = list(converter._original_source_patterns) + list(converter._original_target_patterns)
        for group in config_groups.values():
            targets = getattr(group, "targets", None) or []
            ignore = getattr(group, "ignore", None) or []
            targets = [targets] if isinstance(targets, str) else targets
            ignore = [ignore] if isinstance(ignore, str) else ignore

            if not targets:
                continue

            for target in targets:
                is_regex = target.startswith("re:")
                t_pat = target.removeprefix("re:") if is_regex else target

                for p in patterns_to_check:
                    p_sample = p.replace("*", "0").removesuffix(".weight")
                    samples = [p_sample, f".{p_sample}", f"model.{p_sample}", p.replace("*", "0")]
                    if is_regex:
                        if any(re.search(t_pat, s) for s in samples):
                            if not any(
                                (
                                    re.search(ign.removeprefix("re:"), p_sample)
                                    if ign.startswith("re:")
                                    else ign in p_sample
                                )
                                for ign in ignore
                            ):
                                return True
                    else:
                        if any(t_pat in s or t_pat == s for s in samples):
                            if not any(ign in p_sample for ign in ignore):
                                return True

                if model is not None:
                    for name, mod in model.named_modules():
                        mod_class = type(mod).__name__
                        matches_target = False
                        if is_regex:
                            if re.search(t_pat, name) or re.search(t_pat, f"{name}.0"):
                                matches_target = True
                        else:
                            if t_pat == name or t_pat == mod_class:
                                matches_target = True

                        if matches_target:
                            is_ignored = False
                            for ign in ignore:
                                if ign.startswith("re:") and (
                                    re.search(ign.removeprefix("re:"), name)
                                    or re.search(ign.removeprefix("re:"), f"{name}.0")
                                ):
                                    is_ignored = True
                                    break
                                elif not ign.startswith("re:") and (ign == name or ign == mod_class):
                                    is_ignored = True
                                    break
                            if not is_ignored:
                                for tp in converter._original_target_patterns:
                                    clean_tp = tp.removesuffix(".weight").replace("*", "")
                                    if clean_tp in name or any(part in name for part in clean_tp.split(".") if part):
                                        return True
        return False

    def get_load_operations(self):
        # No `deserialize`: the serialized tensors load as is into the params compressed-tensors creates
        return {"dequantize": CompressedTensorsDequantize}

    def update_weight_conversions(self, weight_conversions, model, checkpoint_keys=None):
        updated = super().update_weight_conversions(weight_conversions, model, checkpoint_keys=checkpoint_keys)
        if not is_torch_available() or CompressedTensorsWeightConverter is None:
            return updated

        for i, conv in enumerate(updated):
            if isinstance(conv, WeightConverter) and any(
                op.__class__.__name__ == "CompressedTensorsDequantize" for op in conv.operations
            ):
                orig = next(
                    (
                        c
                        for c in weight_conversions
                        if isinstance(c, WeightConverter)
                        and c.target_patterns == conv.target_patterns
                        and getattr(c, "scope_prefix", None) == getattr(conv, "scope_prefix", None)
                    ),
                    None,
                )
                base_sources = orig.source_patterns if orig is not None else conv.source_patterns
                new_conv = CompressedTensorsWeightConverter(
                    source_patterns=conv.source_patterns,
                    target_patterns=conv.target_patterns,
                    operations=conv.operations,
                    force_cpu=conv.force_cpu,
                    base_source_patterns=base_sources,
                )
                new_conv.scope_prefix = conv.scope_prefix
                new_conv.base_model_prefix = conv.base_model_prefix
                updated[i] = new_conv
        return updated
