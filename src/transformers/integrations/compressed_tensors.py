# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"CompressedTensors integration file"

import torch
from torch import nn

from ..core_model_loading import ConversionOps, _IdentityOp
from ..utils import logging
from ..utils.import_utils import is_torch_greater_or_equal


logger = logging.get_logger(__name__)


_EXPERT_COMPONENTS_BY_FORMAT = {
    "float-quantized": ("weight", "weight_scale"),
    "pack-quantized": ("weight_packed", "weight_scale", "weight_shape"),
    "nvfp4-pack-quantized": (
        "weight_packed",
        "weight_scale",
        "weight_global_scale",
        "input_global_scale",
    ),
}


def get_expert_components(scheme):
    """Return the checkpoint components used by an expert quantization scheme."""
    from compressed_tensors.compressors.format import infer_module_format

    format = scheme.format or infer_module_format(nn.Linear, scheme)
    return _EXPERT_COMPONENTS_BY_FORMAT.get(getattr(format, "value", format), ())


def get_scheme(quantization_config, module_name):
    """Return the quantization scheme matching a concrete module name."""
    from compressed_tensors.utils.match import match_name

    groups = list(quantization_config.config_groups.values())
    for group in groups:
        if any(target != "Linear" and match_name(module_name, target) for target in group.targets):
            return group
    # The concrete expert projections represented by checkpoint names are Linear modules.
    return next((group for group in groups if "Linear" in group.targets), None)


def get_experts_scheme(quantization_config):
    """Resolve which config group quantizes the MoE experts. Mixed configs quantize
    different layers with different schemes (e.g. Kimi: FP8 attention + INT4 experts), so
    the experts' scheme cannot be assumed global: the experts' group is the one whose
    `re:` targets mention the expert modules.
    """
    groups = list(quantization_config.config_groups.values())
    for group in groups:
        if any(target.startswith("re:") and "experts" in target for target in group.targets):
            return group
    # Class-name targets (e.g. "Linear"): expert projections are Linear submodules in
    # the checkpoint layout, so the first such group covers them.
    for group in groups:
        if any(not target.startswith("re:") for target in group.targets):
            return group
    return groups[0]


class DecompressExperts(ConversionOps):
    """
    Dequantize MoE layers when they are in new layout, because they aren't `nn.Module` anymore!

    Collects each expert's compressed weight and format-specific metadata, then uses the
    matching compressed-tensors compressor to restore dense weights.

    Requires MoE conversion to be defined on conversion mapping, so that decompressed weights
    are stacked/merged for all experts.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        from compressed_tensors.compressors import BaseCompressor
        from compressed_tensors.compressors.format import infer_module_format

        ct_quantization_config = self.hf_quantizer.compressor.quantization_config
        model = kwargs.get("model")
        target_dtype = model.get_parameter(full_layer_name).dtype if model is not None else None
        processed_out = {}
        for key, value in input_dict.items():
            if not key.endswith((".weight$", ".weight_packed$")):
                continue

            packed = key.endswith(".weight_packed$")
            weight_suffix = ".weight_packed$" if packed else ".weight$"
            base_key = key.removesuffix(weight_suffix)
            projection = base_key.rsplit(".", 1)[-1]
            expert_parent = full_layer_name.rsplit(".", 1)[0]
            checkpoint_module_name = f"{expert_parent}.0.{projection}"
            quantization_scheme = get_scheme(ct_quantization_config, checkpoint_module_name)
            if quantization_scheme is None:
                processed_out[key] = value
                continue

            format = quantization_scheme.format or infer_module_format(nn.Linear, quantization_scheme)
            compressor = BaseCompressor.get_value_from_registry(format)
            quantized = value if isinstance(value, list) else [value]
            companion_names = get_expert_components(quantization_scheme)[1:]
            companions = {
                name: input_dict[f"{base_key}.{name}$"]
                for name in companion_names
                if f"{base_key}.{name}$" in input_dict
            }
            if not packed and "weight_scale" not in companions:
                # An ignored expert projection stays dense and carries no quantization
                # metadata. It still matches the plain ``weight`` conversion source.
                processed_out[key] = value
                continue

            output = None
            for i, quant in enumerate(quantized):
                state_dict = {"weight_packed" if packed else "weight": quant}
                for name, tensors in companions.items():
                    state_dict[name] = tensors[i] if isinstance(tensors, list) else tensors

                # Prefer the checkpoint's stored per-expert `weight_shape`
                # (out-dim, in-dim); it is exact for any bit width. Fall back to
                # rebuilding from the packed tensor only when it is missing/empty,
                # e.g. left empty on most ranks under TP/EP sharding.
                if (
                    packed
                    and ("weight_shape" not in state_dict or not state_dict["weight_shape"].numel())
                    and quantization_scheme.weights.type == "int"
                ):
                    pack_factor = 32 // quantization_scheme.weights.num_bits
                    state_dict["weight_shape"] = torch.tensor(
                        [quant.shape[0], quant.shape[1] * pack_factor], device=quant.device
                    )
                dense = compressor.decompress(state_dict, quantization_scheme)["weight"]
                if target_dtype is not None:
                    # The compressor has no output-dtype argument and returns a format-specific
                    # dtype. The loader installs converted tensors directly, so match the model.
                    dense = dense.to(target_dtype)

                if output is None:
                    # Allocate the final stack directly to avoid retaining each dense expert and
                    # copying them all again in the following `MergeModulelist` operation.
                    output = torch.empty(
                        (len(quantized), *dense.shape),
                        dtype=dense.dtype,
                        device=dense.device,
                    )
                output[i].copy_(dense)
                # Release this dense tensor before decompressing the next expert.
                del dense

            if output is not None:
                processed_out[key] = output

        return processed_out

    @property
    def reverse_op(self) -> "ConversionOps":
        return _IdentityOp()


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


def quantize_fp8_per_row(x: torch.Tensor):
    """Quantize a 2D tensor to FP8 per-row using pure PyTorch (torch.compile compatible).

    Args:
        x: Input tensor of shape (num_rows, hidden_dim).

    Returns:
        Tuple of (x_fp8, scales) where x_fp8 is the quantized tensor and
        scales is a 1D tensor of shape (num_rows,) with per-row scales.
    """
    x_float = x.to(torch.float32)
    row_max = x_float.abs().amax(dim=-1)  # (num_rows,)
    # Avoid division by zero for all-zero rows
    safe_max = torch.where(row_max > 0, row_max, torch.ones_like(row_max))
    scales = safe_max / _FP8_MAX  # float32
    x_fp8 = torch.clamp(x_float / scales.unsqueeze(-1), min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)
    return x_fp8, scales


def _scaled_mm_rowwise(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-wise scaled FP8 matmul dispatcher that uses the public `torch.nn.functional.scaled_mm`
    (torch >= 2.10) if available, else falls back to `torch._scaled_mm`.

    Args:
        input (`torch.Tensor`): FP8 input of shape (M, K), scaled per-row.
        weight (`torch.Tensor`): FP8 weight of shape (K, N), scaled per-column.
        scale_a (`torch.Tensor`): Input scales of shape (M, 1), float32.
        scale_b (`torch.Tensor`): Weight scales of shape (1, N), float32.
    Returns:
        `torch.Tensor`: Output tensor of shape (M, N) in `out_dtype`.
    """
    # `torch.nn.functional.scaled_mm` is the public API for scaled matmuls, added in torch 2.10.
    if is_torch_greater_or_equal("2.10", accept_dev=True):
        ScalingType = torch.nn.functional.ScalingType
        return torch.nn.functional.scaled_mm(
            input,
            weight,
            scale_a,
            ScalingType.RowWise,
            scale_b,
            ScalingType.RowWise,
            bias=bias,
            output_dtype=out_dtype,
        )
    return torch._scaled_mm(input, weight, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype, bias=bias)


class CompressedTensorsFP8Linear(nn.Linear):
    """Linear layer for compressed-tensors FP8 models.

    Stores weights in FP8 format and uses a row-wise scaled matmul (`_scaled_mm_rowwise`) for FP8 matmul.
    Activations are always dynamically quantized per-row via ``quantize_fp8_per_row``,
    which works for both ``dynamic`` and ``static`` checkpoints (the checkpoint's
    static input scale, if any, is simply not needed). The weight scale (per-channel
    or per-tensor) is stored as ``weight_scale``.
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        super().__init__(in_features, out_features)

        self.has_bias = has_bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=_FP8_DTYPE))
        self.weight_scale = nn.Parameter(torch.zeros((1, out_features), dtype=torch.float32))

        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Save shape for restoring after squashing batch dims, then flatten once
        # (the kernel operates on a 2D tensor).
        output_shape = (*input.shape[:-1], -1)
        x = input.reshape(-1, input.shape[-1])

        x_quantized, x_scale = quantize_fp8_per_row(x)

        output = _scaled_mm_rowwise(
            x_quantized,
            self.weight.t(),
            scale_a=x_scale.unsqueeze(-1),
            scale_b=self.weight_scale,
            out_dtype=input.dtype,
            bias=self.bias,
        )
        return output.reshape(output_shape)


def replace_with_compressed_tensors_fp8_linear(model, targets=None, ignore=None, modules_to_not_convert=None):
    """Replace nn.Linear modules with CompressedTensorsFP8Linear for compressed-tensors FP8 loading.

    Quantization may only cover part of the model: `targets` / `ignore` are the
    compressed-tensors patterns (class names or `re:` regexes) from the config — only
    matching modules are replaced, everything else keeps its dense weights.
    `modules_to_not_convert` holds additional transformers-side exclusions
    (e.g. `_keep_in_fp32_modules`).
    """
    from compressed_tensors.utils.match import match_named_modules

    # Imported here to avoid a circular import (quantizers/__init__ imports the
    # compressed-tensors quantizer, which imports this module).
    from ..quantizers.quantizers_utils import should_convert_module

    matched = None
    if targets is not None:
        matched = {name for name, _ in match_named_modules(model, targets, ignore=ignore)}

    replaced_names = []
    for module_name, module in model.named_modules():
        if matched is not None and module_name not in matched:
            continue
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        if isinstance(module, nn.Linear):
            # This method already runs under a `torch.device("meta")` context manager,
            # so the new module is created on meta automatically.
            new_module = CompressedTensorsFP8Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                has_bias=module.bias is not None,
            )
            model.set_submodule(module_name, new_module)
            replaced_names.append(module_name)

    if not replaced_names:
        logger.warning(
            "You are loading your model using compressed-tensors FP8 but no linear modules were found. "
            "Please double check your model architecture."
        )
    return replaced_names


def _get_fp8_linear(model, full_layer_name):
    """Return the `CompressedTensorsFP8Linear` owning `full_layer_name` (a `weight_scale` key),
    or None if the key belongs to another kind of module (e.g. a non-FP8 group of a mixed config)."""
    module = model.get_submodule(full_layer_name.rsplit(".", 1)[0])
    return module if isinstance(module, CompressedTensorsFP8Linear) else None


class ConvertFP8LinearScale(ConversionOps):
    """Reshape the checkpoint ``weight_scale`` into the row-wise kernel layout at load time.

    compressed-tensors checkpoints store the dequantization scale as ``(out_features, 1)``
    (channel strategy) or a single element (tensor strategy). The row-wise scaled matmul in
    :class:`CompressedTensorsFP8Linear` needs a contiguous float32 ``(1, out_features)`` scale;
    converting once at load time avoids a transpose + expand + copy on every forward call.
    Scales of modules that were not replaced with :class:`CompressedTensorsFP8Linear` pass
    through untouched.
    """

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs):
        key, scale = next(iter(input_dict.items()))
        scale = scale[0] if isinstance(scale, list) else scale
        module = _get_fp8_linear(model, full_layer_name)
        if module is None:
            return {key: scale}
        scale = scale.to(torch.float32).reshape(1, -1)
        if scale.shape[1] != module.out_features:
            # Tensor strategy: a single scale for all rows. The row-wise kernel cannot mix a
            # row-wise input scale with a tensor-wise weight scale, so expand it once here.
            scale = scale.expand(1, module.out_features).contiguous()
        return {key: scale}

    @property
    def reverse_op(self) -> ConversionOps:
        return RevertFP8LinearScale()


class RevertFP8LinearScale(ConversionOps):
    """Save-time inverse of :class:`ConvertFP8LinearScale`: restore the checkpoint layout,
    ``(out_features, 1)`` for the channel strategy and ``(1, 1)`` for the tensor strategy.
    The strategy is read from the config group targeting the module, so the saved checkpoint
    stays loadable by other compressed-tensors consumers (llm-compressor, vLLM)."""

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs):
        from compressed_tensors.utils.match import is_match

        key, scale = next(iter(input_dict.items()))
        scale = scale[0] if isinstance(scale, list) else scale
        module = _get_fp8_linear(model, full_layer_name)
        if module is None:
            return {key: scale}

        module_name = full_layer_name.rsplit(".", 1)[0]
        ct_config = model.config.quantization_config.quantization_config
        strategy = None
        for group in ct_config.config_groups.values():
            if group.weights is not None and is_match(module_name, module, group.targets, ct_config.ignore or ()):
                strategy = group.weights.strategy
                break
        if strategy == "tensor":
            # The scale was expanded from a single element at load time; collapse it back.
            return {key: scale[:, :1].clone()}
        return {key: scale.t().contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        return ConvertFP8LinearScale()
