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


# compressed-tensors formats whose MoE experts can run without ever being decompressed, mapped to the
# experts implementation that runs them.
PACKED_EXPERTS_FORMATS = {
    "mxfp4-pack-quantized": "mxfp4",
    "nvfp4-pack-quantized": "nvfp4",
}


def get_packed_experts_format(quantization_config) -> str | None:
    """The packed-FP4 format the MoE experts can run in, or `None` for every other scheme.

    Only the formats in `PACKED_EXPERTS_FORMATS` have a runtime that reads their packed bytes; anything
    else (FP8, int4 `pack-quantized`, …) still has to be decompressed at load time.
    """
    scheme = get_experts_scheme(quantization_config)
    weights = getattr(scheme, "weights", None)
    # `weight_zero_point` has no place in the packed layout the runtimes read.
    if weights is None or not getattr(weights, "symmetric", True):
        return None
    # Published checkpoints often leave the per-group `format` out and only set the checkpoint-level one.
    format = scheme.format or quantization_config.format
    format = getattr(format, "value", format)
    return format if format in PACKED_EXPERTS_FORMATS else None


def is_packed_experts_module(module) -> bool:
    """Whether `module` is a fused-experts module whose weights can be swapped for packed ones.

    The projections are read off the module itself rather than from config fields, so latent MoEs whose
    experts do not run in `config.hidden_size` work too. `has_gate` is set by the `use_experts_implementation`
    decorator, and its absence means the module has no dispatch hook to route to the packed runtime.
    """
    return (
        hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
        and hasattr(module, "has_gate")
        and hasattr(module, "num_experts")
        and not getattr(module, "is_transposed", False)
    )


def _broadcast_global_scales(global_scales: dict, input_dict: dict, sources: list[str]) -> dict:
    """Give each per-projection NVFP4 global scale one row per output channel.

    The checkpoint stores it as a single number per expert projection, which the merge operations cannot
    concatenate along the output dimension the way they concatenate weights and group scales. Expanding
    it to `(out_dim, 1)` first — a handful of floats per expert — lets it ride along unchanged.
    """
    broadcast = {}
    for source in sources:
        packed_source = source.replace("weight_global_scale", "weight_packed")
        broadcast[source] = [
            scale.reshape(1, 1).expand(packed.shape[0], 1).contiguous()
            for scale, packed in zip(global_scales[source], input_dict[packed_source])
        ]
    return broadcast


class LoadPackedExperts(ConversionOps):
    """Load packed-FP4 MoE expert weights without decompressing them.

    The checkpoint holds one `weight_packed` / `weight_scale` pair per expert projection, plus a
    `weight_global_scale` for NVFP4. Each of those goes through the model's own expert-merging operations
    separately, so that a `Concatenate` fusing the gate and up projections does not also try to fuse
    weights with scales, and the merged stacks are then attached to the experts module.

    Nothing is handed back to the loader: like the GPT-OSS mxfp4 path, the module is updated in place,
    because a packed weight is not the `nn.Parameter` the module declared.
    """

    def __init__(self, hf_quantizer, scheme, format: str, operations: list[ConversionOps]):
        self.hf_quantizer = hf_quantizer
        self.scheme = scheme
        self.format = format
        # The model's own converter ops (e.g. `MergeModulelist` + `Concatenate`), replayed per component.
        self.operations = operations

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        model=None,
        full_layer_name: str | None = None,
        missing_keys: set[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        from ..quantizers.quantizers_utils import get_module_from_name

        merged = {}
        for component in ("weight_packed", "weight_scale", "weight_global_scale"):
            component_sources = [p for p in source_patterns if p.rstrip("$").endswith(component)]
            component_dict = {p: input_dict[p] for p in component_sources if p in input_dict}
            if not component_dict:
                continue
            if component == "weight_global_scale":
                component_dict = _broadcast_global_scales(component_dict, input_dict, component_sources)
            for operation in self.operations:
                component_dict = operation.convert(
                    component_dict,
                    source_patterns=component_sources,
                    target_patterns=target_patterns,
                    full_layer_name=full_layer_name,
                    model=model,
                    **kwargs,
                )
            merged[component] = next(iter(component_dict.values()))

        module, proj = get_module_from_name(model, full_layer_name)
        if self.format == "mxfp4-pack-quantized":
            from ..integrations.mxfp4 import attach_packed_mxfp4_proj

            attach_packed_mxfp4_proj(module, proj, merged["weight_packed"], merged["weight_scale"])
        else:
            from ..integrations.nvfp4 import attach_packed_nvfp4_proj

            if "weight_global_scale" not in merged:
                raise ValueError(
                    f"`{full_layer_name}` is quantized to {self.format} but the checkpoint has no "
                    "`weight_global_scale` for it, without which its weights cannot be reconstructed."
                )
            attach_packed_nvfp4_proj(
                module, proj, merged["weight_packed"], merged["weight_scale"], merged["weight_global_scale"]
            )

        if missing_keys is not None:
            missing_keys.discard(full_layer_name)
        module._is_hf_initialized = True
        return {}

    @property
    def reverse_op(self) -> "ConversionOps":
        return _IdentityOp()


class DecompressExperts(ConversionOps):
    """
    Dequantize MoE layers when they are in new layout, because they aren't `nn.Module` anymore!

    Takes packed weights and scales from the loaded state dict, creates a dummy Module
    to take advantage of higher-lvl API `decompress_module` and dequantizes all weights.

    Requires MoE conversion to be defined on conversion mapping, so that decompressed weights
    are stacked/merged for all experts.
    """

    def __init__(self, hf_quantizer, scheme=None):
        self.hf_quantizer = hf_quantizer
        # The config group quantizing these experts; mixed configs (e.g. Kimi: FP8
        # attention + INT4 experts) quantize different layers with different schemes.
        self.scheme = scheme

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

        quantization_scheme = self.scheme or get_experts_scheme(ct_quantization_config)

        weights_args = quantization_scheme.weights
        if weights_args.type == "float" and weights_args.num_bits == 8:
            # FP8 experts are plain (fp8 weight, scale) pairs — nothing is packed. Fold the
            # scale into each per-expert weight so the downstream merge sees dense tensors.
            processed_out = {}
            for key, value in input_dict.items():
                if "weight_scale" in key:
                    continue
                scale_key = key.replace(".weight", ".weight_scale")
                if scale_key not in input_dict:
                    # Weights that ship no scale (e.g. norms) pass through untouched.
                    processed_out[key] = value
                    continue
                quantized = value if isinstance(value, list) else [value]
                scales = input_dict[scale_key]
                scales = scales if isinstance(scales, list) else [scales]
                output = None
                for i, (quant, scale) in enumerate(zip(quantized, scales)):
                    dense = (quant.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
                    if output is None:
                        # Pre-allocate the stacked buffer (see the packed path below).
                        output = torch.empty((len(quantized), *dense.shape), dtype=dense.dtype, device=dense.device)
                    output[i].copy_(dense)
                    del dense
                if output is not None:
                    processed_out[key] = output
            return processed_out

        format = quantization_scheme.format or infer_module_format(nn.Linear, quantization_scheme)
        compressor = BaseCompressor.get_value_from_registry(format)

        class DummyModule(nn.Module):
            def __init__(self, weight, scale, shape, global_scale=None):
                super().__init__()
                self.weight_packed = nn.Parameter(weight, requires_grad=False)
                self.weight_scale = nn.Parameter(scale, requires_grad=False)
                self.weight_shape = nn.Parameter(shape, requires_grad=False)
                if global_scale is not None:
                    # NVFP4 divides its group scales by one more factor per tensor. Without it the
                    # decompressed weights come out scaled by several thousand.
                    self.weight_global_scale = nn.Parameter(global_scale, requires_grad=False)

        # `pack_factor` low-bit weights are packed per int32 along the packed dim.
        # Used only as a fallback for `weight_shape` (see below): rebuilding the
        # unpacked in-dim as `packed_cols * pack_factor` is exact only when
        # `num_bits` divides 32; for 3/5/6/7-bit dense packing it under-counts.
        pack_factor = 32 // quantization_scheme.weights.num_bits

        # Per-expert compressed projections of size (input-dim; output-dim)
        processed_out = {}
        for key, value in input_dict.items():
            if "weight_packed" not in key:
                continue
            quantized = value
            scales = input_dict[key.replace("weight_packed", "weight_scale")]
            shapes = input_dict.get(key.replace("weight_packed", "weight_shape"))
            global_scales = input_dict.get(key.replace("weight_packed", "weight_global_scale"))

            # Pre-allocate the stacked output buffer to reduce cuda mem fragmentation
            # Without pre-allocation the loop accumulates N tensors per expert and next
            # `MergeModulelist` stacks the full list for MoE kernels compatibility, i.e. x2 memory
            output = None
            for i, (quant, scale) in enumerate(zip(quantized, scales)):
                # Prefer the checkpoint's stored per-expert `weight_shape`
                # (out-dim, in-dim); it is exact for any bit width. Fall back to
                # rebuilding from the packed tensor only when it is missing/empty,
                # e.g. left empty on most ranks under TP/EP sharding.
                stored_shape = None if shapes is None else shapes[i]
                if stored_shape is not None and stored_shape.numel():
                    shape = stored_shape
                else:
                    shape = torch.tensor([quant.shape[0], quant.shape[1] * pack_factor])
                module = DummyModule(quant, scale, shape, None if global_scales is None else global_scales[i])
                module.quantization_scheme = quantization_scheme
                compressor.decompress_module(module)

                if output is None:
                    # Use the first expert's decompressed shape/dtype to allocate full buffer.
                    output = torch.empty(
                        (len(quantized), *module.weight.shape),
                        dtype=module.weight.dtype,
                        device=module.weight.device,
                    )
                output[i].copy_(module.weight)
                # explicitly free intermediate tensors so it does not accumulate across iterations
                del module

            del quantized, scales
            if output is not None:
                # Return a single pre-stacked tensor instead of a list. `MergeModulelist`
                # passes it through without an extra `torch.stack` copy -> no x2 memory overhead
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
