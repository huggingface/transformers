from __future__ import annotations

from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)

PRISM_BITS = 1
PRISM_PACK_BITS = 32


def _packed_values_per_group(group_size: int) -> int:
    if group_size <= 0 or group_size % PRISM_PACK_BITS != 0:
        raise ValueError(f"Prism requires group_size to be a positive multiple of {PRISM_PACK_BITS}, got {group_size}")
    return group_size // PRISM_PACK_BITS


def _bit_shifts(device: "torch.device") -> "torch.Tensor":
    return torch.arange(PRISM_PACK_BITS, device=device, dtype=torch.int64)


def unpack_prism_weights(packed: "torch.Tensor") -> "torch.Tensor":
    shifts = _bit_shifts(packed.device)
    return ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 1).reshape(*packed.shape[:-1], -1)


def _select_compute_dtype(tensor: "torch.Tensor", fallback: "torch.dtype") -> "torch.dtype":
    if tensor.is_floating_point():
        if tensor.device.type == "cpu" and tensor.dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return tensor.dtype
    if tensor.device.type == "cpu" and fallback in {torch.float16, torch.bfloat16}:
        return torch.float32
    return fallback


class _PrismAffine1BitMixin:
    def __init__(self, group_size: int) -> None:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self.group_size = group_size
        self.packed_values_per_group = _packed_values_per_group(group_size)

    @property
    def num_groups(self) -> int:
        return self.scales.shape[1]

    def _validate_buffer_shapes(self) -> None:
        if self.weight.dtype != torch.uint32:
            raise ValueError(f"Prism packed weights must use torch.uint32, got {self.weight.dtype}")
        if self.scales.shape != self.biases.shape:
            raise ValueError(
                f"Prism scales and biases must have identical shapes, got {tuple(self.scales.shape)} and {tuple(self.biases.shape)}"
            )
        if self.weight.shape[-1] != self.num_groups * self.packed_values_per_group:
            raise ValueError(
                "Prism packed weight shape is incompatible with scales/biases: "
                f"packed_cols={self.weight.shape[-1]}, groups={self.num_groups}, "
                f"packed_per_group={self.packed_values_per_group}"
            )

    def dequantize_weight(self, dtype: "torch.dtype | None" = None) -> "torch.Tensor":
        self._validate_buffer_shapes()
        target_dtype = self.scales.dtype if dtype is None else dtype
        rows = self.weight.shape[0]
        out = torch.empty((rows, self.num_groups * self.group_size), device=self.weight.device, dtype=target_dtype)
        for group_idx in range(self.num_groups):
            packed_start = group_idx * self.packed_values_per_group
            packed_end = packed_start + self.packed_values_per_group
            col_start = group_idx * self.group_size
            col_end = col_start + self.group_size
            bits = unpack_prism_weights(self.weight[:, packed_start:packed_end]).to(target_dtype)
            scale = self.scales[:, group_idx : group_idx + 1].to(target_dtype)
            bias = self.biases[:, group_idx : group_idx + 1].to(target_dtype)
            out[:, col_start:col_end] = bits * scale + bias
        return out


class PrismLinear(nn.Module, _PrismAffine1BitMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        group_size: int = 128,
        device=None,
        dtype=None,
    ):
        nn.Module.__init__(self)
        _PrismAffine1BitMixin.__init__(self, group_size=group_size)
        if in_features % self.group_size != 0:
            raise ValueError(
                f"PrismLinear requires in_features divisible by group_size={self.group_size}, got {in_features}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.zeros(
                (out_features, (in_features // self.group_size) * self.packed_values_per_group),
                dtype=torch.uint32,
                device=device,
            ),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.zeros((out_features, in_features // self.group_size), dtype=dtype, device=device),
            requires_grad=False,
        )
        self.biases = nn.Parameter(
            torch.zeros((out_features, in_features // self.group_size), dtype=dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=dtype, device=device), requires_grad=False)
        else:
            self.bias = None
        self.requires_grad_(False)

    def forward(self, input: "torch.Tensor") -> "torch.Tensor":
        self._validate_buffer_shapes()
        original_shape = input.shape[:-1]
        flat_input = input.reshape(-1, self.in_features)
        compute_dtype = _select_compute_dtype(flat_input, self.scales.dtype)
        flat_input = flat_input.to(compute_dtype)
        output = torch.zeros((flat_input.shape[0], self.out_features), dtype=compute_dtype, device=flat_input.device)

        for group_idx in range(self.num_groups):
            col_start = group_idx * self.group_size
            col_end = col_start + self.group_size
            packed_start = group_idx * self.packed_values_per_group
            packed_end = packed_start + self.packed_values_per_group

            input_group = flat_input[:, col_start:col_end]
            packed_group = self.weight[:, packed_start:packed_end]
            bits = unpack_prism_weights(packed_group).to(compute_dtype)
            scale = self.scales[:, group_idx].to(compute_dtype)
            bias = self.biases[:, group_idx].to(compute_dtype)
            partial = input_group @ bits.transpose(0, 1)
            output += partial * scale.unsqueeze(0)
            output += input_group.sum(dim=-1, keepdim=True) * bias.unsqueeze(0)

        if self.bias is not None:
            output = output + self.bias.to(compute_dtype)

        output = output.reshape(*original_shape, self.out_features)
        return output.to(input.dtype if input.is_floating_point() else self.scales.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )


class PrismEmbedding(nn.Module, _PrismAffine1BitMixin):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        group_size: int = 128,
        device=None,
        dtype=None,
    ):
        nn.Module.__init__(self)
        _PrismAffine1BitMixin.__init__(self, group_size=group_size)
        if embedding_dim % self.group_size != 0:
            raise ValueError(
                f"PrismEmbedding requires embedding_dim divisible by group_size={self.group_size}, got {embedding_dim}"
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(
            torch.zeros(
                (num_embeddings, (embedding_dim // self.group_size) * self.packed_values_per_group),
                dtype=torch.uint32,
                device=device,
            ),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.zeros((num_embeddings, embedding_dim // self.group_size), dtype=dtype, device=device),
            requires_grad=False,
        )
        self.biases = nn.Parameter(
            torch.zeros((num_embeddings, embedding_dim // self.group_size), dtype=dtype, device=device),
            requires_grad=False,
        )
        self.requires_grad_(False)

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        self._validate_buffer_shapes()
        flat_ids = input_ids.reshape(-1)
        packed_rows = self.weight.index_select(0, flat_ids)
        scales_rows = self.scales.index_select(0, flat_ids)
        biases_rows = self.biases.index_select(0, flat_ids)
        output = torch.empty((flat_ids.shape[0], self.embedding_dim), dtype=self.scales.dtype, device=self.weight.device)

        for group_idx in range(self.num_groups):
            col_start = group_idx * self.group_size
            col_end = col_start + self.group_size
            packed_start = group_idx * self.packed_values_per_group
            packed_end = packed_start + self.packed_values_per_group
            bits = unpack_prism_weights(packed_rows[:, packed_start:packed_end]).to(self.scales.dtype)
            output[:, col_start:col_end] = (
                bits * scales_rows[:, group_idx : group_idx + 1] + biases_rows[:, group_idx : group_idx + 1]
            )

        return output.reshape(*input_ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"group_size={self.group_size}, padding_idx={self.padding_idx}"
        )


def augment_prism_tied_weights(model) -> None:
    tied = dict(getattr(model, "_tied_weights_keys", {}) or {})
    updated = False
    for target_name, source_name in list(tied.items()):
        if not target_name.endswith(".weight") or not source_name.endswith(".weight"):
            continue
        target_prefix = target_name.removesuffix(".weight")
        source_prefix = source_name.removesuffix(".weight")
        for suffix in ("scales", "biases"):
            tied_target = f"{target_prefix}.{suffix}"
            tied_source = f"{source_prefix}.{suffix}"
            if tied_target not in tied:
                tied[tied_target] = tied_source
                updated = True

    if updated:
        model._tied_weights_keys = tied
        model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(all_submodels=True)


def replace_with_prism_modules(model, modules_to_not_convert: list[str] | None = None, quantization_config=None):
    has_been_replaced = False
    group_size = getattr(quantization_config, "group_size", 128)

    for module_name, module in model.named_modules():
        if module_name == "" or not should_convert_module(module_name, modules_to_not_convert):
            continue

        with torch.device("meta"):
            if isinstance(module, nn.Linear):
                model.set_submodule(
                    module_name,
                    PrismLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        group_size=group_size,
                    ),
                )
                has_been_replaced = True
            elif isinstance(module, nn.Embedding):
                model.set_submodule(
                    module_name,
                    PrismEmbedding(
                        num_embeddings=module.num_embeddings,
                        embedding_dim=module.embedding_dim,
                        padding_idx=module.padding_idx,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
                        group_size=group_size,
                    ),
                )
                has_been_replaced = True

    if has_been_replaced:
        augment_prism_tied_weights(model)
    else:
        logger.warning(
            "You are loading your model using prism but no linear or embedding modules were found in your model."
        )

    return model


__all__ = [
    "PRISM_BITS",
    "PrismEmbedding",
    "PrismLinear",
    "augment_prism_tied_weights",
    "replace_with_prism_modules",
    "unpack_prism_weights",
]
