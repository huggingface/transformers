# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Conversion ops for the transforms llama.cpp applies when writing a GGUF file.

Each op undoes one such transform, so that the loaded weights are exactly the weights
transformers expects. They are ordinary `ConversionOps`: they take a dense tensor and return a
dense tensor, and the loading pipeline runs them like any other conversion.
"""

import torch

from ...core_model_loading import ConversionOps


class SubtractOne(ConversionOps):
    """Undo llama.cpp storing zero-centred RMSNorm weights as `w + 1`.

    Runs in fp32 on purpose. The file holds these norms as F32 and llama.cpp computed `w + 1` in
    fp32, so subtracting in fp32 recovers `w` exactly; the cast to the model dtype then reproduces
    the original value. Subtracting after a bf16 cast loses ~1 ULP near 1.0.
    """

    def __init__(self, offset: float = 1.0):
        self.offset = offset

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: (tensor.float() - self.offset).to(tensor.dtype)}

    @property
    def reverse_op(self) -> ConversionOps:
        return SubtractOne(offset=-self.offset)


class LogNegate(ConversionOps):
    """`A_log = log(-a)`, undoing llama.cpp storing `ssm_a = -exp(A_log)`.

    Not bit-exact: `exp` then `log` is a 1-2 ULP round trip.
    """

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: torch.log(-tensor.float()).to(tensor.dtype)}


class Unsqueeze(ConversionOps):
    """Add a size-1 dim, undoing llama.cpp squeezing `conv1d` from `(C, 1, K)` to `(C, K)`."""

    def __init__(self, dim: int):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: tensor.unsqueeze(self.dim)}

    @property
    def reverse_op(self) -> ConversionOps:
        return Squeeze(dim=self.dim)


class Squeeze(ConversionOps):
    """Reverse of `Unsqueeze`."""

    def __init__(self, dim: int):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: tensor.squeeze(self.dim)}


class PermuteRows(ConversionOps):
    """Reorder rows (dim 0), optionally only those from `offset` onwards.

    llama.cpp can store head-indexed tensors in a different head order than transformers. Where
    that axis is the tensor's *output* axis, undoing it is a row permutation.

    `offset` covers tensors whose leading rows must stay put: e.g. Qwen3.5's fused
    `in_proj_qkv`, where only the v-block is reordered.
    """

    def __init__(self, permutation: torch.Tensor, offset: int = 0):
        self.permutation = permutation
        self.offset = offset

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        perm = self.permutation.to(tensor.device)
        if self.offset:
            head, tail = tensor[: self.offset], tensor[self.offset :]
            tensor = torch.cat([head, tail[perm]], dim=0)
        else:
            tensor = tensor[perm]
        return {target_patterns[0]: tensor.contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        return PermuteRows(torch.argsort(self.permutation), offset=self.offset)


class PermuteInputFeatures(ConversionOps):
    """Reorder columns (dim 1), for a tensor that *consumes* an axis llama.cpp reordered.

    Same logical reordering as `PermuteRows`, but it lands on `in_features` because this tensor is
    on the consuming side (Qwen3.5's `linear_attn.out_proj`).
    """

    def __init__(self, permutation: torch.Tensor):
        self.permutation = permutation

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        perm = self.permutation.to(tensor.device)
        return {target_patterns[0]: tensor[:, perm].contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        return PermuteInputFeatures(torch.argsort(self.permutation))


def _single_tensor(input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """These ops are all one-to-one; unwrap the single (possibly listed) tensor."""
    if len(input_dict) != 1:
        raise ValueError(f"expected a single source tensor, got {list(input_dict)}")
    tensors = next(iter(input_dict.values()))
    return tensors[0] if isinstance(tensors, list) else tensors
