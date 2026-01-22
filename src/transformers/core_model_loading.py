# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Core helpers for loading model checkpoints."""

from __future__ import annotations

import math
import os
import re
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, MutableMapping, MutableSet
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, Any

import torch

from .integrations.accelerate import get_device, offload_weight
from .integrations.tensor_parallel import ALL_PARALLEL_STYLES
from .utils import is_env_variable_true, is_torch_greater_or_equal, logging


_torch_distributed_available = torch.distributed.is_available()
_is_dtensor_available = _torch_distributed_available and is_torch_greater_or_equal("2.5")
if _is_dtensor_available:
    from torch.distributed.tensor import DTensor, Replicate

if TYPE_CHECKING:
    from .integrations.tensor_parallel import TensorParallelLayer
    from .modeling_utils import PreTrainedModel
    from .quantizers import HfQuantizer


logger = logging.get_logger(__name__)


def process_target_pattern(pattern: str) -> tuple[str, str | None]:
    """
    Process a target pattern for reverse mapping (when targets become sources).

    This handles several edge cases in checkpoint conversion mappings:
    - Removes `^` prefix and `$` suffix (start/end of string anchors)
    - Removes negative lookahead/lookbehind assertions
    - Detects capturing groups and replaces them with `\\1` backreference

    Args:
        pattern: The target pattern to process for reverse mapping.

    Returns:
        A tuple of (processed_pattern, captured_group) where captured_group is
        the original capturing group found (e.g., "(encoder|decoder)") or None.
    """
    # Some mapping contains `^` to notify start of string when matching -> remove it during reverse mapping
    pattern = pattern.removeprefix("^")
    # Some mapping contains `$` to notify end of string when matching -> remove it during reverse mapping
    pattern = pattern.removesuffix("$")
    # Remove negative lookahead/behind if any. This is ugly but needed for reverse mapping of
    # Qwen2.5, Sam3, Ernie4.5 VL MoE!
    pattern = re.sub(r"\(\?.+\)", "", pattern)
    # Allow capturing groups in patterns, i.e. to add/remove a prefix to all keys (e.g. timm_wrapper, sam3)
    capturing_group_match = re.search(r"\(.+?\)", pattern)
    captured_group = None
    if capturing_group_match:
        captured_group = capturing_group_match.group(0)
        pattern = pattern.replace(captured_group, r"\1", 1)
    return pattern, captured_group


def build_glob_alternation(
    globs: list[WeightRenaming | WeightConverter | str],
) -> tuple[re.Pattern, dict[str, str], dict[str, str]]:
    """
    Build a single alternation regex with one named group per glob.
    """
    src_group_to_glob: dict[str, str] = {}
    tgt_group_to_glob: dict[str, str] = {}
    branches: list[str] = []
    i = 0
    for glob in globs:
        if isinstance(glob, (WeightRenaming, WeightConverter)):
            for src in glob.source_patterns:
                group_name = f"g{i}"
                src_group_to_glob[group_name] = src
                i += 1
                body = src.replace("*", r".*")
                branches.append(f"(?P<{group_name}>{body})")
                tgt_group_to_glob[group_name] = glob.target_patterns[0]  # we index witht the first target
        else:
            group_name = f"g{i}"
            src_group_to_glob[group_name] = glob
            i += 1
            body = glob
            body = body.replace("*", r".*")
            branches.append(f"(?P<{group_name}>{body})")
            tgt_group_to_glob[group_name] = glob

    alternation = re.compile("|".join(branches))
    return alternation, src_group_to_glob, tgt_group_to_glob


class ConversionOps:
    """Base class for weight conversion operations."""

    def __repr__(self):
        if hasattr(self, "dim"):
            return f"{self.__class__.__name__}(dim={self.dim})"
        else:
            return f"{self.__class__.__name__}"

    @abstractmethod
    def convert(
        self, input_dict: dict[str, Any], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, list[torch.Tensor]]:
        raise NotImplementedError

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError


class Chunk(ConversionOps):
    """Split a tensor along ``dim`` into equally sized chunks."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        targets = self.get_target_patterns(input_dict, target_patterns)
        sizes = len(targets)
        chunks = torch.chunk(tensor, sizes, dim=self.dim)
        return dict(zip(targets, chunks))

    def get_target_patterns(self, input_dict: dict, target_patterns: list[str]) -> list[str]:
        # Here we always return the target patterns
        if len(input_dict) > 1 or len(target_patterns) == 1:
            raise ValueError("Undefined Operation encountered!")
        return target_patterns

    @property
    def reverse_op(self) -> ConversionOps:
        return Concatenate(self.dim)


class Concatenate(ConversionOps):
    """Concatenate tensors along `dim`."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = self.get_target_pattern(target_patterns)
        all_tensors = []
        # Very important to keep the relative order of the source patterns here, so we iterate over them not the
        # input directly as it's unordered!
        for source_pattern in source_patterns:
            tensors = input_dict[source_pattern]
            if isinstance(tensors, list):
                all_tensors.extend(tensors)
            else:
                all_tensors.append(tensors)
        return {target_pattern: torch.cat(all_tensors, dim=self.dim)}

    def get_target_pattern(self, target_patterns: list[str]) -> str:
        # Here we always return the target pattern
        if len(target_patterns) > 1:
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return Chunk(self.dim)


class MergeModulelist(ConversionOps):
    """
    Merge a list of tensors into a single tensor along the first dimension.
    We explicitly define this because for EP or TP you want to make sure you know what you are doing!

    """

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        for source_pattern, tensors in input_dict.items():
            target_pattern = self.get_target_pattern(input_dict, source_pattern, target_patterns)
            merged[target_pattern] = torch.stack(tensors, dim=self.dim)
        return merged

    def get_target_pattern(self, input_dict: dict, source_pattern: str, target_patterns: list[str]) -> str:
        # Here it's a single operation, so we use the target
        if len(input_dict) == 1:
            if len(target_patterns) == 1:
                return target_patterns[0]
            else:
                raise ValueError("Undefined Operation encountered!")
        #  Here it's the first operation in a chain, so we use the source as they were replaced before in the chain
        else:
            return source_pattern

    @property
    def reverse_op(self) -> ConversionOps:
        return SplitModulelist(self.dim)


class SplitModulelist(ConversionOps):
    """Inverse of :class:`MergeModulelist` using explicit split sizes per group."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        all_tensors = {}
        for source_pattern, tensors in input_dict.items():
            tensor = tensors[0] if isinstance(tensors, list) else tensors
            # We split in the number of tensors present in the given dim
            sizes = tensor.size(self.dim)
            targets = self.get_target_patterns(input_dict, source_pattern, target_patterns, sizes)
            chunks = torch.chunk(tensor, sizes, dim=self.dim)
            # We squeeze each chunk here as well to make sure to give them their original shape
            all_tensors.update({target: chunk.squeeze() for target, chunk in zip(targets, chunks)})
        return all_tensors

    def get_target_patterns(
        self, input_dict: dict, source_pattern: str, target_patterns: list[str], sizes: int
    ) -> list[str]:
        # Here it's a single operation, so we use the target
        if len(input_dict) == 1:
            if len(target_patterns) == 1:
                return [target_patterns[0].replace("*", f"{i}") for i in range(sizes)]
            else:
                raise ValueError("Undefined Operation encountered!")
        # Here it's the last operation in a chain, so we use the source as they were replaced before in the chain
        else:
            return [source_pattern.replace("*", f"{i}") for i in range(sizes)]

    @property
    def reverse_op(self) -> ConversionOps:
        return MergeModulelist(self.dim)


class Transpose(ConversionOps):
    """
    Transposes the given tensor along dim0 and dim1.
    """

    def __init__(self, dim0: int = 0, dim1: int = 1):
        self.dim0 = dim0
        self.dim1 = dim1

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        target_pattern = self.get_target_pattern(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: torch.transpose(tensor, dim0=self.dim0, dim1=self.dim1).contiguous()}

    def get_target_pattern(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str]
    ) -> str:
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        # Here it's the first operation of a chain, so return the source
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            else:
                raise ValueError("Undefined Operation encountered!")
        # Here it's the only operation, or the last operation in a chain, so we return the target
        else:
            return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return Transpose(dim0=self.dim1, dim1=self.dim0)


class PermuteForRope(ConversionOps):
    """
    Applies the permutation required to convert complex RoPE weights to the split sin/cos format.
    """

    def __init__(self):
        pass

    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        dim1, dim2 = tensor.shape
        n_heads = self.config.getattr("num_attention_heads", 1)

        tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        tensor = tensor.transpose(1, 2).reshape(dim1, dim2)
        return tensor

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        self.config = config
        output: dict[str, list[torch.Tensor]] = {}
        for key, tensors in input_dict.items():
            if len(tensors) != 1:
                raise ValueError("PermuteForRope expects a single tensor per key.")
            output[key] = [self._apply(tensors[0])]
        return output


class ErnieFuseAndSplitTextVisionExperts(ConversionOps):
    r"""
    Special operation that splits a module list over all keys and fuses over the number of original modules.

    Example with 2 original modules "Gate" and "Up" with 2 target keys "Text" and "Vision":

                 ModuleList 1            ModuleList 2
                [   Gate    ]            [   Up    ]
                |           |            |         |
          [Gate_Text] [Gate_Vision] [Up_Text]   [Up_Vision]
              \                  \  /                /
               \                 \ /                /
                \               /  \               /
                 \            /     \             /
                 [GateUp_Text]      [GateUp_Vision]

    The splits are equal and are defined by the amount of target keys.
    The final fusions are defined by the amount of original module lists.
    """

    def __init__(self, stack_dim: int = 0, concat_dim: int = 1):
        self.stack_dim = stack_dim
        self.concat_dim = concat_dim

    def split_list_into_chunks(self, tensor_list: list[torch.Tensor], chunks: int = 2):
        split_size = math.ceil(len(tensor_list) / chunks)  # best effort split size
        return [tensor_list[i * split_size : (i + 1) * split_size] for i in range(chunks)]

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        valid_keys = input_dict.keys()
        split_and_fused = defaultdict(list)
        for key in source_patterns:
            if key not in valid_keys:
                raise ValueError(
                    f"Expected pattern {key} in collected tensors but only found tensors for: {valid_keys}"
                )

            tensors = input_dict.get(key, [])
            split_tensor_lists = self.split_list_into_chunks(tensors, chunks=len(target_patterns))
            stacked_tensors = (torch.stack(tensor_group, dim=self.stack_dim) for tensor_group in split_tensor_lists)
            for idx, tensor_group in enumerate(stacked_tensors):
                split_and_fused[target_patterns[idx]].append(tensor_group)

        for k, v in split_and_fused.items():
            split_and_fused[k] = torch.cat(v, dim=self.concat_dim)

        return split_and_fused

    @property
    def reverse_op(self) -> ConversionOps:
        return ErnieSplitAndDecoupleTextVisionExperts(stack_dim=self.stack_dim, concat_dim=self.concat_dim)


class ErnieSplitAndDecoupleTextVisionExperts(ConversionOps):
    r"""
    Special operation that splits a fused module list over all original modules and
    then decouples them into a mixed module list each over all keys.

    Example with 2 original modules "Gate" and "Up" with 2 target keys "Text" and "Vision":

                    [GateUp_Text]     [GateUp_Vision]
                  /              \   /             \
                 /                \ /               \
                /                / \                 \
               /                /   \                 \
          [Gate_Text] [Gate_Vision] [Up_Text]   [Up_Vision]
                |           |            |         |
                [   Gate    ]            [   Up    ]
                 ModuleList 1            ModuleList 2

    The splits are equal and are defined by the amount of original module lists.
    The final decoupled module lists are defined by the amount of keys.
    """

    def __init__(self, stack_dim: int = 0, concat_dim: int = 1):
        self.stack_dim = stack_dim
        self.concat_dim = concat_dim

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        fused_modules = len(target_patterns)
        valid_keys = input_dict.keys()
        split_tensors = []
        for key in source_patterns:
            if key not in valid_keys:
                raise ValueError(
                    f"Expected pattern {key} in collected tensors but only found tensors for: {valid_keys}"
                )

            # Assuming that we get single sized lists here to index with 0
            split_tensors.append(input_dict[key][0].chunk(fused_modules, dim=self.concat_dim))

        decoupled = {}
        for idx, key in enumerate(target_patterns):
            tensor_groups = [
                list(torch.unbind(tensor_group[idx], dim=self.stack_dim)) for tensor_group in split_tensors
            ]
            tensor_list = list(chain.from_iterable(tensor_groups))
            targets = [key.replace("*", f"{i}") for i in range(len(tensor_list))]
            decoupled |= dict(zip(targets, tensor_list))

        return decoupled

    @property
    def reverse_op(self) -> ConversionOps:
        return ErnieFuseAndSplitTextVisionExperts(stack_dim=self.stack_dim, concat_dim=self.concat_dim)


class Force16BytesAlignment(ConversionOps):
    """
    Ensures that the given tensor is 16-bytes aligned in memory and clones it if not.
    This garantees 16-bytes alignmenet for kernels / implementations that use TMA or SIMD instructions like torch._grouped_mm.
    """

    @torch.no_grad()
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        target_pattern = self.get_target_pattern(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        tensor = tensor.clone() if tensor.data_ptr() % 16 != 0 else tensor
        return {target_pattern: tensor}

    def get_target_pattern(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str]
    ) -> str:
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        # Here it's the first operation of a chain, so return the source
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            else:
                raise ValueError("Undefined Operation encountered!")
        # Here it's the only operation, or the last operation in a chain, so we return the target
        else:
            return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return Force16BytesAlignment()


@dataclass(slots=True)
class WeightTransform:
    source_patterns: str | list[str] = field(init=True)
    target_patterns: str | list[str] = field(init=True)
    compiled_sources: re.Pattern = field(init=False)

    distributed_operation: TensorParallelLayer | None = None
    quantization_operation: ConversionOps | None = None

    collected_tensors: dict[str, list[Future]] = field(default_factory=lambda: defaultdict(list), init=False)
    layer_targets: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set), init=False)

    def __setattr__(self, name: str, value: Any):
        if name == "source_patterns":
            if isinstance(value, str):
                value = [value]
            normalized = []
            for pattern in value:
                if r"\1" in pattern:
                    pattern = pattern.replace(r"\1", r"(.+)")
                normalized.append(pattern)
            object.__setattr__(self, name, normalized)
            self._rebuild_compiled_sources()
            return
        if name == "target_patterns":
            if isinstance(value, str):
                value = [value]
            normalized = []
            for pattern in value:
                # Some mapping contains `^` to notify start of string when matching -> remove it during reverse mapping
                pattern = pattern.removeprefix("^")
                # Some mapping contains `$` to notify end of string when matching -> remove it during reverse mapping
                pattern = pattern.removesuffix("$")
                # Remove negative lookahead/behind if any. This is ugly but needed for reverse mapping of
                # Qwen2.5, Sam3, Ernie4.5 VL MoE!
                pattern = re.sub(r"\(\?.+\)", "", pattern)
                # Allow capturing groups in patterns, i.e. to add/remove a prefix to all keys (e.g. timm_wrapper, sam3)
                if r"(.+)" in pattern:
                    pattern = pattern.replace(r"(.+)", r"\1")
                normalized.append(pattern)
            object.__setattr__(self, name, normalized)
            return
        object.__setattr__(self, name, value)

    def _rebuild_compiled_sources(self):
        branches = []
        for i, source_pattern in enumerate(self.source_patterns):
            group_name = f"g{i}"
            pattern = source_pattern.replace(".*.", r"\..*\.")
            branches.append(f"(?P<{group_name}>{pattern})")
        object.__setattr__(self, "compiled_sources", re.compile("|".join(branches)))


    def __post_init__(self):
        if isinstance(self.source_patterns, str):
            self.source_patterns = [self.source_patterns]
        if isinstance(self.target_patterns, str):
            self.target_patterns = [self.target_patterns]

        # Due to how our `_checkpoint_conversion_mapping` mappings are written, we need a few exceptions here
        # when instantiating the reverse mapping (i.e. the targets become sources, and sources become targets)
        # The issues lie in the sources usually, so here we need to check the targets for the reversed mapping

        # Process target_patterns: detect capturing groups and replace with \1
        # Store the original capturing group patterns for reverse mapping
        target_capturing_groups: list[str] = []
        for i, pattern in enumerate(self.target_patterns):
            self.target_patterns[i], captured_group = process_target_pattern(pattern)
            if captured_group is not None:
                target_capturing_groups.append(captured_group)

        # Validate that we only have one unique capturing group pattern across all targets
        # This ensures deterministic reverse mapping when sources have \1 backreferences
        unique_capturing_groups = set(target_capturing_groups)
        if len(unique_capturing_groups) > 1:
            raise ValueError(
                f"Multiple different capturing groups found in target_patterns: {unique_capturing_groups}. "
                f"All target patterns must use the same capturing group pattern."
            )
        unique_capturing_group = unique_capturing_groups.pop() if unique_capturing_groups else None

        # We also need to check capturing groups in the sources during reverse mapping (e.g. timm_wrapper, sam3)
        for i, pattern in enumerate(self.source_patterns):
            if r"\1" in pattern:
                if unique_capturing_group is None:
                    raise ValueError(
                        f"Source pattern '{pattern}' contains \\1 backreference, but no capturing groups "
                        f"found in target_patterns."
                    )
                # Use the unique capturing group from target_patterns for all sources
                pattern = pattern.replace(r"\1", unique_capturing_group, 1)
            self.source_patterns[i] = pattern

        # Construct the regex we will use to rename keys from the sources to the targets
        self._rebuild_compiled_sources()

    def add_tensor(self, target_key: str, source_key: str, source_pattern: str, future: Future):
        self.collected_tensors[source_pattern].append(future)
        self.layer_targets[target_key].add(source_key)

    def rename_source_key(self, source_key: str) -> tuple[str, str | re.Pattern]:
        """
        Return a tuple (renamed_key, source_pattern_producing_the_match).
        Try renaming `source_key` according to the source and target patterns of the current WeightTransform.
        In case of a one-to-many transform, i.e. we have several target patterns, the matching source pattern
        will be replaced by the first of all the target patterns (they are then correctly expanded in the Operations).
        """
        # Try matching one of the alternation branches
        match_object = self.compiled_sources.search(source_key)
        if match_object is None:
            return source_key, None

        # Find the source that produced the match (it's the first group that matched, as the search stops after first branch match)
        matching_group_name = next(name for name, val in match_object.groupdict().items() if val is not None)
        source_pattern_that_matched = self.source_patterns[int(matching_group_name[1:])]
        # If we matched, we always replace with the first target pattern, in case we have several (one to many transform)
        replacement = self.target_patterns[0]
        # Allow capturing groups in patterns, i.e. to add a prefix to all keys (e.g. timm_wrapper, sam3)
        if r"\1" in replacement:
            # The index of the internal group we need to replace is the index of the matched named group as it comes
            # inside that matched named group
            replaced_group_idx = self.compiled_sources.groupindex[matching_group_name] + 1
            replacement = replacement.replace(r"\1", match_object.group(replaced_group_idx))
        renamed_key = source_key.replace(match_object.group(0), replacement)
        return renamed_key, source_pattern_that_matched

    def reverse_transform(self) -> WeightTransform:
        """Reverse the current `WeightTransform` instance, to be able to save with the opposite weight transformations."""
        # TODO: check this and relax when quantizer have `reverse_op`
        if self.quantization_operation is not None:
            raise ValueError("Cannot reverse the transform with TP or quantization")

        kwargs = {}
        # Add the reverse ops if applicable (it needs to be provided at __init__)
        if hasattr(self, "operations"):
            # All reverse ops, in reverse order
            kwargs["operations"] = [op.reverse_op for op in self.operations[::-1]]

        reverse_transform = self.__class__(
            source_patterns=self.target_patterns, target_patterns=self.source_patterns, **kwargs
        )

        return reverse_transform

    def materialize_tensors(self) -> dict[str, list[torch.Tensor]]:
        """
        Materialize all the tensors that were saved in `self.collected_tensors`. This function removes them from the
        internal attribute to avoid keeping them in memory during the different `self.convert` operations, and return
        a new dictionary (otherwise we use more memory than needed during loading).

        We basically have 3 cases here:
        - async loading (default): the tensors are Future instances that we need to wait for
        - sync loading: the tensors are Callable, we need to call the Callable to actually load them from disk
        - saving: the tensors are already torch.Tensor instances (the existing model weights)
        """
        collected_tensors = {}
        for key in set(self.collected_tensors.keys()):
            # Remove from internal attribute
            tensors = self.collected_tensors.pop(key)
            # Async loading
            if isinstance(tensors[0], Future):
                tensors = [future.result() for future in tensors]
            # Sync loading
            elif callable(tensors[0]):
                tensors = [func() for func in tensors]
            # Add them to the new dictionary
            collected_tensors[key] = tensors

        return collected_tensors


@dataclass(slots=True)
class WeightRenaming(WeightTransform):
    # Special case of WeightTransform that only renames keys without any conversion.

    def convert(
        self,
        layer_name: str,
        model=None,
        config=None,
        hf_quantizer=None,
        missing_keys: MutableSet[str] | None = None,
        conversion_errors: MutableMapping[str, str] | None = None,
    ):
        # Collect the tensors here - we use a new dictionary to avoid keeping them in memory in the internal
        # attribute during the whole process
        collected_tensors = self.materialize_tensors()

        # Perform renaming op (for a simple WeightRenaming, `self.source_patterns` and `self.target_patterns` can
        # only be of length 1, and are actually the full key names - we also have only 1 single related tensor)
        target_key = self.target_patterns[0]
        collected_tensors = {target_key: collected_tensors[self.source_patterns[0]]}

        if hf_quantizer is not None and self.quantization_operation is not None:
            with log_conversion_errors(
                layer_name, conversion_errors, (len(collected_tensors), layer_name), self.quantization_operation
            ):
                collected_tensors = self.quantization_operation.convert(
                    collected_tensors,
                    source_patterns=self.source_patterns,
                    target_patterns=self.target_patterns,
                    full_layer_name=target_key,
                    model=model,
                    config=config,
                    missing_keys=missing_keys,
                )

        return collected_tensors, conversion_errors


# List of classes that are known to be able to use m:n
_INTERNAL_MANY_TO_MANY_CONVERSIONS = (
    ErnieFuseAndSplitTextVisionExperts,
    ErnieSplitAndDecoupleTextVisionExperts,
)


@dataclass(slots=True)
class WeightConverter(WeightTransform):
    operations: list[ConversionOps] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if bool(len(self.source_patterns) - 1) + bool(len(self.target_patterns) - 1) >= 2:
            # We allow many-to-many only if we use an internal operation that can handle it
            if not any(isinstance(op, _INTERNAL_MANY_TO_MANY_CONVERSIONS) for op in self.operations):
                raise ValueError(
                    f"source keys={self.source_patterns}, target_patterns={self.target_patterns} but you can only have one to many, one to one or many to one."
                )
        if not self.operations:
            raise ValueError("WeightConverter requires at least one operation.")

    def convert(
        self,
        layer_name: str,
        model=None,
        config=None,
        hf_quantizer=None,
        missing_keys: MutableSet[str] | None = None,
        conversion_errors: MutableMapping[str, str] | None = None,
    ):
        # Collect the tensors here - we use a new dictionary to avoid keeping them in memory in the internal
        # attribute during the whole process
        collected_tensors = self.materialize_tensors()

        for op in self.operations:
            with log_conversion_errors(layer_name, conversion_errors, (len(collected_tensors), layer_name), op):
                collected_tensors = op.convert(
                    collected_tensors,
                    source_patterns=self.source_patterns,
                    target_patterns=self.target_patterns,
                    # Additional kwargs, ususally not used
                    full_layer_name=layer_name,
                    model=model,
                    config=config,
                    missing_keys=missing_keys,
                )

        # Tensors are returned from ops with the target patterns, we need to expand them to full name.
        # This means we need to grab the prefix and suffix to add to every target key
        full_name = layer_name
        if ".*." in layer_name:
            full_name = layer_name.replace(".*.", ".0.")

        try:
            prefix, _, suffix = next(full_name.partition(k) for k in collected_tensors.keys() if k in full_name)
            # Rename the tensors
            collected_tensors = {prefix + k + suffix: v for k, v in collected_tensors.items()}
        # some quantizers need to already rename in `convert` as they cannot only rely on prefix and suffix
        except StopIteration:
            pass

        if hf_quantizer is not None and self.quantization_operation is not None:
            with log_conversion_errors(
                layer_name, conversion_errors, (len(collected_tensors), layer_name), self.quantization_operation
            ):
                collected_tensors = self.quantization_operation.convert(
                    collected_tensors,
                    source_patterns=self.source_patterns,
                    target_patterns=self.target_patterns,
                    full_layer_name=layer_name,
                    config=config,
                    model=model,
                    missing_keys=missing_keys,
                )
        return collected_tensors, conversion_errors


# For I/O bound operations (i.e. here reading files), it is better to have fewer threads, e.g. 4 is a good default.
# Having too many is actually harming performances quite a lot, i.e. using 16 can sometimes lead to taking TWICE
# as much time to load the same model
GLOBAL_WORKERS = min(4, os.cpu_count() or 4)


def _materialize_copy(tensor: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    # This slicing is what actually loads the tensor from the safetensors slice object
    tensor = tensor[...]
    if dtype is not None or device is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor


def spawn_materialize(
    thread_pool: ThreadPoolExecutor | None, tensor: torch.Tensor, device=None, dtype=None
) -> Future | Callable:
    """Materialize a tensor from file asynchronously if `thread_pool` is provided, or return a Callable that will
    load the tensor synchronously when called."""

    def _job():
        return _materialize_copy(tensor, device, dtype)

    if thread_pool is not None:
        return thread_pool.submit(_job)
    else:
        # Return the Callable here, not the Tensor itself, so we actually delay loading to avoid saturating cpu
        # memory during Conversion
        return _job


def spawn_tp_materialize(
    thread_pool: ThreadPoolExecutor | None, tensor: torch.Tensor, sharding_method, tensor_idx, device=None, dtype=None
) -> Future | Callable:
    """Materialize and shard a tensor (according to the TP-plan) from file asynchronously if `thread_pool` is provided, or
    return a Callable that will load the tensor synchronously when called."""

    def _job():
        return sharding_method.shard_tensor(tensor, tensor_idx=tensor_idx, device=device, dtype=dtype)

    if thread_pool is not None:
        return thread_pool.submit(_job)
    else:
        # Return the Callable here, not the Tensor itself, so we actually delay loading to avoid saturating cpu
        # memory during Conversion
        return _job


def dot_natural_key(s: str):
    parts = s.split(".")
    for i, p in enumerate(parts):
        # whole-segment digits -> int; otherwise leave as str
        if p.isdigit():
            parts[i] = int(p)
    return parts


@contextmanager
def log_conversion_errors(
    first_target_key: str,
    conversion_errors: MutableMapping[str, str] | None,
    extras: Any = None,
    op: list[ConversionOps] | ConversionOps | None = None,
):
    """Catch all exceptions during `convert` calls, and log the errors for later. Re-raise a `SkipParameters` exception
    that will be catched later to skip the parameters that raised the original Exception."""
    try:
        yield
    except Exception as e:
        # During reverse mapping, we do not log and skip errors
        if conversion_errors is None:
            raise e

        def _format_op_name(curr_op: list[ConversionOps] | ConversionOps | None) -> str | None:
            if curr_op is None:
                return None
            if isinstance(curr_op, (list, tuple, set)):
                names = [o.__class__.__name__ for o in curr_op if o is not None]
                if not names:
                    return None
                return ", ".join(names)
            return curr_op.__class__.__name__

        op_name = _format_op_name(op)
        if isinstance(extras, tuple) and len(extras) == 2:
            length, target_keys = extras
            descriptor = f"{op_name} " if op_name else ""
            conversion_errors[first_target_key] = (
                f"{e}\nError: {descriptor}on tensors destined for {target_keys}. Ckpt contains: {length}"
            )
        elif isinstance(extras, str):
            suffix = f" via {op_name}" if op_name else ""
            conversion_errors[first_target_key] = f"{e}\nError{suffix} when processing parameter {extras}"
        elif extras is None and op_name:
            conversion_errors[first_target_key] = f"{op_name}: {e}"
        else:
            conversion_errors[first_target_key] = f"{extras} |Error: {e}"

        # Raise a specific Exception that we can catch easily
        raise SkipParameters()


def set_param_for_module(
    model: PreTrainedModel,
    target_name: str,
    param_value: torch.Tensor,
    mismatch_keys: MutableSet[tuple[str, torch.Size, torch.Size]],
    missing_keys: MutableSet[str],
    unexpected_keys: MutableSet[str],
    distributed_operation: TensorParallelLayer | None,
    hf_quantizer: HfQuantizer,
):
    module_path, _, param_name = target_name.rpartition(".")
    module_obj = model.get_submodule(module_path) if module_path else model

    ref = getattr(module_obj, param_name)
    if ref is None:
        unexpected_keys.add(target_name)
    else:
        if not isinstance(param_value, torch.nn.Parameter):
            if distributed_operation is not None:
                if getattr(distributed_operation, "use_dtensor", False):
                    param_value = DTensor.from_local(
                        param_value,
                        distributed_operation.device_mesh,
                        getattr(distributed_operation, "shard", Replicate()),
                        run_check=False,
                        shape=ref.size(),
                        stride=ref.stride(),
                    )
            if param_name not in module_obj._buffers:
                param_value = torch.nn.Parameter(param_value, requires_grad=param_value.is_floating_point())

        # Remove from missing keys (it's either mismatched, or all good)
        missing_keys.discard(target_name)
        if ref is not None and ref.shape != param_value.shape and hf_quantizer is None:
            mismatch_keys.add((target_name, param_value.shape, ref.shape))
        else:
            # super important otherwise _init_weight will re-init the param
            param_value._is_hf_initialized = True
            setattr(module_obj, param_name, param_value)


def offload_and_maybe_resave_param(
    target_name: str,
    param: torch.Tensor,
    missing_keys: MutableSet[str],
    disk_offload_folder: str,
    disk_offload_index: dict,
    applied_ops: WeightConverter | WeightRenaming,
) -> dict:
    """Takes care of correctly offloading `param`. If it's not already present in the `disk_offload_index`, or if any
    WeightConverter operations have been applied, it will resave the new parameter. Otherwise, it will use the original
    `disk_offload_index` for this given param."""
    # We need to remove from missing keys
    missing_keys.discard(target_name)
    # If not already offloaded, or if we applied any special Operation except Renaming, we need to re-save
    if target_name not in disk_offload_index or isinstance(applied_ops, WeightConverter):
        disk_offload_index = offload_weight(param, target_name, disk_offload_folder, disk_offload_index)
    return disk_offload_index


class SkipParameters(Exception):
    """Control-flow sentinel: abort processing of the current parameters only (that were supposed to be created
    by a WeightConverter)."""

    pass


def rename_source_key(
    source_key: str,
    weight_renamings: list[WeightRenaming],
    weight_converters: list[WeightConverter],
    prefix: str | None = None,
    meta_state_dict: dict | None = None,
) -> tuple[str, str | None]:
    """
    Rename a source key given all the renaming and weight conversion patterns we have. Also takes care of adding/removing
    the base model prefix during loading if necesary.
    """
    renamed_key = source_key
    # 1. apply all renamings in turns (if multiple match, it's the responsibility of the mappings to make sure they
    # are coherent)
    for renaming in weight_renamings:
        renamed_key, _ = renaming.rename_source_key(renamed_key)

    # 2. apply renaming through weight conversions on the key if we have any WeightConverter (here we stop after
    # the first match, as we assume only 1 converter can match any source key)
    source_pattern = None
    for converter in weight_converters:
        renamed_key, source_pattern = converter.rename_source_key(renamed_key)
        if source_pattern is not None:
            break

    # 3. check if we need to add or remove prefix if necesary (only during loading, not saving)
    if prefix is not None and meta_state_dict is not None:
        if (
            renamed_key.startswith(prefix)
            and meta_state_dict.get(re.sub(f"^{prefix}.", "", renamed_key, count=1)) is not None
        ):
            renamed_key = re.sub(f"^{prefix}.", "", renamed_key, count=1)
        elif meta_state_dict.get(f"{prefix}.{renamed_key}") is not None:
            renamed_key = f"{prefix}.{renamed_key}"

    return renamed_key, source_pattern


def convert_and_load_state_dict_in_model(
    model: PreTrainedModel,
    state_dict: dict[str, Any],
    load_config: Any,
    tp_plan: dict[str, str] | None,
    dtype_plan: dict | None = None,
    disk_offload_index: dict | None = None,
):
    r"""
    We build a mapping from the keys obtained by renaming each of the checkpoint keys according to the weight_mapping rules.
    Then we load the tensors into the model, applying any conversion operations as needed.

    The `param_name_to_load` will look like this:
    {
        "model.layers.0.attention.q.weight": # Notice here there is only the first key of the target keys
            WeightConverter(
                source_patterns=["qkv"],
                target_patterns=["q", "k","v"],
                operations=[Chunk(dim=0, chunks=3)]),
                collected_tensors={
                    "qkv": [Future]},
                layer_targets={
                    "model.layers.0.attention.q.weight": {"model.layers.0.attention.qkv.weight"},
                    "model.layers.0.attention.k.weight": {"model.layers.0.attention.qkv.weight"},
                    "model.layers.0.attention.v.weight": {"model.layers.0.attention.qkv.weight"},
                }
            ),
        ...
    }

    We make sure that the keys are the full keys. The only "nit" here is that 1 key can map to multiple target keys (e.g. qkv -> q, k, v).
    In that case the weight converter will take care of doing the appropriate renaming.

    For example for:
    ```python
    WeightConverter(
        source_patterns=["mlp.experts.*.gate_proj.weight","mlp.experts.*.up_proj.weight"],
        target_patterns="mlp.experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    )
    ```
    we would have the following collected tensors:
    ```python
    collected_tensors = {
        "mlp.experts.*.gate_proj.weight": [Future, Future, Future, Future, Future, Future, Future, Future],
        "mlp.experts.*.up_proj.weight": [Future, Future, Future, Future, Future, Future, Future, Future],
    }
    ```
    The first op, `MergeModulelist`, would stack the 8 tensors of each source but will not "rename" them into the fused target name.
    The second op, `Concatenate`, would then rename the fused tensor into the final target name.

    If we want to split `qkv` we would have:
    ```python
    collected_tensors = {
        "attention.qkv.weight": [Future], # here its the full SOURCE keys.
    }
    ```
    The `Chunk` operation would then split the single tensor into 3 and rename them accordingly and update the collected tensors to:
    ```python
    realized_values = {
        "attention.q.weight": [Tensor],
        "attention.k.weight": [Tensor],
        "attention.v.weight": [Tensor],
    }
    ```

    Now that this is done, we can quantize / dequantize accordingly the collected_tensors.

    For some quantization methods, we need to gather different tensors:

    ```python
    # for "medmekk/llama-3.2-1b-float8-torchao"
    WeightConverter(
        source_patterns=[":qdata", ":scale"],
        target_patterns="",
        operations=[TorchaoDeserialize()],
    )
    ```
    This will collect all tensors that have the same prefix, but end with `:qdata` or `:scale`. This will give us:
    ```python
    all_weight_mapping = {
        "model.layers.13.self_attn.o_proj.weight": WeightConverter(
            source_patterns=[":qdata", ":scale"],
            target_patterns="",
            operations=[TorchaoDeserialize()],
            collected_tensors={
                ":qdata": [Future],
                ":scale": [Future],
            },
        ...
    }
    ```

    """
    prefix = model.base_model_prefix
    tp_plan = tp_plan or {}
    device_map = load_config.device_map or {"": "cpu"}
    hf_quantizer = load_config.hf_quantizer
    dtype = load_config.dtype
    device_mesh = load_config.device_mesh
    disk_offload_folder = load_config.disk_offload_folder
    offload_buffers = load_config.offload_buffers
    dtype_plan = dtype_plan or {}
    weight_mapping = load_config.weight_mapping or []
    meta_model_state_dict = model.state_dict()
    model_buffers = {k for k, _ in model.named_buffers()}

    missing_keys = set(meta_model_state_dict.keys())
    conversion_errors = {}
    mismatch_keys = set()
    unexpected_keys = set()

    # We use threading by default, if not explicitly deactivated via env variable. If we have to offload,
    # we cannot use it either to control the memory as we are under memory constraints, so we need to be sequential
    if is_env_variable_true("HF_DEACTIVATE_ASYNC_LOAD") or "disk" in device_map.values():
        thread_pool = None
    else:
        thread_pool = ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)

    renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]
    param_name_to_load: dict[str, WeightRenaming | WeightConverter] = {}

    # build '(?P<g0>.*.*\\.block_sparse_moe\\..*)' and group to source {'g0': '*.block_sparse_moe.'}
    # and target to source {'g0': '*.mlp.'}. This allows us to quickly find which pattern matched.
    if tp_plan != {}:
        tp_plan_alt, tp_plan_by_group_name, _ = build_glob_alternation(list(tp_plan.keys()))
    if dtype_plan != {}:
        dtype_policy_alt, dtype_policy_by_group_name, _ = build_glob_alternation(list(dtype_plan.keys()))

    pattern_to_converter = {k: converter for converter in converters for k in converter.source_patterns}

    state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
    for original_key, tensor in state_dict:
        # 1. Rename the key according to all renaming pattern and optional weight converter patterns
        renamed_key, source_pattern = rename_source_key(
            original_key, renamings, converters, prefix, meta_model_state_dict
        )

        # 2. finally, collect the tensor into the proper converter
        if renamed_key in missing_keys:
            empty_param = meta_model_state_dict.get(renamed_key)
            # If we enter here, we have a WeightConverter operation to perform
            if source_pattern is not None:
                new_converter = deepcopy(pattern_to_converter[source_pattern])
                # each target key gets its own converter instance
                mapping = param_name_to_load.setdefault(renamed_key, new_converter)
            # Otherwise, only potential renaming
            else:
                mapping = param_name_to_load.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
                source_pattern = original_key

            # 3. Handle dtype casting
            if (
                hf_quantizer
                and not hf_quantizer.pre_quantized
                and hf_quantizer.param_needs_quantization(model, renamed_key)
            ):
                mapping.quantization_operation = hf_quantizer.get_quantize_ops()

            _dtype = dtype
            if hf_quantizer and hf_quantizer.pre_quantized and original_key != renamed_key:
                # if the key was renamed as it is not available in the state dict otherwise, it means that we are deserializing it,
                # so we need to make sure to load the tensor with the same dtype from the checkpoint
                # TODO: make the condition more srict for native fp8 model such as qwen2moe fp8
                _dtype = None
            elif dtype_plan != {} and dtype_policy_alt.search(renamed_key):
                matched_dtype_pattern = dtype_policy_alt.search(renamed_key)
                if matched_dtype_pattern is not None:
                    _dtype = dtype_plan[dtype_policy_by_group_name[matched_dtype_pattern.lastgroup]]
            elif empty_param is not None and empty_param.dtype != _dtype:
                _dtype = empty_param.dtype  # usually correct when initializing

            # 4. Handle TP sharding or device_map placement
            future_or_tensor = None
            if device_mesh:
                if matched_tp_pattern := tp_plan_alt.search(renamed_key):
                    matched_tp_pattern = tp_plan_by_group_name[matched_tp_pattern.lastgroup]
                    if getattr(mapping, "distributed_operation", None) is None:
                        tp_layer = ALL_PARALLEL_STYLES[model.tp_plan[matched_tp_pattern]].__class__
                        mapping.distributed_operation = tp_layer(
                            device_mesh=device_mesh, rank=device_mesh.get_local_rank(), empty_param=empty_param.clone()
                        )
                    shard_index = len(mapping.collected_tensors.get(original_key, []))
                    future_or_tensor = spawn_tp_materialize(
                        thread_pool,
                        tensor,
                        mapping.distributed_operation,
                        shard_index,
                        device_map[""],
                        _dtype,
                    )

            if future_or_tensor is None:
                param_device = get_device(device_map, renamed_key, valid_torch_device=True)
                future_or_tensor = spawn_materialize(thread_pool, tensor, param_device, _dtype)

            mapping.add_tensor(renamed_key, original_key, source_pattern, future_or_tensor)
        elif source_pattern is not None:  # add all target keys as unexpected
            mapping = pattern_to_converter[source_pattern]
            for k in mapping.target_patterns:
                unexpected_keys.add(renamed_key.replace(mapping.target_patterns[0], k))
        else:
            unexpected_keys.add(renamed_key)

    try:
        total_entries = len(param_name_to_load)
        with logging.tqdm(total=total_entries, desc="Loading weights") as pbar:
            for first_param_name, mapping in param_name_to_load.items():
                pbar.update(1)
                pbar.set_postfix({"Materializing param": first_param_name})
                pbar.refresh()
                try:
                    realized_value, conversion_errors = mapping.convert(
                        first_param_name,
                        model=model,
                        config=model.config,
                        hf_quantizer=hf_quantizer,
                        missing_keys=missing_keys,
                        conversion_errors=conversion_errors,
                    )
                    for target_name, param in realized_value.items():
                        param = param[0] if isinstance(param, list) else param
                        param_device = get_device(device_map, target_name)
                        # Offloading support
                        if param_device == "disk" and (target_name not in model_buffers or offload_buffers):
                            disk_offload_index = offload_and_maybe_resave_param(
                                target_name, param, missing_keys, disk_offload_folder, disk_offload_index, mapping
                            )
                        else:
                            set_param_for_module(
                                model,
                                target_name,
                                param,
                                mismatch_keys,
                                missing_keys,
                                unexpected_keys,
                                mapping.distributed_operation,
                                hf_quantizer,
                            )

                    # Cleanup all the tensors that were gathered before next iteration
                    del realized_value

                except SkipParameters:
                    continue

    # Close the pool, independently of whether the code was interrupted or finished successfully
    finally:
        if thread_pool is not None:
            # `cancel_futures=True` in case the program was interupted, to avoid wasting time on exit
            thread_pool.shutdown(wait=False, cancel_futures=True)

    # Keep the current weight conversion mapping for later saving (in case it was coming directly from the user)
    model._weight_conversions = weight_mapping
    return missing_keys, unexpected_keys, mismatch_keys, disk_offload_index, conversion_errors


def revert_weight_conversion(model: PreTrainedModel, state_dict: dict[str, torch.Tensor]):
    """
    Revert the conversion mapping that was used to load the model with `from_pretrained`, or the default one
    if the model was created in another way and is part of the default mappings.
    """
    weight_conversions = getattr(model, "_weight_conversions", None)
    # In this case, the model was not created with `from_pretrained` -> let's check if it's in the hardcoded
    # mappings, and recreate the mapping from there if it is
    if weight_conversions is None:
        from .conversion_mapping import get_model_conversion_mapping

        # Do not resave with the legacy renaming, if present
        weight_conversions = get_model_conversion_mapping(model, add_legacy=False)
        weight_conversions = weight_conversions if len(weight_conversions) > 0 else None

    # We did not find any operations to perform -> quick escape
    if weight_conversions is None:
        return state_dict

    # Reverse all Transform to correctly match keys
    reverse_weight_conversion = [conversion.reverse_transform() for conversion in weight_conversions]
    # If we are still here, we need to create the (reverse) conversion mapping from scratch
    renamings = [entry for entry in reverse_weight_conversion if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in reverse_weight_conversion if isinstance(entry, WeightConverter)]
    pattern_to_converter = {k: converter for converter in converters for k in converter.source_patterns}
    conversion_mapping = {}

    state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
    for original_key, tensor in state_dict:
        # Rename the key according to all renaming pattern and optional weight converter patterns
        renamed_key, source_pattern = rename_source_key(original_key, renamings, converters)
        if source_pattern is not None:
            new_converter = deepcopy(pattern_to_converter[source_pattern])
            # each target key gets its own converter instance
            mapping = conversion_mapping.setdefault(renamed_key, new_converter)
        else:
            mapping = conversion_mapping.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
            source_pattern = original_key

        mapping.add_tensor(renamed_key, original_key, source_pattern, tensor)

    new_state_dict = {}
    for first_param_name, reversed_converter in conversion_mapping.items():
        # Apply the reverse converter
        realized_value, _ = reversed_converter.convert(first_param_name, model=model, config=model.config)
        for target_name, param in realized_value.items():
            param = param[0] if isinstance(param, list) else param
            new_state_dict[target_name] = param

    return new_state_dict
