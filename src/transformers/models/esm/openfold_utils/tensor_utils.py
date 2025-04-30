# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload

import torch
import torch.nn as nn
import torch.types


def add(m1: torch.Tensor, m2: torch.Tensor, inplace: bool) -> torch.Tensor:
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: torch.Tensor, inds: List[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int) -> torch.Tensor:
    return t.reshape(t.shape[:-no_dims] + (-1,))


def masked_mean(mask: torch.Tensor, value: torch.Tensor, dim: int, eps: float = 1e-4) -> torch.Tensor:
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def pts_to_distogram(
    pts: torch.Tensor, min_bin: torch.types.Number = 2.3125, max_bin: torch.types.Number = 21.6875, no_bins: int = 64
) -> torch.Tensor:
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=pts.device)
    dists = torch.sqrt(torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1))
    return torch.bucketize(dists, boundaries)


def dict_multimap(fn: Callable[[list], Any], dicts: List[dict]) -> dict:
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if isinstance(v, dict):
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x: torch.Tensor, v_bins: torch.Tensor) -> torch.Tensor:
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def batched_gather(data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0) -> torch.Tensor:
    ranges: List[Union[slice, torch.Tensor]] = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims: List[Union[slice, torch.Tensor]] = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    # Matt note: Editing this to get around the behaviour of using a list as an array index changing
    # in recent Numpy versions
    return data[tuple(ranges)]


T = TypeVar("T")


# With tree_map, a poor man's JAX tree_map
def dict_map(
    fn: Callable[[T], Any], dic: Dict[Any, Union[dict, list, tuple, T]], leaf_type: Type[T]
) -> Dict[Any, Union[dict, list, tuple, Any]]:
    new_dict: Dict[Any, Union[dict, list, tuple, Any]] = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


@overload
def tree_map(fn: Callable[[T], Any], tree: T, leaf_type: Type[T]) -> Any: ...


@overload
def tree_map(fn: Callable[[T], Any], tree: dict, leaf_type: Type[T]) -> dict: ...


@overload
def tree_map(fn: Callable[[T], Any], tree: list, leaf_type: Type[T]) -> list: ...


@overload
def tree_map(fn: Callable[[T], Any], tree: tuple, leaf_type: Type[T]) -> tuple: ...


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple(tree_map(fn, x, leaf_type) for x in tree)
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise TypeError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)
