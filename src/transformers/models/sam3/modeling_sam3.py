import collections
import collections.abc
import copy
import gc
import gzip
import html
import inspect
import io
import logging
import math
import os
import string
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from dataclasses import field as field_ptr_behaviour
from functools import lru_cache, partial, wraps
from itertools import repeat
from typing import Any, Callable, Optional, TypeVar, Union, get_args, get_origin

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision
from iopath.common.file_io import g_pathmgr
from PIL import Image
from timm.models.layers import DropPath
from torch import Tensor
from torch.nn import MultiheadAttention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._pytree import tree_map_only
from torchvision.ops.roi_align import RoIAlign
from typing_extensions import override


### Utils ------------------------------------------------------------- #
# Type variables for better type hinting
T = TypeVar("T")
Module = TypeVar("Module", bound=nn.Module)


MyTensor = Union[torch.Tensor, list[Any]]


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width


def load_image_as_single_frame(
    image_path,
    image_size,
    offload_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
):
    """Load an image as a single-frame video."""
    images, image_height, image_width = _load_img_as_tensor(image_path, image_size)
    images = images.unsqueeze(0).half()

    img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
    if not offload_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, image_height, image_width


@dataclass
class BatchedPointer:
    stage_ids: MyTensor
    stage_ids__type = torch.long
    query_ids: MyTensor
    query_ids__type = torch.long
    object_ids: MyTensor
    object_ids__type = torch.long
    ptr_mask: MyTensor
    ptr_mask__type = torch.bool
    ptr_types: MyTensor
    ptr_types__type = torch.long


@dataclass
class FindStage:
    img_ids: MyTensor
    img_ids__type = torch.long
    text_ids: MyTensor
    text_ids__type = torch.long

    input_boxes: MyTensor
    input_boxes__type = torch.float
    input_boxes_mask: MyTensor
    input_boxes_mask__type = torch.bool
    input_boxes_label: MyTensor
    input_boxes_label__type = torch.long

    input_points: MyTensor
    input_points__type = torch.float
    input_points_mask: MyTensor
    input_points_mask__type = torch.bool

    ptrs: Optional[BatchedPointer]
    ptrs_seg: Optional[BatchedPointer]
    # We track the object ids referred to by this query.
    # This is beneficial for tracking in videos without the need for pointers.
    object_ids: Optional[list[list]] = None  # List of objects per query

    input_boxes_before_embed: Optional[MyTensor] = None
    input_boxes_before_embed__type = torch.float

    input_points_before_embed: Optional[MyTensor] = None
    input_points_before_embed__type = torch.float


@dataclass
class GetStage:
    text_inputs: list[str]
    text_output: list[str]
    ptrs_x: BatchedPointer
    ptrs_y: BatchedPointer


@dataclass
class BatchedFindTarget:
    # The number of boxes in each find query
    num_boxes: MyTensor
    num_boxes__type = torch.long

    # Target boxes in normalized CxCywh format
    boxes: MyTensor
    boxes__type = torch.float
    # Target boxes in normalized CxCywh format but in padded representation
    # as used in BinaryHungarianMatcherV2 (unlike the packed ones in `boxes`)
    boxes_padded: MyTensor
    boxes_padded__type = torch.float

    # For hybrid matching, we repeat the boxes
    repeated_boxes: MyTensor
    repeated_boxes__type = torch.float

    # Target Segmentation masks
    segments: Optional[MyTensor]
    segments__type = torch.bool

    # Target Semantic Segmentation masks
    semantic_segments: Optional[MyTensor]
    semantic_segments__type = torch.bool

    is_valid_segment: Optional[MyTensor]
    is_valid_segment__type = torch.bool

    # Whether annotations are exhaustive for each query
    is_exhaustive: MyTensor
    is_exhaustive__type = torch.bool

    # The object id for each ground-truth box, in both packed and padded representations
    object_ids: MyTensor
    object_ids__type = torch.long
    object_ids_padded: MyTensor
    object_ids_padded__type = torch.long


@dataclass
class BatchedInferenceMetadata:
    """All metadata required to post-process a find stage"""

    # Coco id that corresponds to the "image" for evaluation by the coco evaluator
    coco_image_id: MyTensor
    coco_image_id__type = torch.long

    # id in the original dataset, such that we can use the original evaluator
    original_image_id: MyTensor
    original_image_id__type = torch.long

    # Original category id (if we want to use the original evaluator)
    original_category_id: MyTensor
    original_category_id__type = torch.int

    # Size of the raw image (height, width)
    original_size: MyTensor
    original_size__type = torch.long

    # id of the object in the media (track_id for a video)
    object_id: MyTensor
    object_id__type = torch.long

    # index of the frame in the media (0 in the case of a single-frame media)
    frame_index: MyTensor
    frame_index__type = torch.long

    # Adding for relations inference
    get_text_input: list[Optional[str]]

    # Adding for TA conditional inference
    is_conditioning_only: list[Optional[bool]]


@dataclass
class PointerExtractBehaviour:
    """This class contains configuration for the pointer extraction mechanism"""

    # If this is true, we will extract embeddings for the objects that match the ground truth
    # This is the behaviour used in training, and in some cases in evaluation
    # Note that if this true, the rest of the options are ignored
    match_to_gt: bool = True

    # --------- All options below are ignored if match_to_gt is true ---------

    # If this is true, the pointers will be sorted by confidence.
    # This will change the semantics of the object_id in the pointer description:
    #  - If this is false, then object_id = i will return the i-th object embedding (in the model's internal order)
    #  - If this is true, then object_id = i will return the i-th most confident object embedding
    sort_by_confidence: bool = True

    # If this is > 0, then only objects that have a confidence above this threshold will be returned
    # Note that this means some pointers may be missing.
    score_threshold: float = 0.0


class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def clone(self):
        new_tensors = self.tensors.clone()
        new_mask = None if self.mask is None else self.mask.clone()
        return NestedTensor(new_tensors, new_mask)

    @property
    def device(self):
        return self.tensors.device

    @property
    def shape(self):
        return self.tensors.shape

    # custom memory pinning method on custom type
    def pin_memory(self, device=None):
        self.tensors = self.tensors.pin_memory(device)
        if self.mask is not None:
            self.mask = self.mask.pin_memory(device)
        return self

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list, rounding=None):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            if rounding is not None:
                # Round to an even size to avoid rounding issues in fpn
                p = rounding
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, h, w

            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            mask_required = False
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
                if img.shape[1] != h or img.shape[2] != w:
                    mask_required = True

            if not mask_required:
                mask = None
        else:
            raise ValueError("not supported")
        return cls(tensor, mask)

    def to_tensor_list(self):
        """
        Undo the batching and padding, returning a list of tensors of
        different sizes.
        """
        data = self.tensors
        masks = self.mask.to(torch.int)
        # mask is 0 in the tensor and 1 outside it; argmax finds the smallest 1 index
        # and can be used to get heights and widths. Since padding is always to the
        # bottom and right, we can take the first value along x to get height and
        # visa versa for y and width. To handle cases with no padding, pad masks
        # out by one element so an edge is always found.
        masks = torch.nn.functional(masks, (0, 1, 0, 1), value=1)
        heights = torch.argmax(masks, dim=-2)
        assert torch.all((heights == heights[:, 0:1]) | (heights == 0)), "Image mask is not a box."
        heights = heights[:, 0]
        widths = torch.argmax(masks, dim=-1)
        assert torch.all((widths == widths[:, 0:1]) | (widths == 0)), "Image mask is not a box."
        widths = widths[:, 0]
        return [data[i, :, :h, :w] for i, (h, w) in enumerate(zip(heights, widths))]

    def __repr__(self):
        return repr(self.tensors)


@dataclass
class BatchedDatapoint:
    img_batch: NestedTensor
    find_text_batch: list[str]
    find_inputs: list[FindStage]
    find_targets: list[BatchedFindTarget]
    find_metadatas: list[BatchedInferenceMetadata]
    get_queries: GetStage
    ptr_behaviour: PointerExtractBehaviour = field_ptr_behaviour(default_factory=lambda: PointerExtractBehaviour())
    raw_images: Optional[list[Any]] = None

    def pin_memory(self, device=None):
        return recursive_pin_memory(self, device)


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def convert_my_tensors(obj):
    def is_optional_field(field) -> bool:
        return get_origin(field) is Union and type(None) in get_args(field)

    for field in fields(obj):
        if is_dataclass(getattr(obj, field.name)):
            convert_my_tensors(getattr(obj, field.name))
            continue

        field_type = field.type
        if is_optional_field(field.type):
            field_type = Union[get_args(field.type)[:-1]]  # Get the Optional field type

        if field_type != MyTensor or getattr(obj, field.name) is None:
            continue

        elif len(getattr(obj, field.name)) and isinstance(getattr(obj, field.name)[0], torch.Tensor):
            stack_dim = 0
            if field.name in [
                "input_boxes_before_embed",
                "input_boxes",
                "input_boxes_label",
            ]:
                stack_dim = 1
            setattr(
                obj,
                field.name,
                torch.stack(getattr(obj, field.name), dim=stack_dim).to(getattr(obj, field.name + "__type")),
            )
        else:
            setattr(
                obj,
                field.name,
                torch.as_tensor(getattr(obj, field.name), dtype=getattr(obj, field.name + "__type")),
            )
    return obj


def recursive_to(data, *args, **kwargs):
    if isinstance(data, torch.Tensor):
        ret = data.to(*args, **kwargs)
    elif isinstance(data, Mapping):
        ret = type(data)()
        for key in data:
            ret[key] = recursive_to(data[key], *args, **kwargs)
    elif isinstance(data, tuple):
        ret = ()
        for value in data:
            ret += (recursive_to(value, *args, **kwargs),)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        ret = type(data)()
        for value in data:
            ret.append(recursive_to(value, *args, **kwargs))
    elif is_dataclass(data):
        ret_cls = type(data)
        ret_fields = {field.name: recursive_to(getattr(data, field.name), *args, **kwargs) for field in fields(data)}
        ret = ret_cls(**ret_fields)
    else:
        ret = data
    return ret


def recursive_pin_memory(data, device=None):
    """Pinning function that also supports dataclasses."""

    if isinstance(data, torch.Tensor):
        return data.pin_memory(device)
    elif isinstance(data, (str, bytes)):
        return data
    elif isinstance(data, collections.abc.Mapping):
        pinned_data = {k: recursive_pin_memory(sample, device) for k, sample in data.items()}
        try:
            return type(data)(pinned_data)  # type: ignore[call-arg]
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return pinned_data
    elif isinstance(data, collections.abc.Sequence):
        pinned_data = [recursive_pin_memory(sample, device) for sample in data]  # type: ignore[assignment]
        try:
            return type(data)(pinned_data)  # type: ignore[call-arg]
        except TypeError:
            # The sequence type may not support `__init__(iterable)` (e.g., `range`).
            return pinned_data
    elif is_dataclass(data):
        pinned_data = {field.name: recursive_pin_memory(getattr(data, field.name), device) for field in fields(data)}
        return type(data)(**pinned_data)
    elif hasattr(data, "pin_memory"):
        return data.pin_memory(device)

    return data


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(pos_tensor, num_feats=256):
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos


def activation_ckpt_wrapper(module: Union[nn.Module, Callable]) -> Callable:
    """
    Wraps a given module to enable or disable activation checkpointing.

    Activation checkpointing (gradient checkpointing) trades compute for memory by
    recomputing intermediate activations during the backward pass instead of storing
    them in memory during the forward pass.

    When activation checkpointing is enabled, the wrapper expects only keyword arguments,
    and it maps these to positional arguments based on the module's signature.

    Args:
        module: The module or function to wrap with activation checkpointing

    Returns:
        A wrapped callable that supports activation checkpointing

    Usage:
        The returned wrapper function can be called with the same arguments as the
        original module, with an additional `act_ckpt_enable` keyword argument to control
        activation checkpointing and optional `use_reentrant` parameter.

    Example:
        ```python
        wrapped_module = activation_ckpt_wrapper(my_module)
        output = wrapped_module(x=input_tensor, y=another_tensor, act_ckpt_enable=True)
        ```
    """

    @wraps(module)
    def act_ckpt_wrapper(*args, act_ckpt_enable: bool = True, use_reentrant: bool = False, **kwargs):
        if act_ckpt_enable:
            if len(args) > 0:
                raise ValueError("This wrapper expects keyword arguments only when `act_ckpt_enable=True`")
            # Get the signature of the target function/module
            callable_fn = module.forward if isinstance(module, nn.Module) else module
            sig = inspect.signature(callable_fn)
            # Create a mapping of parameter names to their default values
            param_defaults = {name: param.default for name, param in sig.parameters.items()}
            args = []
            for p_name in param_defaults.keys():
                if p_name in kwargs:
                    args.append(kwargs.pop(p_name))
                elif param_defaults[p_name] is not inspect.Parameter.empty:
                    # Set arg to default value if it's not in kwargs. Useful for primitive types or args that default to None
                    args.append(param_defaults[p_name])
                elif sig.parameters[p_name].kind is not inspect.Parameter.VAR_KEYWORD:  # Skip **kwargs parameter
                    raise ValueError(f"Missing positional argument: {p_name}")

            # Scan remaining kwargs for torch.Tensor
            remaining_keys = list(kwargs.keys())
            for key in remaining_keys:
                if isinstance(kwargs[key], torch.Tensor):
                    # Remove the tensor from kwargs, assuming it's not required by the module.
                    # If it is required, the module's signature should be modified to accept it as a positional or keyword argument.
                    kwargs[key] = "_REMOVED_BY_ACT_CKPT_WRAPPER_"

            ret = checkpoint.checkpoint(module, *args, use_reentrant=use_reentrant, **kwargs)
        else:
            ret = module(*args, **kwargs)

        return ret

    return act_ckpt_wrapper


def clone_output_wrapper(f: Callable[..., T]) -> Callable[..., T]:
    """
    Clone the CUDA output tensors of a function to avoid in-place operations.

    This wrapper is useful when working with torch.compile to prevent errors
    related to in-place operations on tensors.

    Args:
        f: The function whose CUDA tensor outputs should be cloned

    Returns:
        A wrapped function that clones any CUDA tensor outputs
    """

    @wraps(f)
    def wrapped(*args, **kwargs):
        outputs = f(*args, **kwargs)
        return tree_map_only(torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs)

    return wrapped


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [(x), (y), (x + w), (y + h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_cxcywh(x):
    x, y, w, h = x.unbind(-1)
    b = [(x + 0.5 * w), (y + 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x, y, X, Y = x.unbind(-1)
    b = [(x), (y), (X - x), (Y - y)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """
    Batched version of box area. Boxes should be in [x0, y0, x1, y1] format.

    Inputs:
    - boxes: Tensor of shape (..., 4)

    Returns:
    - areas: Tensor of shape (...,)
    """
    x0, y0, x1, y1 = boxes.unbind(-1)
    return (x1 - x0) * (y1 - y0)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # whether to add the output as a residual connection to the input
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        precompute_resolution: Optional[int] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
        # Precompute positional encodings under `precompute_resolution` to fill the cache
        # and avoid symbolic shape tracing errors in torch.compile in PyTorch 2.4 nightly.
        if precompute_resolution is not None:
            # We precompute pos enc for stride 4, 8, 16 and 32 to fill `self.cache`.
            precompute_sizes = [
                (precompute_resolution // 4, precompute_resolution // 4),
                (precompute_resolution // 8, precompute_resolution // 8),
                (precompute_resolution // 16, precompute_resolution // 16),
                (precompute_resolution // 32, precompute_resolution // 32),
            ]
            for size in precompute_sizes:
                tensors = torch.zeros((1, 1) + size, device="cuda")
                self.forward(NestedTensor(tensors, None))
                # further clone and detach it in the cache (just to be safe)
                self.cache[size] = self.cache[size].clone().detach()

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask

        cache_key = None
        if mask is None:
            cache_key = (x.shape[-2], x.shape[-1])
            if cache_key in self.cache:
                return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
            y_embed = (
                torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
                .view(1, -1, 1)
                .repeat(x.shape[0], 1, x.shape[-1])
            )
            x_embed = (
                torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
                .view(1, 1, -1)
                .repeat(x.shape[0], x.shape[-2], 1)
            )
        else:
            not_mask = ~mask
            # assert not_mask.all().item()
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        if cache_key is not None:
            self.cache[cache_key] = pos[0]
        return pos


### End utils ------------------------------------------------------------- #

### Tokenizer ---------------------------------------------------------------------------- #

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEFAULT_CONTEXT_LENGTH = 77


@lru_cache
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def _clean_canonicalize(x):
    # basic, remove whitespace, remove punctuation, lower case
    return canonicalize_text(basic_clean(x))


def _clean_lower(x):
    # basic, remove whitespace, lower case
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
    # basic, remove whitespace
    return whitespace_clean(basic_clean(x))


def get_clean_fn(type: str):
    if type == "canonicalize":
        return _clean_canonicalize
    elif type == "lower":
        return _clean_lower
    elif type == "whitespace":
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."


def canonicalize_text(text, *, keep_punctuation_exact_string=None):
    """Returns canonicalized `text` (lowercase and punctuation removed).
    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94
    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SimpleTokenizer(object):
    def __init__(
        self,
        bpe_path: Union[str, os.PathLike],
        additional_special_tokens: Optional[list[str]] = None,
        context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
        clean: str = "lower",
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with g_pathmgr.open(bpe_path, "rb") as fh:
            bpe_bytes = io.BytesIO(fh.read())
            merges = gzip.open(bpe_bytes).read().decode("utf-8").split("\n")
        # merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace").replace("</w>", " ")
        return text

    def __call__(self, texts: Union[str, list[str]], context_length: Optional[int] = None) -> torch.LongTensor:
        """Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]
        context_length = context_length or self.context_length
        assert context_length, "Please set a valid context length"
        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = self.eot_token_id
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result


### End Tokenizer ---------------------------------------------------------------------------- #


### Segmentation Head ------------------------------------------------------------- #
class LinearPresenceHead(nn.Sequential):
    def __init__(self, d_model):
        # a hack to make `LinearPresenceHead` compatible with old checkpoints
        super().__init__(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(self, hs, prompt, prompt_mask):
        return super().forward(hs)


class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim, mask_dim):
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, obj_queries, pixel_embed):
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum("bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed)
            else:
                mask_preds = torch.einsum("bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed)
        else:
            # Assumed to have aux masks
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum("lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed)
            else:
                mask_preds = torch.einsum("lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed)

        return mask_preds


class SegmentationHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        use_encoder_inputs=False,
        aux_masks=False,
        no_dec=False,
        pixel_decoder=None,
        act_ckpt=False,
        shared_conv=False,
        compile_mode_pixel_decoder=None,
    ):
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)
        else:
            self.mask_predictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim)

        self.act_ckpt = act_ckpt

        # used to update the output dictionary
        self.instance_keys = ["pred_masks"]

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def _embed_pixels(
        self,
        backbone_feats: NestedTensor,
        image_ids,
        encoder_hidden_states,
    ) -> torch.Tensor:
        feature_device = backbone_feats[0].tensors.device  # features could be on CPU
        model_device = self.device
        image_ids_ = image_ids.to(feature_device)
        if self.use_encoder_inputs:
            if backbone_feats[0].tensors.shape[0] > 1:
                # For bs > 1, we construct the per query backbone features
                backbone_visual_feats = []
                for feat in backbone_feats:
                    # Copy the img features per query (pixel decoder won't share img feats)
                    backbone_visual_feats.append(
                        NestedTensor(
                            feat.tensors[image_ids_, ...],
                            (feat.mask[image_ids_, ...] if feat.mask is not None else None),
                        ).to(model_device)
                    )
            else:
                # Bs=1, we rely on broadcasting for query-based processing
                backbone_visual_feats = [bb_feat.clone() for bb_feat in backbone_feats]
            # Extract visual embeddings
            encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
            spatial_dim = math.prod(backbone_feats[-1].tensors.shape[-2:])
            encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(
                -1, *backbone_feats[-1].tensors.shape[1:]
            )

            backbone_visual_feats[-1].tensors = encoder_visual_embed
            if self.act_ckpt:
                pixel_embed = checkpoint.checkpoint(self.pixel_decoder, backbone_visual_feats, use_reentrant=False)
            else:
                pixel_embed = self.pixel_decoder(backbone_visual_feats)
        else:
            backbone_feats = [x.to(model_device) for x in backbone_feats]
            pixel_embed = self.pixel_decoder(backbone_feats)
            if pixel_embed.shape[0] == 1:
                # For batch_size=1 training, we can avoid the indexing to save memory
                pixel_embed = pixel_embed.squeeze(0)
            else:
                pixel_embed = pixel_embed[image_ids, ...]
        return pixel_embed

    def forward(
        self,
        backbone_feats: NestedTensor,
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        if self.no_dec:
            mask_pred = self.mask_predictor(pixel_embed)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}


class PixelDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_upsampling_stages,
        interpolation_mode="nearest",
        shared_conv=False,
        compile_mode=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode
        conv_layers = []
        norms = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
            norms.append(nn.GroupNorm(8, self.hidden_dim))  # TODO: Make the norm configurable

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norms)
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].out_channels
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, dynamic=True, fullgraph=True)
            # Needed to make checkpointing happy. But we don't know if the module is checkpointed, so we disable it by default.
            torch._dynamo.config.optimize_ddp = False

    def forward(self, backbone_feats: NestedTensor):
        # Assumes backbone features are already projected (C == hidden dim)

        prev_fpn = backbone_feats[-1].tensors
        fpn_feats = backbone_feats[:-1]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat.tensors
            prev_fpn = curr_fpn + F.interpolate(prev_fpn, size=curr_fpn.shape[-2:], mode=self.interpolation_mode)
            if self.shared_conv:
                # only one conv layer
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

        return prev_fpn


class UniversalSegmentationHead(SegmentationHead):
    """This module handles semantic+instance segmentation"""

    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        pixel_decoder,
        aux_masks=False,
        no_dec=False,
        act_ckpt=False,
        presence_head: bool = False,
        dot_product_scorer=None,
        cross_attend_prompt=None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, "Specifying a dot product scorer without a presence head is likely a mistake"

        self.presence_head = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer if dot_product_scorer is not None else LinearPresenceHead(self.d_model)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, self.d_model, kernel_size=1)

    def forward(
        self,
        backbone_feats: NestedTensor,
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, Optional[torch.Tensor]]:
        assert encoder_hidden_states is not None
        bs = encoder_hidden_states.shape[1]

        if self.cross_attend_prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = (
                self.presence_head(
                    pooled_enc.view(1, bs, 1, self.d_model),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        instance_embeds = self.instance_seg_head(pixel_embed)

        if self.no_dec:
            mask_pred = self.mask_predictor(instance_embeds)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, instance_embeds)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        return {
            "pred_masks": mask_pred,
            "semantic_seg": self.semantic_seg_head(pixel_embed),
            "presence_logit": presence_logit,
        }


### End Segmentation Head ------------------------------------------------------------- #


### Sequence Geometry Encoder ------------------------------------------------------------- #


class Prompt:
    """Utility class to manipulate geometric prompts.

    We expect the sequences in pytorch convention, that is sequence first, batch second
    The dimensions are expected as follows:
    box_embeddings shape: N_boxes x B x C_box
    box_mask shape: B x N_boxes. Can be None if nothing is masked out
    point_embeddings shape: N_points x B x C_point
    point_mask shape: B x N_points. Can be None if nothing is masked out
    mask_embeddings shape: N_masks x B x 1 x H_mask x W_mask
    mask_mask shape: B x N_masks. Can be None if nothing is masked out

    We also store positive/negative labels. These tensors are also stored batch-first
    If they are None, we'll assume positive labels everywhere
    box_labels: long tensor of shape N_boxes x B
    point_labels: long tensor of shape N_points x B
    mask_labels: long tensor of shape N_masks x B
    """

    def __init__(
        self,
        box_embeddings=None,
        box_mask=None,
        point_embeddings=None,
        point_mask=None,
        box_labels=None,
        point_labels=None,
        mask_embeddings=None,
        mask_mask=None,  # Attention mask for mask prompt
        mask_labels=None,
    ):
        # Check for null prompt
        if box_embeddings is None and point_embeddings is None and mask_embeddings is None:
            self.box_embeddings = None
            self.box_labels = None
            self.box_mask = None
            self.point_embeddings = None
            self.point_labels = None
            self.point_mask = None
            self.mask_embeddings = None
            self.mask_mask = None
            # Masks are assumed positive only for now.
            self.mask_labels = None
            return
        # Get sequence lengths and device
        box_seq_len, point_seq_len, mask_seq_len, bs, device = self._init_seq_len_and_device(
            box_embeddings, point_embeddings, mask_embeddings
        )

        # Initialize embeds, labels, attention masks.
        box_embeddings, box_labels, box_mask = self._init_box(
            box_embeddings, box_labels, box_mask, box_seq_len, bs, device
        )
        point_embeddings, point_labels, point_mask = self._init_point(
            point_embeddings, point_labels, point_mask, point_seq_len, bs, device
        )
        mask_embeddings, mask_labels, mask_mask = self._init_mask(
            mask_embeddings, mask_labels, mask_mask, mask_seq_len, bs, device
        )

        # Dimension checks
        assert box_embeddings is not None and list(box_embeddings.shape[:2]) == [
            box_seq_len,
            bs,
        ], f"Wrong dimension for box embeddings. Expected [{box_seq_len}, {bs}, *] got {box_embeddings.shape}"
        assert box_mask is not None and list(box_mask.shape) == [
            bs,
            box_seq_len,
        ], f"Wrong dimension for box mask. Expected [{bs}, {box_seq_len}] got {box_mask.shape}"
        assert point_embeddings is not None and list(point_embeddings.shape[:2]) == [
            point_seq_len,
            bs,
        ], f"Wrong dimension for point embeddings. Expected [{point_seq_len}, {bs}, *] got {point_embeddings.shape}"
        assert point_mask is not None and list(point_mask.shape) == [
            bs,
            point_seq_len,
        ], f"Wrong dimension for point mask. Expected [{bs}, {point_seq_len}] got {point_mask.shape}"
        assert box_labels is not None and list(box_labels.shape) == [
            box_seq_len,
            bs,
        ], f"Wrong dimension for box labels. Expected [{box_seq_len}, {bs}] got {box_labels.shape}"
        assert point_labels is not None and list(point_labels.shape) == [
            point_seq_len,
            bs,
        ], f"Wrong dimension for point labels. Expected [{point_seq_len}, {bs}] got {point_labels.shape}"
        assert (
            # Allowed to be None, we leave it to the encoder to check for validity before encoding.
            mask_embeddings is None
            or list(mask_embeddings.shape[:2])
            == [
                mask_seq_len,
                bs,
            ]
        ), f"Wrong dimension for mask embeddings. Expected [{mask_seq_len}, {bs}, *] got {mask_embeddings.shape}"
        assert mask_mask is None or list(mask_mask.shape) == [
            bs,
            mask_seq_len,
        ], f"Wrong dimension for mask attn. mask. Expected [{bs}, {mask_seq_len}] got {mask_mask.shape}"

        # Device checks
        assert box_embeddings is not None and box_embeddings.device == device, (
            f"Expected box embeddings to be on device {device}, got {box_embeddings.device}"
        )
        assert box_mask is not None and box_mask.device == device, (
            f"Expected box mask to be on device {device}, got {box_mask.device}"
        )
        assert box_labels is not None and box_labels.device == device, (
            f"Expected box labels to be on device {device}, got {box_labels.device}"
        )
        assert point_embeddings is not None and point_embeddings.device == device, (
            f"Expected point embeddings to be on device {device}, got {point_embeddings.device}"
        )
        assert point_mask is not None and point_mask.device == device, (
            f"Expected point mask to be on device {device}, got {point_mask.device}"
        )
        assert point_labels is not None and point_labels.device == device, (
            f"Expected point labels to be on device {device}, got {point_labels.device}"
        )
        assert mask_embeddings is None or mask_embeddings.device == device, (
            f"Expected mask embeddings to be on device {device}, got {mask_embeddings.device}"
        )
        assert mask_mask is None or mask_mask.device == device, (
            f"Expected mask attn. mask to be on device {device}, got {mask_mask.device}"
        )

        self.box_embeddings = box_embeddings
        self.point_embeddings = point_embeddings
        self.box_mask = box_mask
        self.point_mask = point_mask
        self.box_labels = box_labels
        self.point_labels = point_labels
        self.mask_embeddings = mask_embeddings
        self.mask_labels = mask_labels
        self.mask_mask = mask_mask

    def _init_seq_len_and_device(self, box_embeddings, point_embeddings, mask_embeddings):
        box_seq_len = point_seq_len = mask_seq_len = 0
        bs = None
        device = None
        if box_embeddings is not None:
            bs = box_embeddings.shape[1]
            box_seq_len = box_embeddings.shape[0]
            device = box_embeddings.device

        if point_embeddings is not None:
            point_seq_len = point_embeddings.shape[0]
            if bs is not None:
                assert bs == point_embeddings.shape[1], (
                    f"Batch size mismatch between box and point embeddings. Got {bs} and {point_embeddings.shape[1]}."
                )
            else:
                bs = point_embeddings.shape[1]
            if device is not None:
                assert device == point_embeddings.device, "Device mismatch between box and point embeddings"
            else:
                device = point_embeddings.device

        if mask_embeddings is not None:
            mask_seq_len = mask_embeddings.shape[0]
            if bs is not None:
                assert bs == mask_embeddings.shape[1], (
                    f"Batch size mismatch between box/point and mask embedding. Got {bs} and {mask_embeddings.shape[1]}"
                )
            else:
                bs = mask_embeddings.shape[1]
            if device is not None:
                assert device == mask_embeddings.device, "Device mismatch between box/point and mask embeddings."
            else:
                device = mask_embeddings.device

        return box_seq_len, point_seq_len, mask_seq_len, bs, device

    def _init_box(self, box_embeddings, box_labels, box_mask, box_seq_len, bs, device):
        if box_embeddings is None:
            box_embeddings = torch.zeros(box_seq_len, bs, 4, device=device)
        if box_labels is None:
            box_labels = torch.ones(box_seq_len, bs, device=device, dtype=torch.long)
        if box_mask is None:
            box_mask = torch.zeros(bs, box_seq_len, device=device, dtype=torch.bool)
        return box_embeddings, box_labels, box_mask

    def _init_point(self, point_embeddings, point_labels, point_mask, point_seq_len, bs, device):
        """
        Identical to _init_box. Except that C=2 for points (vs. 4 for boxes).
        """
        if point_embeddings is None:
            point_embeddings = torch.zeros(point_seq_len, bs, 2, device=device)
        if point_labels is None:
            point_labels = torch.ones(point_seq_len, bs, device=device, dtype=torch.long)
        if point_mask is None:
            point_mask = torch.zeros(bs, point_seq_len, device=device, dtype=torch.bool)
        return point_embeddings, point_labels, point_mask

    def _init_mask(self, mask_embeddings, mask_labels, mask_mask, mask_seq_len, bs, device):
        # NOTE: Mask embeddings can be of arbitrary resolution, so we don't initialize it here.
        # In case we append new mask, we check that its resolution matches exisiting ones (if any).
        # In case mask_embeddings is None, we should never encode it.
        if mask_labels is None:
            mask_labels = torch.ones(mask_seq_len, bs, device=device, dtype=torch.long)
        if mask_mask is None:
            mask_mask = torch.zeros(bs, mask_seq_len, device=device, dtype=torch.bool)
        return mask_embeddings, mask_labels, mask_mask

    def append_boxes(self, boxes, labels, mask=None):
        if self.box_embeddings is None:
            self.box_embeddings = boxes
            self.box_labels = labels
            self.box_mask = mask
            return

        bs = self.box_embeddings.shape[1]
        assert boxes.shape[1] == labels.shape[1] == bs
        assert list(boxes.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = torch.zeros(bs, boxes.shape[0], dtype=torch.bool, device=boxes.device)

        self.box_labels, _ = concat_padded_sequences(
            self.box_labels.unsqueeze(-1), self.box_mask, labels.unsqueeze(-1), mask
        )
        self.box_labels = self.box_labels.squeeze(-1)
        self.box_embeddings, self.box_mask = concat_padded_sequences(self.box_embeddings, self.box_mask, boxes, mask)

    def append_points(self, points, labels, mask=None):
        if self.point_embeddings is None:
            self.point_embeddings = points
            self.point_labels = labels
            self.point_mask = mask
            return

        bs = self.point_embeddings.shape[1]
        assert points.shape[1] == labels.shape[1] == bs
        assert list(points.shape[:2]) == list(labels.shape[:2])
        if mask is None:
            mask = torch.zeros(bs, points.shape[0], dtype=torch.bool, device=points.device)

        self.point_labels, _ = concat_padded_sequences(
            self.point_labels.unsqueeze(-1), self.point_mask, labels.unsqueeze(-1), mask
        )
        self.point_labels = self.point_labels.squeeze(-1)
        self.point_embeddings, self.point_mask = concat_padded_sequences(
            self.point_embeddings, self.point_mask, points, mask
        )

    def append_masks(self, masks, labels=None, attn_mask=None):
        if labels is not None:
            assert list(masks.shape[:2]) == list(labels.shape[:2])
        if self.mask_embeddings is None:
            self.mask_embeddings = masks
            mask_seq_len, bs = masks.shape[:2]
            if labels is None:
                self.mask_labels = torch.ones(mask_seq_len, bs, device=masks.device, dtype=torch.long)
            else:
                self.mask_labels = labels
            if attn_mask is None:
                self.mask_mask = torch.zeros(bs, mask_seq_len, device=masks.device, dtype=torch.bool)
            else:
                self.mask_mask = attn_mask
        else:
            raise NotImplementedError("Only one mask per prompt is supported.")

    def clone(self):
        return Prompt(
            box_embeddings=(None if self.box_embeddings is None else self.box_embeddings.clone()),
            box_mask=None if self.box_mask is None else self.box_mask.clone(),
            point_embeddings=(None if self.point_embeddings is None else self.point_embeddings.clone()),
            point_mask=None if self.point_mask is None else self.point_mask.clone(),
            box_labels=None if self.box_labels is None else self.box_labels.clone(),
            point_labels=(None if self.point_labels is None else self.point_labels.clone()),
        )


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class SimpleFuser(nn.Module):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # normally x: (N, C, H, W)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleMaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
        # Option to interpolate the input mask first before downsampling using convs. In that case, the total_stride is assumed to be after interpolation.
        # If set to input resolution or None, we don't interpolate. We default to None to be safe (for older configs or if not explicitly set)
        interpol_size=None,
    ):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.interpol_size = interpol_size
        if self.interpol_size is not None:
            assert isinstance(self.interpol_size, (list, tuple)), (
                f"Unsupported type {type(self.interpol_size)}. Should be a list or tuple."
            )
            self.interpol_size = list(interpol_size)
            assert len(self.interpol_size) == 2

    def forward(self, x: torch.Tensor):
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = F.interpolate(
                x.float(),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
        return self.encoder(x)


def is_right_padded(mask):
    """Given a padding mask (following pytorch convention, 1s for padded values),
    returns whether the padding is on the right or not."""
    return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    """
    Concatenates two right-padded sequences, such that the resulting sequence
    is contiguous and also right-padded.

    Following pytorch's convention, tensors are sequence first, and the mask are
    batch first, with 1s for padded values.

    :param seq1: A tensor of shape (seq1_length, batch_size, hidden_size).
    :param mask1: A tensor of shape (batch_size, seq1_length).
    :param seq2: A tensor of shape (seq2_length, batch_size,  hidden_size).
    :param mask2: A tensor of shape (batch_size, seq2_length).
    :param return_index: If True, also returns the index of the ids of the element of seq2
        in the concatenated sequence. This can be used to retrieve the elements of seq2
    :return: A tuple (concatenated_sequence, concatenated_mask) if return_index is False,
        otherwise (concatenated_sequence, concatenated_mask, index).
    """
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length, batch_size, hidden_size = seq2.shape

    assert batch_size == seq1.size(1) == seq2.size(1) == mask1.size(0) == mask2.size(0)
    assert hidden_size == seq1.size(2) == seq2.size(2)
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)

    torch._assert_async(is_right_padded(mask1))
    torch._assert_async(is_right_padded(mask2))

    actual_seq1_lengths = (~mask1).sum(dim=-1)
    actual_seq2_lengths = (~mask2).sum(dim=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length
    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1) >= final_lengths[:, None]
    )

    # (max_len, batch_size, hidden_size)
    concatenated_sequence = torch.zeros((max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype)
    concatenated_sequence[:seq1_length, :, :] = seq1

    # At this point, the element of seq1 are in the right place
    # We just need to shift the elements of seq2

    index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
    index = index + actual_seq1_lengths[None]

    concatenated_sequence = concatenated_sequence.scatter(0, index[:, :, None].expand(-1, -1, hidden_size), seq2)

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


class MaskEncoder(nn.Module):
    """
    Base class for mask encoders.
    """

    def __init__(
        self,
        mask_downsampler: nn.Module,
        position_encoding: nn.Module,
    ):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.position_encoding = position_encoding

    def forward(self, masks, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        masks = self.mask_downsampler(masks)
        masks_nt = NestedTensor(
            tensors=masks,
            mask=None,
        )
        masks_pos = self.position_encoding(masks_nt).to(masks_nt.tensors.dtype)

        return masks, masks_pos


class FusedMaskEncoder(MaskEncoder):
    """
    Identical to memory.SimpleMaskEncoder but follows the interface of geometry_encoders.MaskEncoder.
    We also remove the `skip_mask_sigmoid` option (to be handled outside the MaskEncoder).
    Fuses backbone image features with mask features.
    """

    def __init__(
        self,
        mask_downsampler: nn.Module,
        position_encoding: nn.Module,
        fuser: nn.Module,
        in_dim: int = 256,
        out_dim: int = 256,
    ):
        super().__init__(mask_downsampler, position_encoding)
        self.fuser = fuser
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)

    @override
    def forward(
        self,
        masks: torch.Tensor,
        pix_feat: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        x_pos = NestedTensor(
            tensors=x,
            mask=None,
        )
        pos = self.position_encoding(x_pos).to(x_pos.tensors.dtype)

        return x, pos


class SequenceGeometryEncoder(nn.Module):
    """
    This a fully fledged encoder for geometric prompts.
    It assumes boxes are passed in the "normalized CxCyWH" format, and points in normalized xy
    This allows flexibility in how to encode the features (eg do pooling)

    Points and boxes can be encoded with any of the three possibilities:
     - direct projection: we just compute a linear from coordinate space to d_model
     - pooling: pool features from the backbone in the requested location.
                For boxes, it's a roi align
                For points it's a grid sample
     - pos encoder: Take the position encoding of the point or box center

    These three options are mutually compatible. If several are selected, we'll take a simple addition

    As an alternative, we offer the possibility to encode points only.
    In that case, the boxes are converted to two points for the top left and bottom right corners (with appropriate labels)

    On top of these encodings, we offer the possibility to further encode the prompt sequence with a transformer.
    """

    def __init__(
        self,
        encode_boxes_as_points: bool,
        points_direct_project: bool,
        points_pool: bool,
        points_pos_enc: bool,
        boxes_direct_project: bool,
        boxes_pool: bool,
        boxes_pos_enc: bool,
        d_model: int,
        pos_enc,
        num_layers: int,
        layer: nn.Module,
        roi_size: int = 7,  # for boxes pool
        add_cls: bool = True,
        add_post_encode_proj: bool = True,
        mask_encoder: MaskEncoder = None,
        add_mask_label: bool = False,
        use_act_ckpt: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_enc = pos_enc
        self.encode_boxes_as_points = encode_boxes_as_points
        self.roi_size = roi_size
        # There usually are two labels: positive and negatives.
        # If we encode boxes as points, we have 3 types of points: regular, top left, bottom right
        # These 3 types can be positives or negatives, hence 2*3 = 6 labels
        num_labels = 6 if self.encode_boxes_as_points else 2
        self.label_embed = torch.nn.Embedding(num_labels, self.d_model)

        # This is a cls token, can be used for pooling if need be.
        # It also ensures that the encoded sequences are always non-empty
        self.cls_embed = None
        if add_cls:
            self.cls_embed = torch.nn.Embedding(1, self.d_model)

        assert points_direct_project or points_pos_enc or points_pool, "Error: need at least one way to encode points"
        assert encode_boxes_as_points or boxes_direct_project or boxes_pos_enc or boxes_pool, (
            "Error: need at least one way to encode boxes"
        )

        self.points_direct_project = None
        if points_direct_project:
            self.points_direct_project = nn.Linear(2, self.d_model)
        self.points_pool_project = None
        if points_pool:
            self.points_pool_project = nn.Linear(self.d_model, self.d_model)
        self.points_pos_enc_project = None
        if points_pos_enc:
            self.points_pos_enc_project = nn.Linear(self.d_model, self.d_model)

        self.boxes_direct_project = None
        self.boxes_pool_project = None
        self.boxes_pos_enc_project = None
        if not encode_boxes_as_points:
            if boxes_direct_project:
                self.boxes_direct_project = nn.Linear(4, self.d_model)
            if boxes_pool:
                self.boxes_pool_project = nn.Conv2d(self.d_model, self.d_model, self.roi_size)
            if boxes_pos_enc:
                self.boxes_pos_enc_project = nn.Linear(self.d_model + 2, self.d_model)

        self.final_proj = None
        if add_post_encode_proj:
            self.final_proj = nn.Linear(self.d_model, self.d_model)
            self.norm = nn.LayerNorm(self.d_model)

        self.img_pre_norm = nn.Identity()
        if self.points_pool_project is not None or self.boxes_pool_project is not None:
            self.img_pre_norm = nn.LayerNorm(self.d_model)

        self.encode = None
        if num_layers > 0:
            assert add_cls, "It's currently highly recommended to add a CLS when using a transformer"
            self.encode = get_clones(layer, num_layers)
            self.encode_norm = nn.LayerNorm(self.d_model)

        if mask_encoder is not None:
            assert isinstance(mask_encoder, MaskEncoder), (
                f"Expected mask_encoder of type MaskEncoder. Got {type(mask_encoder)}."
            )
            if add_mask_label:
                self.mask_label_embed = torch.nn.Embedding(2, self.d_model)
        self.add_mask_label = add_mask_label
        self.mask_encoder = mask_encoder
        self.use_act_ckpt = use_act_ckpt

    def _encode_points(self, points, points_mask, points_labels, img_feats):
        points_embed = None
        n_points, bs = points.shape[:2]

        if self.points_direct_project is not None:
            proj = self.points_direct_project(points)
            assert points_embed is None
            points_embed = proj

        if self.points_pool_project is not None:
            # points are [Num_points, bs, 2], normalized in [0, 1]
            # the grid needs to be [Bs, H_out, W_out, 2] normalized in [-1,1]
            # Will take H_out = num_points, w_out = 1
            grid = points.transpose(0, 1).unsqueeze(2)
            # re normalize to [-1, 1]
            grid = (grid * 2) - 1
            sampled = torch.nn.functional.grid_sample(img_feats, grid)
            assert list(sampled.shape) == [bs, self.d_model, n_points, 1]
            sampled = sampled.squeeze(-1).permute(2, 0, 1)
            proj = self.points_pool_project(sampled)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        if self.points_pos_enc_project is not None:
            x, y = points.unbind(-1)
            enc_x, enc_y = self.pos_enc._encode_xy(x.flatten(), y.flatten())
            enc_x = enc_x.view(n_points, bs, enc_x.shape[-1])
            enc_y = enc_y.view(n_points, bs, enc_y.shape[-1])
            enc = torch.cat([enc_x, enc_y], -1)

            proj = self.points_pos_enc_project(enc)
            if points_embed is None:
                points_embed = proj
            else:
                points_embed = points_embed + proj

        type_embed = self.label_embed(points_labels.long())
        return type_embed + points_embed, points_mask

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats):
        boxes_embed = None
        n_boxes, bs = boxes.shape[:2]

        if self.boxes_direct_project is not None:
            proj = self.boxes_direct_project(boxes)
            assert boxes_embed is None
            boxes_embed = proj

        if self.boxes_pool_project is not None:
            H, W = img_feats.shape[-2:]

            # boxes are [Num_boxes, bs, 4], normalized in [0, 1]
            # We need to denormalize, and convert to [x, y, x, y]
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device).view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale
            sampled = torchvision.ops.roi_align(img_feats, boxes_xyxy.float().transpose(0, 1).unbind(0), self.roi_size)
            assert list(sampled.shape) == [
                bs * n_boxes,
                self.d_model,
                self.roi_size,
                self.roi_size,
            ]
            proj = self.boxes_pool_project(sampled)
            proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        if self.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
            enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

            proj = self.boxes_pos_enc_project(enc)
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        type_embed = self.label_embed(boxes_labels.long())
        return type_embed + boxes_embed, boxes_mask

    def _encode_masks(
        self,
        masks: torch.Tensor,
        attn_mask: torch.Tensor,
        mask_labels: torch.Tensor,
        img_feats: torch.Tensor = None,
    ):
        n_masks, bs = masks.shape[:2]
        assert n_masks == 1, (
            "We assume one mask per prompt for now. Code should still be functional if this assertion is removed."
        )
        assert list(attn_mask.shape) == [
            bs,
            n_masks,
        ], f"Expected attn_mask to be of shape {bs}x{n_masks}. Got {list(attn_mask.shape)}."
        masks, pos = self.mask_encoder(
            masks=masks.flatten(0, 1).float(),
            pix_feat=img_feats,
        )
        H, W = masks.shape[-2:]
        n_tokens_per_mask = H * W
        # NOTE: We directly add pos enc here as we usually don't keep track of pos encoding
        # for the concatenated prompt (text, other geometric prompts).
        # Might need to do some refactoring for more flexibility.
        masks = masks + pos
        masks = masks.view(n_masks, bs, *masks.shape[1:]).flatten(-2)  # n_masks x bs x C x H*W
        masks = masks.permute(0, 3, 1, 2).flatten(0, 1)  # n_masks * H*W x bs x C
        attn_mask = attn_mask.repeat_interleave(n_tokens_per_mask, dim=1)
        if self.add_mask_label:
            masks = masks + self.mask_label_embed(mask_labels.long())
        return masks, attn_mask

    def forward(self, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None):
        points = geo_prompt.point_embeddings
        points_mask = geo_prompt.point_mask
        points_labels = geo_prompt.point_labels
        boxes = geo_prompt.box_embeddings
        boxes_mask = geo_prompt.box_mask
        boxes_labels = geo_prompt.box_labels
        masks = geo_prompt.mask_embeddings
        masks_mask = geo_prompt.mask_mask
        masks_labels = geo_prompt.mask_labels
        seq_first_img_feats = img_feats[-1]  # [H*W, B, C]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1] if img_pos_embeds is not None else torch.zeros_like(seq_first_img_feats)
        )

        if self.points_pool_project or self.boxes_pool_project:
            assert len(img_feats) == len(img_sizes)
            cur_img_feat = img_feats[-1]
            cur_img_feat = self.img_pre_norm(cur_img_feat)
            H, W = img_sizes[-1]
            assert cur_img_feat.shape[0] == H * W
            N, C = cur_img_feat.shape[-2:]
            # Put back in NxCxHxW
            cur_img_feat = cur_img_feat.permute(1, 2, 0)
            cur_img_feat = cur_img_feat.view(N, C, H, W)
            img_feats = cur_img_feat

        if self.encode_boxes_as_points:
            assert boxes is not None
            assert geo_prompt.box_mask is not None
            assert geo_prompt.box_labels is not None
            assert boxes.shape[-1] == 4

            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            top_left, bottom_right = boxes_xyxy.split(split_size=2, dim=-1)

            labels_tl = geo_prompt.box_labels + 2
            labels_br = geo_prompt.box_labels + 4

            # Append to the existing points
            points, _ = concat_padded_sequences(points, points_mask, top_left, boxes_mask)
            points_labels, points_mask = concat_padded_sequences(
                points_labels.unsqueeze(-1),
                points_mask,
                labels_tl.unsqueeze(-1),
                boxes_mask,
            )
            points_labels = points_labels.squeeze(-1)

            points, _ = concat_padded_sequences(points, points_mask, bottom_right, boxes_mask)
            points_labels, points_mask = concat_padded_sequences(
                points_labels.unsqueeze(-1),
                points_mask,
                labels_br.unsqueeze(-1),
                boxes_mask,
            )
            points_labels = points_labels.squeeze(-1)

        final_embeds, final_mask = self._encode_points(
            points=points,
            points_mask=points_mask,
            points_labels=points_labels,
            img_feats=img_feats,
        )

        if not self.encode_boxes_as_points:
            boxes_embeds, boxes_mask = self._encode_boxes(
                boxes=boxes,
                boxes_mask=boxes_mask,
                boxes_labels=boxes_labels,
                img_feats=img_feats,
            )

            final_embeds, final_mask = concat_padded_sequences(final_embeds, final_mask, boxes_embeds, boxes_mask)

        if masks is not None and self.mask_encoder is not None:
            masks_embed, masks_mask = self._encode_masks(
                masks=masks,
                attn_mask=masks_mask,
                mask_labels=masks_labels,
                img_feats=img_feats,
            )
            if points.size(0) == boxes.size(0) == 0:
                return masks_embed, masks_mask
        bs = final_embeds.shape[1]
        assert final_mask.shape[0] == bs
        if self.cls_embed is not None:
            cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
            cls_mask = torch.zeros(bs, 1, dtype=final_mask.dtype, device=final_mask.device)
            final_embeds, final_mask = concat_padded_sequences(final_embeds, final_mask, cls, cls_mask)

        if self.final_proj is not None:
            final_embeds = self.norm(self.final_proj(final_embeds))

        if self.encode is not None:
            for lay in self.encode:
                if isinstance(lay, TransformerEncoderLayer):  # Note: TransformerDecoderLayer renamed for code release
                    final_embeds = activation_ckpt_wrapper(lay)(
                        tgt=final_embeds,
                        memory=seq_first_img_feats,
                        tgt_key_padding_mask=final_mask,
                        pos=seq_first_img_pos_embeds,
                        act_ckpt_enable=self.training and self.use_act_ckpt,
                    )
                else:
                    raise NotImplementedError(
                        f"Only `TransformerEncoderLayer` or `TransformerDecoderLayer` are supported. Got {type(lay)}"
                    )

            final_embeds = self.encode_norm(final_embeds)
        # Finally, concat mask embeddings if any
        if masks is not None and self.mask_encoder is not None:
            final_embeds, final_mask = concat_padded_sequences(final_embeds, final_mask, masks_embed, masks_mask)
        return final_embeds, final_mask


### End Sequence Geometry Encoder ------------------------------------------------------------- #

### Transformer Wrapper ------------------------------------------------------------- #


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        cross_attention: nn.Module,
        n_heads: int,
        use_text_cross_attention: bool = False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = cross_attention
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        with torch.cuda.amp.autocast(enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        memory_text: Optional[Tensor] = None,  # num_token, bs, d_model
        text_attention_mask: Optional[Tensor] = None,  # bs, num_token
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
        # dac
        dac=False,
        dac_use_selfatt_ln=True,
        # skip inside deformable attn
        identity=0.0,
        **kwargs,  # additional kwargs for compatibility
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        # self attention
        if self.self_attn is not None:
            if dac:
                # we only apply self attention to the first half of the queries
                assert tgt.shape[0] % 2 == 0
                num_o2o_queries = tgt.shape[0] // 2
                tgt_o2o = tgt[:num_o2o_queries]
                tgt_query_pos_o2o = tgt_query_pos[:num_o2o_queries]
                tgt_o2m = tgt[num_o2o_queries:]
            else:
                tgt_o2o = tgt
                tgt_query_pos_o2o = tgt_query_pos

            q = k = self.with_pos_embed(tgt_o2o, tgt_query_pos_o2o)
            tgt2 = self.self_attn(q, k, tgt_o2o, attn_mask=self_attn_mask)[0]
            tgt_o2o = tgt_o2o + self.dropout2(tgt2)
            if dac:
                if not dac_use_selfatt_ln:
                    tgt_o2o = self.norm2(tgt_o2o)
                tgt = torch.cat((tgt_o2o, tgt_o2m), dim=0)  # Recombine
                if dac_use_selfatt_ln:
                    tgt = self.norm2(tgt)
            else:
                tgt = tgt_o2o
                tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text,
                memory_text,
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            key=self.with_pos_embed(memory, memory_pos),
            value=memory,
            attn_mask=cross_attn_mask,
            key_padding_mask=(
                memory_key_padding_mask.transpose(0, 1) if memory_key_padding_mask is not None else None
            ),
        )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        interaction_layer,
        layer,
        num_layers: int,
        num_queries: int,
        return_intermediate: bool,
        box_refine: bool = False,
        num_o2m_queries: int = 0,
        dac: bool = False,
        boxRPB: str = "none",
        # Experimental: An object query for SAM 2 tasks
        instance_query: bool = False,
        # Defines the number of additional instance queries,
        # 1 or 4 are the most likely for single vs multi mask support
        num_instances: int = 1,  # Irrelevant if instance_query is False
        dac_use_selfatt_ln: bool = True,
        use_act_checkpoint: bool = False,
        compile_mode=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.fine_layers = (
            get_clones(interaction_layer, num_layers) if interaction_layer is not None else [None] * num_layers
        )
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.dac = dac
        if dac:
            self.num_o2m_queries = num_queries
            tot_num_queries = num_queries
        else:
            self.num_o2m_queries = num_o2m_queries
            tot_num_queries = num_queries + num_o2m_queries
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Embedding(tot_num_queries, d_model)
        self.instance_query_embed = None
        self.instance_query_reference_points = None
        self.use_instance_query = instance_query
        self.num_instances = num_instances
        if instance_query:
            self.instance_query_embed = nn.Embedding(num_instances, d_model)
        self.box_refine = box_refine
        if box_refine:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

            self.reference_points = nn.Embedding(num_queries, 4)
            if instance_query:
                self.instance_reference_points = nn.Embedding(num_instances, 4)

        assert boxRPB in ["none", "log", "linear", "both"]
        self.boxRPB = boxRPB
        if boxRPB != "none":
            try:
                nheads = self.layers[0].cross_attn_image.num_heads
            except AttributeError:
                nheads = self.layers[0].cross_attn.num_heads

            n_input = 4 if boxRPB == "both" else 2
            self.boxRPB_embed_x = MLP(n_input, d_model, nheads, 2)
            self.boxRPB_embed_y = MLP(n_input, d_model, nheads, 2)
            self.compilable_cord_cache = None
            self.compilable_stored_size = None
            self.coord_cache = {}

        self.roi_pooler = (
            RoIAlign(output_size=7, spatial_scale=1, sampling_ratio=-1, aligned=True)
            if interaction_layer is not None
            else None
        )
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2)
        self.dac_use_selfatt_ln = dac_use_selfatt_ln
        self.use_act_checkpoint = use_act_checkpoint

        nn.init.normal_(self.query_embed.weight.data)
        if self.instance_query_embed is not None:
            nn.init.normal_(self.instance_query_embed.weight.data)

        assert self.roi_pooler is None
        # TODO(chay): add support for turning off box_refine and return_intermediate
        assert self.return_intermediate, "support return_intermediate only"
        assert self.box_refine, "support box refine only"

        self.compile_mode = compile_mode
        self.compiled = False
        # We defer compilation till after the first forward, to first warm-up the boxRPB cache

        # assign layer index to each layer so that some layers can decide what to do
        # based on which layer index they are (e.g. cross attention to memory bank only
        # in selected layers)
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx

    @staticmethod
    def _get_coords(H, W, device):
        coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
        coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
        return coords_h, coords_w

    def _get_rpb_matrix(self, reference_boxes, feat_size):
        H, W = feat_size
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
        bs, num_queries, _ = boxes_xyxy.shape
        if self.compilable_cord_cache is None:
            self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
            self.compilable_stored_size = (H, W)

        if torch.compiler.is_dynamo_compiling() or self.compilable_stored_size == (
            H,
            W,
        ):
            # good, hitting the cache, will be compilable
            coords_h, coords_w = self.compilable_cord_cache
        else:
            # cache miss, will create compilation issue
            # In case we're not compiling, we'll still rely on the dict-based cache
            if feat_size not in self.coord_cache:
                self.coord_cache[feat_size] = self._get_coords(H, W, reference_boxes.device)
            coords_h, coords_w = self.coord_cache[feat_size]

            assert coords_h.shape == (H,)
            assert coords_w.shape == (W,)

        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(bs, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(bs, num_queries, -1, 2)

        if self.boxRPB in ["log", "both"]:
            deltas_x_log = deltas_x * 8  # normalize to -8, 8
            deltas_x_log = torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0) / np.log2(8)

            deltas_y_log = deltas_y * 8  # normalize to -8, 8
            deltas_y_log = torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0) / np.log2(8)
            if self.boxRPB == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)

        deltas_x = activation_ckpt_wrapper(self.boxRPB_embed_x)(
            x=deltas_x,
            act_ckpt_enable=self.training and self.use_act_checkpoint,
        )  # bs, num_queries, W, n_heads
        deltas_y = activation_ckpt_wrapper(self.boxRPB_embed_y)(
            x=deltas_y,
            act_ckpt_enable=self.training and self.use_act_checkpoint,
        )  # bs, num_queries, H, n_heads

        if not torch.compiler.is_dynamo_compiling():
            assert deltas_x.shape[:3] == (bs, num_queries, W)
            assert deltas_y.shape[:3] == (bs, num_queries, H)

        B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)  # bs, num_queries, H, W, n_heads
        if not torch.compiler.is_dynamo_compiling():
            assert B.shape[:4] == (bs, num_queries, H, W)
        B = B.flatten(2, 3)  # bs, num_queries, H*W, n_heads
        B = B.permute(0, 3, 1, 2)  # bs, n_heads, num_queries, H*W
        B = B.contiguous()  # memeff attn likes ordered strides
        if not torch.compiler.is_dynamo_compiling():
            assert B.shape[2:] == (num_queries, H * W)
        return B

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        reference_boxes: Optional[Tensor] = None,  # num_queries, bs, 4
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
        # for text
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        # if `apply_dac` is None, it will default to `self.dac`
        apply_dac: Optional[bool] = None,
        decoder_extra_kwargs: Optional[dict] = None,
        # ROI memory bank
        obj_roi_memory_feat=None,
        obj_roi_memory_mask=None,
        box_head_trk=None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: \\sum{hw}, bs, d_model
            - pos: \\sum{hw}, bs, d_model
            - reference_boxes: nq, bs, 4 (after sigmoid)
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        if memory_mask is not None:
            assert self.boxRPB == "none", (
                "inputting a memory_mask in the presence of boxRPB is unexpected/not implemented"
            )

        apply_dac = apply_dac if apply_dac is not None else self.dac
        if apply_dac:
            assert (tgt.shape[0] == self.num_queries) or (
                self.use_instance_query and (tgt.shape[0] == self.instance_query_embed.num_embeddings)
            )

            tgt = tgt.repeat(2, 1, 1)
            # note that we don't tile tgt_mask, since DAC doesn't
            # use self-attention in o2m queries
            if reference_boxes is not None:
                assert (reference_boxes.shape[0] == self.num_queries) or (
                    self.use_instance_query and (reference_boxes.shape[0] == self.instance_query_embed.num_embeddings)
                )
                reference_boxes = reference_boxes.repeat(2, 1, 1)

        bs = tgt.shape[1]
        intermediate = []

        if self.box_refine:
            if reference_boxes is None:
                # In this case, we're in a one-stage model, so we generate the reference boxes
                reference_boxes = self.reference_points.weight.unsqueeze(1)
                reference_boxes = reference_boxes.repeat(2, bs, 1) if apply_dac else reference_boxes.repeat(1, bs, 1)
                reference_boxes = reference_boxes.sigmoid()
            intermediate_ref_boxes = [reference_boxes]
        else:
            reference_boxes = None
            intermediate_ref_boxes = None

        output = tgt

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = (
                reference_boxes[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )  # nq, bs, nlevel, 4

            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )  # nq, bs, d_model*2

            # conditional query
            query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, d_model

            if self.boxRPB != "none" and reference_boxes is not None:
                assert spatial_shapes.shape[0] == 1, "only single scale support implemented"
                memory_mask = self._get_rpb_matrix(
                    reference_boxes,
                    (spatial_shapes[0, 0], spatial_shapes[0, 1]),
                )
                memory_mask = memory_mask.flatten(0, 1)  # (bs*n_heads, nq, H*W)

            output = activation_ckpt_wrapper(layer)(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                dac=apply_dac,
                dac_use_selfatt_ln=self.dac_use_selfatt_ln,
                **(decoder_extra_kwargs or {}),
                act_ckpt_enable=self.training and self.use_act_checkpoint,
                # ROI memory bank
                obj_roi_memory_feat=obj_roi_memory_feat,
                obj_roi_memory_mask=obj_roi_memory_mask,
            )

            # iter update
            if self.box_refine:
                reference_before_sigmoid = inverse_sigmoid(reference_boxes)
                if box_head_trk is None:
                    delta_unsig = self.bbox_embed(output)
                else:
                    # box_head_trk use a separate box head for tracking queries
                    Q_det = decoder_extra_kwargs["Q_det"]
                    assert output.size(0) >= Q_det
                    delta_unsig_det = self.bbox_embed(output[:Q_det])
                    delta_unsig_trk = box_head_trk(output[Q_det:])
                    delta_unsig = torch.cat([delta_unsig_det, delta_unsig_trk], dim=0)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                reference_boxes = new_reference_points.detach()
                if layer_idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_points)
            else:
                raise NotImplementedError("not implemented yet")

            intermediate.append(self.norm(output))

        if not self.compiled and self.compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=self.compile_mode, fullgraph=True)
            self.compiled = True

        return torch.stack(intermediate), torch.stack(intermediate_ref_boxes)


def pool_text_feat(prompt, prompt_mask, pool_with_mask):
    # prompt has shape (seq, bs, dim)
    if not pool_with_mask:
        return prompt.mean(dim=0)

    # prompt_mask has shape (bs, seq), where False is valid and True is padding
    assert prompt_mask.dim() == 2
    # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
    is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
    # num_valid has shape (bs, 1)
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)

    # mean pool over all the valid tokens
    pooled_text = (prompt * is_valid).sum(dim=0) / num_valid
    return pooled_text


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer that performs self-attention followed by cross-attention.

    This layer was previously called TransformerDecoderLayer but was renamed to better
    reflect its role in the architecture. It processes input sequences through self-attention
    and then cross-attention with another input (typically image features).

    The layer supports both pre-norm and post-norm configurations, as well as
    positional encoding at different stages of the attention mechanism.
    """

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module,
    ):
        """
        Initialize a transformer encoder layer.

        Args:
            activation: Activation function to use in the feedforward network
            cross_attention: Cross-attention module for attending to image features
            d_model: Model dimension/hidden size
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout probability
            pos_enc_at_attn: Whether to add positional encodings at self-attention
            pos_enc_at_cross_attn_keys: Whether to add positional encodings to keys in cross-attention
            pos_enc_at_cross_attn_queries: Whether to add positional encodings to queries in cross-attention
            pre_norm: Whether to use pre-norm (True) or post-norm (False) architecture
            self_attention: Self-attention module
        """
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        self.layer_idx = None

    def forward_post(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass for post-norm architecture.

        In post-norm architecture, normalization is applied after attention and feedforward operations.

        Args:
            tgt: Input tensor to be processed
            memory: Memory tensor for cross-attention
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key padding mask for self-attention
            memory_key_padding_mask: Key padding mask for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query
            **kwargs: Additional keyword arguments

        Returns:
            Processed tensor
        """
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt

        # Self attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: Tensor,
        memory: Tensor,
        dac: bool = False,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        # attn_bias: Optional[Tensor] = None,
        # **kwargs,
    ) -> Tensor:
        """
        Forward pass for pre-norm architecture.

        In pre-norm architecture, normalization is applied before attention and feedforward operations.

        Args:
            tgt: Input tensor to be processed
            memory: Memory tensor for cross-attention
            dac: Whether to use Divide-and-Conquer attention
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key padding mask for self-attention
            memory_key_padding_mask: Key padding mask for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query
            attn_bias: Optional attention bias tensor
            **kwargs: Additional keyword arguments

        Returns:
            Processed tensor
        """
        if dac:
            # we only apply self attention to the first half of the queries
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            # Recombine
            tgt = torch.cat((tgt, other_tgt), dim=0)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            # attn_bias=attn_bias,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        dac: bool = False,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        # attn_bias: Optional[Tensor] = None,
        # **kwds: Any,
    ) -> torch.Tensor:
        """
        Forward pass for the transformer encoder layer.

        Args:
            tgt: Input tensor to be processed
            memory: Memory tensor (e.g., image features) for cross-attention
            dac: Whether to use Divide-and-Conquer attention (only apply self-attention to first half)
            tgt_mask: Mask for self-attention
            memory_mask: Mask for cross-attention
            tgt_key_padding_mask: Key padding mask for self-attention
            memory_key_padding_mask: Key padding mask for cross-attention
            pos: Positional encoding for memory
            query_pos: Positional encoding for query
            attn_bias: Optional attention bias tensor
            **kwds: Additional keyword arguments

        Returns:
            Processed tensor after self-attention, cross-attention, and feedforward network
        """
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt,
            memory,
            dac=dac,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            # attn_bias=attn_bias,
            # **kwds,
        )


class TransformerEncoder(nn.Module):
    """
    Transformer encoder that processes multi-level features.

    This encoder takes multi-level features (e.g., from a backbone network) and processes
    them through a stack of transformer encoder layers. It supports features from multiple
    levels (e.g., different resolutions) and can apply activation checkpointing for memory
    efficiency during training.

    Args:
        layer: The encoder layer to be stacked multiple times
        num_layers: Number of encoder layers to stack
        d_model: Model dimension/hidden size
        num_feature_levels: Number of feature levels to process
        frozen: Whether to freeze the parameters of this module
        use_act_checkpoint: Whether to use activation checkpointing during training
    """

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        frozen: bool = False,
        use_act_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers

        self.num_feature_levels = num_feature_levels
        self.level_embed = None
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.use_act_checkpoint = use_act_checkpoint

        # assign layer index to each layer so that some layers can decide what to do
        # based on which layer index they are (e.g. cross attention to memory bank only
        # in selected layers)
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        with torch.no_grad():
            reference_points_list = []
            for lvl, (H_, W_) in enumerate(spatial_shapes):
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                    torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                )
                ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
                ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
                ref = torch.stack((ref_x, ref_y), -1)
                reference_points_list.append(ref)
            reference_points = torch.cat(reference_points_list, 1)
            reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def _prepare_multilevel_features(self, srcs, masks, pos_embeds):
        assert len(srcs) == self.num_feature_levels, "mismatch between expected and received # of feature levels"

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        has_mask = masks is not None and masks[0] is not None
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            if has_mask:
                mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            if has_mask:
                mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1) if has_mask else None  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        if has_mask:
            valid_ratios = torch.stack([get_valid_ratio(m) for m in masks], 1)
        else:
            valid_ratios = torch.ones(
                (src_flatten.shape[0], self.num_feature_levels, 2),
                device=src_flatten.device,
            )

        return (
            src_flatten,
            mask_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        )

    def forward(
        self,
        src: list[Tensor],
        src_key_padding_masks: Optional[list[Tensor]] = None,
        pos: Optional[list[Tensor]] = None,
        prompt: Optional[Tensor] = None,
        prompt_key_padding_mask: Optional[Tensor] = None,
        encoder_extra_kwargs: Optional[dict] = None,
    ) -> tuple[Tensor, Optional[Tensor], Tensor, Tensor, Tensor, Tensor]:
        """
        Process multi-level features through the transformer encoder.

        Args:
            src: List of multi-level features, each with shape (batch_size, channels, height, width)
            src_key_padding_masks: List of padding masks for each feature level, each with shape (batch_size, height, width)
            pos: List of positional embeddings for each feature level, each with shape (batch_size, channels, height, width)
            prompt: Optional text/prompt features to attend to, with shape (seq_len, batch_size, d_model)
            prompt_key_padding_mask: Optional padding mask for prompt, with shape (batch_size, seq_len)
            encoder_extra_kwargs: Optional additional arguments to pass to each encoder layer

        Returns:
            A tuple containing:
            - output: Processed features with shape (seq_len, batch_size, d_model)
            - key_padding_masks_flatten: Flattened padding masks
            - lvl_pos_embed_flatten: Flattened positional embeddings
            - level_start_index: Starting indices for each feature level
            - spatial_shapes: Spatial dimensions of each feature level
            - valid_ratios: Valid ratios for each feature level
        """
        assert len(src) == self.num_feature_levels, "must be equal to num_feature_levels"
        if src_key_padding_masks is not None:
            assert len(src_key_padding_masks) == self.num_feature_levels
        if pos is not None:
            assert len(pos) == self.num_feature_levels
        # Flatten multilevel feats and add level pos embeds
        (
            src_flatten,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        ) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)

        output = src_flatten
        for layer in self.layers:
            layer_kwargs = {}

            assert isinstance(layer, TransformerEncoderLayer)
            layer_kwargs["memory"] = prompt
            layer_kwargs["memory_key_padding_mask"] = prompt_key_padding_mask
            layer_kwargs["query_pos"] = lvl_pos_embed_flatten
            layer_kwargs["tgt"] = output
            layer_kwargs["tgt_key_padding_mask"] = key_padding_masks_flatten

            if encoder_extra_kwargs is not None:
                layer_kwargs.update(encoder_extra_kwargs)
            output = activation_ckpt_wrapper(layer)(
                **layer_kwargs,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
            )
        # return as seq first
        return (
            output.transpose(0, 1),
            (key_padding_masks_flatten.transpose(0, 1) if key_padding_masks_flatten is not None else None),
            lvl_pos_embed_flatten.transpose(0, 1),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )


class TransformerEncoderFusion(TransformerEncoder):
    """
    Transformer encoder that fuses text and image features.

    This encoder extends TransformerEncoder to handle both text and image features,
    with the ability to add pooled text features to image features for better
    cross-modal fusion. It supports torch.compile for performance optimization.

    Args:
        layer: The encoder layer to be stacked multiple times
        num_layers: Number of encoder layers to stack
        d_model: Model dimension/hidden size
        num_feature_levels: Number of feature levels to process
        add_pooled_text_to_img_feat: Whether to add pooled text features to image features
        pool_text_with_mask: Whether to use the mask when pooling text features
        compile_mode: Mode for torch.compile, or None to disable compilation
        **kwargs: Additional arguments to pass to the parent class
    """

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        add_pooled_text_to_img_feat: bool = True,
        pool_text_with_mask: bool = False,
        compile_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            layer,
            num_layers,
            d_model,
            num_feature_levels,
            **kwargs,
        )
        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = nn.Linear(d_model, d_model)
        self.pool_text_with_mask = pool_text_with_mask
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # Not needed here
        return None

    def forward(
        self,
        src: list[Tensor],
        prompt: Tensor,
        src_key_padding_mask: Optional[list[Tensor]] = None,
        src_pos: Optional[list[Tensor]] = None,
        prompt_key_padding_mask: Optional[Tensor] = None,
        prompt_pos: Optional[Tensor] = None,
        feat_sizes: Optional[list[int]] = None,
        encoder_extra_kwargs: Optional[dict] = None,
    ):
        # Restore spatial shapes of vision
        bs = src[0].shape[1]  # seq first
        if feat_sizes is not None:
            assert len(feat_sizes) == len(src)
            for i, (h, w) in enumerate(feat_sizes):
                src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
                src_key_padding_mask[i] = (
                    src_key_padding_mask[i].reshape(h, w, bs).permute(2, 0, 1)
                    if src_key_padding_mask[i] is not None
                    else None
                )
        else:
            assert all(x.dim == 4 for x in src), "expected list of (bs, c, h, w) tensors"

        if self.add_pooled_text_to_img_feat:
            # Fusion: Add mean pooled text to image features
            pooled_text = pool_text_feat(prompt, prompt_key_padding_mask, self.pool_text_with_mask)
            pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]  # prompt is seq first
            src = [x.add_(pooled_text) for x in src]

        (
            out,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
        ) = super().forward(
            src,
            src_key_padding_masks=src_key_padding_mask,
            pos=src_pos,
            prompt=prompt.transpose(0, 1),
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

        return {
            "memory": out,
            "padding_mask": key_padding_masks_flatten,
            "pos_embed": lvl_pos_embed_flatten,
            "memory_text": prompt,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
        }


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",  # ["none"] only for now
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries

        # for two stage
        assert two_stage_type in ["none"], f"unknown param {two_stage_type} of two_stage_type"
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if "box_embed" not in n and "query_embed" not in n and "reference_points" not in n:
                    nn.init.xavier_uniform_(p)


### End Transformer Wrapper ------------------------------------------------------------- #

### VETextEncoder ------------------------------------------------------------- #


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        # Attention
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        # LayerNorm, LayerScale
        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        # MLP
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        if attn_mask is not None:
            # Leave boolean masks as is
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q_x.dtype)

        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


# text transformer only
class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        compile_mode: Optional[str] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(layers)
            ]
        )

        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for _, r in enumerate(self.resblocks):
            x = r(
                x,
                attn_mask=attn_mask,
            )
        return x


def text_global_pool(
    x: torch.Tensor, text: Optional[torch.Tensor] = None, pool_type: str = "argmax"
) -> tuple[torch.Tensor, torch.Tensor]:
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        output_dim: int = 512,
        no_causal_mask: bool = False,
        pool_type: str = "none",  # no pooling
        proj_bias: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        output_tokens: bool = False,
        use_ln_post: bool = True,
        compile_mode: Optional[str] = None,
    ):
        super().__init__()
        assert pool_type in ("first", "last", "argmax", "none")
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            compile_mode=compile_mode,
        )
        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer("attn_mask", self.build_causal_mask(), persistent=False)
        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def build_causal_mask(self) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        seq_len = text.shape[1]
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)

        x = self.ln_final(x)
        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return pooled, tokens
        return pooled


class VETextEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        tokenizer: Callable,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
        compile_mode: Optional[str] = None,
    ):
        super().__init__()
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        self.tokenizer = tokenizer

        self.encoder = TextTransformer(
            context_length=self.context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            # we want the tokens, not just the pooled output
            output_tokens=True,
            use_ln_post=use_ln_post,
            compile_mode=compile_mode,
        )
        self.resizer = nn.Linear(self.encoder.width, d_model)

    def forward(
        self,
        text: Union[list[str], tuple[torch.Tensor, torch.Tensor, dict]],
        input_boxes: Optional[list] = None,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(text[0], str):
            # no use case for this
            assert input_boxes is None or len(input_boxes) == 0, "not supported"

            # Encode the text
            tokenized = self.tokenizer(text, context_length=self.context_length).to(device)  # [b, seq_len]
            text_attention_mask = (tokenized != 0).bool()
            print("tokenized", tokenized)
            print("text_attention_mask", text_attention_mask)

            # manually embed the tokens
            inputs_embeds = self.encoder.token_embedding(tokenized)  # [b, seq_len, d=1024]
            _, text_memory = self.encoder(tokenized)  # [b, seq_len, d=1024]

            assert text_memory.shape[1] == inputs_embeds.shape[1]
            # Invert attention mask because its the opposite in pytorch transformer
            text_attention_mask = text_attention_mask.ne(1)
            # Transpose memory because pytorch's attention expects sequence first
            text_memory = text_memory.transpose(0, 1)
            # Resize the encoder hidden states to be of the same d_model as the decoder
            text_memory_resized = self.resizer(text_memory)
        else:
            # The text is already encoded, use as is.
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
            assert input_boxes is None or len(input_boxes) == 0, "Can't replace boxes in text if it's already encoded"

        # Note that the input_embeds are returned in pytorch's convention (sequence first)
        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(0, 1),
        )


### End VETextEncoder ---------------------------------------------------------------------------- #


### ViTDet ---------------------------------------------------------------------------- #


def init_t_xy(end_x: int, end_y: int, scale: float = 1.0, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
) -> torch.Tensor:
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]) -> Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: Tensor) -> Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def get_abs_pos(
    abs_pos: Tensor,
    has_cls_token: bool,
    hw: tuple[int, int],
    retain_cls_token: bool = False,
    tiling: bool = False,
) -> Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
        retain_cls_token: whether to retain the cls_token
        tiling: whether to tile the embeddings, *instead* of interpolation (a la abs_win)
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C),
        if retain_cls_token is False, otherwise (1, 1+H*W, C)
    """
    if retain_cls_token:
        assert has_cls_token

    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]

    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            new_abs_pos = new_abs_pos.tile([1, 1] + [x // y + 1 for x, y in zip((h, w), new_abs_pos.shape[2:])])[
                :, :, :h, :w
            ]
        else:
            new_abs_pos = F.interpolate(
                new_abs_pos,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )

        if not retain_cls_token:
            return new_abs_pos.permute(0, 2, 3, 1)
        else:
            # add cls_token back, flatten spatial dims
            assert has_cls_token
            return torch.cat(
                [cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)],
                dim=1,
            )

    else:
        if not retain_cls_token:
            return abs_pos.reshape(1, h, w, -1)
        else:
            assert has_cls_token
            return torch.cat([cls_pos, abs_pos], dim=1)


def concat_rel_pos(
    q: Tensor,
    k: Tensor,
    q_hw: tuple[int, int],
    k_hw: tuple[int, int],
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    rescale: bool = False,
    relative_coords: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """
    Concatenate rel pos coeffs to the q & k tensors, so that qk^T is now
    effectively including rel pos biases.
    Args:
        q (Tensor): q tensor with shape (B, L_q, C).
        k (Tensor): k tensor with shape (B, L_k, C).
        q_hw, k_hw: These are spatial size of q & k tensors.
        rel_pos_h, rel_pos_w: These are relative pos embeddings/params of height, width.
        rescale (bool): whether to rescale. e.g. for use when using sdpa, pytorch will
            scale by the wrong factor due to the concat.
    Returns:
        q, k: But, padded so that qk^T accounts for rel pos biases
    """
    q_h, q_w = q_hw
    k_h, k_w = k_hw

    assert (q_h == q_w) and (k_h == k_w), "only square inputs supported"

    if relative_coords is not None:
        Rh = rel_pos_h[relative_coords]
        Rw = rel_pos_w[relative_coords]
    else:
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    old_scale = dim**0.5
    new_scale = (dim + k_h + k_w) ** 0.5 if rescale else old_scale  # for sdpa
    # attn will be divided by new_scale, but we want to divide q by old_scale
    scale_ratio = new_scale / old_scale

    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) * new_scale  # (B, q_h, q_w, k_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) * new_scale  # (B, q_h, q_w, k_w)

    eye_h = torch.eye(k_h, dtype=q.dtype, device=q.device)
    eye_w = torch.eye(k_w, dtype=q.dtype, device=q.device)

    eye_h = eye_h.view(1, k_h, 1, k_h).expand([B, k_h, k_w, k_h])
    eye_w = eye_w.view(1, 1, k_w, k_w).expand([B, k_h, k_w, k_w])

    q = torch.cat([r_q * scale_ratio, rel_h, rel_w], dim=-1).view(B, q_h * q_w, -1)
    k = torch.cat([k.view(B, k_h, k_w, -1), eye_h, eye_w], dim=-1).view(B, k_h * k_w, -1)

    return q, k


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings and 2d-rope."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[tuple[int, int]] = None,
        cls_token: bool = False,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_pt_size: Optional[tuple[int, int]] = None,
        rope_interp: bool = False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size or rope size.
            attn_type: Type of attention operation, e.g. "vanilla", "vanilla-xformer".
            cls_token: whether a cls_token is present.
            use_rope: whether to use rope 2d (indep of use_rel_pos, as it can be used together)
            rope_theta: control frequencies of rope
            rope_pt_size: size of rope in previous stage of training, needed for interpolation or tiling
            rope_interp: whether to interpolate (or extrapolate) rope to match input size
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cls_token = cls_token

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # rel_pos embeddings and rope
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size

        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.rope_pt_size = rope_pt_size
        self.rope_interp = rope_interp

        # init rel_pos embeddings and rope
        self._setup_rel_pos(rel_pos_zero_init)
        self._setup_rope_freqs()

    def _setup_rel_pos(self, rel_pos_zero_init: bool = True) -> None:
        if not self.use_rel_pos:
            self.rel_pos_h = None
            self.rel_pos_w = None
            return

        assert self.input_size is not None
        assert self.cls_token is False, "not supported"
        # initialize relative positional embeddings
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * self.input_size[0] - 1, self.head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * self.input_size[1] - 1, self.head_dim))

        if not rel_pos_zero_init:
            trunc_normal_(self.rel_pos_h, std=0.02)
            trunc_normal_(self.rel_pos_w, std=0.02)

        # Precompute the relative coords
        H, W = self.input_size
        q_coords = torch.arange(H)[:, None]
        k_coords = torch.arange(W)[None, :]
        relative_coords = (q_coords - k_coords) + (H - 1)
        self.register_buffer("relative_coords", relative_coords.long())

    def _setup_rope_freqs(self) -> None:
        if not self.use_rope:
            self.freqs_cis = None
            return

        assert self.input_size is not None
        # determine rope input size
        if self.rope_pt_size is None:
            self.rope_pt_size = self.input_size

        # initialize 2d rope freqs
        self.compute_cis = partial(
            compute_axial_cis,
            dim=self.head_dim,
            theta=self.rope_theta,
        )

        # interpolate rope
        scale_pos = 1.0
        if self.rope_interp:
            scale_pos = self.rope_pt_size[0] / self.input_size[0]
        # get scaled freqs_cis
        freqs_cis = self.compute_cis(
            end_x=self.input_size[0],
            end_y=self.input_size[1],
            scale_pos=scale_pos,
        )
        if self.cls_token:
            t = torch.zeros(
                self.head_dim // 2,
                dtype=torch.float32,
                device=freqs_cis.device,
            )
            cls_freqs_cis = torch.polar(torch.ones_like(t), t)[None, :]
            freqs_cis = torch.cat([cls_freqs_cis, freqs_cis], dim=0)

        self.register_buffer("freqs_cis", freqs_cis)

    def _apply_rope(self, q, k) -> tuple[Tensor, Tensor]:
        if not self.use_rope:
            return q, k

        assert self.freqs_cis is not None
        # TODO: allow recomputation of freqs_cis on the fly

        return apply_rotary_enc(q, k, freqs_cis=self.freqs_cis)

    def forward(self, x: Tensor) -> Tensor:
        s = 1 if self.cls_token else 0  # used to exclude cls_token
        if x.ndim == 4:
            B, H, W, _ = x.shape
            assert s == 0  # no cls_token
            L = H * W
            ndim = 4
        else:
            assert x.ndim == 3
            B, L, _ = x.shape
            ndim = 3
            H = W = math.sqrt(L - s)

        # qkv with shape (3, B, nHead, L, C)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
        # q, k, v with shape (B, nHead, L, C)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # handle rope and rel pos embeddings
        q, k = self._apply_rope(q, k)
        if self.use_rel_pos:
            q, k = concat_rel_pos(
                q.flatten(0, 1),
                k.flatten(0, 1),
                (H, W),
                x.shape[1:3],
                self.rel_pos_h,
                self.rel_pos_w,
                rescale=True,
                relative_coords=self.relative_coords,
            )

            # sdpa expects [B, nheads, H*W, C] so we transpose back
            q = q.reshape(B, self.num_heads, H * W, -1)
            k = k.reshape(B, self.num_heads, H * W, -1)

        x = F.scaled_dot_product_attention(q, k, v)

        if ndim == 4:
            x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        else:
            x = x.view(B, self.num_heads, L, -1).permute(0, 2, 1, 3).reshape(B, L, -1)

        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[tuple[int, int]] = None,
        use_rope: bool = False,
        rope_pt_size: Optional[tuple[int, int]] = None,
        rope_tiled: bool = False,
        rope_interp: bool = False,
        use_ve_rope: bool = False,
        cls_token: bool = False,
        dropout: float = 0.0,
        init_values: Optional[float] = None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            dropout (float): Dropout rate.
            cls_token: whether a cls_token is present.
            use_rope: whether to use rope 2d (indep of use_rel_pos, as it can be used together)
            rope_pt_size: size of rope in previous stage of training, needed for interpolation or tiling
            rope_interp: whether to interpolate (or extrapolate) rope to match target input size,
                expected to specify source size as rope_pt_size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rope=use_rope,
            rope_pt_size=rope_pt_size,
            rope_interp=rope_interp,
            cls_token=cls_token,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=(dropout, 0.0),
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.ls1(self.attn(x))
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.dropout(self.drop_path(x))
        x = x + self.dropout(self.drop_path(self.ls2(self.mlp(self.norm2(x)))))

        return x


class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: Union[Callable[..., nn.Module], str] = "LayerNorm",
        act_layer: Callable[..., nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        tile_abs_pos: bool = True,
        rel_pos_blocks: Union[tuple[int, ...], bool] = (2, 5, 8, 11),
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_att_blocks: tuple[int, ...] = (2, 5, 8, 11),
        use_rope: bool = False,
        rope_pt_size: Optional[int] = None,
        use_interp_rope: bool = False,
        pretrain_img_size: int = 224,
        pretrain_use_cls_token: bool = True,
        retain_cls_token: bool = True,
        dropout: float = 0.0,
        return_interm_layers: bool = False,
        init_values: Optional[float] = None,  # for layerscale
        ln_pre: bool = False,
        ln_post: bool = False,
        bias_patch_embed: bool = True,
        compile_mode: Optional[str] = None,
    ):
        """
        Args:
            img_size (int): Input image size. Only relevant for rel pos or rope.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            tile_abs_pos (bool): If True, tile absolute positional embeddings instead of interpolation.
            rel_pos_blocks (list): Blocks which have rel pos embeddings.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_att_blocks (list): Indexes for blocks using global attention (other blocks use window attention).
            use_rope (bool): whether to use rope 2d (indep of rel_pos_blocks, as it can be used together).
            rope_pt_size (int): size of rope in previous stage of training, needed for interpolation or tiling.
            use_interp_rope: whether to interpolate (or extrapolate) rope to match target input size,
                expected to specify source size as rope_pt_size.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretraining models use class token.
            retain_cls_token: whether cls_token should be retained.
            dropout (float): Dropout rate. Applied in residual blocks of attn, mlp and inside the mlp.

            return_interm_layers (bool): Whether to return intermediate layers (all global attention blocks).
            init_values: layer scale init, None for no layer scale.

            ln_pre (bool): If True, apply layer norm before transformer blocks.
            ln_post (bool): If True, apply layer norm after transformer blocks.
            bias_patch_embed (bool): bias in conv for patch embed?
            compile_mode (str): mode to compile the forward
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        window_block_indexes = [i for i in range(depth) if i not in global_att_blocks]
        self.full_attn_ids = list(global_att_blocks)
        self.rel_pos_blocks = [False] * depth
        if isinstance(rel_pos_blocks, bool) and rel_pos_blocks:
            self.rel_pos_blocks = [True] * depth
        else:
            for i in rel_pos_blocks:
                self.rel_pos_blocks[i] = True

        self.retain_cls_token = retain_cls_token
        if self.retain_cls_token:
            assert pretrain_use_cls_token
            assert len(window_block_indexes) == 0, "windowing not supported with cls token"

            assert sum(self.rel_pos_blocks) == 0, "rel pos not supported with cls token"

            scale = embed_dim**-0.5
            self.class_embedding = nn.Parameter(scale * torch.randn(1, 1, embed_dim))

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-5)

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=bias_patch_embed,
        )

        # Handle absolute positional embedding
        self.tile_abs_pos = tile_abs_pos
        self.use_abs_pos = use_abs_pos
        if self.tile_abs_pos:
            assert self.use_abs_pos

        if self.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        cur_stage = 1
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=self.rel_pos_blocks[i],
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                use_rope=use_rope,
                rope_pt_size=((window_size, window_size) if rope_pt_size is None else (rope_pt_size, rope_pt_size)),
                rope_interp=use_interp_rope,
                cls_token=self.retain_cls_token,
                dropout=dropout,
                init_values=init_values,
            )

            if i not in window_block_indexes:
                cur_stage += 1

            self.blocks.append(block)

        self.return_interm_layers = return_interm_layers
        self.channel_list = [embed_dim] * len(self.full_attn_ids) if return_interm_layers else [embed_dim]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.ln_pre = norm_layer(embed_dim) if ln_pre else nn.Identity()
        self.ln_post = norm_layer(embed_dim) if ln_post else nn.Identity()

        self.apply(self._init_weights)

        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, tensor_list: NestedTensor) -> list[NestedTensor]:
        x = tensor_list.tensors

        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if self.retain_cls_token:
            # If cls_token is retained, we don't
            # maintain spatial shape
            x = torch.cat([self.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1

        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed,
                self.pretrain_use_cls_token,
                (h, w),
                self.retain_cls_token,
                tiling=self.tile_abs_pos,
            )

        x = self.ln_pre(x)

        outputs = []
        masks = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.full_attn_ids[-1]) or (self.return_interm_layers and i in self.full_attn_ids):
                if i == self.full_attn_ids[-1]:
                    x = self.ln_post(x)

                feats = x[:, s:]
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)
                else:
                    assert feats.ndim == 3
                    h = w = math.sqrt(feats.shape[1])
                    feats = feats.reshape(feats.shape[0], h, w, feats.shape[-1]).permute(0, 3, 1, 2)

                # Optimization, if the mask is all False, just ignore it
                if tensor_list.mask is not None and tensor_list.mask.any() and masks is None:
                    masks = F.interpolate(tensor_list.mask[None].float(), size=feats.shape[-2:]).bool()[0]
                outputs.append(NestedTensor(feats, masks))

        return outputs

    def get_layer_id(self, layer_name: str) -> int:
        # https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
        num_layers = self.get_num_layers()

        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("ln_pre") != -1:
            return 0
        elif layer_name.find("pos_embed") != -1 or layer_name.find("cls_token") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)


class OriginalViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
    ):
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = True  # neck_norm is None
        dim = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

    def forward(self, tensor_list: NestedTensor):
        xs = self.trunk(tensor_list)
        out = []
        pos = []
        x = xs[-1]  # simpleFPN
        for _, conv in enumerate(self.convs):
            x_out = NestedTensor(conv(x.tensors), x.mask)
            out.append(x_out)
            pos.append(self.position_encoding(x_out).to(x_out.tensors.dtype))
        return out, pos


### End ViTDet ---------------------------------------------------------------------------- #


### NonFusionVLBackbone ---------------------------------------------------------------------------- #


class NonFusionVLBackbone(nn.Module):
    """A vision-language backbone that does not fuse the vision and language"""

    def __init__(
        self,
        visual,
        text,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp=0,
    ):
        """Initialize the backbone combiner.

        :param visual: The vision backbone to use
        :param text: The text encoder to use
        """
        super().__init__()
        self.vision_backbone = torch.compile(visual) if compile_visual else visual
        self.language_backbone = text
        self.scalp = scalp
        # allow running activation checkpointing on the entire vision and language backbones
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone

    def forward(
        self,
        samples: NestedTensor,
        captions: list[str],
        input_boxes: Optional[torch.Tensor] = None,
        additional_text: Optional[list[str]] = None,
    ):
        """Forward pass of the backbone combiner.

        :param samples: The input images
        :param captions: The input captions
        :param input_boxes: If the text contains place-holders for boxes, this
            parameter contains the tensor containing their spatial features
        :param additional_text: This can be used to encode some additional text
            (different from the captions) in the same forward of the backbone
        :return: Output dictionary with the following keys:
            - vision_features: The output of the vision backbone
            - language_features: The output of the language backbone
            - vision_mask: The attention mask of the vision backbone
            - language_mask: The attention mask of the language backbone
            - vision_pos_enc: The positional encoding of the vision backbone
            - (optional) additional_text_features: The output of the language
                backbone for the additional text
            - (optional) additional_text_mask: The attention mask of the
                language backbone for the additional text
        """
        output = self.forward_image(samples)
        device = output["vision_features"].device
        output.update(self.forward_text(captions, input_boxes, additional_text, device))
        return output

    def forward_image(self, samples: NestedTensor):
        return activation_ckpt_wrapper(self._forward_image_no_act_ckpt)(
            samples=samples,
            act_ckpt_enable=self.act_ckpt_whole_vision_backbone and self.training,
        )

    def _forward_image_no_act_ckpt(self, samples):
        # Forward through backbone
        features, pos = self.vision_backbone(samples)
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        src, mask = features[-1].decompose()

        output = {
            "vision_features": src,
            "vision_mask": mask,
            "vision_pos_enc": pos,
            "backbone_fpn": features,  # Temporary, to not break anything else
        }
        return output

    def forward_text(self, captions, input_boxes=None, additional_text=None, device="cuda"):
        return activation_ckpt_wrapper(self._forward_text_no_ack_ckpt)(
            captions=captions,
            input_boxes=input_boxes,
            additional_text=additional_text,
            device=device,
            act_ckpt_enable=self.act_ckpt_whole_language_backbone and self.training,
        )

    def _forward_text_no_ack_ckpt(
        self,
        captions,
        input_boxes=None,
        additional_text=None,
        device="cuda",
    ):
        output = {}

        # Forward through text_encoder
        text_to_encode = copy.copy(captions)
        if additional_text is not None:
            # if there are additional_text, we piggy-back them into this forward.
            # They'll be used later for output alignment
            text_to_encode += additional_text

        sdpa_context = sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )

        with sdpa_context:
            text_attention_mask, text_memory, text_embeds = self.language_backbone(
                text_to_encode, input_boxes, device=device
            )

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[-len(additional_text) :]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds  # Text embeddings before forward to the encoder

        return output


### End NonFusionVLBackbone ---------------------------------------------------------------------------- #


### DotProductScoring ---------------------------------------------------------------------------- #
class DotProductScoring(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, torch.nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp  # an optional MLP projection for prompt
        self.prompt_proj = torch.nn.Linear(d_model, d_proj)
        self.hs_proj = torch.nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
        is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
        # num_valid has shape (bs, 1)
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        # mean pool over all the valid tokens -- pooled_prompt has shape (bs, proj_dim)
        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        # hs has shape (num_layer, bs, num_query, d_model)
        # prompt has shape (seq, bs, d_model)
        # prompt_mask has shape (bs, seq), where 1 is valid and 0 is padding
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2

        # apply MLP on prompt if specified
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)

        # first, get the mean-pooled version of the prompt
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)

        # then, project pooled_prompt and hs to d_proj dimensions
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # (bs, d_proj)
        proj_hs = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # finally, get dot-product scores of shape (num_layer, bs, num_query, 1)
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale

        # clamp scores to a max value to avoid numerical issues in loss or matcher
        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


### End DotProductScoring ---------------------------------------------------------------------------- #


### Sam3Image ---------------------------------------------------------------------------- #


def _update_out(out, out_name, out_value, auxiliary=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


class Sam3Image(torch.nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_output: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head

        self.o2m_mask_predict = o2m_mask_predict

        self.dot_prod_scoring = dot_prod_scoring

        # verify the number of queries for O2O and O2M
        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)

        self.use_instance_query = use_instance_query
        self.multimask_output = multimask_output

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def _get_img_feats(self, backbone_out, img_ids):
        """Retrieve correct image features from backbone output."""
        if "backbone_fpn" in backbone_out:
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                # this is for the video case. We only have a partial forward of all features
                img_ids = backbone_out["id_mapping"][img_ids]
                # If this assert fails, it likely means we're requesting different img_ids (perhaps a different frame?)
                # We currently don't expect this to happen. We could technically trigger a recompute here,
                # but likely at the cost of a cpu<->gpu sync point, which would deteriorate perf
                torch._assert_async((img_ids >= 0).all())

            vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
            vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
            vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]  # (H, W) shapes
            # index and flatten visual features NxCxHxW => HWxNxC (batch-first => seq-first)
            img_feats = [x.tensors[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats]
            img_masks = [None if x.mask is None else x.mask[img_ids].flatten(1) for x in vis_feats]
            img_pos_embeds = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
            return backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes

        # Image features not available in backbone output, so we compute them on the fly
        # This case likely occurs for video. In that case, we want to forward only the current frame
        img_batch = backbone_out["img_batch_all_stages"]
        if img_ids.numel() > 1:
            # Only forward backbone on unique image ids to avoid repetitive computation
            unique_ids, _ = torch.unique(img_ids, return_inverse=True)
        else:
            unique_ids, _ = img_ids, slice(None)
        # Compute the image features on those unique image ids
        # note: we allow using a list (or other indexable types) of tensors as img_batch.tensors
        # (e.g. for async frame loading in demo). In this case we index img_batch.tensors directly
        if isinstance(img_batch.tensors, torch.Tensor):
            image = img_batch.tensors[unique_ids]
        elif unique_ids.numel() == 1:
            image = img_batch.tensors[unique_ids.item()].unsqueeze(0)
        else:
            image = torch.stack([img_batch.tensors[i] for i in unique_ids.tolist()])
        # `img_batch` might be fp16 and offloaded to CPU
        image = image.to(dtype=torch.float32, device=self.device)
        image_mask = img_batch.mask[unique_ids] if img_batch.mask is not None else None
        image_tensors = NestedTensor(tensors=image, mask=image_mask)
        # Next time we call this function, we want to remember which indices we computed
        id_mapping = torch.full((len(img_batch.tensors),), -1, dtype=torch.long, device=self.device)
        id_mapping[unique_ids] = torch.arange(len(unique_ids), device=self.device)
        backbone_out = {
            **backbone_out,
            **self.backbone.forward_image(image_tensors),
            "id_mapping": id_mapping,
        }
        assert "backbone_fpn" in backbone_out
        return self._get_img_feats(backbone_out, img_ids=img_ids)

    def _get_feat_tuple(self, backbone_out, find_input):
        img_ids, img_feat_inds = find_input.img_ids, slice(None)
        return self._get_img_feats(backbone_out, img_ids), img_feat_inds

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
        prev_mask_pred=None,
    ):
        # index text features (note that regardless of early or late fusion, the batch size of
        # `txt_feats` is always the number of *prompts* in the encoder)
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        feat_tuple, _ = self._get_feat_tuple(backbone_out, find_input)
        backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes = feat_tuple

        if prev_mask_pred is not None:
            # TODO: Support Multi-scale? for now, mutli-scale will break other things (like decoder boxRPB), so it won't go silently.
            img_feats = [img_feats[-1] + prev_mask_pred]
        # Encode geometry
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros((0, *geo_feats.shape[1:]), device=geo_feats.device)
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )
        if encode_text:
            prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        else:
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out

    def _run_encoder(
        self,
        backbone_out,
        find_input,
        prompt,
        prompt_mask,
        prev_mask_pred=None,
        encoder_extra_kwargs: Optional[dict] = None,
    ):
        feat_tuple, img_feat_inds = self._get_feat_tuple(backbone_out, find_input)
        backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes = feat_tuple
        if prev_mask_pred is not None:
            # TODO: Support Multi-scale? for now, mutli-scale will break other things (like decoder boxRPB), so it won't go silently.
            img_feats = [img_feats[-1] + prev_mask_pred]
        # Run the encoder
        prompt_pos_embed = torch.zeros_like(prompt)
        # make a copy of the image feature lists since the encoder may modify these lists in-place
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=img_masks.copy(),
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        encoder_out = {
            # encoded image features
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            "img_feat_inds": img_feat_inds,
            # encoded text features (or other prompts)
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple

    def _update_scores_and_boxes(self, out, hs, reference_boxes, prompt, prompt_mask, apply_dac=None):
        apply_dac = apply_dac if apply_dac is not None else self.transformer.decoder.dac
        num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
        num_o2m = hs.size(2) - num_o2o
        assert num_o2m == (num_o2o if apply_dac else 0)
        out["queries"] = hs[-1][:, :num_o2o]  # remove o2m queries if there are any
        # score prediction
        # if self.use_dot_prod_scoring:
        outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)
        # else:
        #     outputs_class = self.class_embed(hs)
        # box prediction
        anchor_box_offsets = self.transformer.decoder.bbox_embed(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        _update_out(out, "pred_logits", outputs_class[:, :, :num_o2o])
        _update_out(out, "pred_boxes", outputs_coord[:, :, :num_o2o])
        _update_out(out, "pred_boxes_xyxy", outputs_boxes_xyxy[:, :, :num_o2o])
        if num_o2m > 0:
            _update_out(out, "pred_logits_o2m", outputs_class[:, :, num_o2o:])
            _update_out(out, "pred_boxes_o2m", outputs_coord[:, :, num_o2o:])
            _update_out(out, "pred_boxes_xyxy_o2m", outputs_boxes_xyxy[:, :, num_o2o:])

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        vis_feat_sizes,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
        apply_dac=None,
    ):
        # Apply segmentation head (w/ bfloat16 autocast just like the rest of the model)
        apply_dac = apply_dac if apply_dac is not None else self.transformer.decoder.dac
        if self.segmentation_head is not None:
            num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
            num_o2m = hs.size(2) - num_o2o
            obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
            seg_head_outputs = activation_ckpt_wrapper(self.segmentation_head)(
                backbone_feats=backbone_out["backbone_fpn"],
                obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                act_ckpt_enable=False,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            aux_masks = False  # self.aux_loss and self.segmentation_head.aux_masks
            for k, v in seg_head_outputs.items():
                if k in self.segmentation_head.instance_keys:
                    _update_out(out, k, v[:, :num_o2o], auxiliary=aux_masks)
                    if self.o2m_mask_predict and num_o2m > 0:  # handle o2m mask prediction
                        _update_out(out, f"{k}_o2m", v[:, num_o2o:], auxiliary=aux_masks)
                else:
                    out[k] = v

    def _get_best_mask(self, out):
        prev_mask_idx = out["pred_logits"].argmax(dim=1).squeeze(1)
        batch_idx = torch.arange(out["pred_logits"].shape[0], device=prev_mask_idx.device)
        prev_mask_pred = out["pred_masks"][batch_idx, prev_mask_idx][:, None]
        # Downsample mask to match image resolution.
        prev_mask_pred = self.geometry_encoder.mask_encoder.mask_downsampler(prev_mask_pred)
        prev_mask_pred = prev_mask_pred.flatten(-2).permute(2, 0, 1)

        return prev_mask_pred

    def forward_video_grounding(
        self,
        backbone_out,
        find_input,
        find_target,
        frame_idx,
        previous_stages_out,
        geometric_prompt: Prompt = None,
        run_encoder: bool = True,
        prev_encoder_out: dict = None,
        visual_prompt=None,
        visual_prompt_mask=None,
        is_instance_prompt=False,
        act_ckpt_enable: bool = False,
        track_in_reverse: bool = False,  # track in reverse time order (for demo usage)
        multimask_output: bool = False,
        prev_mask_pred: torch.Tensor = None,
    ):
        """Only activation checkpointing the inner part of video grounding forward."""
        num_prompts = find_input.img_ids.size(0)
        prev_frame_idx = frame_idx + 1 if track_in_reverse else frame_idx - 1

        prev_tracking_queries = self._init_tracking_queries(
            B=num_prompts,
            is_instance_prompt=is_instance_prompt,
            multimask_output=multimask_output,
        )

        prompt, prompt_mask, backbone_out = self._encode_prompt(
            backbone_out,
            find_input,
            geometric_prompt,
            visual_prompt_embed=visual_prompt,
            visual_prompt_mask=visual_prompt_mask,
            prev_mask_pred=prev_mask_pred,
        )
        # Run the encoder
        if run_encoder:
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out,
                find_input,
                prompt,
                prompt_mask,
                prev_mask_pred=prev_mask_pred,
            )
        else:
            assert prev_encoder_out is not None, (
                "Something went wrong. If `run_encoder` is False, encoder outputs from previous step should be passed."
            )
            backbone_out, encoder_out = (
                prev_encoder_out["backbone_out"],
                prev_encoder_out["encoder_out"],
            )

        img_feat_inds = encoder_out["img_feat_inds"]
        # (Note that here we directly index the encoder output visual features into per-query
        # feature maps, so the batch size in decoder will always be the number of text prompts
        # rather than the number of images even under late fusion. This is because for video
        # tracking, the tracking queries will have batch size equal to the number of text prompts
        # anyway. So there is no way to reduce the decoder batch size to be the number of images
        # under late fusion, which is unlike the case of image grounding.)
        encoder_hidden_states = encoder_out["encoder_hidden_states"][:, img_feat_inds]
        pos_embed = encoder_out["pos_embed"][:, img_feat_inds]
        assert encoder_hidden_states.size(1) == num_prompts
        assert pos_embed.size(1) == num_prompts
        src_mask = None
        if encoder_out["padding_mask"] is not None:
            src_mask = encoder_out["padding_mask"][img_feat_inds]  # mask is batch-first
            assert src_mask.size(0) == num_prompts

        out = {
            "encoder_hidden_states": encoder_hidden_states,
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        # Run the decoder
        out, hs = self._run_decoder_for_tracking(
            memory=encoder_hidden_states,
            pos_embed=pos_embed,
            src_mask=src_mask,
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
            tracking_queries=prev_tracking_queries,
            is_instance_prompt=is_instance_prompt,
            is_multimask_output=multimask_output,
            apply_dac=False,
        )

        # Run segmentation heads
        self._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=find_input.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
            apply_dac=False,
        )

        out = self._postprocess_out(out, multimask_output=multimask_output)
        return out, backbone_out

    def _postprocess_out(self, out: dict, multimask_output: bool = False):
        # TODO: Drop some keys to save memory
        # For multimask output, during eval we return the single best mask with the
        # dict keys expected by the evaluators, but also return the multimasks output with new keys.
        num_mask_preds = out["pred_masks"].size(1)
        if not self.training and multimask_output and num_mask_preds > 1:
            out["multi_pred_logits"] = out["pred_logits"]
            out["multi_pred_masks"] = out["pred_masks"]
            out["multi_pred_boxes"] = out["pred_boxes"]
            out["multi_pred_boxes_xyxy"] = out["pred_boxes_xyxy"]

            best_mask_idx = out["pred_logits"].argmax(1).squeeze(1)
            batch_idx = torch.arange(len(best_mask_idx), device=best_mask_idx.device)

            out["pred_logits"] = out["pred_logits"][batch_idx, best_mask_idx].unsqueeze(1)
            out["pred_masks"] = out["pred_masks"][batch_idx, best_mask_idx].unsqueeze(1)
            out["pred_boxes"] = out["pred_boxes"][batch_idx, best_mask_idx].unsqueeze(1)
            out["pred_boxes_xyxy"] = out["pred_boxes_xyxy"][batch_idx, best_mask_idx].unsqueeze(1)

        return out

    def _run_decoder_for_tracking(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
        tracking_queries,
        apply_dac=None,
        **kwargs,
    ):
        # In Video OWL-ViT style tracking, we directly feed previous frame's decoder
        # output embeddings from as current frame's decoder inputs.
        tgt = tracking_queries["embed"]
        reference_boxes = tracking_queries["reference_boxes"]

        hs, reference_boxes = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_mask,
            pos=pos_embed,
            reference_boxes=reference_boxes,
            level_start_index=encoder_out["level_start_index"],
            spatial_shapes=encoder_out["spatial_shapes"],
            valid_ratios=encoder_out["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=apply_dac,
        )
        hs = hs.transpose(1, 2)  # seq-first to batch-first
        reference_boxes = reference_boxes.transpose(1, 2)  # seq-first to batch-first
        self._update_scores_and_boxes(out, hs, reference_boxes, prompt, prompt_mask, apply_dac)
        # in Video OWL-ViT style tracking, all output queries are valid
        scores = out["pred_logits"].squeeze(-1)
        out["pred_is_valid"] = torch.ones_like(scores, dtype=torch.bool)  # (B, Q) shape
        # the previously tracked object ids for all (det + track) output queries
        out["pred_old_obj_ids"] = tracking_queries["object_ids"]
        return out, hs

    def _init_tracking_queries(self, B, is_instance_prompt, multimask_output=False):
        """Initialize the tracking queries for the first frame of a video."""
        # Following Video OWL-ViT, on the first frame, the tracking queries are initialized
        # using the learned detection queries.
        if is_instance_prompt and self.use_instance_query:
            query_embed = self.transformer.decoder.instance_query_embed.weight
            query_embed = query_embed[1:] if multimask_output else query_embed[:1]
            reference_boxes = self.transformer.decoder.instance_reference_points.weight
            reference_boxes = reference_boxes[1:] if multimask_output else reference_boxes[:1]
        else:
            query_embed = self.transformer.decoder.query_embed.weight
            reference_boxes = self.transformer.decoder.reference_points.weight

        reference_boxes = reference_boxes.unsqueeze(1).expand(-1, B, -1).sigmoid()
        device = query_embed.device
        init_embed = query_embed.unsqueeze(1).expand(-1, B, -1)  # (Q, B, D), seq-first
        # Initial object ids are all -1, meaning that they are not tracking any objects yet
        Q = query_embed.size(0)
        init_obj_ids = -torch.ones(B, Q, dtype=torch.long, device=device)
        # Initialize the keep-alive countdown for each query. If the tracked object of a query
        # goes out of frame or gets occluded, we maintain its tracking object id for this countdown
        # number of frames before resetting its object id to -1.
        keep_alive_countdown = -torch.ones_like(init_obj_ids)  # (B, Q) shape
        tracking_queries = {
            "embed": init_embed,  # (Q, B, D) shape, seq-first
            "reference_boxes": reference_boxes,  # (Q, B, D) shape, seq-first
            "object_ids": init_obj_ids,  # (B, Q) shape
            "keep_alive_countdown": keep_alive_countdown,  # (B, Q) shape
            # the maximum object id assigned so far (to assign new ids during inference)
            "max_object_id": torch.zeros(B, 1, dtype=torch.long, device=device),
        }
        return tracking_queries


class Sam3ImageInteractiveDemo(Sam3Image):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.compile_model = compile_model

        # Debug added for owlvit
        self.use_aot_mem = False
        self.use_obj_mem_bank = False

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_to_cpu=False,
    ):
        """Initialize an inference state from `resource_path` (an image or a video)."""

        images, orig_height, orig_width = load_image_as_single_frame(
            image_path=resource_path,
            image_size=self.image_size,
            offload_to_cpu=offload_to_cpu,
        )
        inference_state = {}
        inference_state["image_size"] = self.image_size
        inference_state["num_frames"] = len(images)
        inference_state["device"] = torch.device("cuda")
        # the original video height and width, used for resizing final output scores
        inference_state["orig_height"] = orig_height
        inference_state["orig_width"] = orig_width
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # inputs on each frame
        self._construct_initial_input_batch(inference_state, images)
        return inference_state

    def _construct_initial_input_batch(self, inference_state, images):
        """Construct an initial `BatchedDatapoint` instance as input."""
        # 1) img_batch
        num_frames = len(images)
        device = inference_state["device"]
        img_batch = NestedTensor(tensors=images, mask=None)

        # 2) find_text_batch
        # "<text placeholder>" will be replaced by the actual text prompt when adding prompts
        find_text_batch = ["<text placeholder>", "visual", "geometric"]

        # 3) find_inputs
        input_box_embedding_dim = 258  # historical default
        input_points_embedding_dim = 257  # historical default
        dummy_ptrs = BatchedPointer(stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[])
        stages = [
            FindStage(
                img_ids=[stage_id],
                text_ids=[0],
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_before_embed=[torch.zeros(4)],
                input_boxes_mask=[1],
                input_boxes_label=[0],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_before_embed=[torch.empty(0, 3)],
                input_points_mask=[torch.empty(0)],
                ptrs=dummy_ptrs,
                ptrs_seg=dummy_ptrs,
                object_ids=[],
            )
            for stage_id in range(num_frames)
        ]
        for i in range(len(stages)):
            stages[i] = convert_my_tensors(stages[i])

        # construct the final `BatchedDatapoint` and cast to GPU
        input_batch = BatchedDatapoint(
            img_batch=img_batch,
            find_text_batch=find_text_batch,
            find_inputs=stages,
            find_targets=[None] * num_frames,
            get_queries=None,
            find_metadatas=[None] * num_frames,
        )
        input_batch = recursive_to(input_batch, device, non_blocking=True)
        inference_state["input_batch"] = input_batch

        # construct the placeholder interactive prompts and tracking queries
        bs = 1
        inference_state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, bs, 4, device=device),
            box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
            point_embeddings=torch.zeros(0, bs, 2, device=device),
            point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
        )

        # constructing an output list in inference state (we start with an empty list)
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["text_prompt"] = None
        inference_state["per_frame_raw_point_input"] = [None] * num_frames
        inference_state["per_frame_raw_box_input"] = [None] * num_frames
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames
        inference_state["per_frame_cur_step"] = [0] * num_frames
        if self.use_aot_mem:
            inference_state["aot_mem_per_frame"] = {"spatial": {}, "pointer": {}}
        if self.use_obj_mem_bank:
            inference_state["obj_mem_per_frame"] = {
                "roi": {},
                "roi_zoomed_out": {},
                "global": {},
            }

        # placeholders for cached outputs
        # (note: currently, a single visual prompt embedding is shared for all frames)
        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state,
        frame_idx=0,
        text_str=None,
        clear_old_points=True,
        points=None,
        point_labels=None,
        boxes_xywh=None,
        box_labels=None,
        clear_old_boxes=True,
        output_prob_thresh=0.5,
        instance_prompt=False,
    ):
        """
        Add text, point or box prompts on a single frame. This method returns the inference
        outputs only on the prompted frame.

        Note that text prompts are NOT associated with a particular frame (i.e. they apply
        to all frames). However, we only run inference on the frame specified in `frame_idx`.
        """
        device = inference_state["device"]
        num_frames = inference_state["num_frames"]
        assert text_str is not None or points is not None or boxes_xywh is not None, (
            "at least one type of prompt (text, points, boxes) must be provided"
        )
        assert 0 <= frame_idx < num_frames, f"{frame_idx=} is out of range for a total of {num_frames} frames"

        # 1) add text prompt
        if text_str is not None:
            # currently we do not allow simultaneously adding text prompt and visual
            # prompt both as initial prompt (since visual prompt uses the text "visual")
            if any(p is not None for p in inference_state["per_frame_visual_prompt"]):
                raise RuntimeError(
                    "Text and visual prompts (box as an initial prompt) cannot be used together. "
                    "Please reset the session."
                )

            inference_state["text_prompt"] = text_str
            # add the text prompt into the input batch (to be applied to *all* frames)
            inference_state["input_batch"].find_text_batch[0] = text_str
            for t in range(inference_state["num_frames"]):
                text_id = self.TEXT_ID_FOR_TEXT
                inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id

        # 2) add geometric prompt (points or boxes)
        # start with an empty geometric_prompt (we later add previous point and box prompts
        # from "per_frame_raw_point_input" and "per_frame_raw_box_input" below)
        geometric_prompt = inference_state["constants"]["empty_geometric_prompt"].clone()

        if points is not None and boxes_xywh is not None:
            raise RuntimeError("Cannot add both point and box prompts at the same time. ")

        if points is not None and not instance_prompt:
            raise RuntimeError("Point prompts are only supported for instance tracking. ")

        if instance_prompt and (text_str is not None or boxes_xywh is not None):
            raise RuntimeError("Text and box prompts are not supported for instance tracking. ")

        new_visual_prompt = None

        # 2.1) handle point prompt
        assert (points is not None) == (point_labels is not None)
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32)
            point_labels = torch.as_tensor(point_labels, dtype=torch.long)
            assert points.dim() == 2
            assert points.size(0) > 0 and points.size(-1) == 2
            assert point_labels.dim() == 1 and point_labels.size(0) == points.size(0)
            assert torch.all(points >= 0).item() and torch.all(points <= 1).item()
            # append previous points under `clear_old_points=False`
            prev_point_input = inference_state["per_frame_raw_point_input"][frame_idx]
            if prev_point_input is not None and not clear_old_points:
                prev_points, prev_point_labels = prev_point_input
                points = torch.cat([prev_points, points], dim=0)
                point_labels = torch.cat([prev_point_labels, point_labels], dim=0)
            new_point_input = points, point_labels
            inference_state["per_frame_raw_point_input"][frame_idx] = new_point_input
            # add a batch dimensions (note that it's sequence first)
            points = points.unsqueeze(1).to(device)
            point_labels = point_labels.unsqueeze(1).to(device)
            geometric_prompt.append_points(points=points, labels=point_labels)
            new_visual_prompt = None

            for t in range(inference_state["num_frames"]):
                inference_state["input_batch"].find_inputs[t].text_ids[...] = self.TEXT_ID_FOR_GEOMETRIC

        # 2.2) handle box prompt
        assert (boxes_xywh is not None) == (box_labels is not None)
        if boxes_xywh is not None:
            boxes_xywh = torch.as_tensor(boxes_xywh, dtype=torch.float32)
            box_labels = torch.as_tensor(box_labels, dtype=torch.long)
            # input boxes are expected to be [xmin, ymin, width, height] format
            # in normalized coordinates of range 0~1, similar to FA
            assert boxes_xywh.dim() == 2
            assert boxes_xywh.size(0) > 0 and boxes_xywh.size(-1) == 4
            assert box_labels.dim() == 1 and box_labels.size(0) == boxes_xywh.size(0)
            boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)
            assert (boxes_xywh >= 0).all().item() and (boxes_xywh <= 1).all().item()
            assert (boxes_cxcywh >= 0).all().item() and (boxes_cxcywh <= 1).all().item()
            # append previous boxes under `clear_old_boxes=False`
            prev_box_input = inference_state["per_frame_raw_box_input"][frame_idx]
            if prev_box_input is not None and not clear_old_boxes:
                prev_boxes_cxcywh, prev_box_labels = prev_box_input
                boxes_cxcywh = torch.cat([prev_boxes_cxcywh, boxes_cxcywh], dim=0)
                box_labels = torch.cat([prev_box_labels, box_labels], dim=0)
            new_box_input = boxes_cxcywh, box_labels
            inference_state["per_frame_raw_box_input"][frame_idx] = new_box_input

            # handle the case of visual prompt (also added as an input box from the UI)
            boxes_cxcywh, box_labels, new_visual_prompt = self._get_visual_prompt(
                inference_state, frame_idx, boxes_cxcywh, box_labels
            )
            # add a batch dimensions (note that it's sequence first)
            boxes_cxcywh = boxes_cxcywh.unsqueeze(1).to(device)
            box_labels = box_labels.unsqueeze(1).to(device)
            geometric_prompt.append_boxes(boxes=boxes_cxcywh, labels=box_labels)

        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        # 3) run inference on this frame
        inference_state["backbone_out"] = self._init_backbone_out(inference_state)
        if new_visual_prompt is not None:
            # currently we do not allow simultaneously adding text prompt and visual
            # prompt both as initial prompt (since visual prompt uses the text "visual")
            if inference_state["text_prompt"] is not None:
                raise RuntimeError(
                    "Text and visual prompts (box as an initial prompt) cannot be used together. "
                    "Please reset the session."
                )

            # add the visual prompt into the input batch and encode it (currently the added
            # visual prompt is applied to *all* frames, i.e. not just this prompted frame)
            for t in range(inference_state["num_frames"]):
                text_id = self.TEXT_ID_FOR_VISUAL
                inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id
            # currently visual prompt is encoded the same way (`_encode_prompt`) as geometric prompt
            visual_prompt_embed, visual_prompt_mask, _backbone_out = self._encode_prompt(
                backbone_out=inference_state["backbone_out"],
                find_input=inference_state["input_batch"].find_inputs[frame_idx],
                geometric_prompt=new_visual_prompt,
                encode_text=False,
            )
            inference_state["visual_prompt_embed"] = visual_prompt_embed
            inference_state["visual_prompt_mask"] = visual_prompt_mask

        reverse = False  # TODO for now, we always track forward when adding prompts
        out = self._run_single_frame_inference(
            inference_state,
            frame_idx,
            reverse,
            is_instance_processing=instance_prompt,
        )
        return self._postprocess_output(inference_state, out, output_prob_thresh, pop=False)

    def _get_visual_prompt(self, inference_state, frame_idx, boxes_cxcywh, box_labels):
        """
        Handle the case of visual prompt. Currently, in the inference API we do not
        explicitly distinguish between initial box as visual prompt vs subsequent boxes
        or boxes after inference for refinement.
        """
        # If the frame hasn't had any inference results before (prompting or propagation),
        # we treat the first added box prompt as a visual prompt; otherwise, we treat
        # the first box just as a refinement prompt.
        is_new_visual_prompt = (
            inference_state["per_frame_visual_prompt"][frame_idx] is None
            and inference_state["previous_stages_out"][frame_idx] is None
        )
        if is_new_visual_prompt:
            # take the first box prompt as a visual prompt
            device = inference_state["device"]
            new_visual_prompt = Prompt(
                box_embeddings=boxes_cxcywh[None, 0:1, :].to(device),  # (seq, bs, 4)
                box_mask=None,
                box_labels=box_labels[None, 0:1].to(device),  # (seq, bs)
                point_embeddings=None,
                point_mask=None,
                point_labels=None,
            )
            inference_state["per_frame_visual_prompt"][frame_idx] = new_visual_prompt
        else:
            new_visual_prompt = None

        # `boxes_cxcywh` and `box_labels` contains all the raw box inputs added so far
        # strip any visual prompt from the input boxes (for geometric prompt encoding)
        if inference_state["per_frame_visual_prompt"][frame_idx] is not None:
            boxes_cxcywh = boxes_cxcywh[1:]
            box_labels = box_labels[1:]

        return boxes_cxcywh, box_labels, new_visual_prompt

    def _init_backbone_out(self, inference_state):
        """
        Initialize a backbone_out dictionary and extract the text features.

        Note that the visual features of each frame are not extracted here. They will be
        extracted on the fly when running inference on each frame.
        """
        input = inference_state["input_batch"]
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)
        return backbone_out

    def _run_single_frame_inference(self, inference_state, frame_idx, reverse, is_instance_processing=False):
        """
        Perform inference on a single frame and get its inference results. This would
        also update `inference_state`.
        """
        input = inference_state["input_batch"]
        find_input = input.find_inputs[frame_idx]
        find_target = None
        num_frames = inference_state["num_frames"]
        is_video_batch = num_frames > 1

        backbone_out = inference_state["backbone_out"]
        geometric_prompt = inference_state["per_frame_geometric_prompt"][frame_idx]
        if geometric_prompt is None:
            geometric_prompt = inference_state["constants"]["empty_geometric_prompt"]
        previous_stages_out = inference_state["previous_stages_out"]
        prev_encoder_out = None
        if previous_stages_out[frame_idx] is not None:
            prev_encoder_out = previous_stages_out[frame_idx].get("prev_encoder_out")
        cur_step = inference_state["per_frame_cur_step"][frame_idx]

        if self.use_aot_mem:
            aot_mem_per_frame = inference_state["aot_mem_per_frame"]
            aot_mem_id_bank = self._aot_mem_get_rand_permuted_id_bank(is_video_batch)
        else:
            aot_mem_per_frame, aot_mem_id_bank = None, None
        obj_mem_per_frame = inference_state.get("obj_mem_per_frame", None)

        prev_mask_pred = None
        if (
            inference_state["previous_stages_out"][frame_idx]
            # and self.use_prev_mask
            and is_instance_processing
        ):
            prev_mask_pred = self._get_best_mask(inference_state["previous_stages_out"][frame_idx])

        out, _ = self.forward_video_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            frame_idx=frame_idx,
            # num_frames=num_frames,
            previous_stages_out=previous_stages_out,
            geometric_prompt=geometric_prompt.clone(),
            # run_encoder=self.interactivity_in_encoder or cur_step == 0,
            prev_encoder_out=prev_encoder_out,
            visual_prompt=inference_state["visual_prompt_embed"],
            visual_prompt_mask=inference_state["visual_prompt_mask"],
            is_instance_prompt=is_instance_processing,
            track_in_reverse=reverse,
            prev_mask_pred=prev_mask_pred,
        )
        inference_state["previous_stages_out"][frame_idx] = out
        inference_state["per_frame_cur_step"][frame_idx] = cur_step + 1

        return out

    def _postprocess_output(
        self,
        inference_state,
        out,
        output_prob_thresh,
        pop=True,
    ):
        """Post-process the single-frame output into the desired numpy result format."""
        prompt_idx = 0
        if pop:
            out_scores = out.pop("pred_logits")[prompt_idx].squeeze(-1)
            out_boxes_xyxy = out.pop("pred_boxes_xyxy")[prompt_idx]
            # out_obj_ids = out.pop("pred_object_ids")[prompt_idx]
            out_masks = out.pop("pred_masks")[prompt_idx]

            # remove a few unused keys (to reduce GPU memory usage)
            unused_output_keys = [
                "pred_boxes",
                "pred_is_valid",
                "pred_old_obj_ids",
                "semantic_seg",
                "presence_logit",
            ]
            for k in unused_output_keys:
                out.pop(k, None)
        else:
            out_scores = out["pred_logits"][prompt_idx].squeeze(-1)
            out_boxes_xyxy = out["pred_boxes_xyxy"][prompt_idx]
            # out_obj_ids = out["pred_object_ids"][prompt_idx]
            out_masks = out["pred_masks"][prompt_idx]

        # only take the entries above the score threshold
        out_probs = out_scores.sigmoid()  # output in probabilities in 0~1
        # keep = out_obj_ids >= 0
        if output_prob_thresh is not None:
            # keep = torch.logical_and(out_probs > output_prob_thresh, keep)
            keep = out_probs > output_prob_thresh
        out_probs = out_probs[keep]
        out_boxes_xyxy = out_boxes_xyxy[keep]
        # out_obj_ids = out_obj_ids[keep]
        out_masks = out_masks[keep]

        out_boxes_xywh = box_xyxy_to_xywh(out_boxes_xyxy)  # output in XYWH box format
        num_out_obj = out_masks.size(0)
        if num_out_obj > 0:
            out_masks_orig_size = torch.nn.functional.interpolate(
                out_masks.unsqueeze(0),
                size=(inference_state["orig_height"], inference_state["orig_width"]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            out_binary_masks = out_masks_orig_size > 0.0
        else:
            # in case there is no object, `torch.nn.functional.interpolate` would raise
            # an error on an empty tensor, so we treat it specially here
            out_binary_masks = torch.zeros(
                0,
                inference_state["orig_height"],
                inference_state["orig_width"],
                dtype=torch.bool,
            )

        # We directly convert the outputs to CPU numpy format so that it is easy for
        # the server to send them across processes and construct the final response.
        frame_outputs = {
            "out_probs": out_probs.float().cpu().numpy(),
            "out_boxes_xywh": out_boxes_xywh.float().cpu().numpy(),
            # "out_obj_ids": out_obj_ids.cpu().numpy(),
            "out_binary_masks": out_binary_masks.cpu().numpy(),
        }

        return frame_outputs

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Revert `inference_state` to what it was right after initialization."""
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = 0
            # constructing an output list in inference state (we start with an empty list)
            inference_state["previous_stages_out"][t] = None
            inference_state["per_frame_raw_point_input"][t] = None
            inference_state["per_frame_raw_box_input"][t] = None
            inference_state["per_frame_visual_prompt"][t] = None
            inference_state["per_frame_geometric_prompt"][t] = None
            inference_state["per_frame_cur_step"][t] = 0

        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        if self.use_aot_mem:
            inference_state["aot_mem_per_frame"]["spatial"].clear()
            inference_state["aot_mem_per_frame"]["pointer"].clear()
        if self.use_obj_mem_bank:
            inference_state["obj_mem_per_frame"]["roi"].clear()
            inference_state["obj_mem_per_frame"]["roi_zoomed_out"].clear()
            inference_state["obj_mem_per_frame"]["global"].clear()
        gc.collect()

    def _compile_model(self):
        """Compile the SAM model with torch.compile for speedup."""
        is_compiled = getattr(self, "_model_is_compiled", False)
        if is_compiled or not self.compile_model:
            return

        import torch._dynamo

        # a larger cache size to hold varying number of shapes for torch.compile
        # see https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_dynamo/config.py#L42-L49
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True

        self.backbone.vision_backbone.forward = clone_output_wrapper(
            torch.compile(
                self.backbone.vision_backbone.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.transformer.encoder.forward = clone_output_wrapper(
            torch.compile(
                self.transformer.encoder.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.transformer.decoder.forward = clone_output_wrapper(
            torch.compile(
                self.transformer.decoder.forward,
                fullgraph=True,
                mode="max-autotune",
                dynamic=True,  # the decoder uses dynamic shapes
            )
        )
        self._model_is_compiled = True

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def warm_up_compilation(self):
        """
        Warm up the model by running a dummy inference to compile the model. This is
        useful to avoid the compilation overhead in the first inference call.
        """
        if not self.compile_model:
            return

        if self.device.type != "cuda":
            raise RuntimeError(f"The model must be on CUDA for warm-up compilation, got {self.device=}.")

        # Get a random video
        orig_tracking_score_thresh = self.tracking_score_thresh
        inference_state = self.init_state(resource_path="<load-dummy-video-30>")
        # use different tracking score thresholds for each round to simulate different number of output objects
        tracking_score_thresh_list = [0.0, -3.0, -3.5, -4.0]
        num_rounds = len(tracking_score_thresh_list)
        for i, thresh in enumerate(tracking_score_thresh_list):
            self.tracking_score_thresh = thresh
            # start at different locations for each round
            start_frame_idx = (inference_state["num_frames"] * i) // num_rounds
            logging.warning(f"{i + 1}/{num_rounds} warming up model compilation")
            self.add_prompt(inference_state, frame_idx=start_frame_idx, text_str="cat")
            logging.warning(
                f"{i + 1}/{num_rounds} warming up model compilation with propagation -- this might take a while"
            )
            for _ in self.propagate_in_video(inference_state, start_frame_idx, reverse=False):
                pass
            for _ in self.propagate_in_video(inference_state, start_frame_idx, reverse=True):
                pass
            self.reset_state(inference_state)
            logging.warning(
                f"{i + 1}/{num_rounds} warming up model compilation -- completed round {i + 1} out of {num_rounds}"
            )

        self.tracking_score_thresh = orig_tracking_score_thresh
        logging.warning("Warm-up compilation completed.")


def build_sam3_image_model(
    bpe_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
):
    """
    This function replaces the Hydra-based configuration in sam3_image_v1.4.yaml
    for image - only setting.

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary

    Returns:
        A SAM3 image model for interactive segmentation
    """
    # Create position encoding for visual backbone
    position_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000)

    # Create ViT backbone
    vit_backbone = ViT(
        img_size=1008,
        # weights_path=None,
        # freezing="NoFreeze",
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        # use_act_checkpoint=True,
        compile_mode="default",
    )

    # Create ViT neck
    vit_neck = OriginalViTDetNeck(
        position_encoding=position_encoding,
        # neck_norm=None,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
    )

    # Create text tokenizer
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)

    # Create text encoder
    text_encoder = VETextEncoder(
        # frozen=False,
        tokenizer=tokenizer,
        d_model=256,
        # weights_path=None,
        width=1024,
        heads=16,
        layers=24,
        # use_act_checkpoint=True
    )

    # Create visual-language backbone
    backbone = NonFusionVLBackbone(visual=vit_neck, text=text_encoder, scalp=1)

    # Create transformer encoder layer
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            # attn_type=AttentionType.Vanilla,
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            # attn_type=AttentionType.Vanilla,
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    # Create transformer encoder
    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )

    # Create transformer decoder layer
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,  #  attn_type=AttentionType.Vanilla,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    # Create transformer decoder
    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        use_act_checkpoint=True,
        instance_query=True,
        num_instances=4,
    )

    # Create transformer
    transformer = TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)

    # Create MLP for dot product scorer
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )

    # Create dot product scorer
    dot_prod_scoring = DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)

    # Create pixel decoder for segmentation head
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode="default",
    )

    # Create cross attention for segmentation head
    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,  # attn_type=AttentionType.Vanilla,
    )

    # Create segmentation head
    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=True,
        dot_product_scorer=DotProductScoring(
            d_model=256,
            d_proj=256,
            prompt_mlp=MLP(
                input_dim=256,
                hidden_dim=2048,
                output_dim=256,
                num_layers=2,
                dropout=0.1,
                residual=True,
                out_norm=nn.LayerNorm(256),
            ),
        ),
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )

    # Create position encoding for geometry encoder
    geo_pos_enc = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000)

    # Create mask downsampler
    mask_downsampler = SimpleMaskDownSampler(
        interpol_size=[288, 288], kernel_size=3, stride=2, padding=1, total_stride=4
    )

    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    # Create fuser
    fuser = SimpleFuser(layer=cx_block, num_layers=2)

    # Create mask encoder
    mask_encoder = FusedMaskEncoder(
        out_dim=256,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=1008,
        ),
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            # attn_type=AttentionType.Vanilla,
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            # attn_type=AttentionType.Vanilla,
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
        mask_encoder=mask_encoder,
    )

    # Create the SAM3 model
    model = Sam3ImageInteractiveDemo(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=dot_prod_scoring,
        use_instance_query=True,
        multimask_output=True,
    )

    # move to eval mode
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt, strict=True)

    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()

    return model


### End Sam3Image ---------------------------------------------------------------------------- #
