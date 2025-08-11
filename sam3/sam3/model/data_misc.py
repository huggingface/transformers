# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.
"""

import collections

from dataclasses import dataclass, field as field_ptr_behaviour, fields, is_dataclass
from typing import Any, get_args, get_origin, List, Mapping, Optional, Sequence, Union

import torch

from .model_misc import NestedTensor

MyTensor = Union[torch.Tensor, List[Any]]


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
    object_ids: Optional[List[List]] = None  # List of objects per query

    input_boxes_before_embed: Optional[MyTensor] = None
    input_boxes_before_embed__type = torch.float

    input_points_before_embed: Optional[MyTensor] = None
    input_points_before_embed__type = torch.float


@dataclass
class GetStage:
    text_inputs: List[str]
    text_output: List[str]
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
    get_text_input: List[Optional[str]]

    # Adding for TA conditional inference
    is_conditioning_only: List[Optional[bool]]


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


@dataclass
class BatchedDatapoint:
    img_batch: NestedTensor
    find_text_batch: List[str]
    find_inputs: List[FindStage]
    find_targets: List[BatchedFindTarget]
    find_metadatas: List[BatchedInferenceMetadata]
    get_queries: GetStage
    ptr_behaviour: PointerExtractBehaviour = field_ptr_behaviour(
        default_factory=lambda: PointerExtractBehaviour()
    )
    raw_images: Optional[List[Any]] = None

    def pin_memory(self, device=None):
        return recursive_pin_memory(self, device)


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

        elif len(getattr(obj, field.name)) and isinstance(
            getattr(obj, field.name)[0], torch.Tensor
        ):
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
                torch.stack(getattr(obj, field.name), dim=stack_dim).to(
                    getattr(obj, field.name + "__type")
                ),
            )
        else:
            setattr(
                obj,
                field.name,
                torch.as_tensor(
                    getattr(obj, field.name), dtype=getattr(obj, field.name + "__type")
                ),
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
        ret_fields = {
            field.name: recursive_to(getattr(data, field.name), *args, **kwargs)
            for field in fields(data)
        }
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
        pinned_data = {
            k: recursive_pin_memory(sample, device) for k, sample in data.items()
        }
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
        pinned_data = {
            field.name: recursive_pin_memory(getattr(data, field.name), device)
            for field in fields(data)
        }
        return type(data)(**pinned_data)
    elif hasattr(data, "pin_memory"):
        return data.pin_memory(device)

    return data
