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
"""What models taking images, videos or audio share: their per-item getters, built on the flat ones (`MultimodalMixin`),
and the conversion of `mm_encoder_outputs` between the two layouts."""

import copy
import functools

import torch

from .utils import is_torchdynamo_compiling, logging
from .utils.generic import can_return_tuple


logger = logging.get_logger(__name__)


def flatten_items(features: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
    """The `(num_items, num_tokens, hidden_size)` features of fixed-size images or videos as one sequence, and the
    `split_sizes` of each item."""
    return features.flatten(0, 1), features.new_full(features.shape[:1], features.shape[1], dtype=torch.long)


def placeholder_features(outputs):
    """The features of encoder `outputs` that fill the placeholder tokens: `pooler_output`, or the first deepstack
    layer's for models only returning deepstack features (Granite 4 Vision)."""
    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    deepstack_features = getattr(outputs, "deepstack_features", None)
    return deepstack_features[0] if deepstack_features else None


def split_modality_outputs(outputs):
    """Splits flat `get_*_embeddings` outputs per item by their `split_sizes`."""
    split_sizes = outputs.split_sizes.tolist()
    # Some models (e.g. Granite 4 Vision) only return deepstack features, without a pooled output
    if outputs.pooler_output is not None:
        outputs.pooler_output = torch.split(outputs.pooler_output, split_sizes)
    if getattr(outputs, "deepstack_features", None) is not None:
        outputs.deepstack_features = [torch.split(features, split_sizes) for features in outputs.deepstack_features]
    return outputs


def _features_getter(modality: str):
    """`get_<modality>_features`: `get_<modality>_embeddings` split per item."""

    @can_return_tuple
    def get_features(self, *args, **kwargs):
        return split_modality_outputs(getattr(self, f"get_{modality}_embeddings")(*args, return_dict=True, **kwargs))

    get_features.__name__ = f"get_{modality}_features"
    get_features.__qualname__ = f"MultimodalMixin.{get_features.__name__}"
    get_features.__doc__ = f"Each {modality}'s features, from `get_{modality}_embeddings`."
    return get_features


def _embeddings_getter(modality: str):
    """`get_<modality>_embeddings` of a head: its base model's."""
    name = f"get_{modality}_embeddings"

    def get_embeddings(self, *args, **kwargs):
        # Only a head has a base model to reach
        if getattr(self, "model", None) is None:
            raise AttributeError(f"{type(self).__name__} has no `{name}`")
        return getattr(self.model, name)(*args, **kwargs)

    get_embeddings.__name__ = name
    get_embeddings.__qualname__ = f"MultimodalMixin.{name}"
    get_embeddings.__doc__ = f"The base model's `{name}`."
    return get_embeddings


class _ModalityGetter:
    """A getter of `MultimodalMixin` that only exists on models taking its modality as input, so `hasattr(model, ...)`
    tells which inputs a model takes."""

    def __init__(self, modality: str, getter):
        self.modality = modality
        self.getter = getter

    def __get__(self, instance, owner):
        # On the class it always resolves, as `dir()` lists it: tools walking a class's attributes (the docs builder)
        # call `getattr` on each name
        if instance is None:
            return self.getter
        input_modalities = owner.input_modalities
        input_modalities = (input_modalities,) if isinstance(input_modalities, str) else input_modalities
        # A model defining its own per-item getter (GLM-Image's feature maps) has no flat level for the mixin to build on
        features_owner = next(cls for cls in owner.__mro__ if f"get_{self.modality}_features" in cls.__dict__)
        if self.modality not in input_modalities or features_owner is not MultimodalMixin:
            raise AttributeError(f"{owner.__name__} has no `{self.getter.__name__}`")
        return self.getter.__get__(instance, owner)


class MultimodalMixin:
    """
    `get_image_features`, `get_video_features` and `get_audio_features` from a model's flat `get_image_embeddings`,
    `get_video_embeddings` and `get_audio_embeddings`, split per item by the `split_sizes` they return. A model with a
    head reaches its base model's. Only models taking the modality as input have them, so `hasattr(model, ...)` tells
    which inputs a model takes.
    """

    model: "MultimodalMixin"

    get_image_embeddings = _ModalityGetter("image", _embeddings_getter("image"))
    get_video_embeddings = _ModalityGetter("video", _embeddings_getter("video"))
    get_audio_embeddings = _ModalityGetter("audio", _embeddings_getter("audio"))
    get_image_features = _ModalityGetter("image", _features_getter("image"))
    get_video_features = _ModalityGetter("video", _features_getter("video"))
    get_audio_features = _ModalityGetter("audio", _features_getter("audio"))


def merge_modality_outputs(outputs):
    """Concatenates per-item outputs back into the flat layout of `get_*_embeddings`."""
    if not isinstance(placeholder_features(outputs), (tuple, list)):
        return outputs
    deepstack_features = getattr(outputs, "deepstack_features", None)
    merged = copy.copy(outputs)
    if outputs.pooler_output is not None:
        merged.pooler_output = torch.cat(outputs.pooler_output)
    if deepstack_features is not None:
        merged.deepstack_features = [torch.cat(features) for features in deepstack_features]
    return merged


def merge_mm_encoder_outputs(forward):
    """Lets `forward` take per-item `mm_encoder_outputs`, as returned by `get_*_features`, by merging them into the
    flat layout it consumes. Only for backward compatibility with users passing them, and deprecated."""

    @functools.wraps(forward)
    def wrapper(self, *args, mm_encoder_outputs=None, **kwargs):
        # Idefics' deprecated `mm_encoder_outputs` can be a bare tensor: only a dict of outputs is merged
        if isinstance(mm_encoder_outputs, dict) and mm_encoder_outputs:
            merged = {modality: merge_modality_outputs(outputs) for modality, outputs in mm_encoder_outputs.items()}
            is_per_item = any(merged[modality] is not outputs for modality, outputs in mm_encoder_outputs.items())
            if is_per_item and not is_torchdynamo_compiling():
                logger.warning_once(
                    "Passing per-item `mm_encoder_outputs` (from `get_*_features`) is deprecated and will be removed in "
                    "v5.29: pass the flat outputs of `get_*_embeddings` (`get_image_embeddings`, ...)."
                )
            mm_encoder_outputs = merged
        return forward(self, *args, mm_encoder_outputs=mm_encoder_outputs, **kwargs)

    return wrapper
