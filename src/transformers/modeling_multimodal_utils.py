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
from typing import Any

import torch

from .utils import is_torchdynamo_compiling, logging
from .utils.generic import can_return_tuple


logger = logging.get_logger(__name__)


def placeholder_features(outputs):
    """The features of encoder `outputs` that fill the placeholder tokens: `pooler_output`, or the first deepstack
    layer's for models only returning deepstack features (Granite 4 Vision)."""
    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    deepstack_features = getattr(outputs, "deepstack_features", None)
    return deepstack_features[0] if deepstack_features else None


def map_features(outputs, function):
    """Applies `function` to the pooled and deepstack features of encoder `outputs`, in place."""
    if outputs.pooler_output is not None:
        outputs.pooler_output = function(outputs.pooler_output)
    if getattr(outputs, "deepstack_features", None) is not None:
        outputs.deepstack_features = [function(features) for features in outputs.deepstack_features]
    return outputs


def split_modality_outputs(outputs):
    """Splits flat `get_*_embeddings` outputs per item by their `split_sizes`."""
    split_sizes = outputs.split_sizes.tolist()
    return map_features(outputs, lambda features: torch.split(features, split_sizes))


class MultimodalMixin:
    """
    `get_image_features`, `get_video_features` and `get_audio_features` from a model's flat `get_image_embeddings`,
    `get_video_embeddings` and `get_audio_embeddings`, split per item by the `split_sizes` they return. A model with a
    head reaches its base model's getters and `get_data_dependent_inputs`.
    """

    model: "MultimodalMixin"
    base_model: "MultimodalMixin"

    def _base_model_getter(self, name: str):
        # Only a head has a base model to reach: a base model without its own getter takes no such input
        if getattr(self, "model", None) is None:
            raise AttributeError(f"{type(self).__name__} has no `{name}`, as it takes no such input")
        return getattr(self.model, name)

    def get_image_embeddings(self, *args, **kwargs):
        return self._base_model_getter("get_image_embeddings")(*args, **kwargs)

    def get_video_embeddings(self, *args, **kwargs):
        return self._base_model_getter("get_video_embeddings")(*args, **kwargs)

    def get_audio_embeddings(self, *args, **kwargs):
        return self._base_model_getter("get_audio_embeddings")(*args, **kwargs)

    @can_return_tuple
    def get_image_features(self, *args, **kwargs):
        """Each image's features, from `get_image_embeddings`."""
        return split_modality_outputs(self.get_image_embeddings(*args, return_dict=True, **kwargs))

    @can_return_tuple
    def get_video_features(self, *args, **kwargs):
        """Each video's features, from `get_video_embeddings`."""
        return split_modality_outputs(self.get_video_embeddings(*args, return_dict=True, **kwargs))

    @can_return_tuple
    def get_audio_features(self, *args, **kwargs):
        """Each audio's features, from `get_audio_embeddings`."""
        return split_modality_outputs(self.get_audio_embeddings(*args, return_dict=True, **kwargs))

    def get_data_dependent_inputs(self, **kwargs) -> dict[str, Any]:
        """The inputs the forward computes with data-dependent ops (host syncs, loops over images). Passed back to it,
        they keep it traceable by `torch.compile(fullgraph=True)` and `torch.export`. None unless the model computes
        some."""
        base_model = self.base_model
        if base_model is self or not hasattr(base_model, "get_data_dependent_inputs"):
            return {}
        return base_model.get_data_dependent_inputs(**kwargs)


def merge_modality_outputs(outputs):
    """Concatenates per-item outputs back into the flat layout of `get_*_embeddings`."""
    if not isinstance(placeholder_features(outputs), (tuple, list)):
        return outputs
    return map_features(copy.copy(outputs), torch.cat)


def merge_mm_encoder_outputs(forward):
    """Lets `forward` take per-item `mm_encoder_outputs`, as returned by `get_*_features`, by merging them into the
    flat layout it consumes. Only for backward compatibility with users passing them, and deprecated."""

    @functools.wraps(forward)
    def wrapper(self, *args, mm_encoder_outputs=None, **kwargs):
        # Idefics' deprecated `mm_encoder_outputs` can be a bare tensor: only a dict of outputs is merged
        if isinstance(mm_encoder_outputs, dict) and mm_encoder_outputs:
            merged = {modality: merge_modality_outputs(outputs) for modality, outputs in mm_encoder_outputs.items()}
            per_item = [
                modality for modality, outputs in mm_encoder_outputs.items() if merged[modality] is not outputs
            ]
            if per_item and not is_torchdynamo_compiling():
                modality = per_item[0]
                logger.warning_once(
                    f"Passing per-item `mm_encoder_outputs['{modality}']` (from `get_{modality}_features`) is deprecated "
                    f"and will be removed in v5.29: pass the flat outputs of `get_{modality}_embeddings`."
                )
            mm_encoder_outputs = merged
        return forward(self, *args, mm_encoder_outputs=mm_encoder_outputs, **kwargs)

    return wrapper
