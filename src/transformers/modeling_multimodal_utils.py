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
"""Shared getters of models that take images or videos."""

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


def split_modality_outputs(outputs):
    """Splits flat `get_*_embeddings` outputs per item by their `split_sizes`."""
    split_sizes = outputs.split_sizes.tolist()
    # Some models (e.g. Granite 4 Vision) only return deepstack features, without a pooled output
    if outputs.pooler_output is not None:
        outputs.pooler_output = torch.split(outputs.pooler_output, split_sizes)
    if getattr(outputs, "deepstack_features", None) is not None:
        outputs.deepstack_features = [torch.split(features, split_sizes) for features in outputs.deepstack_features]
    return outputs


def merge_modality_outputs(outputs):
    """Concatenates per-item outputs back into the flat layout of `get_*_embeddings`."""
    deepstack_features = getattr(outputs, "deepstack_features", None)
    features = outputs.pooler_output if outputs.pooler_output is not None else (deepstack_features or [None])[0]
    if not isinstance(features, (tuple, list)):
        return outputs
    merged = copy.copy(outputs)
    if outputs.pooler_output is not None:
        merged.pooler_output = torch.cat(outputs.pooler_output)
    if deepstack_features is not None:
        merged.deepstack_features = [torch.cat(features) for features in deepstack_features]
    return merged


class MultimodalMixin:
    """
    `get_image_features`, `get_video_features` and `get_audio_features` from a model's flat `get_image_embeddings`,
    `get_video_embeddings` and `get_audio_embeddings`, split per item by the `split_sizes` they return. A model with a
    head reaches its base model's.
    """

    model: "MultimodalMixin"

    def get_image_embeddings(self, *args, **kwargs):
        return self.model.get_image_embeddings(*args, **kwargs)

    def get_video_embeddings(self, *args, **kwargs):
        return self.model.get_video_embeddings(*args, **kwargs)

    def get_audio_embeddings(self, *args, **kwargs):
        return self.model.get_audio_embeddings(*args, **kwargs)

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
                    "Passing per-item `mm_encoder_outputs` (from `get_image_features` / `get_video_features`) is "
                    "deprecated and will be removed: pass the flat outputs of `get_image_embeddings` / "
                    "`get_video_embeddings`."
                )
            mm_encoder_outputs = merged
        return forward(self, *args, mm_encoder_outputs=mm_encoder_outputs, **kwargs)

    return wrapper
