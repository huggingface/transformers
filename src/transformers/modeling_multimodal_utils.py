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

import functools

import torch

from .utils.generic import can_return_tuple


class MultimodalMixin:
    """
    `get_image_features` and `get_video_features` from a model's flat `get_image_embeddings` and `get_video_embeddings`,
    split per image or video by the `split_sizes` they return. A model with a head reaches its base model's.
    """

    model: "MultimodalMixin"
    base_model: "MultimodalMixin"

    def get_image_embeddings(self, *args, **kwargs):
        return self.model.get_image_embeddings(*args, **kwargs)

    def get_video_embeddings(self, *args, **kwargs):
        return self.model.get_video_embeddings(*args, **kwargs)

    @staticmethod
    def split_modality_outputs(outputs):
        """Splits flat `get_*_embeddings` outputs per item by their `split_sizes`."""
        split_sizes = outputs.split_sizes.tolist()
        outputs.pooler_output = torch.split(outputs.pooler_output, split_sizes)
        if getattr(outputs, "deepstack_features", None) is not None:
            outputs.deepstack_features = [
                torch.split(features, split_sizes) for features in outputs.deepstack_features
            ]
        return outputs

    @staticmethod
    def merge_modality_outputs(outputs):
        """Concatenates per-item outputs back into the flat layout of `get_*_embeddings`."""
        if not isinstance(outputs.pooler_output, (tuple, list)):
            return outputs
        merged = {"pooler_output": torch.cat(outputs.pooler_output)}
        if getattr(outputs, "deepstack_features", None) is not None:
            merged["deepstack_features"] = [torch.cat(features) for features in outputs.deepstack_features]
        return type(outputs)(**{**outputs, **merged})

    @can_return_tuple
    def get_image_features(self, *args, **kwargs):
        """Each image's features, from `get_image_embeddings`."""
        outputs = self.get_image_embeddings(*args, return_dict=True, **kwargs)
        return getattr(self.base_model, "split_modality_outputs", self.split_modality_outputs)(outputs)

    @can_return_tuple
    def get_video_features(self, *args, **kwargs):
        """Each video's features, from `get_video_embeddings`."""
        outputs = self.get_video_embeddings(*args, return_dict=True, **kwargs)
        return getattr(self.base_model, "split_modality_outputs", self.split_modality_outputs)(outputs)


def merge_mm_encoder_outputs(forward):
    """Lets `forward` take per-item `mm_encoder_outputs` (passed by users, or split by beam search) by merging them
    into the flat layout it consumes."""

    @functools.wraps(forward)
    def wrapper(self, *args, mm_encoder_outputs=None, **kwargs):
        if mm_encoder_outputs:
            merge = getattr(self, "merge_modality_outputs", MultimodalMixin.merge_modality_outputs)
            mm_encoder_outputs = {modality: merge(outputs) for modality, outputs in mm_encoder_outputs.items()}
        return forward(self, *args, mm_encoder_outputs=mm_encoder_outputs, **kwargs)

    return wrapper
