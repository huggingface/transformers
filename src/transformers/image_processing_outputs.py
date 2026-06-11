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
from typing import TYPE_CHECKING

from .image_processing_base import BatchFeature


if TYPE_CHECKING:
    import torch


class SemanticSegmentationPostProcessorOutput(BatchFeature):
    """
    Output of a semantic segmentation post-processing step.

    This class is derived from a python dictionary and can be used as a dictionary.

    Attributes:
        segmentation (`torch.LongTensor` of shape `(height, width)`):
            Predicted class label for each pixel. Height and width match ``target_sizes`` when provided,
            otherwise the model's native logit size.
        segmentation_scores (`torch.FloatTensor` of shape `(num_labels, height, width)`):
            Raw classification scores for each class at every pixel position.
            Height and width are the same as for ``segmentation``. May be `None` in cases where per-class
            scores are not available.
    """

    segmentation: "torch.Tensor"
    segmentation_scores: "torch.Tensor | None"
