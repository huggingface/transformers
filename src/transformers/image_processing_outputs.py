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

from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING

from .feature_extraction_utils import BatchFeature


if TYPE_CHECKING:
    import torch


class PostProcessorOutput(BatchFeature):
    """Base class for image processor post-processing outputs.

    Behaves like a dict with additional attribute access. Fields are accessible via
    dict-style access (``output["segmentation"]``) and attribute access (``output.segmentation``).
    Device/dtype casting is supported via ``output.to(...)`` and tensor conversion via
    ``output.convert_to_tensors("np")``. Dtype casting is only supported for float tensors.

    Subclasses must use the ``@dataclass`` decorator and define typed fields for each output.

    Example::

        @dataclass
        class SemanticSegmentationPostProcessorOutput(PostProcessorOutput):
            segmentation: torch.Tensor
            segmentation_scores: torch.Tensor
    """

    skip_tensor_conversion: "list[str] | set[str] | None"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses must use the @dataclass decorator.
        # Note that dataclass subclasses generate their own __init__ and bypass this one.
        is_post_process_output_subclass = self.__class__ != PostProcessorOutput

        if is_post_process_output_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                f" Subclasses of {PostProcessorOutput.__name__} must use the @dataclass decorator."
            )

    def __post_init__(self):
        # Dataclasses generate their own __init__ which bypasses BatchFeature.__init__.
        # We set any potentially missing attributes here to match BatchFeature.__init__.
        if "data" not in self.__dict__:
            # "data" is already populated through dataclass __init__.
            # Add it here in case the dataclass doesn't have any fields.
            super().__setattr__("data", {})
        if "skip_tensor_conversion" not in self.__dict__:
            super().__setattr__("skip_tensor_conversion", None)

    def __setattr__(self, name: str, value) -> None:
        # Dataclass field values live in BatchFeature.data
        if is_dataclass(self) and name in {field.name for field in fields(self)}:
            if "data" not in self.__dict__:
                super().__setattr__("data", {})
            self.data[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        # Dataclass field values live in BatchFeature.data
        if is_dataclass(self) and "data" in self.__dict__ and name in self.data:
            del self.data[name]
        else:
            super().__delattr__(name)


@dataclass
class SemanticSegmentationPostProcessorOutput(PostProcessorOutput):
    """
    Output of a semantic segmentation post-processing step.

    Args:
        segmentation (`torch.LongTensor` of shape `(height, width)`):
            Predicted class label for each pixel. Height and width match ``target_sizes`` when provided,
            otherwise the model's native logit size.
        segmentation_scores (`torch.FloatTensor` of shape `(num_classes, height, width)`):
            Raw classification scores for each class at every pixel position.
            Height and width are the same as for ``segmentation``. May be `None` in cases where per-class
            scores are not available.
    """

    segmentation: "torch.Tensor"
    segmentation_scores: "torch.Tensor | None"
