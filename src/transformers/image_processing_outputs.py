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

from collections import UserDict
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch


class PostProcessOutput(UserDict):
    """Base class for image processor post-processing outputs.

    Behaves like a dict, so fields are accessible via both attribute access (``output.segmentation``)
    and dict-style access (``output["segmentation"]``).

    Subclasses must use the ``@dataclass`` decorator and define typed fields for each output.

    Example::

        @dataclass
        class SemanticSegmentationPostProcessOutput(PostProcessOutput):
            segmentation: torch.Tensor
            segmentation_scores: torch.Tensor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of PostProcessOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(PostProcessOutput, PostProcessOutput) when False is needed
        # Just need to check that the current class is not PostProcessOutput
        is_post_process_output_subclass = self.__class__ != PostProcessOutput

        if is_post_process_output_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                f" Subclasses of {PostProcessOutput.__name__} must use the @dataclass decorator."
            )

    def __post_init__(self):
        # @dataclass generates its own __init__ which replaces UserDict.__init__, so self.data
        # is never initialized. __post_init__ runs at the end of the generated __init__ after
        # all fields are assigned, making it the right place to populate self.data so that
        # dict-style access (obj["key"], iteration, len, etc.) works correctly.
        self.data = {field.name: getattr(self, field.name) for field in fields(self)}

    def __setattr__(self, name: str, value) -> None:
        if is_dataclass(self):
            if name == "data" and "data" in self.__dict__:
                raise AttributeError(
                    f"Cannot set 'data' directly on {self.__class__.__name__}. Update individual fields instead."
                )
            # Keep self.data in sync when a field is updated after construction (e.g. obj.scores = x).
            elif "data" in self.__dict__ and name in {field.name for field in fields(self)}:
                super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        if is_dataclass(self) and key in {field.name for field in fields(self)}:
            object.__delattr__(self, key)

    def __delattr__(self, name: str) -> None:
        if is_dataclass(self) and name == "data":
            raise AttributeError(f"Cannot delete 'data' on {self.__class__.__name__}.")
        object.__delattr__(self, name)
        if "data" in self.__dict__ and name in self.data:
            del self.data[name]


@dataclass
class SemanticSegmentationPostProcessOutput(PostProcessOutput):
    """
    Output of a semantic segmentation post-processing step.

    Args:
        segmentation (`torch.LongTensor` of shape `(height, width)`):
            Predicted class label for each pixel. Height and width match ``target_sizes`` when provided,
            otherwise the model's native logit size.
        segmentation_scores (`torch.FloatTensor` of shape `(num_classes, height, width)`):
            Raw classification scores for each class at every pixel position.
            Height and width are the same as for ``segmentation``.
    """

    segmentation: "torch.Tensor"
    segmentation_scores: "torch.Tensor"
