# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

"""Configuration for Backbone models"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_backbone_utils import BackboneConfigMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class TimmBackboneConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration for a timm backbone [`TimmBackbone`].

    It is used to instantiate a timm backbone model according to the specified arguments, defining the model.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        backbone (`str`, *optional*):
            The timm checkpoint to load.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        features_only (`bool`, *optional*, defaults to `True`):
            Whether to output only the features or also the logits.
        out_indices (`list[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). Will default to the last stage if unset.
        freeze_batch_norm_2d (`bool`, *optional*, defaults to `False`):
            Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`.

    Example:
    ```python
    >>> from transformers import TimmBackboneConfig, TimmBackbone

    >>> # Initializing a timm backbone
    >>> configuration = TimmBackboneConfig("resnet50")

    >>> # Initializing a model from the configuration
    >>> model = TimmBackbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "timm_backbone"

    def __init__(
        self,
        backbone=None,
        num_channels=3,
        features_only=True,
        out_indices=None,
        freeze_batch_norm_2d=False,
        output_stride=None,
        **kwargs,
    ):
        self.backbone = backbone
        self.num_channels = num_channels
        self.features_only = features_only
        self.out_indices = out_indices if out_indices is not None else [-1]
        self.output_stride = output_stride
        self.freeze_batch_norm_2d = freeze_batch_norm_2d

        # self._out_features = kwargs.pop("out_features", None)
        super().__init__(**kwargs)

    @property
    def out_indices(self):
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: tuple[int, ...] | list[int]):
        """
        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.
        """
        self._out_indices = list(out_indices) if out_indices is not None else out_indices
        if getattr(self, "stage_names", None) is not None:
            self._out_features = None
            self.align_output_features_output_indices()

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: list[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        self._out_features = out_features
        if getattr(self, "stage_names", None) is not None:
            self._out_indices = None
            self.align_output_features_output_indices()


__all__ = ["TimmBackboneConfig"]
