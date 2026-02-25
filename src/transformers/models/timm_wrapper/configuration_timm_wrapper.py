# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Configuration for TimmWrapper models"""

from typing import Any

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import is_timm_available, logging


if is_timm_available():
    pass


logger = logging.get_logger(__name__)


class TimmWrapperConfig(PreTrainedConfig, BackboneConfigMixin):
    r"""
    This is the configuration class to store the configuration for a timm backbone [`TimmWrapper`].

    It is used to instantiate a timm model according to the specified arguments, defining the model.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Config loads imagenet label descriptions and stores them in `id2label` attribute, `label2id` attribute for default
    imagenet models is set to `None` due to occlusions in the label descriptions.

    Args:
        architecture (`str`, *optional*, defaults to `"resnet50"`):
            The timm architecture to load.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        do_pooling (`bool`, *optional*, defaults to `True`):
            Whether to do pooling for the last_hidden_state in `TimmWrapperModel` or not.
        freeze_batch_norm_2d (`bool`, *optional*, defaults to `False`):
            Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`.
        model_args (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `timm.create_model` function. e.g. `model_args={"depth": 3}`
            for `timm/vit_base_patch32_clip_448.laion2b_ft_in12k_in1k` to create a model with 3 blocks. Defaults to `None`.

    Example:
    ```python
    >>> from transformers import TimmWrapperModel

    >>> # Initializing a timm model
    >>> model = TimmWrapperModel.from_pretrained("timm/resnet18.a1_in1k")

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "timm_wrapper"

    def __init__(
        self,
        architecture: str = "resnet50",
        initializer_range: float = 0.02,
        do_pooling: bool = True,
        freeze_batch_norm_2d=False,
        model_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        is_backbone_config = kwargs.get("backbone") is not None
        self.architecture = kwargs.pop("backbone") if is_backbone_config else architecture
        self.initializer_range = initializer_range
        self.do_pooling = do_pooling
        self.freeze_batch_norm_2d = freeze_batch_norm_2d
        if model_args is None and is_backbone_config:
            model_args = {
                "features_only": kwargs.pop("features_only", True),
                "in_chans": kwargs.pop("num_channels", 3),
                "output_stride": kwargs.get("output_stride"),
            }

        self.model_args = model_args  # named "model_args" for BC with timm
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        # Create a copy to avoid mutating the original dict
        config_dict = config_dict.copy()

        label_names = config_dict.get("label_names")
        is_custom_model = "num_labels" in kwargs or "id2label" in kwargs
        if label_names is not None and not is_custom_model:
            kwargs["id2label"] = dict(enumerate(label_names))

            # if all label names are unique, create label2id mapping as well
            if len(set(label_names)) == len(label_names):
                kwargs["label2id"] = {name: i for i, name in enumerate(label_names)}
            else:
                kwargs["label2id"] = None

        # timm config stores the `num_classes` attribute in both the root of config and in the "pretrained_cfg" dict.
        # We are removing these attributes in order to have the native `transformers` num_labels attribute in config
        # and to avoid duplicate attributes. Noe that `num_labels` has priority over `num_classes` in config_dict
        if config_dict.get("num_classes") is not None and kwargs.get("num_labels") is None:
            kwargs["num_labels"] = config_dict.pop("num_classes", None)

            # Pop in nested `pretrained_cfg` as well
            if "pretrained_cfg" in config_dict and "num_classes" in config_dict["pretrained_cfg"]:
                config_dict["pretrained_cfg"].pop("num_classes", None)

        return super().from_dict(config_dict, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output.setdefault("num_classes", self.num_labels)
        output.setdefault("label_names", list(self.id2label.values()))
        output.pop("id2label", None)
        output.pop("label2id", None)
        return output

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
            self.set_output_features_output_indices(out_features=None, out_indices=out_indices)

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
            self.set_output_features_output_indices(out_features=out_features, out_indices=None)


__all__ = ["TimmWrapperConfig"]
