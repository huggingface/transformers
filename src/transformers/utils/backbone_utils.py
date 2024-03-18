# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

""" Collection of utils to be used by backbones and their components."""

import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union


class BackboneType(enum.Enum):
    TIMM = "timm"
    TRANSFORMERS = "transformers"


def verify_out_features_out_indices(
    out_features: Optional[Iterable[str]], out_indices: Optional[Iterable[int]], stage_names: Optional[Iterable[str]]
):
    """
    Verify that out_indices and out_features are valid for the given stage_names.
    """
    if stage_names is None:
        raise ValueError("Stage_names must be set for transformers backbones")

    if out_features is not None:
        if not isinstance(out_features, (list,)):
            raise ValueError(f"out_features must be a list got {type(out_features)}")
        if any(feat not in stage_names for feat in out_features):
            raise ValueError(f"out_features must be a subset of stage_names: {stage_names} got {out_features}")
        if len(out_features) != len(set(out_features)):
            raise ValueError(f"out_features must not contain any duplicates, got {out_features}")
        if out_features != (sorted_feats := [feat for feat in stage_names if feat in out_features]):
            raise ValueError(
                f"out_features must be in the same order as stage_names, expected {sorted_feats} got {out_features}"
            )

    if out_indices is not None:
        if not isinstance(out_indices, (list, tuple)):
            raise ValueError(f"out_indices must be a list or tuple, got {type(out_indices)}")
        # Convert negative indices to their positive equivalent: [-1,] -> [len(stage_names) - 1,]
        positive_indices = tuple(idx % len(stage_names) if idx < 0 else idx for idx in out_indices)
        if any(idx for idx in positive_indices if idx not in range(len(stage_names))):
            raise ValueError(f"out_indices must be valid indices for stage_names {stage_names}, got {out_indices}")
        if len(positive_indices) != len(set(positive_indices)):
            msg = f"out_indices must not contain any duplicates, got {out_indices}"
            msg += f"(equivalent to {positive_indices}))" if positive_indices != out_indices else ""
            raise ValueError(msg)
        if positive_indices != tuple(sorted(positive_indices)):
            sorted_negative = tuple(idx for _, idx in sorted(zip(positive_indices, out_indices), key=lambda x: x[0]))
            raise ValueError(
                f"out_indices must be in the same order as stage_names, expected {sorted_negative} got {out_indices}"
            )

    if out_features is not None and out_indices is not None:
        if len(out_features) != len(out_indices):
            raise ValueError("out_features and out_indices should have the same length if both are set")
        if out_features != [stage_names[idx] for idx in out_indices]:
            raise ValueError("out_features and out_indices should correspond to the same stages if both are set")


def _align_output_features_output_indices(
    out_features: Optional[List[str]],
    out_indices: Optional[Union[List[int], Tuple[int]]],
    stage_names: List[str],
):
    """
    Finds the corresponding `out_features` and `out_indices` for the given `stage_names`.

    The logic is as follows:
        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the
        `out_indices`.
        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the
        `out_features`.
        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.
        - `out_indices` and `out_features` set: input `out_indices` and `out_features` are returned.

    Args:
        out_features (`List[str]`): The names of the features for the backbone to output.
        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.
        stage_names (`List[str]`): The names of the stages of the backbone.
    """
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) - 1]
        out_features = [stage_names[-1]]
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]
    return out_features, out_indices


def get_aligned_output_features_output_indices(
    out_features: Optional[List[str]],
    out_indices: Optional[Union[List[int], Tuple[int]]],
    stage_names: List[str],
) -> Tuple[List[str], List[int]]:
    """
    Get the `out_features` and `out_indices` so that they are aligned.

    The logic is as follows:
        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the
        `out_indices`.
        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the
        `out_features`.
        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.
        - `out_indices` and `out_features` set: they are verified to be aligned.

    Args:
        out_features (`List[str]`): The names of the features for the backbone to output.
        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.
        stage_names (`List[str]`): The names of the stages of the backbone.
    """
    # First verify that the out_features and out_indices are valid
    verify_out_features_out_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
    output_features, output_indices = _align_output_features_output_indices(
        out_features=out_features, out_indices=out_indices, stage_names=stage_names
    )
    # Verify that the aligned out_features and out_indices are valid
    verify_out_features_out_indices(out_features=output_features, out_indices=output_indices, stage_names=stage_names)
    return output_features, output_indices


class BackboneMixin:
    backbone_type: Optional[BackboneType] = None

    def _init_timm_backbone(self, config) -> None:
        """
        Initialize the backbone model from timm The backbone must already be loaded to self._backbone
        """
        if getattr(self, "_backbone", None) is None:
            raise ValueError("self._backbone must be set before calling _init_timm_backbone")

        # These will diagree with the defaults for the transformers models e.g. for resnet50
        # the transformer model has out_features = ['stem', 'stage1', 'stage2', 'stage3', 'stage4']
        # the timm model has out_features = ['act', 'layer1', 'layer2', 'layer3', 'layer4']
        self.stage_names = [stage["module"] for stage in self._backbone.feature_info.info]
        self.num_features = [stage["num_chs"] for stage in self._backbone.feature_info.info]
        out_indices = self._backbone.feature_info.out_indices
        out_features = self._backbone.feature_info.module_name()

        # We verify the out indices and out features are valid
        verify_out_features_out_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        self._out_features, self._out_indices = out_features, out_indices

    def _init_transformers_backbone(self, config) -> None:
        stage_names = getattr(config, "stage_names")
        out_features = getattr(config, "out_features", None)
        out_indices = getattr(config, "out_indices", None)

        self.stage_names = stage_names
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=stage_names
        )
        # Number of channels for each stage. This is set in the transformer backbone model init
        self.num_features = None

    def _init_backbone(self, config) -> None:
        """
        Method to initialize the backbone. This method is called by the constructor of the base class after the
        pretrained model weights have been loaded.
        """
        self.config = config

        self.use_timm_backbone = getattr(config, "use_timm_backbone", False)
        self.backbone_type = BackboneType.TIMM if self.use_timm_backbone else BackboneType.TRANSFORMERS

        if self.backbone_type == BackboneType.TIMM:
            self._init_timm_backbone(config)
        elif self.backbone_type == BackboneType.TRANSFORMERS:
            self._init_transformers_backbone(config)
        else:
            raise ValueError(f"backbone_type {self.backbone_type} not supported.")

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: List[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=None, stage_names=self.stage_names
        )

    @property
    def out_indices(self):
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        """
        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=None, out_indices=out_indices, stage_names=self.stage_names
        )

    @property
    def out_feature_channels(self):
        # the current backbones will output the number of channels for each stage
        # even if that stage is not in the out_features list.
        return {stage: self.num_features[i] for i, stage in enumerate(self.stage_names)}

    @property
    def channels(self):
        return [self.out_feature_channels[name] for name in self.out_features]

    def forward_with_filtered_kwargs(self, *args, **kwargs):
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}
        return self(*args, **filtered_kwargs)

    def forward(
        self,
        pixel_values,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        raise NotImplementedError("This method should be implemented by the derived class.")

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to
        include the `out_features` and `out_indices` attributes.
        """
        output = super().to_dict()
        output["out_features"] = output.pop("_out_features")
        output["out_indices"] = output.pop("_out_indices")
        return output


class BackboneConfigMixin:
    """
    A Mixin to support handling the `out_features` and `out_indices` attributes for the backbone configurations.
    """

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: List[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=None, stage_names=self.stage_names
        )

    @property
    def out_indices(self):
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        """
        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=None, out_indices=out_indices, stage_names=self.stage_names
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to
        include the `out_features` and `out_indices` attributes.
        """
        output = super().to_dict()
        output["out_features"] = output.pop("_out_features")
        output["out_indices"] = output.pop("_out_indices")
        return output


def load_backbone(config):
    """
    Loads the backbone model from a config object.

    If the config is from the backbone model itself, then we return a backbone model with randomly initialized
    weights.

    If the config is from the parent model of the backbone model itself, then we load the pretrained backbone weights
    if specified.
    """
    from transformers import AutoBackbone, AutoConfig

    backbone_config = getattr(config, "backbone_config", None)
    use_timm_backbone = getattr(config, "use_timm_backbone", None)
    use_pretrained_backbone = getattr(config, "use_pretrained_backbone", None)
    backbone_checkpoint = getattr(config, "backbone", None)
    backbone_kwargs = getattr(config, "backbone_kwargs", None)

    backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs

    if backbone_kwargs and backbone_config is not None:
        raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

    # If there is a backbone_config and a backbone checkpoint, and use_pretrained_backbone=False then the desired
    # behaviour is ill-defined: do you want to load from the checkpoint's config or the backbone_config?
    if backbone_config is not None and backbone_checkpoint is not None and use_pretrained_backbone is not None:
        raise ValueError("Cannot specify both config.backbone_config and config.backbone")

    # If any of thhe following are set, then the config passed in is from a model which contains a backbone.
    if (
        backbone_config is None
        and use_timm_backbone is None
        and backbone_checkpoint is None
        and backbone_checkpoint is None
    ):
        return AutoBackbone.from_config(config=config, **backbone_kwargs)

    # config from the parent model that has a backbone
    if use_timm_backbone:
        if backbone_checkpoint is None:
            raise ValueError("config.backbone must be set if use_timm_backbone is True")
        # Because of how timm backbones were originally added to models, we need to pass in use_pretrained_backbone
        # to determine whether to load the pretrained weights.
        backbone = AutoBackbone.from_pretrained(
            backbone_checkpoint,
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            **backbone_kwargs,
        )
    elif use_pretrained_backbone:
        if backbone_checkpoint is None:
            raise ValueError("config.backbone must be set if use_pretrained_backbone is True")
        backbone = AutoBackbone.from_pretrained(backbone_checkpoint, **backbone_kwargs)
    else:
        if backbone_config is None and backbone_checkpoint is None:
            raise ValueError("Either config.backbone_config or config.backbone must be set")
        if backbone_config is None:
            backbone_config = AutoConfig.from_pretrained(backbone_checkpoint, **backbone_kwargs)
        backbone = AutoBackbone.from_config(config=backbone_config)
    return backbone
