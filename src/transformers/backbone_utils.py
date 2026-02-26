# Copyright 2026 The HuggingFace Inc. team.
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

"""Collection of utils to be used by backbones and their components."""

import enum
import inspect

from huggingface_hub import repo_exists

from .utils import logging


logger = logging.get_logger(__name__)


class BackboneType(enum.Enum):
    TIMM = "timm"
    TRANSFORMERS = "transformers"


class BackboneConfigMixin:
    """
    A Mixin to support handling the `out_features` and `out_indices` attributes for the backbone configurations.
    """

    def set_output_features_output_indices(
        self,
        out_features: list | None,
        out_indices: list | None,
    ):
        """
        Sets output indices and features to new values and aligns them with the given `stage_names`.
        If one of the inputs is not given, find the corresponding `out_features` or `out_indices`
        for the given `stage_names`.

        Args:
            out_features (`list[str]`, *optional*):
                The names of the features for the backbone to output. Defaults to `config._out_features` if not provided.
            out_indices (`list[int]` or `tuple[int]`, *optional*):
                The indices of the features for the backbone to output. Defaults to `config._out_indices` if not provided.
        """
        self._out_features = out_features
        self._out_indices = list(out_indices) if isinstance(out_indices, tuple) else out_indices

        # First verify that the out_features and out_indices are valid
        self.verify_out_features_out_indices()

        # Align output features with indices
        out_features, out_indices = self._out_features, self._out_indices
        if out_indices is None and out_features is None:
            out_indices = [len(self.stage_names) - 1]
            out_features = [self.stage_names[-1]]
        elif out_indices is None and out_features is not None:
            out_indices = [self.stage_names.index(layer) for layer in out_features]
        elif out_features is None and out_indices is not None:
            out_features = [self.stage_names[idx] for idx in out_indices]

        # Update values and verify that the aligned out_features and out_indices are valid
        self._out_features, self._out_indices = out_features, out_indices
        self.verify_out_features_out_indices()

    def verify_out_features_out_indices(self):
        """
        Verify that out_indices and out_features are valid for the given stage_names.
        """
        if self.stage_names is None:
            raise ValueError("Stage_names must be set for transformers backbones")

        if self._out_features is not None:
            if not isinstance(self._out_features, (list,)):
                raise ValueError(f"out_features must be a list got {type(self._out_features)}")
            if any(feat not in self.stage_names for feat in self._out_features):
                raise ValueError(
                    f"out_features must be a subset of stage_names: {self.stage_names} got {self._out_features}"
                )
            if len(self._out_features) != len(set(self._out_features)):
                raise ValueError(f"out_features must not contain any duplicates, got {self._out_features}")
            if self._out_features != (
                sorted_feats := [feat for feat in self.stage_names if feat in self._out_features]
            ):
                raise ValueError(
                    f"out_features must be in the same order as stage_names, expected {sorted_feats} got {self._out_features}"
                )

        if self._out_indices is not None:
            if not isinstance(self._out_indices, list):
                raise ValueError(f"out_indices must be a list, got {type(self._out_indices)}")
            # Convert negative indices to their positive equivalent: [-1,] -> [len(stage_names) - 1,]
            positive_indices = tuple(idx % len(self.stage_names) if idx < 0 else idx for idx in self._out_indices)
            if any(idx for idx in positive_indices if idx not in range(len(self.stage_names))):
                raise ValueError(
                    f"out_indices must be valid indices for stage_names {self.stage_names}, got {self._out_indices}"
                )
            if len(positive_indices) != len(set(positive_indices)):
                msg = f"out_indices must not contain any duplicates, got {self._out_indices}"
                msg += f"(equivalent to {positive_indices}))" if positive_indices != self._out_indices else ""
                raise ValueError(msg)
            if positive_indices != tuple(sorted(positive_indices)):
                sorted_negative = [
                    idx for _, idx in sorted(zip(positive_indices, self._out_indices), key=lambda x: x[0])
                ]
                raise ValueError(
                    f"out_indices must be in the same order as stage_names, expected {sorted_negative} got {self._out_indices}"
                )

        if self._out_features is not None and self._out_indices is not None:
            if len(self._out_features) != len(self._out_indices):
                raise ValueError("out_features and out_indices should have the same length if both are set")
            if self._out_features != [self.stage_names[idx] for idx in self._out_indices]:
                raise ValueError("out_features and out_indices should correspond to the same stages if both are set")

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: list[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        self.set_output_features_output_indices(out_features=out_features, out_indices=None)

    @property
    def out_indices(self):
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: tuple[int, ...] | list[int]):
        """
        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.
        """
        out_indices = list(out_indices) if out_indices is not None else out_indices
        self.set_output_features_output_indices(out_features=None, out_indices=out_indices)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PreTrainedConfig` to
        include the `out_features` and `out_indices` attributes.
        """
        output = super().to_dict()
        output["out_features"] = output.pop("_out_features", None)
        output["out_indices"] = output.pop("_out_indices", None)
        return output


class BackboneMixin:
    backbone_type: BackboneType | None = None

    # Attribute to indicate if the backbone has attention and can return attention outputs.
    # Should be set to `False` for conv-based models to be able to run `forward_with_filtered_kwargs`
    has_attentions: bool = True

    def __init__(self, *args, **kwargs) -> None:
        """
        Method to initialize the backbone. This method is called by the constructor of the base class after the
        pretrained model weights have been loaded.
        """
        super().__init__(*args, **kwargs)
        timm_backbone = kwargs.pop("timm_backbone", None)
        if timm_backbone is not None:
            self.backbone_type = BackboneType.TIMM
        else:
            self.backbone_type = BackboneType.TRANSFORMERS

        if self.backbone_type == BackboneType.TIMM:
            self._init_timm_backbone(backbone=timm_backbone)
        elif self.backbone_type == BackboneType.TRANSFORMERS:
            self._init_transformers_backbone()
        else:
            raise ValueError(f"backbone_type {self.backbone_type} not supported.")

    def _init_timm_backbone(self, backbone) -> None:
        """
        Initialize the backbone model from timm. The backbone must already be loaded to backbone
        """

        out_features_from_config = getattr(self.config, "out_features", None)
        stage_names_from_config = getattr(self.config, "stage_names", None)

        # These will disagree with the defaults for the transformers models e.g. for resnet50
        # the transformer model has out_features = ['stem', 'stage1', 'stage2', 'stage3', 'stage4']
        # the timm model has out_features = ['act', 'layer1', 'layer2', 'layer3', 'layer4']
        self.stage_names = [stage["module"] for stage in backbone.feature_info.info]
        self.num_features = [stage["num_chs"] for stage in backbone.feature_info.info]

        out_indices = list(backbone.feature_info.out_indices)
        out_features = backbone.feature_info.module_name()

        if out_features_from_config is not None and out_features_from_config != out_features:
            raise ValueError(
                f"Config has `out_features` set to {out_features_from_config} which doesn't match `out_features` "
                "from backbone's feature_info. Please check if your checkpoint has correct out features/indices saved."
            )

        if stage_names_from_config is not None and stage_names_from_config != self.stage_names:
            raise ValueError(
                f"Config has `stage_names` set to {stage_names_from_config} which doesn't match `stage_names` "
                "from backbone's feature_info. Please check if your checkpoint has correct `stage_names` saved."
            )

        # We set, align and verify out indices, out features and stage names
        self.config.stage_names = self.stage_names
        self.config.set_output_features_output_indices(out_features, out_indices)

    def _init_transformers_backbone(self) -> None:
        self.stage_names = self.config.stage_names
        self.config.verify_out_features_out_indices()
        # Number of channels for each stage. This is set in the transformer backbone model init
        self.num_features = None

    @property
    def out_features(self):
        return self.config._out_features

    @out_features.setter
    def out_features(self, out_features: list[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        self.config.out_features = out_features

    @property
    def out_indices(self):
        return self.config._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: tuple[int] | list[int]):
        """
        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.
        """
        self.config.out_indices = out_indices

    @property
    def out_feature_channels(self):
        # the current backbones will output the number of channels for each stage
        # even if that stage is not in the out_features list.
        return {stage: self.num_features[i] for i, stage in enumerate(self.stage_names)}

    @property
    def channels(self):
        return [self.out_feature_channels[name] for name in self.out_features]

    def forward_with_filtered_kwargs(self, *args, **kwargs):
        if not self.has_attentions:
            kwargs.pop("output_attentions", None)
        if self.backbone_type == BackboneType.TIMM:
            signature = dict(inspect.signature(self.forward).parameters)
            kwargs = {k: v for k, v in kwargs.items() if k in signature}
        return self(*args, **kwargs)

    def forward(
        self,
        pixel_values,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
    ):
        raise NotImplementedError("This method should be implemented by the derived class.")


def consolidate_backbone_kwargs_to_config(
    backbone_config,
    default_backbone: str | None = None,
    default_config_type: str | None = None,
    default_config_kwargs: dict | None = None,
    timm_default_kwargs: dict | None = None,
    **kwargs,
):
    # Lazy import to avoid circular import issues. Can be imported properly
    # after deleting ref to `BackboneMixin` in `utils/backbone_utils.py`
    from .configuration_utils import PreTrainedConfig
    from .models.auto import CONFIG_MAPPING

    use_timm_backbone = kwargs.pop("use_timm_backbone", True)
    backbone_kwargs = kwargs.pop("backbone_kwargs", {})
    backbone = kwargs.pop("backbone") if kwargs.get("backbone") is not None else default_backbone
    kwargs.pop("use_pretrained_backbone", None)

    # Init timm backbone with hardcoded values for BC. If everything is set to `None` and there is
    # a default timm config, we use it to init the backbone.
    if (
        timm_default_kwargs is not None
        and use_timm_backbone
        and backbone is not None
        and backbone_config is None
        and not backbone_kwargs
    ):
        backbone_config = CONFIG_MAPPING["timm_backbone"](backbone=backbone, **timm_default_kwargs)
    elif backbone is not None and backbone_config is None:
        if repo_exists(backbone):
            config_dict, _ = PreTrainedConfig.get_config_dict(backbone)
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            config_dict.update(backbone_kwargs)
            backbone_config = config_class(**config_dict)
        else:
            backbone_config = CONFIG_MAPPING["timm_backbone"](backbone=backbone, **backbone_kwargs)
    elif backbone_config is None and default_config_type is not None:
        logger.info(
            f"`backbone_config` is `None`. Initializing the config with the default `{default_config_type}` vision config."
        )
        default_config_kwargs = default_config_kwargs or {}
        backbone_config = CONFIG_MAPPING[default_config_type](**default_config_kwargs)
    elif isinstance(backbone_config, dict):
        backbone_model_type = backbone_config.get("model_type")
        config_class = CONFIG_MAPPING[backbone_model_type]
        backbone_config = config_class.from_dict(backbone_config)

    return backbone_config, kwargs


def load_backbone(config):
    """
    Loads the backbone model from a config object.

    If the config is from the backbone model itself, then we return a backbone model with randomly initialized
    weights.

    If the config is from the parent model of the backbone model itself, then we load the pretrained backbone weights
    if specified.
    """
    from transformers import AutoBackbone

    backbone_config = getattr(config, "backbone_config", None)

    if backbone_config is None:
        backbone = AutoBackbone.from_config(config=config)
    else:
        backbone = AutoBackbone.from_config(config=backbone_config)
    return backbone
