# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""DepthPro model configuration"""

from copy import deepcopy

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class DepthProConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DepthProModel`]. It is used to instantiate a
    DepthPro model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DepthPro
    [apple/DepthPro](https://huggingface.co/apple/DepthPro) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            The number of channels before fusion.
        patch_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each patch. This is also the image_size for backbone model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        intermediate_hook_ids (`List[int]`, *optional*, defaults to `[11, 5]`):
            Indices of the intermediate hidden states from the patch encoder to use for fusion.
        intermediate_feature_dims (`List[int]`, *optional*, defaults to `[256, 256]`):
            Hidden state dimensions during upsampling for each intermediate hidden state in `intermediate_hook_ids`.
        scaled_images_ratios (`List[float]`, *optional*, defaults to `[0.25, 0.5, 1]`):
            Ratios of scaled images to be used by the patch encoder.
        scaled_images_overlap_ratios (`List[float]`, *optional*, defaults to `[0.0, 0.5, 0.25]`):
            Overlap ratios between patches for each scaled image in `scaled_images_ratios`.
        scaled_images_feature_dims (`List[int]`, *optional*, defaults to `[1024, 1024, 512]`):
            Hidden state dimensions during upsampling for each scaled image in `scaled_images_ratios`.
        merge_padding_value (`int`, *optional*, defaults to 3):
            When merging smaller patches back to the image size, overlapping sections of this size are removed.
        use_batch_norm_in_fusion_residual (`bool`, *optional*, defaults to `False`):
            Whether to use batch normalization in the pre-activate residual units of the fusion blocks.
        use_bias_in_fusion_residual (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the pre-activate residual units of the fusion blocks.
        use_fov_model (`bool`, *optional*, defaults to `False`):
            Whether to use `DepthProFovModel` to generate the field of view.
        num_fov_head_layers (`int`, *optional*, defaults to 2):
            Number of convolution layers in the head of `DepthProFovModel`.
        image_model_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the image encoder model, which is loaded using the [`AutoModel`] API.
            By default, Dinov2 model is used as backbone.
        patch_model_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the patch encoder model, which is loaded using the [`AutoModel`] API.
            By default, Dinov2 model is used as backbone.
        fov_model_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the fov encoder model, which is loaded using the [`AutoModel`] API.
            By default, Dinov2 model is used as backbone.

    Example:

    ```python
    >>> from transformers import DepthProConfig, DepthProModel

    >>> # Initializing a DepthPro apple/DepthPro style configuration
    >>> configuration = DepthProConfig()

    >>> # Initializing a model (with random weights) from the apple/DepthPro style configuration
    >>> model = DepthProModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "depth_pro"
    sub_configs = {"image_model_config": AutoConfig, "patch_model_config": AutoConfig, "fov_model_config": AutoConfig}

    def __init__(
        self,
        fusion_hidden_size=256,
        patch_size=384,
        initializer_range=0.02,
        intermediate_hook_ids=[11, 5],
        intermediate_feature_dims=[256, 256],
        scaled_images_ratios=[0.25, 0.5, 1],
        scaled_images_overlap_ratios=[0.0, 0.5, 0.25],
        scaled_images_feature_dims=[1024, 1024, 512],
        merge_padding_value=3,
        use_batch_norm_in_fusion_residual=False,
        use_bias_in_fusion_residual=True,
        use_fov_model=False,
        num_fov_head_layers=2,
        image_model_config=None,
        patch_model_config=None,
        fov_model_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # scaled_images_ratios is sorted
        if scaled_images_ratios != sorted(scaled_images_ratios):
            raise ValueError(
                f"Values in scaled_images_ratios={scaled_images_ratios} " "should be sorted from low to high"
            )

        # scaled_images_ratios, scaled_images_overlap_ratios, scaled_images_feature_dims should be consistent
        if not (len(scaled_images_ratios) == len(scaled_images_overlap_ratios) == len(scaled_images_feature_dims)):
            raise ValueError(
                f"len(scaled_images_ratios)={len(scaled_images_ratios)} and "
                f"len(scaled_images_overlap_ratios)={len(scaled_images_overlap_ratios)} and "
                f"len(scaled_images_feature_dims)={len(scaled_images_feature_dims)}, "
                f"should match in config."
            )

        # intermediate_hook_ids, intermediate_feature_dims should be consistent
        if not (len(intermediate_hook_ids) == len(intermediate_feature_dims)):
            raise ValueError(
                f"len(intermediate_hook_ids)={len(intermediate_hook_ids)} and "
                f"len(intermediate_feature_dims)={len(intermediate_feature_dims)}, "
                f"should match in config."
            )

        # fusion_hidden_size should be consistent with num_fov_head_layers
        if fusion_hidden_size // 2**num_fov_head_layers == 0:
            raise ValueError(
                f"fusion_hidden_size={fusion_hidden_size} should be consistent with num_fov_head_layers={num_fov_head_layers} "
                "i.e fusion_hidden_size // 2**num_fov_head_layers > 0"
            )

        self.fusion_hidden_size = fusion_hidden_size
        self.patch_size = patch_size
        self.initializer_range = initializer_range
        self.use_batch_norm_in_fusion_residual = use_batch_norm_in_fusion_residual
        self.use_bias_in_fusion_residual = use_bias_in_fusion_residual
        self.use_fov_model = use_fov_model
        self.num_fov_head_layers = num_fov_head_layers
        self.intermediate_hook_ids = intermediate_hook_ids
        self.intermediate_feature_dims = intermediate_feature_dims
        self.scaled_images_ratios = scaled_images_ratios
        self.scaled_images_overlap_ratios = scaled_images_overlap_ratios
        self.scaled_images_feature_dims = scaled_images_feature_dims
        self.merge_padding_value = merge_padding_value
        self.image_model_config = image_model_config
        self.patch_model_config = patch_model_config
        self.fov_model_config = fov_model_config

        for sub_config_key in self.sub_configs.keys():
            sub_config = getattr(self, sub_config_key)

            if sub_config is None:
                sub_config = CONFIG_MAPPING["dinov2"](image_size=patch_size)
                logger.info(
                    f"`{sub_config_key}` is `None`. Initializing `{sub_config_key}` with the `Dinov2Config` "
                    f"with default values except `{sub_config_key}.image_size` is set to `config.patch_size`."
                )
            elif isinstance(sub_config, dict):
                sub_config = deepcopy(sub_config)
                if "model_type" not in sub_config:
                    raise KeyError(
                        f"The `model_type` key is missing in the `{sub_config_key}` dictionary. Please provide the model type."
                    )
                elif sub_config["model_type"] not in CONFIG_MAPPING:
                    raise ValueError(
                        f"The model type `{sub_config['model_type']}` in `{sub_config_key}` is not supported. Please provide a valid model type."
                    )
                image_size = sub_config.get("image_size")
                if image_size != patch_size:
                    logger.info(
                        f"The `image_size` in `{sub_config_key}` is set to `{image_size}`, "
                        f"but it does not match the required `patch_size` of `{patch_size}`. "
                        f"Updating `image_size` to `{patch_size}` for consistency. "
                        f"Ensure that `image_size` aligns with `patch_size` in the configuration."
                    )
                    sub_config.update({"image_size": patch_size})
                sub_config = CONFIG_MAPPING[sub_config["model_type"]](**sub_config)
            elif isinstance(sub_config, PretrainedConfig):
                sub_config = sub_config
                image_size = getattr(sub_config, "image_size", None)
                if image_size != patch_size:
                    raise ValueError(
                        f"`config.{sub_config_key}.image_size={image_size}` should match `config.patch_size={patch_size}`."
                    )
            else:
                raise TypeError(
                    f"Invalid type for `sub_config`. Expected `PretrainedConfig`, `dict`, or `None`, but got {type(sub_config)}."
                )

            setattr(self, sub_config_key, sub_config)


__all__ = ["DepthProConfig"]
