# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""ViT Hybrid model configuration"""

from ....configuration_utils import PretrainedConfig
from ....utils import logging
from ...auto.configuration_auto import CONFIG_MAPPING
from ...bit import BitConfig


logger = logging.get_logger(__name__)


class ViTHybridConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTHybridModel`]. It is used to instantiate a ViT
    Hybrid model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ViT Hybrid
    [google/vit-hybrid-base-bit-384](https://huggingface.co/google/vit-hybrid-base-bit-384) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`Union[dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the backbone in a dictionary or the config object of the backbone.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        use_timm_backbone (`bool`, *optional*, defaults to `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 1):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        backbone_featmap_shape (`list[int]`, *optional*, defaults to `[1, 1024, 24, 24]`):
            Used only for the `hybrid` embedding type. The shape of the feature maps of the backbone.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.

    Example:

    ```python
    >>> from transformers import ViTHybridConfig, ViTHybridModel

    >>> # Initializing a ViT Hybrid vit-hybrid-base-bit-384 style configuration
    >>> configuration = ViTHybridConfig()

    >>> # Initializing a model (with random weights) from the vit-hybrid-base-bit-384 style configuration
    >>> model = ViTHybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vit-hybrid"

    def __init__(
        self,
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        backbone_kwargs=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=1,
        num_channels=3,
        backbone_featmap_shape=[1, 1024, 24, 24],
        qkv_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")

        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with a `BiT` backbone.")
            backbone_config = {
                "global_padding": "same",
                "layer_type": "bottleneck",
                "depths": [3, 4, 9],
                "out_features": ["stage3"],
                "embedding_dynamic_padding": True,
            }

        if backbone_kwargs is not None and backbone_kwargs and backbone_config is not None:
            raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

        if isinstance(backbone_config, dict):
            if "model_type" in backbone_config:
                backbone_config_class = CONFIG_MAPPING[backbone_config["model_type"]]
            else:
                logger.info(
                    "`model_type` is not found in `backbone_config`. Use `Bit` as the backbone configuration class."
                )
                backbone_config_class = BitConfig
            backbone_config = backbone_config_class(**backbone_config)

        self.backbone_featmap_shape = backbone_featmap_shape
        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias

    @property
    def sub_configs(self):
        return (
            {"backbone_config": type(self.backbone_config)}
            if getattr(self, "backbone_config", None) is not None
            else {}
        )


__all__ = ["ViTHybridConfig"]
