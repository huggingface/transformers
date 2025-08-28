# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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


from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


class InternVLVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternVLVisionModel`]. It is used to instantiate an InternVLVisionModel
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield
    a similar configuration to that of the InternVL3-1B.
    e.g. [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries, keys and values.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply normalization to the queries and keys before the attention operation.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        projection_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the projection layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of normalization to use in the encoder. Can be `"layer_norm"` or `"rms_norm"`.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`int` or `list[int]`, *optional*, defaults to `[448, 448]`):
            The size (resolution) of each image.
        patch_size (`int` or `list[int]`, *optional*, defaults to `[14, 14]`):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        use_absolute_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to use BERT-style absolute position embeddings.
        layer_scale_init_value (`float`, *optional*, defaults to 0.1):
            Scale to use in the self-attention layers. 0.1 for base, 1e-5 for large. Set 0 to disable layer scale.
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
            CLS token, before applying the classification head.

    Example:

    ```python
    >>> from transformers import InternVLVisionConfig, InternVLVisionModel

    >>> # Initializing a InternVLVisionModel OpenGVLab/InternVL3-1B-hf style configuration
    >>> configuration = InternVLVisionConfig()

    >>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
    >>> model = InternVLVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internvl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        attention_bias=False,
        use_qk_norm=False,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
        initializer_range=0.02,
        norm_type="layer_norm",
        layer_norm_eps=1e-06,
        image_size=[448, 448],
        patch_size=[14, 14],
        num_channels=3,
        use_mask_token=False,
        use_absolute_position_embeddings=True,
        layer_scale_init_value=0.1,
        use_mean_pooling=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_bias = attention_bias
        self.use_qk_norm = use_qk_norm
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.initializer_range = initializer_range
        self.norm_type = norm_type
        self.layer_norm_eps = layer_norm_eps

        image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size

        self.num_channels = num_channels
        self.use_mask_token = use_mask_token
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.layer_scale_init_value = layer_scale_init_value
        self.use_mean_pooling = use_mean_pooling


class InternVLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternVLForConditionalGeneration`]. It is used to instantiate a
    InternVL model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of InternVL3-1B.
    e.g. [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `InternVisonConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        image_token_id (`int`, *optional*, defaults to 151667):
            The image token index to encode the image prompt.
        image_seq_length (`int`, *optional*, defaults to 256):
            Number of image tokens to use per image patch.
        downsample_ratio (`float`, *optional*, defaults to 0.5):
            Factor by which to downsample the image.
        projector_hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the projector.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            The index of the layer to use as the image features.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.

    ```python
    >>> from transformers import InternVLForConditionalGeneration, InternVLConfig

    >>> # Initializing a InternVL style configuration
    >>> configuration = InternVLConfig()

    >>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
    >>> model = InternVLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internvl"
    sub_configs = {"text_config": AutoConfig, "vision_config": InternVLVisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151667,
        image_seq_length=256,
        downsample_ratio=0.5,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
        vision_feature_select_strategy="default",
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.downsample_ratio = downsample_ratio
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy

        if isinstance(vision_config, dict):
            self.vision_config = InternVLVisionConfig(**vision_config)
        elif isinstance(vision_config, InternVLVisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = InternVLVisionConfig()

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["InternVLVisionConfig", "InternVLConfig"]
