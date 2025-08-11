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
from ..internvl.configuration_internvl import InternVLVisionConfig


class InternS1VisionConfig(InternVLVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternS1VisionModel`]. It is used to instantiate
    an InternS1VisionModel model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Intern-S1.
    e.g. [internlm/Intern-S1](https://huggingface.co/internlm/Intern-S1)

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
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate for the Transformer encoder. It is applied to the residual connections
            of each layer in the Transformer encoder.
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
    >>> from transformers import InternS1VisionConfig, InternS1VisionModel

    >>> # Initializing a InternS1VisionModel
    >>> configuration = InternS1VisionConfig()

    >>> # Initializing a model (with random weights) from configuration
    >>> model = InternS1VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "interns1_vision"

    def __init__(
        self,
        *args,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.drop_path_rate = drop_path_rate


class InternS1Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternS1ForConditionalGeneration`].
    It is used to instantiate a InternS1 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Intern-S1.
    e.g. [internlm/Intern-S1](https://huggingface.co/internlm/Intern-S1)

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
    >>> from transformers import InternS1ForConditionalGeneration, InternS1Config

    >>> # Initializing a InternS1 style configuration
    >>> configuration = InternS1Config()

    >>> # Initializing a model (with random weights) from configuration
    >>> model = InternS1ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "interns1"
    sub_configs = {"text_config": AutoConfig, "vision_config": InternS1VisionConfig}

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
            self.vision_config = InternS1VisionConfig(**vision_config)
        elif isinstance(vision_config, InternS1VisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = InternS1VisionConfig()

        if isinstance(text_config, dict):
            if "model_type" not in text_config:
                raise ValueError(
                    "If `text_config` is a dictionary, it must contain the key `model_type` to specify the model type."
                )
            text_config["model_type"] = text_config["model_type"]
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            raise ValueError(
                "If `text_config` is None, you must pass a valid configuration object or dictionary for the text model."
            )

        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["InternS1VisionConfig", "InternS1Config"]
