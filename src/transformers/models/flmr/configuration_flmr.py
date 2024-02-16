# coding=utf-8
# Copyright 2010, FLMR authors, The Hugging Face Team.
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
""" FLMR model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

FLMR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "LinWeizheDragon/PreFLMR_ViT-L": "https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/resolve/main/config.json",
    "LinWeizheDragon/FLMR": "https://huggingface.co/LinWeizheDragon/FLMR/resolve/main/config.json",
}


# Copied from transformers.models.clip.configuration_clip.CLIPVisionConfig with CLIP -> FLMR
class FLMRVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FLMRVisionModel`]. It is used to instantiate a
    FLMR vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the FLMR
    [openai/flmr-vit-base-patch32](https://huggingface.co/openai/flmr-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import FLMRVisionConfig, FLMRVisionModel

    >>> # Initializing a FLMRVisionConfig with openai/flmr-vit-base-patch32 style configuration
    >>> configuration = FLMRVisionConfig()

    >>> # Initializing a FLMRVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = FLMRVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flmr_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from a CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


# Copied from transformers.models.dpr.configuration_dpr.DPRConfig with DPR -> FLMR
class FLMRTextConfig(PretrainedConfig):
    r"""
    [`FLMRTextConfig`] is the configuration class to store the configuration of a *FLMRTextModel*.

    This is the configuration class to store the configuration of a [`FLMRTextModel`]. It is used to instantiate the components of the FLMR model according to the specified arguments,
    defining the model component architectures. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the DPRContextEncoder
    [facebook/dpr-ctx_encoder-single-nq-base](https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base)
    architecture.

    This class is a subclass of [`BertConfig`]. Please check the superclass for the documentation of all kwargs.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the FLMR model. Defines the different tokens that can be represented by the *inputs_ids*
            passed to the forward method of [`BertModel`].
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
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the *token_type_ids* passed into [`BertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        projection_dim (`int`, *optional*, defaults to 0):
            Dimension of the projection for the context and question encoders. If it is set to zero (default), then no
            projection is done.

    Example:

    ```python
    >>> from transformers import FLMRConfig, FLMRTextModel

    >>> # Initializing a FLMR facebook/dpr-ctx_encoder-single-nq-base style configuration
    >>> configuration = FLMRConfig()

    >>> # Initializing a model (with random weights) from the facebook/dpr-ctx_encoder-single-nq-base style configuration
    >>> model = FLMRTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flmr_text_model"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        projection_dim: int = 0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.projection_dim = projection_dim
        self.position_embedding_type = position_embedding_type


class FLMRConfig(PretrainedConfig):
    r"""
    [`FLMRConfig`] is the configuration class to store the configuration of a *FLMRModelForRetrieval*.

    This is the configuration class to store the configuration of a [`FLMRModelForRetrieval`]. It is used to instantiate the components of the FLMR model according to the specified arguments,
    defining the model component architectures. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the FLMR
    [BByrneLab/PreFLMR_ViT-G](https://huggingface.co/BByrneLab/PreFLMR_ViT-G)
    architecture.

    Args:
        vision_config (:class:`~transformers.FLMRVisionConfig`, *optional*):
            Configuration for the vision encoder.
        text_config (:class:`~transformers.FLMRTextConfig`, *optional*):
            Configuration for the text encoder.
        mask_punctuation (:obj:`bool`, *optional*, defaults to :obj:`True`):
            Whether to mask punctuation tokens in the input.
        mapping_network_prefix_length (:obj:`int`, *optional*, defaults to 32):
            The output length of the linear mapping network.
        dim (:obj:`int`, *optional*, defaults to 128):
            The late-interaction dimension of the model. The output of the text encoder, vision encoder, transformer mapping network should all be projected to this dimension for late-interaction scoring.
        use_vision_encoder (:obj:`bool`, *optional*, defaults to :obj:`True`):
            Whether to load the vision encoder. When no vision encoder is loaded, `image_features` should be used in the forward pass rather than `pixel_values`.
        initializer_range (:obj:`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        separate_query_and_context_text_encoder (:obj:`bool`, *optional*, defaults to :obj:`False`):
            Whether to use separate text encoders for query and context.
        separate_query_and_context_vision_encoder (:obj:`bool`, *optional*, defaults to :obj:`False`):
            Whether to use separate vision encoders for query and context.
        query_concat_output_from_vision_encoder (:obj:`bool`, *optional*, defaults to :obj:`True`):
            Whether to concatenate the output from the vision encoder to the output from the text encoder for the query.
        query_concat_output_from_text_encoder (:obj:`bool`, *optional*, defaults to :obj:`True`):
            Whether to concatenate the output from the text encoder to the output from the vision encoder for the query.
        context_concat_output_from_vision_encoder (:obj:`bool`, *optional*, defaults to :obj:`False`):
            Whether to concatenate the output from the vision encoder to the output from the text encoder for the context.
        context_concat_output_from_text_encoder (:obj:`bool`, *optional*, defaults to :obj:`True`):
            Whether to concatenate the output from the text encoder to the output from the vision encoder for the context.
        use_transformer_mapping_network (:obj:`bool`, *optional*, defaults to :obj:`False`):
            Whether to add a transformer mapping network to map the features from the vision encoder to the embedding space. This option is used in PreFLMR.
        transformer_mapping_config_base (:obj:`str`, *optional*):
            The base configuration for the transformer mapping network. This option is used in PreFLMR. An example of this argument is `bert-base-uncased`.
        transformer_mapping_num_hidden_layers (:obj:`int`, *optional*):
            The number of hidden layers in the transformer mapping network. This option is used in PreFLMR.
        load_cpu_extension (:obj:`bool`, *optional*, defaults to :obj:`False`):
            Whether to load the CPU extension. Only set this to `True` if a CPU is used in training and inference. In any case, GPU is recommended for training and inference.
        mask_instruction_token (:obj:`str`, *optional*):
            The token that indicates the end of the input instruction. All tokens before this token (the first one in a sequence) will be masked. This option is used in PreFLMR.
        transformer_mapping_cross_attention_length (:obj:`int`, *optional*, defaults to 32):
            The length of the cross attention in the transformer mapping network. This option is used in PreFLMR.
        vision_model_version (:obj:`str`, *optional*, defaults to :obj:`"openai/clip-vit-base-patch32"`):
            The version of the vision model being used in this FLMR model.
            This option is used in performing retrieval only. Though it does not affect the model architecture, it is highly recommended to set this argument so that it properly reflects the version of the vision model being used in the FLMR model. This arugment will be saved in the model configuration, and it can be read by the indexing engine. The indexing engine will use this argument to initialize an image processor, which can process the input image files. Find more details under `examples/research_projects/flmr-retrieval`.

    Example:

    ```python
    >>> from transformers import FLMRConfig, FLMRModelForRetrieval

    >>> # Initializing a FLMR weizhelin/flmr style configuration
    >>> configuration = FLMRConfig()

    >>> # Initializing a model (with random weights) from the FLMR style configuration
    >>> model = FLMRModelForRetrieval(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flmr"

    def __init__(
        self,
        vision_config: FLMRVisionConfig = None,
        text_config: FLMRTextConfig = None,
        mask_punctuation: bool = True,
        mapping_network_prefix_length: int = 32,
        dim: int = 128,
        use_vision_encoder: bool = True,
        initializer_range: float = 0.02,
        separate_query_and_context_text_encoder: bool = False,
        separate_query_and_context_vision_encoder: bool = False,
        query_concat_output_from_vision_encoder: bool = True,
        query_concat_output_from_text_encoder: bool = True,
        context_concat_output_from_vision_encoder: bool = False,
        context_concat_output_from_text_encoder: bool = True,
        use_transformer_mapping_network: bool = False,
        transformer_mapping_config_base: str = None,
        transformer_mapping_num_hidden_layers: int = None,
        load_cpu_extension: bool = False,
        mask_instruction_token: str = None,
        transformer_mapping_cross_attention_length: int = 32,
        vision_model_version: str = "openai/clip-vit-base-patch32",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
        if text_config is None:
            text_config = {}

        if not isinstance(vision_config, FLMRVisionConfig):
            vision_config = FLMRVisionConfig(**vision_config)
        if not isinstance(text_config, FLMRTextConfig):
            text_config = FLMRTextConfig(**text_config)

        self.vision_config = vision_config
        self.text_config = text_config
        self.dim = dim
        self.initializer_range = initializer_range
        self.mask_punctuation = mask_punctuation
        self.mapping_network_prefix_length = mapping_network_prefix_length
        self.use_vision_encoder = use_vision_encoder
        self.separate_query_and_context_text_encoder = separate_query_and_context_text_encoder
        self.separate_query_and_context_vision_encoder = separate_query_and_context_vision_encoder
        self.query_concat_output_from_vision_encoder = query_concat_output_from_vision_encoder
        self.query_concat_output_from_text_encoder = query_concat_output_from_text_encoder
        self.context_concat_output_from_vision_encoder = context_concat_output_from_vision_encoder
        self.context_concat_output_from_text_encoder = context_concat_output_from_text_encoder
        self.use_transformer_mapping_network = use_transformer_mapping_network
        self.transformer_mapping_config_base = transformer_mapping_config_base
        self.transformer_mapping_num_hidden_layers = transformer_mapping_num_hidden_layers
        self.load_cpu_extension = load_cpu_extension
        self.mask_instruction_token = mask_instruction_token
        self.transformer_mapping_cross_attention_length = transformer_mapping_cross_attention_length
        self.vision_model_version = vision_model_version

    @classmethod
    def from_text_vision_configs(cls, text_config: FLMRTextConfig, vision_config: FLMRVisionConfig, **kwargs):
        r"""
        Instantiate a [`FLMRConfig`] (or a derived class) from FLMR text model configuration and FLMR vision model
        configuration.

        Returns:
            [`FLMRConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config, vision_config=vision_config, **kwargs)
