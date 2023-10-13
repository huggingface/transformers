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
""" LLaVA model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging

from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shauray/Llava-Llama-2-7B-hf": "https://huggingface.co/shauray/Llava-Llama-2-7B-hf/resolve/main/config.json",
}


class LlavaTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
            is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        pad_token_id (`int`, *optional*, defaults to `None`):
            The ID of the padding token used for input sequences. This token is typically used to pad sequences to a
            common length during tokenization.
        bos_token_id (`int`, *optional*, defaults to `1`):
            The ID of the beginning-of-sequence (BOS) token. The BOS token is used to mark the start of a sequence and
            is often used in language generation tasks.
        eos_token_id (`int`, *optional*, defaults to `2`):
            The ID of the end-of-sequence (EOS) token. The EOS token is used to mark the end of a sequence and is often
            used in language generation tasks.
        mm_hidden_size (`int`, *optional*, defaults to `1024`):
            Dimension of the hidden representations.
        attention_bias (`bool`, *optional*, defaults to `False`):
            The type of attention bias to apply during model computation. This parameter controls how the model attends
            to certain positions in the input sequence.
    """
    model_type = "llava_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        pretraining_tp=1,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        mm_hidden_size=1024,
        attention_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size

        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from InstructBlipConfig
        if config_dict.get("model_type") == "llava":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class LlavaVisionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptModel`]. It is used to instantiate a Mpt model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        d_model (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        mm_hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations for vision model.
        n_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        vocab_size (`int`, *optional*, defaults to 50282):
            Vocabulary size of the Mpt model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MptModel`]. Check [this
            discussion](https://huggingface.co/bigscience/mpt/discussions/120#633d28389addb8530b406c2a) on how the
            `vocab_size` has been defined.
        projector (`str`, *optional*, defaults to `"Linear"`): Checks if the model in v1.5 or v1.0

    Example:

    """

    model_type = "llava_vision"
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        d_model: int = 4096,
        mm_hidden_size: int = 1024,
        n_heads: int = 32,
        n_layers: int = 32,
        use_cache: bool = True,
        vocab_size: int = 50282,
        projector: str = "Linear",
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
        self.mm_hidden_size = mm_hidden_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.projector = projector
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
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from InstructBlipConfig
        if config_dict.get("model_type") == "llava":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class LlavaConfig(PretrainedConfig):
    r"""
    [`LlavaConfig`] is the configuration class to store the configuration of a [`LlavaForCausalLM`]. It is used to
    instantiate a Llava model according to the specified arguments, defining the llama model and a llava model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture. objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`LlavaTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`LlavaVisionConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers.models.llava.configuration_llava import LlavaTextConfig
    >>> from transformers import LlavaVisionConfig, LlavaConfig, LlavaForCausalLM

    >>> # Initializing a LlavaConfig with shauray/Llava-Llama-2-7B-hf style configuration
    >>> configuration = LlavaConfig()

    >>> # Initializing a LlavaForCausalLM (with random weights) from the shauray/Llava-Llama-2-7B-hf style configuration
    >>> model = LlavaForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Initializing Llava, Llama configurations
    >>> text_config = LlavaTextConfig()
    >>> vision_config = LlavaVisionConfig()

    >>> config = LlavaConfig.from_llava_configs(
    ...     vision_config,
    ...     text_config,
    ... )
    ```"""

    model_type = "llava"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. initializing the LlavaTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the LlavaVisionConfig with default values.")
            
        text_model_type = text_config["model_type"] if "model_type" in text_config else "llama"
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)


        #self.text_config = LlavaTextConfig(**text_config)
        self.vision_config = LlavaVisionConfig(**vision_config)

        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_llava_configs(
        cls,
        text_config: PretrainedConfig,
        #text_config: LlavaTextConfig,
        vision_config: LlavaVisionConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`LlavaConfig`] (or a derived class) from a Llama and a Llava model,

        Returns:
            [`LlavaConfig`]: An instance of a configuration object
        """

        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )
