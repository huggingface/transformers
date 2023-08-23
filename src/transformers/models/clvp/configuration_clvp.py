# coding=utf-8
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
""" CLVP model configuration"""

import copy
import os
from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    pass

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "susnato/clvp_dev": "https://huggingface.co/susnato/clvp_dev/resolve/main/config.json",
}


class CLVPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLVPTextModel`]. It is used to instantiate a CLVP
    text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the text encoder of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256):
            Vocabulary size of the CLVP text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`CLVPTextModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the text projection vector.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the feed-forward layers in [`CLVPMLP`].
        use_rotary_embedding (`bool`, *optional*, defaults to `True`):
            Whether to use rotary_embedding or not.
        summary_type (`str`, *optional*, defaults to `"mean"`):
            What strategy to use to get pooler_output from the last_hidden_state. `"last"`, `"first"`, `"mean"` and
            `"cls_index"` are supported.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import CLVPTextConfig, CLVPTextModel

    >>> # Initializing a CLVPTextConfig with susnato/clvp_dev style configuration
    >>> configuration = CLVPTextConfig()

    >>> # Initializing a CLVPTextModel (with random weights) from the susnato/clvp_dev style configuration
    >>> model = CLVPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "clvp_text_model"

    def __init__(
        self,
        vocab_size=256,
        hidden_size=768,
        intermediate_size=1536,
        projection_dim=768,
        num_hidden_layers=20,
        num_attention_heads=12,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.1,
        dropout=0.1,
        use_rotary_embedding=True,
        summary_type="mean",
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.use_rotary_embedding = use_rotary_embedding
        self.summary_type = summary_type

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from CLVPConfig
        if config_dict.get("model_type") == "clvp":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CLVPSpeechConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLVPSpeechModel`]. It is used to instantiate a
    CLVP speech encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the speech encoder of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 8192):
            Vocabulary size of the CLVP speech model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`CLVPSpeechModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 768):
            Dimensionality of the speech projection vector.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the feed forward layers in [`CLVPMLP`].
        use_rotary_embedding (`bool`, *optional*, defaults to `True`):
            Whether to use rotary_embedding or not.
        summary_type (`str`, *optional*, defaults to `"mean"`):
            What strategy to use to get pooler_output from the last_hidden_state. `"last"`, `"first"`, `"mean"` and
            `"cls_index"` are supported.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import CLVPSpeechConfig, CLVPSpeechModel

    >>> # Initializing a CLVPSpeechConfig with susnato/clvp_dev style configuration
    >>> configuration = CLVPSpeechConfig()

    >>> # Initializing a CLVPSpeechModel (with random weights) from the susnato/clvp_dev style configuration
    >>> model = CLVPSpeechModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clvp_speech_model"

    def __init__(
        self,
        vocab_size=8192,
        hidden_size=768,
        intermediate_size=1536,
        projection_dim=768,
        num_hidden_layers=20,
        num_attention_heads=12,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.1,
        dropout=0.1,
        use_rotary_embedding=True,
        summary_type="mean",
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.use_rotary_embedding = use_rotary_embedding
        self.summary_type = summary_type

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the speech config dict if we are loading from CLVPConfig
        if config_dict.get("model_type") == "clvp":
            config_dict = config_dict["speech_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CLVPAutoRegressiveConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLVPAutoRegressiveModel`]. It is used to instantiate a
    CLVP Auto Regressive Model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Auto Regressive part of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Please note that CLVP uses GPT2 as it's Auto Regressive Model.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`GPT2DoubleHeadsModel`].

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`GPT2DoubleHeadsModel`] and
            [`TFGPT2DoubleHeadsModel`].

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted mel features. This value is used in `CLVPMelEncoder`.
    """

    model_type = "clvp_autoregressive_model"

    def __init__(
            self,
            vocab_size=8194,
            n_positions=1012,
            n_embd=1024,
            n_layer=30,
            n_head=16,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=8192,
            eos_token_id=8193,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
            feature_size=80,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.feature_size = feature_size

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the speech config dict if we are loading from CLVPConfig
        if config_dict.get("model_type") == "clvp":
            config_dict = config_dict["autoregressive_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CLVPConfig(PretrainedConfig):
    r"""
    [`CLVPConfig`] is the configuration class to store the configuration of a [`CLVPModel`]. It is used to instantiate
    a CLVP model according to the specified arguments, defining the text model and speech model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the CLVP
    [susnato/clvp_dev](https://huggingface.co/susnato/clvp_dev) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLVPTextConfig`].
        speech_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLVPSpeechConfig`].
        autoregressive_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLVPAutoRegressiveConfig`].
        projection_dim (`int`, *optional*, defaults to 768):
            Dimentionality of text and speech projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLVP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import CLVPConfig, CLVPModel

    >>> # Initializing a CLVPConfig with susnato/clvp_dev style configuration
    >>> configuration = CLVPConfig()

    >>> # Initializing a CLVPModel (with random weights) from the susnato/clvp_dev style configuration
    >>> model = CLVPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLVPConfig from a CLVPTextConfig, CLVPSpeechConfig and a CLVPAutoRegressiveConfig
    >>> from transformers import CLVPTextConfig, CLVPSpeechConfig, CLVPAutoRegressiveConfig

    >>> # Initializing a CLVPText, CLVPSpeech and CLVPAutoRegressiveConfig configuration
    >>> config_text = CLVPTextConfig()
    >>> config_speech = CLVPSpeechConfig()
    >>> autoregressive_config = CLVPAutoRegressiveConfig()

    >>> config = CLVPConfig.from_text_speech_configs(config_text, config_speech, autoregressive_config)
    ```"""

    model_type = "clvp"
    is_composition = True

    def __init__(
        self, text_config=None, speech_config=None, autoregressive_config=None, projection_dim=768, logit_scale_init_value=2.6592, **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 3 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        speech_config_dict = kwargs.pop("speech_config_dict", None)
        autoregressive_config_dict = kwargs.pop("autoregressive_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[text|speech|autoregressive]_config_dict` to `[text|speech|autoregressive]_config`,
        # we use the values in `[text|speech|autoregressive]_config_dict` to update the values in
        # `[text|speech|autoregressive]_config`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = CLVPTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `CLVPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if speech_config_dict is not None:
            if speech_config is None:
                speech_config = {}

            # This is the complete result when using `speech_config_dict`.
            _speech_config_dict = CLVPSpeechConfig(**speech_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _speech_config_dict:
                _speech_config_dict["id2label"] = {
                    str(key): value for key, value in _speech_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_speech_config_dict` and `speech_config` but being different.
            for key, value in _speech_config_dict.items():
                if key in speech_config and value != speech_config[key] and key not in ["transformers_version"]:
                    # If specified in `speech_config_dict`
                    if key in speech_config_dict:
                        message = (
                            f"`{key}` is found in both `speech_config_dict` and `speech_config` but with different "
                            f'values. The value `speech_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`speech_config_dict` is provided which will be used to initialize `CLVPSpeechConfig`. "
                            f'The value `speech_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `speech_config` with the ones in `_speech_config_dict`.
            speech_config.update(_speech_config_dict)

        if autoregressive_config_dict is not None:
            if autoregressive_config is None:
                autoregressive_config = {}

            # This is the complete result when using `autoregressive_config_dict`.
            _autoregressive_config_dict = CLVPAutoRegressiveConfig(**autoregressive_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _autoregressive_config_dict:
                _autoregressive_config_dict["id2label"] = {
                    str(key): value for key, value in _autoregressive_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_autoregressive_config_dict` and `autoregressive_config` but being different.
            for key, value in _autoregressive_config_dict.items():
                if key in autoregressive_config and value != autoregressive_config[key] and key not in ["transformers_version"]:
                    # If specified in `autoregressive_config_dict`
                    if key in autoregressive_config_dict:
                        message = (
                            f"`{key}` is found in both `autoregressive_config_dict` and `autoregressive_config` but with different "
                            f'values. The value `autoregressive_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`autoregressive_config_dict` is provided which will be used to initialize `CLVPAutoRegressiveConfig`. "
                            f'The value `autoregressive_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `speech_config` with the ones in `_speech_config_dict`.
            autoregressive_config.update(_autoregressive_config_dict)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `CLVPTextConfig` with default values.")

        if speech_config is None:
            speech_config = {}
            logger.info("`speech_config` is `None`. initializing the `CLVPSpeechConfig` with default values.")

        if autoregressive_config is None:
            autoregressive_config = {}
            logger.info("`autoregressive_config` is `None`. initializing the `CLVPAutoRegressiveConfig` with default values.")

        self.text_config = CLVPTextConfig(**text_config)
        self.speech_config = CLVPSpeechConfig(**speech_config)
        self.autoregressive_config = CLVPAutoRegressiveConfig(**autoregressive_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_speech_autoregressive_configs(cls,
                                                text_config: CLVPTextConfig,
                                                speech_config: CLVPSpeechConfig,
                                                autoregressive_config: CLVPAutoRegressiveConfig,
                                                **kwargs):
        r"""
        Instantiate a [`CLVPConfig`] (or a derived class) from clvp text model configuration, clvp speech model
        configuration and clvp autoregressive model configuration.

        Returns:
            [`CLVPConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(),
                   speech_config=speech_config.to_dict(),
                   autoregressive_config=autoregressive_config.to_dict(),
                   **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["speech_config"] = self.speech_config.to_dict()
        output["autoregressive_config"] = self.autoregressive_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
