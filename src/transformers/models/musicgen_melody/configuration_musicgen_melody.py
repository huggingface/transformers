# coding=utf-8
# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""Musicgen Melody model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class MusicgenMelodyDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MusicgenMelodyDecoder`]. It is used to instantiate a
    Musicgen Melody decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Musicgen Melody
    [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the MusicgenMelodyDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`MusicgenMelodyDecoder`].
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer block.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer block.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556)
            for more details.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, text_encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(hidden_size).
        num_codebooks (`int`, *optional*, defaults to 4):
            The number of parallel codebooks forwarded to the model.
        audio_channels (`int`, *optional*, defaults to 1):
            Number of audio channels used by the model (either mono or stereo). Stereo models generate a separate
            audio stream for the left/right output channels. Mono models generate a single audio stream output.
        pad_token_id (`int`, *optional*, defaults to 2048): The id of the *padding* token.
        bos_token_id (`int`, *optional*, defaults to 2048): The id of the *beginning-of-sequence* token.
        eos_token_id (`int`, *optional*): The id of the *end-of-sequence* token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`): Whether to tie word embeddings with the text encoder.
    """

    model_type = "musicgen_melody_decoder"
    base_config_key = "decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=2048,
        max_position_embeddings=2048,
        num_hidden_layers=24,
        ffn_dim=4096,
        num_attention_heads=16,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        hidden_size=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        initializer_factor=0.02,
        scale_embedding=False,
        num_codebooks=4,
        audio_channels=1,
        pad_token_id=2048,
        bos_token_id=2048,
        eos_token_id=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.initializer_factor = initializer_factor
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.num_codebooks = num_codebooks

        if audio_channels not in [1, 2]:
            raise ValueError(f"Expected 1 (mono) or 2 (stereo) audio channels, got {audio_channels} channels.")
        self.audio_channels = audio_channels

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MusicgenMelodyConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MusicgenMelodyModel`]. It is used to instantiate a
    Musicgen Melody model according to the specified arguments, defining the text encoder, audio encoder and Musicgen Melody decoder
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the Musicgen Melody
    [facebook/musicgen-melody](https://huggingface.co/facebook/musicgen-melody) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_chroma (`int`, *optional*, defaults to 12): Number of chroma bins to use.
        chroma_length (`int`, *optional*, defaults to 235):
            Maximum chroma duration if audio is used to condition the model. Corresponds to the maximum duration used during training.
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **text_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the text encoder config.
                - **audio_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the audio encoder config.
                - **decoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the decoder config.

    Example:

    ```python
    >>> from transformers import (
    ...     MusicgenMelodyConfig,
    ...     MusicgenMelodyDecoderConfig,
    ...     T5Config,
    ...     EncodecConfig,
    ...     MusicgenMelodyForConditionalGeneration,
    ... )

    >>> # Initializing text encoder, audio encoder, and decoder model configurations
    >>> text_encoder_config = T5Config()
    >>> audio_encoder_config = EncodecConfig()
    >>> decoder_config = MusicgenMelodyDecoderConfig()

    >>> configuration = MusicgenMelodyConfig.from_sub_models_config(
    ...     text_encoder_config, audio_encoder_config, decoder_config
    ... )

    >>> # Initializing a MusicgenMelodyForConditionalGeneration (with random weights) from the facebook/musicgen-melody style configuration
    >>> model = MusicgenMelodyForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> config_text_encoder = model.config.text_encoder
    >>> config_audio_encoder = model.config.audio_encoder
    >>> config_decoder = model.config.decoder

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("musicgen_melody-model")

    >>> # loading model and config from pretrained folder
    >>> musicgen_melody_config = MusicgenMelodyConfig.from_pretrained("musicgen_melody-model")
    >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("musicgen_melody-model", config=musicgen_melody_config)
    ```"""

    model_type = "musicgen_melody"
    sub_configs = {
        "text_encoder": AutoConfig,
        "audio_encoder": AutoConfig,
        "decoder": MusicgenMelodyDecoderConfig,
    }
    has_no_defaults_at_init = True

    def __init__(
        self,
        num_chroma=12,
        chroma_length=235,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if "text_encoder" not in kwargs or "audio_encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")

        text_encoder_config = kwargs.pop("text_encoder")
        text_encoder_model_type = text_encoder_config.pop("model_type")

        audio_encoder_config = kwargs.pop("audio_encoder")
        audio_encoder_model_type = audio_encoder_config.pop("model_type")

        decoder_config = kwargs.pop("decoder")

        self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **text_encoder_config)
        self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)
        self.decoder = MusicgenMelodyDecoderConfig(**decoder_config)
        self.is_encoder_decoder = False

        self.num_chroma = num_chroma
        self.chroma_length = chroma_length

    @classmethod
    def from_sub_models_config(
        cls,
        text_encoder_config: PretrainedConfig,
        audio_encoder_config: PretrainedConfig,
        decoder_config: MusicgenMelodyDecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MusicgenMelodyConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`MusicgenMelodyConfig`]: An instance of a configuration object
        """

        return cls(
            text_encoder=text_encoder_config.to_dict(),
            audio_encoder=audio_encoder_config.to_dict(),
            decoder=decoder_config.to_dict(),
            **kwargs,
        )

    @property
    # This is a property because you might want to change the codec model on the fly
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate


__all__ = ["MusicgenMelodyConfig", "MusicgenMelodyDecoderConfig"]
