# coding=utf-8
# Copyright 2022 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""Jukebox configuration"""

import os
from typing import Union

from ....configuration_utils import PretrainedConfig
from ....utils import logging


logger = logging.get_logger(__name__)


_LARGE_ATTENTION = [
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "cross_attention",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "cross_attention",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "cross_attention",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "cross_attention",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "cross_attention",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "cross_attention",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "block_attn",
    "transpose_block_attn",
    "prev_block_attn",
    "cross_attention",
]
_RawColumnPreviousRowAttention = ["block_attn", "transpose_block_attn", "prev_block_attn"]
_FullDenseAttention = ["dense_attention"]
_PrimePrimeDenseAttention = ["prime_attn", "prime_attn", "dense_attn"]


def full_dense_attention(layer):
    return _FullDenseAttention[0]


def raw_column_previous_row_attention(layer):
    return _RawColumnPreviousRowAttention[layer % 3]


def large_separated_enc_dec_w_lyrics(layer):
    return _LARGE_ATTENTION[layer % 79]


def enc_dec_with_lyrics(layer):
    if layer % 16 == 15:
        return _PrimePrimeDenseAttention[layer % 3]
    return _RawColumnPreviousRowAttention[layer % 3]


ATTENTION_PATTERNS = {
    "full_dense_attention": full_dense_attention,
    "raw_column_previous_row_attention": raw_column_previous_row_attention,  # Alternate row, column and previous row attn
    "large_separated_enc_dec_w_lyrics": large_separated_enc_dec_w_lyrics,  # Used by large separated_enc_dec model with lyrics
    "enc_dec_with_lyrics": enc_dec_with_lyrics,  # Used by encoder_decoder model with lyrics
}


class JukeboxPriorConfig(PretrainedConfig):
    """
        This is the configuration class to store the configuration of a [`JukeboxPrior`]. It is used to instantiate a
        `JukeboxPrior` according to the specified arguments, defining the model architecture. Instantiating a
        configuration with the defaults will yield a similar configuration to that of the top level prior from the
        [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox
    -1b-lyrics) architecture.

        Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
        documentation from [`PretrainedConfig`] for more information.



    Args:
        act_fn (`str`, *optional*, defaults to `"quick_gelu"`):
            Activation function.
        alignment_head (`int`, *optional*, defaults to 2):
            Head that is responsible of the alignment between lyrics and music. Only used to compute the lyric to audio
            alignment
        alignment_layer (`int`, *optional*, defaults to 68):
            Index of the layer that is responsible of the alignment between lyrics and music. Only used to compute the
            lyric to audio alignment
        attention_multiplier (`float`, *optional*, defaults to 0.25):
            Multiplier coefficient used to define the hidden dimension of the attention layers. 0.25 means that
            0.25*width of the model will be used.
        attention_pattern (`str`, *optional*, defaults to `"enc_dec_with_lyrics"`):
            Which attention pattern to use for the decoder/
        attn_dropout (`int`, *optional*, defaults to 0):
            Dropout probability for the post-attention layer dropout in the decoder.
        attn_res_scale (`bool`, *optional*, defaults to `False`):
            Whether or not to scale the residuals in the attention conditioner block.
        blocks (`int`, *optional*, defaults to 64):
            Number of blocks used in the `block_attn`. A sequence of length seq_len is factored as `[blocks, seq_len //
            blocks]` in the `JukeboxAttention` layer.
        conv_res_scale (`int`, *optional*):
            Whether or not to scale the residuals in the conditioner block. Since the top level prior does not have a
            conditioner, the default value is to None and should not be modified.
        num_layers (`int`, *optional*, defaults to 72):
            Number of layers of the transformer architecture.
        emb_dropout (`int`, *optional*, defaults to 0):
            Embedding dropout used in the lyric decoder.
        encoder_config (`JukeboxPriorConfig`, *optional*) :
            Configuration of the encoder which models the prior on the lyrics.
        encoder_loss_fraction (`float`, *optional*, defaults to 0.4):
            Multiplication factor used in front of the lyric encoder loss.
        hidden_size (`int`, *optional*, defaults to 2048):
            Hidden dimension of the attention layers.
        init_scale (`float`, *optional*, defaults to 0.2):
            Initialization scales for the prior modules.
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether or not the prior is an encoder-decoder model. In case it is not, and `nb_relevant_lyric_tokens` is
            greater than 0, the `encoder` args should be specified for the lyric encoding.
        mask (`bool`, *optional*, defaults to `False`):
            Whether or not to mask the previous positions in the attention.
        max_duration (`int`, *optional*, defaults to 600):
            Maximum supported duration of the generated song in seconds.
        max_nb_genres (`int`, *optional*, defaults to 1):
            Maximum number of genres that can be used to condition the model.
        merged_decoder (`bool`, *optional*, defaults to `True`):
            Whether or not the decoder and the encoder inputs are merged. This is used for the separated
            encoder-decoder architecture
        metadata_conditioning (`bool`, *optional*, defaults to `True)`:
            Whether or not to condition on the artist and genre metadata.
        metadata_dims (`List[int]`, *optional*, defaults to `[604, 7898]`):
            Number of genres and the number of artists that were used to train the embedding layers of the prior
            models.
        min_duration (`int`, *optional*, defaults to 0):
            Minimum duration of the generated audio on which the model was trained.
        mlp_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier coefficient used to define the hidden dimension of the MLP layers. 0.25 means that 0.25*width of
            the model will be used.
        music_vocab_size (`int`, *optional*, defaults to 2048):
            Number of different music tokens. Should be similar to the `JukeboxVQVAEConfig.nb_discrete_codes`.
        n_ctx (`int`, *optional*, defaults to 6144):
            Number of context tokens for each prior. The context tokens are the music tokens that are attended to when
            generating music tokens.
        n_heads (`int`, *optional*, defaults to 2):
                Number of attention heads.
        nb_relevant_lyric_tokens (`int`, *optional*, defaults to 384):
            Number of lyric tokens that are used when sampling a single window of length `n_ctx`
        res_conv_depth (`int`, *optional*, defaults to 3):
            Depth of the `JukeboxDecoderConvBock` used to upsample the previously sampled audio in the
            `JukeboxMusicTokenConditioner`.
        res_conv_width (`int`, *optional*, defaults to 128):
            Width of the `JukeboxDecoderConvBock` used to upsample the previously sampled audio in the
            `JukeboxMusicTokenConditioner`.
        res_convolution_multiplier (`int`, *optional*, defaults to 1):
            Multiplier used to scale the `hidden_dim` of the `JukeboxResConv1DBlock`.
        res_dilation_cycle (`int`, *optional*):
            Dilation cycle used to define the `JukeboxMusicTokenConditioner`. Usually similar to the ones used in the
            corresponding level of the VQVAE. The first prior does not use it as it is not conditioned on upper level
            tokens.
        res_dilation_growth_rate (`int`, *optional*, defaults to 1):
            Dilation grow rate used between each convolutionnal block of the `JukeboxMusicTokenConditioner`
        res_downs_t (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            Downsampling rates used in the audio conditioning network
        res_strides_t (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Striding used in the audio conditioning network
        resid_dropout (`int`, *optional*, defaults to 0):
            Residual dropout used in the attention pattern.
        sampling_rate (`int`, *optional*, defaults to 44100):
            Sampling rate used for training.
        spread (`int`, *optional*):
            Spread used in the `summary_spread_attention` pattern
        timing_dims (`int`, *optional*, defaults to 64):
            Dimension of the timing embedding.
        zero_out (`bool`, *optional*, defaults to `False`):
            Whether or not to zero out convolution weights when initializing.
    """

    model_type = "jukebox_prior"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        act_fn="quick_gelu",
        level=0,
        alignment_head=2,
        alignment_layer=68,
        attention_multiplier=0.25,
        attention_pattern="enc_dec_with_lyrics",
        attn_dropout=0,
        attn_res_scale=False,
        blocks=64,
        conv_res_scale=None,
        num_layers=72,
        emb_dropout=0,
        encoder_config=None,
        encoder_loss_fraction=0.4,
        hidden_size=2048,
        init_scale=0.2,
        is_encoder_decoder=True,
        lyric_vocab_size=80,
        mask=False,
        max_duration=600,
        max_nb_genres=1,
        merged_decoder=True,
        metadata_conditioning=True,
        metadata_dims=[604, 7898],
        min_duration=0,
        mlp_multiplier=1.0,
        music_vocab_size=2048,
        n_ctx=6144,
        n_heads=2,
        nb_relevant_lyric_tokens=384,
        res_conv_depth=3,
        res_conv_width=128,
        res_convolution_multiplier=1,
        res_dilation_cycle=None,
        res_dilation_growth_rate=1,
        res_downs_t=[3, 2, 2],
        res_strides_t=[2, 2, 2],
        resid_dropout=0,
        sampling_rate=44100,
        spread=None,
        timing_dims=64,
        zero_out=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.act_fn = act_fn
        self.alignment_head = alignment_head
        self.alignment_layer = alignment_layer
        self.attention_multiplier = attention_multiplier
        self.attention_pattern = attention_pattern
        self.attn_dropout = attn_dropout
        self.attn_res_scale = attn_res_scale
        self.blocks = blocks
        self.conv_res_scale = conv_res_scale
        self.num_layers = num_layers
        self.emb_dropout = emb_dropout
        self.music_vocab_size = music_vocab_size
        if encoder_config is not None:
            self.encoder_config = JukeboxPriorConfig(**encoder_config)
        else:
            self.encoder_config = None
        self.encoder_loss_fraction = encoder_loss_fraction
        self.init_scale = init_scale
        self.is_encoder_decoder = is_encoder_decoder
        self.lyric_vocab_size = lyric_vocab_size
        self.level = level
        self.mask = mask
        self.max_duration = max_duration
        self.max_nb_genres = max_nb_genres
        self.merged_decoder = merged_decoder
        self.metadata_conditioning = metadata_conditioning
        self.metadata_dims = metadata_dims
        self.min_duration = min_duration
        self.mlp_multiplier = mlp_multiplier
        self.n_ctx = n_ctx
        self.n_heads = n_heads
        self.nb_relevant_lyric_tokens = nb_relevant_lyric_tokens
        self.res_conv_depth = res_conv_depth
        self.res_conv_width = res_conv_width
        self.res_convolution_multiplier = res_convolution_multiplier
        self.res_dilation_cycle = res_dilation_cycle
        self.res_dilation_growth_rate = res_dilation_growth_rate
        self.res_downs_t = res_downs_t
        self.res_strides_t = res_strides_t
        self.resid_dropout = resid_dropout
        self.sampling_rate = sampling_rate
        self.spread = spread
        self.timing_dims = timing_dims
        self.hidden_size = hidden_size
        self.zero_out = zero_out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], level=0, **kwargs):
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the prior config dict if we are loading from JukeboxConfig
        if config_dict.get("model_type") == "jukebox":
            config_dict = config_dict[f"prior_{level}"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class JukeboxVQVAEConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`JukeboxVQVAE`]. It is used to instantiate a
    `JukeboxVQVAE` according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VQVAE from
    [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        act_fn (`str`, *optional*, defaults to `"relu"`):
            Activation function of the model.
        nb_discrete_codes (`int`, *optional*, defaults to 2048):
            Number of codes of the VQVAE.
        commit (`float`, *optional*, defaults to 0.02):
            Commit loss multiplier.
        conv_input_shape (`int`, *optional*, defaults to 1):
            Number of audio channels.
        conv_res_scale (`bool`, *optional*, defaults to `False`):
            Whether or not to scale the residuals of the `JukeboxResConv1DBlock`.
        embed_dim (`int`, *optional*, defaults to 64):
            Embedding dimension of the codebook vectors.
        hop_fraction (`List[int]`, *optional*, defaults to `[0.125, 0.5, 0.5]`):
            Fraction of non-intersecting window used when continuing the sampling process.
        levels (`int`, *optional*, defaults to 3):
            Number of hierarchical levels that used in the VQVAE.
        lmu (`float`, *optional*, defaults to 0.99):
            Used in the codebook update, exponential moving average coefficient. For more detail refer to Appendix A.1
            of the original [VQVAE paper](https://huggingface.co/papers/1711.00937v2.pdf)
        multipliers (`List[int]`, *optional*, defaults to `[2, 1, 1]`):
            Depth and width multipliers used for each level. Used on the `res_conv_width` and `res_conv_depth`
        res_conv_depth (`int`, *optional*, defaults to 4):
            Depth of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        res_conv_width (`int`, *optional*, defaults to 32):
            Width of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        res_convolution_multiplier (`int`, *optional*, defaults to 1):
            Scaling factor of the hidden dimension used in the `JukeboxResConv1DBlock`.
        res_dilation_cycle (`int`, *optional*):
            Dilation cycle value used in the `JukeboxResnet`. If an int is used, each new Conv1 block will have a depth
            reduced by a power of `res_dilation_cycle`.
        res_dilation_growth_rate (`int`, *optional*, defaults to 3):
            Resnet dilation growth rate used in the VQVAE (dilation_growth_rate ** depth)
        res_downs_t (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            Downsampling rate for each level of the hierarchical VQ-VAE.
        res_strides_t (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Stride used for each level of the hierarchical VQ-VAE.
        sample_length (`int`, *optional*, defaults to 1058304):
            Provides the max input shape of the VQVAE. Is used to compute the input shape of each level.
        init_scale (`float`, *optional*, defaults to 0.2):
            Initialization scale.
        zero_out (`bool`, *optional*, defaults to `False`):
            Whether or not to zero out convolution weights when initializing.
    """

    model_type = "jukebox_vqvae"

    def __init__(
        self,
        act_fn="relu",
        nb_discrete_codes=2048,
        commit=0.02,
        conv_input_shape=1,
        conv_res_scale=False,
        embed_dim=64,
        hop_fraction=[0.125, 0.5, 0.5],
        levels=3,
        lmu=0.99,
        multipliers=[2, 1, 1],
        res_conv_depth=4,
        res_conv_width=32,
        res_convolution_multiplier=1,
        res_dilation_cycle=None,
        res_dilation_growth_rate=3,
        res_downs_t=[3, 2, 2],
        res_strides_t=[2, 2, 2],
        sample_length=1058304,
        init_scale=0.2,
        zero_out=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hop_fraction = hop_fraction
        self.conv_input_shape = conv_input_shape
        self.sample_length = sample_length

        # VQVAE parameters (all used)
        self.levels = levels
        self.embed_dim = embed_dim
        self.nb_discrete_codes = nb_discrete_codes
        self.res_conv_width = res_conv_width
        self.res_conv_depth = res_conv_depth
        self.res_convolution_multiplier = res_convolution_multiplier
        self.res_dilation_growth_rate = res_dilation_growth_rate
        self.res_dilation_cycle = res_dilation_cycle
        self.multipliers = multipliers
        self.res_downs_t = res_downs_t
        self.res_strides_t = res_strides_t
        self.lmu = lmu
        self.commit = commit
        self.conv_res_scale = conv_res_scale
        self.act_fn = act_fn
        self.init_scale = init_scale
        self.zero_out = zero_out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "jukebox":
            config_dict = config_dict["vqvae_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class JukeboxConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`JukeboxModel`].

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration with the defaults will
    yield a similar configuration to that of
    [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) architecture.


    The downsampling and stride are used to determine downsampling of the input sequence. For example, downsampling =
    (5,3), and strides = (2, 2) will downsample the audio by 2^5 = 32 to get the first level of codes, and 2**8 = 256
    to get the second level codes. This is mostly true for training the top level prior and the upsamplers.

    Args:
        vqvae_config (`JukeboxVQVAEConfig`, *optional*):
            Configuration for the `JukeboxVQVAE` model.
        prior_config_list (`List[JukeboxPriorConfig]`, *optional*):
            List of the configs for each of the `JukeboxPrior` of the model. The original architecture uses 3 priors.
        nb_priors (`int`, *optional*, defaults to 3):
            Number of prior models that will sequentially sample tokens. Each prior is conditional auto regressive
            (decoder) model, apart from the top prior, which can include a lyric encoder. The available models were
            trained using a top prior and 2 upsampler priors.
        sampling_rate (`int`, *optional*, defaults to 44100):
            Sampling rate of the raw audio.
        timing_dims (`int`, *optional*, defaults to 64):
            Dimensions of the JukeboxRangeEmbedding layer which is equivalent to traditional positional embedding
            layer. The timing embedding layer converts the absolute and relative position in the currently sampled
            audio to a tensor of length `timing_dims` that will be added to the music tokens.
        min_duration (`int`, *optional*, defaults to 0):
            Minimum duration of the audios to generate
        max_duration (`float`, *optional*, defaults to 600.0):
            Maximum duration of the audios to generate
        max_nb_genres (`int`, *optional*, defaults to 5):
            Maximum number of genres that can be used to condition a single sample.
        metadata_conditioning (`bool`, *optional*, defaults to `True`):
            Whether or not to use metadata conditioning, corresponding to the artist, the genre and the min/maximum
            duration.

    Example:

    ```python
    >>> from transformers import JukeboxModel, JukeboxConfig

    >>> # Initializing a Jukebox configuration
    >>> configuration = JukeboxConfig()

    >>> # Initializing a model from the configuration
    >>> model = JukeboxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "jukebox"

    def __init__(
        self,
        vqvae_config=None,
        prior_config_list=None,
        nb_priors=3,
        sampling_rate=44100,
        timing_dims=64,
        min_duration=0,
        max_duration=600.0,
        max_nb_genres=5,
        metadata_conditioning=True,
        **kwargs,
    ):
        if vqvae_config is None:
            vqvae_config = {}
            logger.info("vqvae_config is None. initializing the JukeboxVQVAE with default values.")

        self.vqvae_config = JukeboxVQVAEConfig(**vqvae_config)
        if prior_config_list is not None:
            self.prior_configs = [JukeboxPriorConfig(**prior_config) for prior_config in prior_config_list]
        else:
            self.prior_configs = []
            for prior_idx in range(nb_priors):
                prior_config = kwargs.pop(f"prior_{prior_idx}", None)
                if prior_config is None:
                    prior_config = {}
                    logger.info(
                        f"prior_{prior_idx}'s  config is None. Initializing the JukeboxPriorConfig list with default"
                        " values."
                    )
                self.prior_configs.append(JukeboxPriorConfig(**prior_config))

        self.hop_fraction = self.vqvae_config.hop_fraction

        self.nb_priors = nb_priors

        # Metadata conditioning
        self.max_nb_genres = max_nb_genres
        self.sampling_rate = sampling_rate
        self.timing_dims = timing_dims
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.metadata_conditioning = metadata_conditioning

        super().__init__(**kwargs)

    @classmethod
    def from_configs(cls, prior_configs: list[JukeboxPriorConfig], vqvae_config: JukeboxVQVAEConfig, **kwargs):
        r"""
        Instantiate a [`JukeboxConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`JukeboxConfig`]: An instance of a configuration object
        """
        prior_config_list = [config.to_dict() for config in prior_configs]
        return cls(prior_config_list=prior_config_list, vqvae_config_dict=vqvae_config.to_dict(), **kwargs)

    def to_dict(self):
        # Override the default to_dict to apply to_dict to the list of prior configs.
        result = super().to_dict()
        result["prior_config_list"] = [config.to_dict() for config in result.pop("prior_configs")]
        return result


__all__ = ["JukeboxConfig", "JukeboxPriorConfig", "JukeboxVQVAEConfig"]
