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
""" Jukebox configuration"""

import copy
import os
from typing import List, Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/jukebox-5b-lyrics": "https://huggingface.co/openai/jukebox-5b-lyrics/blob/main/config.json",
    "openai/jukebox-1b-lyrics": "https://huggingface.co/openai/jukebox-1b-lyrics/blob/main/config.json",
}

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

ATTENTION_PATTERNS = {
    "FullDenseAttention": lambda layer: _FullDenseAttention[0],
    "RawColumnPreviousRowAttention": lambda layer: _RawColumnPreviousRowAttention[
        layer % 3
    ],  # Alternate row, column and previous row attn
    "large_separated_enc_dec_w_lyrics": lambda layer: _LARGE_ATTENTION[
        layer % 79
    ],  # Used by large separated_enc_dec model with lyrics
    "single_enc_dec_w_lyrics": lambda layer: _PrimePrimeDenseAttention[layer % 3]
    if layer % 16 == 15
    else _RawColumnPreviousRowAttention[layer % 3],  # Used by single_enc_dec model with lyrics
}


class JukeboxPriorConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`JukeboxPrior`]. It is used to instantiate a
    `JukeboxPriorl` according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the top level prior fro the
    [openai/jukebox-1b-lyrics](https://huggingface.co/openai/ukebox-1b-lyrics) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.



    Args:
        metadata_dims (`List[Tuple[int, int]]`, *optional*, defaults to `[(604, 7898), (120, 4111), (120, 4111)]`):
            List containing the number of genres and the number of artists that were used to train the embedding layers
            of each of the prior models.
        single_enc_dec (`List[bool]`, *optional*, defaults to `[True, False, False]`):
            Whether or not to use a single encoder-decoder architecture or split both modules and have a seperate
            `encoderoder` for each of the priors.
        merged_decoder (`list`, *optional*, defaults to [True, False, False]):
            Whether or not the decoder is merged with the encoder.
        lyric_conditioning (`list`, *optional*, defaults to [True, False, False]):
            Whether or not to use the lyrics as conditioning.
        nb_relevant_lyric_tokens (`list`, *optional*, defaults to [384, 0, 0]):
            Number of tokens that are used when sampling a single window of length `prior_n_ctx`
        zero_out (`bool`, *optional*, defaults to False):
            Zero out weights when initialising.
        depth (`list`, *optional*, defaults to [3, 16, 16]):
            Number of layers to use for the music conditioner.
        width (`list`, *optional*, defaults to [128, 1024, 1024]):
            Width of the audio conditioning layer.
        dilation_growth_rate (`list`, *optional*, defaults to [1, 3, 3]):
            Dilation grow rate used between each convolutionnal block.
        dilation_cycle (`list`, *optional*, defaults to [None, 8, 8]):
            Cycle of dilation to use. Usually similar to the ones used in the VQVAE.
        res_scale (`list`, *optional*, defaults to [None, True, False]):
            Wheter or not to scale the residuals in the audio conditionner block. Since the top level prior doeas not
            have a conditionner, the default value is to None and should not be modified.
        convolution_multiplier (`int`, *optional*, defaults to 1):
            Conditionner multiplier (the input states are mulitplied by that parameter for each convolution.
        downs_t (`tuple`, *optional*, defaults to (3, 2, 2)):
            Downsampling rates used in the audio conditioning network
        strides_t (`tuple`, *optional*, defaults to (2, 2, 2)):
            Striding used in the audio conditioning network
        encoder_spread (`bool`, *optional*, defaults to `False`):
            Spread used in the attention pattern
        encoder_width (`list`, *optional*, defaults to [128, 128, 128]):
            Width of the lyric encoder
        encoder_depth (`list`, *optional*, defaults to [18, 3, 3]):
            Number of encoder blocks used in the lyric encoder
        encoder_heads (`int`, *optional*, defaults to 4):
            Number of heads in the lyric encoder
        encoder_attention_multiplier (`float`, *optional*, defaults to 0.25):
            Multiplier coefficient used to define the hidden dimension of the attention layers. 0.25 means that
            0.25*width of the model will be used.
        encoder_mlp_multiplier (`float`, *optional*, defaults to 1.0):
            Multiplier coefficient used to define the hidden dimension of the MLP layers. 0.25 means that 0.25*width of
            the model will be used.
        encoder_blocks (`int`, *optional*, defaults to 32):
            Sequence of length seq_len is factored as [blocks, seq_len // blocks] in the `JukeboxAttention` layer.
        encoder_init_scale (`list`, *optional*, defaults to [0.1, 0.4, 0.4]):
            Initialisation scales for the lyric encoder modules.
        encoder_loss_fraction (`list`, *optional*, defaults to [0.4, 0.0, 0.0]):
            Multiplication factor used in front of the lyric encoder loss. Each value is for a particular level.
        encoder_attention_pattern (`list`, *optional*, defaults to [2, 0, 0]):
            Which attention pattern to use for the lyric encoder.
        encoder_attn_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the post-attention layer dropout in the lyric encoder.
        encoder_resid_dropout (`float`, *optional*, defaults to 0.0):
            Residual dropout used in the attention pattern of the lyric encoder.
        encoder_emb_dropout (`float`, *optional*, defaults to 0.0):
            Embedding dropout used in the lyric encoder.
        encoder_zero_out (`bool`, *optional*, defaults to `False`):
            Whether or not to set to zeros the weights the MLPs in the lyric encoder.
        encoder_res_scale (`bool`, *optional*, defaults to `False`):
            Residual scaling factor used in the lyric encoder attention patterns.
        encoder_n_vocab (`int`, *optional*, defaults to 79):
            Defines the number of different tokens that can be represented by the `inputs_ids` passed to the
            `encoderoder`
        init_scale (`list`, *optional*, defaults to [0.2, 1, 1]):
            Initialisation scales for the prior modules.
        spread (`bool`, *optional*, defaults to False):
            Spread used in the attention pattern
        zero_out (`bool`, *optional*, defaults to False):
             Whether or not to set to zeros the weights the MLPs of the priors.
        res_scale (`bool`, *optional*, defaults to False):
            Residual scaling factor used in every prior's attention layer.
        n_ctx (`tuple`, *optional*, defaults to (6144, 8192, 8192)):
            Number of context tokens for each prior. The context tokens are the music tokens that are attended to when
            generating music tokens.
        latent_dim (`int`, *optional*, defaults to 2048):
            Dimension of the latent music token space. Default value match the `vqvae_codebook_dimension`.
        width (`list`, *optional*, defaults to [2048, 1920, 1920]):
            Input and output dimension of the attention layers of each prior.
        attention_multiplier (`float`, *optional*, defaults to 0.25):
            Multiplier coefficient used to define the hidden dimension of the attention layers. 0.25 means that
            0.25*width of the model will be used.
        depth (`list`, *optional*, defaults to [72, 72, 72]):
            Depth of each prior. Defines the number of `attn_block`.
        n_heads (`list`, *optional*, defaults to [2, 1, 1]):
            Number of attention heads per prior.
        attention_pattern (`list`, *optional*, defaults to [12, 2, 2]):
            Attention patterns to use in each prior. Depending on the value, cross attention, block attention and
            sparse attention blocks are stacked.
        blocks (`int`, *optional*, defaults to 64):
            Sequence of length seq_len is factored as [blocks, seq_len // blocks] in the `JukeboxAttention` layer.
        alignment_layer (`list`, *optional*, defaults to [68, None, None]):
            Layer corresponding to the alignemnt between the lyrics and the audio.
        alignment_head (`list`, *optional*, defaults to [2, None, None]):
            Index of the attention head which takes care of the alignemnt between the lyrics and the audio.
        attn_dropout (`int`, *optional*, defaults to 0):
            Dropout probability for the post-attention layer dropout of the prior models.
        resid_dropout (`int`, *optional*, defaults to 0):
            Residual dropout probability used in the attention layers of the prior models.
        emb_dropout (`int`, *optional*, defaults to 0):
            Dropout applied to the embedding layer of the priors.
    """

    model_type = "jukebox"
    attribute_map = {
        "hidden_size": "vqvae_codebook_dimension",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        sampling_rate=44100,
        timing_dims=64,
        min_duration=0,
        max_duration=600,
        max_nb_genres=1,
        metadata_conditioning=True,
        zero_out=False,
        res_conv_depth=3,
        res_conv_width=128,
        res_dilation_growth_rate=1,
        res_dilation_cycle=None,
        res_scale=None,
        res_convolution_multiplier=1,
        res_downs_t=(3, 2, 2),
        res_strides_t=(2, 2, 2),
        encoder_spread=None,
        encoder_width=128,
        encoder_depth=18,
        encoder_heads=4,
        encoder_attention_multiplier=0.25,
        encoder_mlp_multiplier=1.0,
        encoder_blocks=32,
        encoder_init_scale=0.1,
        encoder_loss_fraction=[0.4, 0.0, 0.0],
        encoder_attention_pattern="RawColumnPreviousRowAttention",
        encoder_attn_dropout=0.0,
        encoder_resid_dropout=0.0,
        encoder_emb_dropout=0.0,
        encoder_zero_out=False,
        encoder_res_scale=False,
        encoder_n_vocab=79,
        init_scale=0.2,
        n_ctx=6144,
        width=2048,
        depth=72,
        n_heads=2,
        attention_pattern="single_enc_dec_w_lyrics",
        alignment_layer=68,
        alignment_head=2,
        metadata_dims=(604, 7898),
        single_enc_dec=True,
        merged_decoder=True,
        lyric_conditioning=True,
        nb_relevant_lyric_tokens=384,
        embed_dim=2048,
        spread=None,
        blocks=64,
        attention_multiplier=0.25,
        mlp_multiplier=1.0,
        attn_dropout=0,
        resid_dropout=0,
        emb_dropout=0,
        mask=False,
        act_fn="quick_gelu",
        **kwargs
    ):
        self.metadata_dims = metadata_dims
        self.res_conv_depth = res_conv_depth
        self.res_conv_width = res_conv_width
        #  Auto regressive (decoder) kwargs :
        self.attention_pattern = attention_pattern
        self.n_heads = n_heads
        self.depth = depth
        self.width = width
        self.n_ctx = n_ctx
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.emb_dropout = emb_dropout
        self.zero_out = zero_out
        self.res_scale = res_scale
        self.blocks = blocks
        self.attention_multiplier = attention_multiplier
        self.mlp_multiplier = mlp_multiplier
        self.spread = spread
        self.alignment_layer = alignment_layer
        self.alignment_head = alignment_head
        self.init_scale = init_scale

        # Audio conditioning : upsampler parameters
        self.depth = depth
        self.width = width
        self.res_dilation_growth_rate = res_dilation_growth_rate
        self.res_dilation_cycle = res_dilation_cycle
        self.zero_out = zero_out
        self.res_convolution_multiplier = res_convolution_multiplier
        self.res_scale = res_scale
        self.res_downs_t = res_downs_t
        self.res_strides_t = res_strides_t

        # Lyric conditioning
        self.merged_decoder = merged_decoder  # is this equivalent ?
        self.single_enc_dec = single_enc_dec
        self.lyric_conditioning = lyric_conditioning
        self.nb_relevant_lyric_tokens = nb_relevant_lyric_tokens

        self.encoder_attn_dropout = encoder_attn_dropout
        self.encoder_attention_pattern = encoder_attention_pattern
        self.encoder_blocks = encoder_blocks
        self.encoder_depth = encoder_depth
        self.encoder_emb_dropout = encoder_emb_dropout
        self.encoder_heads = encoder_heads
        self.encoder_init_scale = encoder_init_scale
        self.encoder_loss_fraction = encoder_loss_fraction
        self.encoder_attention_multiplier = encoder_attention_multiplier
        self.encoder_mlp_multiplier = encoder_mlp_multiplier
        self.encoder_resid_dropout = encoder_resid_dropout
        self.encoder_res_scale = encoder_res_scale
        self.encoder_spread = encoder_spread
        self.encoder_width = encoder_width
        self.encoder_zero_out = encoder_zero_out
        self.encoder_n_vocab = encoder_n_vocab
        self.mask = mask
        self.act_fn = act_fn

        self.sampling_rate = sampling_rate
        self.timing_dims = timing_dims
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_nb_genres = max_nb_genres
        self.metadata_conditioning = metadata_conditioning

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "jukebox_prior":
            config_dict = config_dict["prior_configs"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class JukeboxVQVAEConfig(PretrainedConfig):
    """
        hop_fraction (`list`, *optional*, defaults to [0.125, 0.5, 0.5]):
            Fraction of non-intersecting window used when continuing the sampling process.
        input_channels:
            number of audio channels
        sample_length:
            on which the VQVAE was trained. Provides the max output shape of the VQVAE
        levels (`int`, *optional*, defaults to 3):
            Number of hierachical levels that used in the VQVAE.
        downs_t (`tuple`, *optional*, defaults to (3, 2, 2)):
            Downsampling rate for each level of the hierachical VQ-VAE.
        strides_t (`tuple`, *optional*, defaults to (2, 2, 2)):
            Stride used for each level of the hierachical VQ-VAE.
        embed_dim (`int`, *optional*, defaults to 64):
            Dimension of the codebook vectors.
        codebook_dimension (`int`, *optional*, defaults to 2048):
            Number of codes to use in each of the VQVAE.
        convolution_multiplier (`int`, *optional*, defaults to 1):
            Projection factor used in the `JukeboxResConv1DBlock`.
        dilation_growth_rate (`int`, *optional*, defaults to 3):
            Resnet dilation growth rate used in the VQVAE (dilation_growth_rate ** depth)
        dilation_cycle (`int`, *optional*, defaults to None):
            Dilation cycle value used in the `JukeboxResnet`. If an int is used, each new Conv1 block will have a depth
            of reduced by a power of `dilation_cycle`.
        multipliers (`tuple`, *optional*, defaults to (2, 1, 1)):
            Depth and width multipliers used for each level. Used on the `conv_block_width` and `conv_block_depth`
        lmu (`float`, *optional*, defaults to 0.99):
            Used in the codebook update, exponential moving average coefficient. For more detail refer to Appendix A.1
            of the original [VQVAE paper](https://arxiv.org/pdf/1711.00937v2.pdf)
        commit (`float`, *optional*, defaults to 0.02):
            Commit loss multiplier.
        conv_block_depth (`int`, *optional*, defaults to 4):
            Depth of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        conv_block_width (`int`, *optional*, defaults to 32):
            Width of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        reverse_decoder_dilation (`int`, *optional*, defaults to 1):
            Whether or not to reverse the dilation rate for the decoder.
    Example:
    """

    def __init__(
        self,
        hop_fraction=[0.125, 0.5, 0.5],
        sample_length=1058304,
        levels=3,
        embed_dim=64,
        codebook_dimension=2048,
        lmu=0.99,
        commit=0.02,
        conv_input_shape=1,
        res_downs_t=(3, 2, 2),
        res_strides_t=(2, 2, 2),
        multipliers=(2, 1, 1),
        res_conv_width=32,
        res_conv_depth=4,
        res_convolution_multiplier=1,
        res_dilation_growth_rate=3,
        res_dilation_cycle=None,
        res_scale=False,
        act_fn="relu",
        **kwargs
    ):
        self.hop_fraction = hop_fraction
        self.conv_input_shape = conv_input_shape
        self.sample_length = sample_length

        # VQVAE parameters (all used)
        self.levels = levels
        self.embed_dim = embed_dim
        self.codebook_dimension = codebook_dimension
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
        self.res_scale = res_scale
        self.act_fn = act_fn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":

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


    The downsampling and stride are used to determine downsampling of the input sequence. For example, downsamoling =
    (5,3), and strides = (2, 2) will downsample the audio by 2**5 = 32 to get the first level of codes, and 2**8 = 256
    to get the second level codes. This is mostly true for training the top level prior and the upsamplers.

    Args:
        sampling_rate (`int`, *optional*, defaults to 44100):
            Sampling rate of the raw audio.
        nb_priors (`int`, *optional*, defaults to 3):
            Number of prior models that will sequentialy sample tokens. Each prior is conditional auto regressive
            (decoder) model, apart from the top prior, which can include a lyric encoder. The available models were
            trained using a top prior and 2 upsampler priors.
        timing_dims (`int`, *optional*, defaults to 64):
            Dimensions of the JukeboxRangeEmbedding layer which is equivalent to traditional positional embedding
            layer. The timing embedding layer converts the absolute and relative position in the currently sampled
            audio to a tensor of lenght `timing_dims` that will be added to the music tokens.
        metadata_conditioning (`bool`, *optional*, defaults to `True`):
            Whether or not to use metadata conditioning, corresponding to the artist, the genre and the min/maximum
            duration.
        single_enc_dec (`List[bool]`, *optional*, defaults to `[True, False, False]`):
            Whether or not to use a single encoder-decoder architecture or split both modules and have a seperate
            `encoderoder` for each of the priors.
        merged_decoder (`list`, *optional*, defaults to [True, False, False]):
            Whether or not the encoders are merged. This means that the input of.
        lyric_conditioning (`list`, *optional*, defaults to [True, False, False]):
            Whether or not to use the lyrics as conditioning.
        nb_relevant_lyric_tokens (`list`, *optional*, defaults to [384, 0, 0]):
            Number of tokens that are used when sampling a single window of length `n_ctx`
        min_duration (`float`, *optional*, defaults to 17.84):
            Minimum duration of the audios to generate
        max_duration (`float`, *optional*, defaults to 600.0):
            Maximum duration of the audios to generate
        max_nb_genres (`int`, *optional*, defaults to 5):
            Maximum number of genres that can be used to condition a single sample.
        init_std (`float`, *optional*, defaults to 0.2):
            Standard deviation used to inital the model.

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
    attribute_map = {
        "hidden_size": "codebook_dimension",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
    }
    is_composition = True

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
        init_std=0.2,
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

        self.init_std = init_std
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
    def from_configs(cls, prior_configs: List[JukeboxPriorConfig], vqvae_config: JukeboxVQVAEConfig, **kwargs):
        r"""
        Instantiate a [`CLIPConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`CLIPConfig`]: An instance of a configuration object
        """
        prior_config_list = [config.to_dict() for config in prior_configs]
        return cls(prior_config_list=prior_config_list, vqvae_config_dict=vqvae_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        for i, config in enumerate(output.pop("prior_configs")):
            output[f"prior_{i}"] = config.to_dict()

        output["vqvae_config"] = self.vqvae_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
