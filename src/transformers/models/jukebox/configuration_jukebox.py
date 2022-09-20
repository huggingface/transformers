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

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/jukebox-5b-lyrics": "https://huggingface.co/openai/jukebox-5b-lyrics/blob/main/config.json",
    "openai/jukebox-1b-lyrics": "https://huggingface.co/openai/jukebox-1b-lyrics/blob/main/config.json",
}


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
        metadata_dims (`list`, *optional*, defaults to [(604, 7898), (120, 4111), (120, 4111)]):
            List containing the number of genres and the number of artists that were used to train the embedding layers
            of each of the prior models.
        nb_priors (`int`, *optional*, defaults to 3):
            Number of prior models that will sequentialy sample tokens. Each prior is conditional auto regressive
            (decoder) model, apart from the top prior, which can include a lyric encoder. The available models were
            trained using a top prior and 2 upsampler priors.
        timing_dims (`int`, *optional*, defaults to 64):
            Dimensions of the JukeboxRangeEmbedding layer which is equivalent to traditional positional embedding
            layer. The timing embedding layer converts the absolute and relative position in the currently sampled
            audio to a tensor of lenght `timing_dims` that will be added to the music tokens.
        single_enc_dec (`list`, *optional*, defaults to [True, False, False]):
            Whether or not to use a single encoder-decoder architecture or split both modules and have a seperate
            `lyric_encoder` for each of the priors.
        metadata_conditioning (`bool`, *optional*, defaults to True):
            Whether or not to use metadata conditioning, corresponding to the artist, the genre and the min/maximum
            duration.
        merged_decoder (`list`, *optional*, defaults to [True, False, False]):
            Whether or not the decoder is merged with the encoder.
        lyric_conditioning (`list`, *optional*, defaults to [True, False, False]):
            Whether or not to use the lyrics as conditioning.
        nb_relevant_lyric_tokens (`list`, *optional*, defaults to [384, 0, 0]):
            Number of tokens that are used when sampling a single window of length `prior_n_ctx`
        min_duration (`float`, *optional*, defaults to 17.84):
            Minimum duration of the audios to generate
        max_duration (`float`, *optional*, defaults to 600.0):
            Maximum duration of the audios to generate
        max_nb_genres (`int`, *optional*, defaults to 5):
            Maximum number of genres that can be used to condition a single sample.
        init_std (`float`, *optional*, defaults to 0.2):
            Standard deviation used to inital the model.
        hop_fraction (`list`, *optional*, defaults to [0.125, 0.5, 0.5]):
            Fraction of non-intersecting window used when continuing the sampling process.
        cond_zero_out (`bool`, *optional*, defaults to False):
            Zero out weights when initialising.
        cond_depth (`list`, *optional*, defaults to [3, 16, 16]):
            Number of layers to use for the music conditioner.
        cond_width (`list`, *optional*, defaults to [128, 1024, 1024]):
            Width of the audio conditioning layer.
        cond_dilation_growth_rate (`list`, *optional*, defaults to [1, 3, 3]):
            Dilation grow rate used between each convolutionnal block.
        cond_dilation_cycle (`list`, *optional*, defaults to [None, 8, 8]):
            Cycle of dilation to use. Usually similar to the ones used in the VQVAE.
        cond_res_scale (`list`, *optional*, defaults to [None, True, False]):
            Wheter or not to scale the residuals in the audio conditionner block. Since the top level prior doeas not
            have a conditionner, the default value is to None and should not be modified.
        cond_m_conv (`int`, *optional*, defaults to 1):
            Conditionner multiplier (the input states are mulitplied by that parameter for each convolution.
        cond_downs_t (`tuple`, *optional*, defaults to (3, 2, 2)):
            Downsampling rates used in the audio conditioning network
        cond_strides_t (`tuple`, *optional*, defaults to (2, 2, 2)):
            Striding used in the audio conditioning network
        lyric_enc_spread (`bool`, *optional*, defaults to False):
            Spread used in the attention pattern
        lyric_enc_width (`list`, *optional*, defaults to [128, 128, 128]):
            Width of the lyric encoder
        lyric_enc_depth (`list`, *optional*, defaults to [18, 3, 3]):
            Number of encoder blocks used in the lyric encoder
        lyric_enc_heads (`int`, *optional*, defaults to 4):
            Number of heads in the lyric encoder
        lyric_enc_m_attn (`float`, *optional*, defaults to 0.25):
            Multiplier coefficient used to define the hidden dimension of the attention layers. 0.25 means that
            0.25*width of the model will be used.
        lyric_enc_m_mlp (`float`, *optional*, defaults to 1.0):
            Multiplier coefficient used to define the hidden dimension of the MLP layers. 0.25 means that 0.25*width of
            the model will be used.
        lyric_enc_blocks (`int`, *optional*, defaults to 32):
            Sequence of length seq_len is factored as [blocks, seq_len // blocks] in the `JukeboxAttention` layer.
        lyric_enc_init_scale (`list`, *optional*, defaults to [0.1, 0.4, 0.4]):
            Initialisation scales for the lyric encoder modules.
        lyric_enc_loss_fraction (`list`, *optional*, defaults to [0.4, 0.0, 0.0]):
            Multiplication factor used in front of the lyric encoder loss. Each value is for a particular level.
        lyric_enc_attn_order (`list`, *optional*, defaults to [2, 0, 0]):
            Which attention pattern to use for the lyric encoder.
        lyric_enc_attn_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for the post-attention layer dropout in the lyric encoder.
        lyric_enc_resid_dropout (`float`, *optional*, defaults to 0.0):
            Residual dropout used in the attention pattern of the lyric encoder.
        lyric_enc_emb_dropout (`float`, *optional*, defaults to 0.0):
            Embedding dropout used in the lyric encoder.
        lyric_enc_zero_out (`bool`, *optional*, defaults to False):
            Whether or not to set to zeros the weights the MLPs in the lyric encoder.
        lyric_enc_res_scale (`bool`, *optional*, defaults to False):
            Residual scaling factor used in the lyric encoder attention patterns.
        lyric_enc_n_vocab (`int`, *optional*, defaults to 79):
            Defines the number of different tokens that can be represented by the `inputs_ids` passed to the
            `lyric_encoder`
        prior_init_scale (`list`, *optional*, defaults to [0.2, 1, 1]):
            Initialisation scales for the prior modules.
        prior_spread (`bool`, *optional*, defaults to False):
            Spread used in the attention pattern
        prior_zero_out (`bool`, *optional*, defaults to False):
             Whether or not to set to zeros the weights the MLPs of the priors.
        prior_res_scale (`bool`, *optional*, defaults to False):
            Residual scaling factor used in every prior's attention layer.
        prior_n_ctx (`tuple`, *optional*, defaults to (6144, 8192, 8192)):
            Number of context tokens for each prior. The context tokens are the music tokens that are attended to when
            generating music tokens.
        prior_latent_dim (`int`, *optional*, defaults to 2048):
            Dimension of the latent music token space. Default value match the `vqvae_codebook_dimension`.
        prior_width (`list`, *optional*, defaults to [2048, 1920, 1920]):
            Input and output dimension of the attention layers of each prior.
        prior_m_attn (`float`, *optional*, defaults to 0.25):
            Multiplier coefficient used to define the hidden dimension of the attention layers. 0.25 means that
            0.25*prior_width of the model will be used.
        prior_depth (`list`, *optional*, defaults to [72, 72, 72]):
            Depth of each prior. Defines the number of `attn_block`.
        prior_n_heads (`list`, *optional*, defaults to [2, 1, 1]):
            Number of attention heads per prior.
        prior_attn_order (`list`, *optional*, defaults to [12, 2, 2]):
            Attention patterns to use in each prior. Depending on the value, cross attention, block attention and
            sparse attention blocks are stacked.
        prior_blocks (`int`, *optional*, defaults to 64):
            Sequence of length seq_len is factored as [blocks, seq_len // blocks] in the `JukeboxAttention` layer.
        prior_alignment_layer (`list`, *optional*, defaults to [68, None, None]):
            Layer corresponding to the alignemnt between the lyrics and the audio.
        prior_alignment_head (`list`, *optional*, defaults to [2, None, None]):
            Index of the attention head which takes care of the alignemnt between the lyrics and the audio.
        prior_attn_dropout (`int`, *optional*, defaults to 0):
            Dropout probability for the post-attention layer dropout of the prior models.
        prior_resid_dropout (`int`, *optional*, defaults to 0):
            Residual dropout probability used in the attention layers of the prior models.
        prior_emb_dropout (`int`, *optional*, defaults to 0):
            Dropout applied to the embedding layer of the priors.
        vqvae_levels (`int`, *optional*, defaults to 3):
            Number of hierachical levels that used in the VQVAE.
        vqvae_downs_t (`tuple`, *optional*, defaults to (3, 2, 2)):
            Downsampling rate for each level of the hierachical VQ-VAE.
        vqvae_strides_t (`tuple`, *optional*, defaults to (2, 2, 2)):
            Stride used for each level of the hierachical VQ-VAE.
        vqvae_emmbedding_width (`int`, *optional*, defaults to 64):
            Dimension of the codebook vectors.
        vqvae_codebook_dimension (`int`, *optional*, defaults to 2048):
            Number of codes to use in each of the VQVAE.
        vqvae_m_conv (`int`, *optional*, defaults to 1):
            Projection factor used in the `JukeboxResConv1DBlock`.
        vqvae_dilation_growth_rate (`int`, *optional*, defaults to 3):
            Resnet dilation growth rate used in the VQVAE (dilation_growth_rate ** depth)
        vqvae_dilation_cycle (`int`, *optional*, defaults to None):
            Dilation cycle value used in the `JukeboxResnet`. If an int is used, each new Conv1 block will have a depth
            of reduced by a power of `vqvae_dilation_cycle`.
        vqvae_multipliers (`tuple`, *optional*, defaults to (2, 1, 1)):
            Depth and width multipliers used for each level. Used on the `vqvae_conv_block_width` and
            `vqvae_conv_block_depth`
        vqvae_lmu (`float`, *optional*, defaults to 0.99):
            Used in the codebook update, exponential moving average coefficient. For more detail refer to Appendix A.1
            of the original [VQVAE paper](https://arxiv.org/pdf/1711.00937v2.pdf)
        vqvae_commit (`float`, *optional*, defaults to 0.02):
            Commit loss multiplier.
        vqvae_conv_block_depth (`int`, *optional*, defaults to 4):
            Depth of the encoder and decoder block. If no `vqvae_multipliers` are used, this is the same for each
            level.
        vqvae_conv_block_width (`int`, *optional*, defaults to 32):
            Width of the encoder and decoder block. If no `vqvae_multipliers` are used, this is the same for each
            level.
        vqvae_reverse_decoder_dilation (`int`, *optional*, defaults to 1):
            Whether or not to reverse the dilation rate for the decoder.
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
    attribute_map = {
        "hidden_size": "vqvae_codebook_dimension",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        sampling_rate=44100,
        metadata_dims=[(604, 7898), (120, 4111), (120, 4111)],
        nb_priors=3,
        timing_dims=64,
        single_enc_dec=[True, False, False],
        metadata_conditioning=True,
        merged_decoder=[True, False, False],
        lyric_conditioning=[True, False, False],
        nb_relevant_lyric_tokens=[384, 0, 0],
        min_duration=17.84,
        max_duration=600.0,
        max_nb_genres=5,
        init_std=0.2,
        hop_fraction=[0.125, 0.5, 0.5],
        cond_zero_out=False,
        cond_depth=[3, 16, 16],
        cond_width=[128, 1024, 1024],
        cond_dilation_growth_rate=[1, 3, 3],
        cond_dilation_cycle=[None, 8, 8],
        cond_res_scale=[None, True, False],
        cond_m_conv=1,
        cond_downs_t=(3, 2, 2),
        cond_strides_t=(2, 2, 2),
        lyric_enc_spread=None,
        lyric_enc_width=[128, 128, 128],
        lyric_enc_depth=[18, 3, 3],
        lyric_enc_heads=4,
        lyric_enc_m_attn=0.25,
        lyric_enc_m_mlp=1.0,
        lyric_enc_blocks=32,
        lyric_enc_init_scale=[0.1, 0.4, 0.4],
        lyric_enc_loss_fraction=[0.4, 0.0, 0.0],
        lyric_enc_attn_order=[2, 0, 0],
        lyric_enc_attn_dropout=0.0,
        lyric_enc_resid_dropout=0.0,
        lyric_enc_emb_dropout=0.0,
        lyric_enc_zero_out=False,
        lyric_enc_res_scale=False,
        lyric_enc_n_vocab=79,
        prior_init_scale=[0.2, 1, 1],
        prior_spread=None,
        prior_zero_out=False,
        prior_res_scale=False,
        prior_n_ctx=(6144, 8192, 8192),
        prior_latent_dim=2048,
        prior_width=[2048, 1920, 1920],
        prior_depth=[72, 72, 72],
        prior_n_heads=[2, 1, 1],
        prior_attn_order=[12, 2, 2],
        prior_blocks=64,
        prior_alignment_layer=[68, None, None],
        prior_alignment_head=[2, None, None],
        prior_m_attn=0.25,
        prior_attn_dropout=0,
        prior_resid_dropout=0,
        prior_emb_dropout=0,
        vqvae_levels=3,
        vqvae_downs_t=(3, 2, 2),
        vqvae_strides_t=(2, 2, 2),
        vqvae_emmbedding_width=64,
        vqvae_codebook_dimension=2048,
        vqvae_width=32,
        vqvae_depth=4,
        vqvae_m_conv=1,
        vqvae_dilation_growth_rate=3,
        vqvae_dilation_cycle=None,
        vqvae_multipliers=(2, 1, 1),
        vqvae_lmu=0.99,
        vqvae_commit=0.02,
        vqvae_conv_block_depth=4,
        vqvae_conv_block_width=32,
        vqvae_reverse_decoder_dilation=1,
        **kwargs,
    ):
        self.init_std = init_std
        self.nb_priors = nb_priors
        self.hop_fraction = hop_fraction

        #  Auto regressive (decoder) kwargs :
        self.prior_attn_order = prior_attn_order
        self.prior_n_heads = prior_n_heads
        self.prior_depth = prior_depth
        self.prior_width = prior_width
        self.prior_n_ctx = prior_n_ctx
        self.prior_latent_dim = prior_latent_dim
        self.prior_attn_dropout = prior_attn_dropout
        self.prior_resid_dropout = prior_resid_dropout
        self.prior_emb_dropout = prior_emb_dropout
        self.prior_zero_out = prior_zero_out
        self.prior_res_scale = prior_res_scale
        self.prior_blocks = prior_blocks
        self.prior_m_attn = prior_m_attn
        self.prior_spread = prior_spread
        self.prior_alignment_layer = prior_alignment_layer
        self.prior_alignment_head = prior_alignment_head
        self.prior_init_scale = prior_init_scale

        # Audio conditioning : upsampler parameters
        self.cond_depth = cond_depth
        self.cond_width = cond_width
        self.cond_dilation_growth_rate = cond_dilation_growth_rate
        self.cond_dilation_cycle = cond_dilation_cycle
        self.cond_zero_out = cond_zero_out
        self.cond_m_conv = cond_m_conv
        self.cond_res_scale = cond_res_scale
        self.cond_downs_t = cond_downs_t
        self.cond_strides_t = cond_strides_t

        # Metadata conditioning
        self.max_nb_genres = max_nb_genres
        self.sampling_rate = sampling_rate
        self.metadata_dims = metadata_dims
        self.timing_dims = timing_dims
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.metadata_conditioning = metadata_conditioning

        # Lyric conditioning
        self.merged_decoder = merged_decoder  # is this equivalent ?
        self.single_enc_dec = single_enc_dec
        self.lyric_conditioning = lyric_conditioning
        self.nb_relevant_lyric_tokens = nb_relevant_lyric_tokens

        self.lyric_enc_attn_dropout = lyric_enc_attn_dropout
        self.lyric_enc_attn_order = lyric_enc_attn_order
        self.lyric_enc_blocks = lyric_enc_blocks
        self.lyric_enc_depth = lyric_enc_depth
        self.lyric_enc_emb_dropout = lyric_enc_emb_dropout
        self.lyric_enc_heads = lyric_enc_heads
        self.lyric_enc_init_scale = lyric_enc_init_scale
        self.lyric_enc_loss_fraction = lyric_enc_loss_fraction
        self.lyric_enc_m_attn = lyric_enc_m_attn
        self.lyric_enc_m_mlp = lyric_enc_m_mlp
        self.lyric_enc_resid_dropout = lyric_enc_resid_dropout
        self.lyric_enc_res_scale = lyric_enc_res_scale
        self.lyric_enc_spread = lyric_enc_spread
        self.lyric_enc_width = lyric_enc_width
        self.lyric_enc_zero_out = lyric_enc_zero_out
        self.lyric_enc_n_vocab = lyric_enc_n_vocab

        # VQVAE parameters (all used)
        self.vqvae_levels = vqvae_levels
        self.vqvae_downs_t = vqvae_downs_t
        self.vqvae_strides_t = vqvae_strides_t
        self.vqvae_emmbedding_width = vqvae_emmbedding_width
        self.vqvae_codebook_dimension = vqvae_codebook_dimension
        self.vqvae_width = vqvae_width
        self.vqvae_depth = vqvae_depth
        self.vqvae_m_conv = vqvae_m_conv
        self.vqvae_dilation_growth_rate = vqvae_dilation_growth_rate
        self.vqvae_dilation_cycle = vqvae_dilation_cycle
        self.vqvae_multipliers = vqvae_multipliers
        self.vqvae_lmu = vqvae_lmu
        self.vqvae_commit = vqvae_commit
        self.vqvae_conv_block_depth = vqvae_conv_block_depth
        self.vqvae_conv_block_width = vqvae_conv_block_width
        self.vqvae_reverse_decoder_dilation = vqvae_reverse_decoder_dilation

        super().__init__(**kwargs)
