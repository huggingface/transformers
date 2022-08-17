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
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        sampling_rate=44100,
        metadata_dims=[(604, 7898), (120, 4111), (120, 4111)],
        copy_input=False,
        nb_priors=3,
        timing_dims=64,
        single_enc_dec=[True, False, False],
        metadata_conditioning=True,
        merged_decoder=[True, False, False],
        lyric_conditioning=[True, False, False],
        nb_relevant_lyric_tokens=[384, 0, 0],
        min_duration=17.84,
        max_duration=600.0,
        fp16_params=True,
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
        prime_spread=None,
        prime_width=[128, 128, 128],
        prime_depth=[18, 3, 3],
        prime_heads=4,
        prime_m_attn=0.25,
        prime_m_mlp=1.0,
        prime_blocks=32,
        prime_init_scale=[0.1, 0.4, 0.4],
        prime_loss_fraction=[0.4, 0.0, 0.0],
        prime_attn_order=[2, 0, 0],
        prime_attn_dropout=0.0,
        prime_resid_dropout=0.0,
        prime_emb_dropout=0.0,
        prime_zero_out=False,
        prime_res_scale=False,
        prime_pos_init=False,
        prime_n_vocab=79,
        prior_init_scale=[0.2, 1, 1],
        prior_spread=None,
        prior_zero_out=False,
        prior_res_scale=False,
        prior_pos_init=False,
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

        self.fp16_params = fp16_params
        self.init_std = init_std
        self.copy_input = copy_input
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
        self.prior_pos_init = prior_pos_init
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

        self.prime_attn_dropout = prime_attn_dropout
        self.prime_attn_order = prime_attn_order
        self.prime_blocks = prime_blocks
        self.prime_depth = prime_depth
        self.prime_emb_dropout = prime_emb_dropout
        self.prime_heads = prime_heads
        self.prime_init_scale = prime_init_scale
        self.prime_loss_fraction = prime_loss_fraction
        self.prime_m_attn = prime_m_attn
        self.prime_m_mlp = prime_m_mlp
        self.prime_pos_init = prime_pos_init
        self.prime_resid_dropout = prime_resid_dropout
        self.prime_res_scale = prime_res_scale
        self.prime_spread = prime_spread
        self.prime_width = prime_width
        self.prime_zero_out = prime_zero_out
        self.prime_n_vocab = prime_n_vocab

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
