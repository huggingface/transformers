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
""" FastSpeech2Conformer model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class FastSpeech2ConformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerModel`]. It is used to instantiate an
    FastSpeech2Conformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the FastSpeech2Conformer
    [fastspeech2_conformer](https://huggingface.co/put a link here) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:


    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerConfig

    >>> # Initializing a FastSpeech2Conformer fastspeech2 style configuration
    >>> configuration = FastSpeech2ConformerConfig()

    >>> # Initializing a model from the fastspeech2 style configuration
    >>> model = FastSpeech2ConformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "fastspeech2_conformer"

    def __init__(
        self,
        acoustic_dim=384,  # from adim
        input_dim=62,  # from idim
        output_dim=80,  # from odim
        num_attention_heads=4,
        encoder_layers=6,
        encoder_linear_units=1536,
        decoder_layers=6,
        decoder_linear_units=1536,
        postnet_layers=5,
        postnet_chans=256,
        postnet_filts=5,
        positionwise_layer_type="conv1d",
        positionwise_conv_kernel_size=1,
        use_batch_norm=True,
        encoder_normalize_before=True,
        decoder_normalize_before=True,
        encoder_concat_after=False,
        decoder_concat_after=False,
        reduction_factor=1,
        # encoder / decoder
        use_macaron_style_in_conformer=True,
        use_cnn_in_conformer=True,
        conformer_enc_kernel_size=7,
        conformer_dec_kernel_size=31,
        # duration predictor
        duration_predictor_layers=2,
        duration_predictor_chans=256,
        duration_predictor_kernel_size=3,
        # energy predictor
        energy_predictor_layers=2,
        energy_predictor_chans=256,
        energy_predictor_kernel_size=3,
        energy_predictor_dropout=0.5,
        energy_embed_kernel_size=1,
        energy_embed_dropout=0.0,
        stop_gradient_from_energy_predictor=False,
        # pitch predictor
        pitch_predictor_layers=5,
        pitch_predictor_chans=256,
        pitch_predictor_kernel_size=5,
        pitch_predictor_dropout=0.5,
        pitch_embed_kernel_size=1,
        pitch_embed_dropout=0.0,
        stop_gradient_from_pitch_predictor=True,
        # training related
        encoder_dropout_rate=0.2,
        encoder_positional_dropout_rate=0.2,
        encoder_attention_dropout_rate=0.2,
        decoder_dropout_rate=0.2,
        decoder_positional_dropout_rate=0.2,
        decoder_attention_dropout_rate=0.2,
        duration_predictor_dropout_rate=0.2,
        postnet_dropout_rate=0.5,
        init_type="xavier_uniform",
        use_masking=True,
        use_weighted_masking=False,
        # additional features
        utt_embed_dim=None,  # confirm this, previously was 64
        lang_embs=None,  # confirm this, previously was 8000
        vocab_size=75,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        local_vars = locals()
        for var_name, var_value in local_vars.items():
            if "kernel_size" in var_name and var_value % 2 == 0:
                raise ValueError(f"`{var_name}` must be odd, but got {var_value} instead.")
        if acoustic_dim % num_attention_heads != 0:
            raise ValueError("The acoustic_dim must be evenly divisible by the number of attention heads.")
        if use_masking and use_weighted_masking:
            raise ValueError("Either use_masking or use_weighted_masking can be True, but not both.")

        super().__init__(bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        self.acoustic_dim = acoustic_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_attention_heads = num_attention_heads
        self.conformer_dec_kernel_size = conformer_dec_kernel_size
        self.conformer_enc_kernel_size = conformer_enc_kernel_size
        self.decoder_normalize_before = decoder_normalize_before
        self.decoder_layers = decoder_layers
        self.decoder_linear_units = decoder_linear_units
        self.duration_predictor_chans = duration_predictor_chans
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.duration_predictor_layers = duration_predictor_layers
        self.encoder_layers = encoder_layers
        self.encoder_normalize_before = encoder_normalize_before
        self.energy_embed_dropout = energy_embed_dropout
        self.energy_embed_kernel_size = energy_embed_kernel_size
        self.energy_predictor_chans = energy_predictor_chans
        self.energy_predictor_dropout = energy_predictor_dropout
        self.energy_predictor_kernel_size = energy_predictor_kernel_size
        self.energy_predictor_layers = energy_predictor_layers
        self.encoder_linear_units = encoder_linear_units
        self.init_type = init_type
        self.lang_embs = lang_embs
        self.pitch_embed_dropout = pitch_embed_dropout
        self.pitch_embed_kernel_size = pitch_embed_kernel_size
        self.pitch_predictor_chans = pitch_predictor_chans
        self.pitch_predictor_dropout = pitch_predictor_dropout
        self.pitch_predictor_kernel_size = pitch_predictor_kernel_size
        self.pitch_predictor_layers = pitch_predictor_layers
        self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
        self.positionwise_layer_type = positionwise_layer_type
        self.postnet_chans = postnet_chans
        self.postnet_dropout_rate = postnet_dropout_rate
        self.postnet_filts = postnet_filts
        self.postnet_layers = postnet_layers
        self.reduction_factor = reduction_factor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.decoder_attention_dropout_rate = decoder_attention_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.decoder_positional_dropout_rate = decoder_positional_dropout_rate
        self.encoder_attention_dropout_rate = encoder_attention_dropout_rate
        self.encoder_dropout_rate = encoder_dropout_rate
        self.encoder_positional_dropout_rate = encoder_positional_dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_cnn_in_conformer = use_cnn_in_conformer
        self.use_macaron_style_in_conformer = use_macaron_style_in_conformer
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.utt_embed_dim = utt_embed_dim
        self.encoder_concat_after = encoder_concat_after
        self.decoder_concat_after = decoder_concat_after
        self.duration_predictor_dropout_rate = duration_predictor_dropout_rate
        self.vocab_size = vocab_size

    @property
    def mel_dim(self):
        # Dimensionality of the prediced mel-spectrograms.
        return 80

    @property
    def var_pred_num_bins(self):
        # Number of bins in the variance predictors.
        return 256
