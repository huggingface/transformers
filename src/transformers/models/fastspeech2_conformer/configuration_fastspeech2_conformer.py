# coding=utf-8
# Copyright 2023 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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

FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "fastspeech2_conformer": "https://huggingface.co/jaketae/fastspeech2-ljspeech/resolve/main/config.json",
    "fastspeech2_conformer": "https://huggingface.co/jaketae/fastspeech2-commonvoice/resolve/main/config.json",
    # See all FastSpeech2Conformer models at https://huggingface.co/models?filter=fastspeech2
}


class FastSpeech2ConformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerModel`]. It is used to instantiate an
    FastSpeech2Conformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the FastSpeech2Conformer
    [fastspeech2](https://huggingface.co/jaketae/fastspeech2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 75):
            Vocabulary size of the FastSpeech2Conformer model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`~FastSpeech2ConformerModel`].
        encoder_embed_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the encoder layers.
        encoder_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the encoder.
        decoder_embed_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the decoder layers.
        decoder_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 2):
            Number of attention heads for each attention layer in the decoder.
        attention_dropout (`float`, *optional*, defaults to 0)
            The dropout ratio for the attention probabilities.
        fft_hidden_dim (`int`, *optional*, defaults to 1024)
            Dimensionality of the feed forward layers.
        fft_kernel_size (`int`, *optional*, defaults to 9)
            Kernel size of the feed forward layers.
        fft_dropout (`float`, *optional*, defaults to 0.5)
            The dropout ratio for the feedforward layers.
        var_pred_hidden_dim (`int`, *optional*, defaults to 256)
            Dimensionality of the hidden size of the variance predictor.
        var_pred_kernel_size (`int`, *optional*, defaults to 3)
            Kernel size of the variance predictor layer.
        var_pred_dropout (`float`, *optional*, defaults to 0.5)
            The dropout ratio for the variance predictor.
        add_postnet (`bool`, *optional*, defaults to `False`)
            Flag that specifies whether or not to add postnet.
        postnet_conv_dim (`int`, *optional*, defaults to 512)
            Dimensionality of the postnet convolution layers.
        postnet_conv_kernel_size (`int`, *optional*, defaults to 5)
            Kernel size of the convolution layers in the postnet.
        postnet_layers (`int`, *optional*, defaults to 5)
            Number of hidden layers in the postnet.
        postnet_dropout (`float`, *optional*, defaults to 0.5)
            The dropout ratio for the postnet.
        pitch_min (`float`, *optional*, defaults to -4.660287183665281)
            The minimum pitch value of the pitch bucket in the variance predictor.
        pitch_max (`float`, *optional*, defaults to 5.733940816898645)
            The maximum pitch value of the pitch bucket in the variance predictor.
        energy_min (`float`, *optional*, defaults to -4.9544901847839355)
            The minimum energy value of the pitch bucket in the variance predictor.
        energy_max (`float`, *optional*, defaults to 3.2244551181793213)
            The maximum energy value of the pitch bucket in the variance predictor.
        speaker_embed_dim (`int`, *optional*, defaults to 64)
            Dimensionality of the speaker identity embedding.
        num_speakers (`int`, *optional*, defaults to 1)
            Number of speakers. Set to 1 if the dataset is a single-speaker dataset. Otherwise, set to the number of
            speakers in the multi-speaker training dataset.
        max_source_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.0625):
            The standard deviation of the truncated_normal_initializer for initializing all embedding weight matrices.
        use_mean (`bool`, *optional*, defaults to `True`)
            Flag that specifies whether or not to denormalize predicted output using cepstral mean and variance
            normalization. For more information, please refer to [cepstral mean and variance
            normalization](https://en.wikipedia.org/wiki/Cepstral_mean_and_variance_normalization).
        use_standard_deviation (`bool`, *optional*, defaults to `True`)
            Flag that specifies whether or not to scale predicted output using cepstral mean and variance
            normalization. For more information, please refer to [cepstral mean and variance
            normalization](https://en.wikipedia.org/wiki/Cepstral_mean_and_variance_normalization).

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
        vocab_size=75,
        encoder_embed_dim=256,
        encoder_attention_heads=2,
        encoder_layers=4,
        decoder_embed_dim=256,
        decoder_attention_heads=2,
        decoder_layers=4,
        attention_dropout=0,
        fft_hidden_dim=1024,
        fft_kernel_size=9,
        fft_dropout=0.2,
        var_pred_hidden_dim=256,
        var_pred_kernel_size=3,
        var_pred_dropout=0.5,
        add_postnet=False,
        postnet_conv_dim=512,
        postnet_conv_kernel_size=5,
        postnet_layers=5,
        postnet_dropout=0.5,
        pitch_min=-4.660287183665281,
        pitch_max=5.733940816898645,
        energy_min=-4.9544901847839355,
        energy_max=3.2244551181793213,
        speaker_embed_dim=64,
        num_speakers=1,
        use_mean=True,
        use_standard_deviation=True,
        max_source_positions=1024,
        initializer_range=0.0625,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        if fft_kernel_size % 2 == 0:
            raise ValueError(f"`fft_kernel_size` must be odd, but got {fft_kernel_size} instead.")
        if postnet_conv_kernel_size % 2 == 0:
            raise ValueError(f"`postnet_conv_kernel_size` must be odd, but got {postnet_conv_kernel_size} instead.")
        if var_pred_kernel_size % 2 == 0:
            raise ValueError(f"`var_pred_kernel_size` must be odd, but got {var_pred_kernel_size} instead.")
        super().__init__(bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layers = encoder_layers
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_layers = decoder_layers
        self.attention_dropout = attention_dropout
        self.fft_hidden_dim = fft_hidden_dim
        self.fft_kernel_size = fft_kernel_size
        self.fft_dropout = fft_dropout
        self.var_pred_hidden_dim = var_pred_hidden_dim
        self.var_pred_kernel_size = var_pred_kernel_size
        self.var_pred_dropout = var_pred_dropout
        self.add_postnet = add_postnet
        self.postnet_conv_dim = postnet_conv_dim
        self.postnet_conv_kernel_size = postnet_conv_kernel_size
        self.postnet_layers = postnet_layers
        self.postnet_dropout = postnet_dropout
        self.pitch_max = pitch_max
        self.pitch_min = pitch_min
        self.energy_max = energy_max
        self.energy_min = energy_min
        self.speaker_embed_dim = speaker_embed_dim
        self.num_speakers = num_speakers
        self.max_source_positions = max_source_positions
        self.initializer_range = initializer_range
        self.use_mean = use_mean
        self.use_standard_deviation = use_standard_deviation

    @property
    def mel_dim(self):
        # Dimensionality of the prediced mel-spectrograms.
        return 80

    @property
    def var_pred_num_bins(self):
        # Number of bins in the variance predictors.
        return 256
