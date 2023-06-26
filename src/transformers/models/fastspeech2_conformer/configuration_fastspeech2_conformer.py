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


FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "connor-henderson/fastspeech2_conformer": "https://huggingface.co/connor-henderson/fastspeech2_conformer/raw/main/config.json",
}

FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "connor-henderson/fastspeech2_conformer_hifigan": "https://huggingface.co/connor-henderson/fastspeech2_conformer_hifigan/raw/main/config.json",
}


class FastSpeech2ConformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerModel`]. It is used to
    instantiate an FastSpeech2Conformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer
    [connor-henderson/fastspeech2_conformer](https://huggingface.co/connor-henderson/fastspeech2_conformer)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerModel, FastSpeech2ConformerConfig

    >>> # Initializing a FastSpeech2Conformer style configuration
    >>> configuration = FastSpeech2ConformerConfig()

    >>> # Initializing a model from the FastSpeech2Conformer style configuration
    >>> model = FastSpeech2ConformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "fastspeech2_conformer"
    attribute_map = {"num_hidden_layers": "encoder_layers"}

    def __init__(
        self,
        hidden_size=384,
        input_dim=78,
        num_mel_bins=80,
        num_attention_heads=2,
        encoder_layers=4,
        encoder_linear_units=1536,
        decoder_layers=4,
        decoder_linear_units=1536,
        speech_decoder_postnet_layers=5,
        speech_decoder_postnet_units=256,
        speech_decoder_postnet_kernel=5,
        positionwise_conv_kernel_size=3,
        use_batch_norm=True,
        encoder_normalize_before=False,
        decoder_normalize_before=False,
        encoder_concat_after=False,
        decoder_concat_after=False,
        reduction_factor=1,
        use_macaron_style_in_conformer=True,
        use_cnn_in_conformer=True,
        encoder_kernel_size=7,
        decoder_kernel_size=31,
        duration_predictor_layers=2,
        duration_predictor_channels=256,
        duration_predictor_kernel_size=3,
        energy_predictor_layers=2,
        energy_predictor_channels=256,
        energy_predictor_kernel_size=3,
        energy_predictor_dropout=0.5,
        energy_embed_kernel_size=1,
        energy_embed_dropout=0.0,
        stop_gradient_from_energy_predictor=False,
        pitch_predictor_layers=5,
        pitch_predictor_channels=256,
        pitch_predictor_kernel_size=5,
        pitch_predictor_dropout=0.5,
        pitch_embed_kernel_size=1,
        pitch_embed_dropout=0.0,
        stop_gradient_from_pitch_predictor=True,
        encoder_dropout_rate=0.2,
        encoder_positional_dropout_rate=0.2,
        encoder_attention_dropout_rate=0.2,
        decoder_dropout_rate=0.2,
        decoder_positional_dropout_rate=0.2,
        decoder_attention_dropout_rate=0.2,
        duration_predictor_dropout_rate=0.2,
        speech_decoder_postnet_dropout=0.5,
        use_masking=True,
        use_weighted_masking=False,
        utt_embed_dim=None,
        lang_embs=None,
        vocab_size=78,
        is_encoder_decoder=True,
        pad_token_id=0,
        bos_token_id=77,
        eos_token_id=77,
        **kwargs,
    ):
        if positionwise_conv_kernel_size % 2 == 0:
            raise ValueError(
                f"positionwise_conv_kernel_size must be odd, but got {positionwise_conv_kernel_size} instead."
            )
        if encoder_kernel_size % 2 == 0:
            raise ValueError(f"encoder_kernel_size must be odd, but got {encoder_kernel_size} instead.")
        if decoder_kernel_size % 2 == 0:
            raise ValueError(f"decoder_kernel_size must be odd, but got {decoder_kernel_size} instead.")
        if duration_predictor_kernel_size % 2 == 0:
            raise ValueError(
                f"duration_predictor_kernel_size must be odd, but got {duration_predictor_kernel_size} instead."
            )
        if energy_predictor_kernel_size % 2 == 0:
            raise ValueError(
                f"energy_predictor_kernel_size must be odd, but got {energy_predictor_kernel_size} instead."
            )
        if energy_embed_kernel_size % 2 == 0:
            raise ValueError(f"energy_embed_kernel_size must be odd, but got {energy_embed_kernel_size} instead.")
        if pitch_predictor_kernel_size % 2 == 0:
            raise ValueError(
                f"pitch_predictor_kernel_size must be odd, but got {pitch_predictor_kernel_size} instead."
            )
        if pitch_embed_kernel_size % 2 == 0:
            raise ValueError(f"pitch_embed_kernel_size must be odd, but got {pitch_embed_kernel_size} instead.")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden_size must be evenly divisible by num_attention_heads.")
        if use_masking and use_weighted_masking:
            raise ValueError("Either use_masking or use_weighted_masking can be True, but not both.")

        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.num_mel_bins = num_mel_bins
        self.num_attention_heads = num_attention_heads
        self.decoder_kernel_size = decoder_kernel_size
        self.encoder_kernel_size = encoder_kernel_size
        self.decoder_normalize_before = decoder_normalize_before
        self.decoder_layers = decoder_layers
        self.decoder_linear_units = decoder_linear_units
        self.duration_predictor_channels = duration_predictor_channels
        self.duration_predictor_kernel_size = duration_predictor_kernel_size
        self.duration_predictor_layers = duration_predictor_layers
        self.encoder_layers = encoder_layers
        self.encoder_normalize_before = encoder_normalize_before
        self.energy_embed_dropout = energy_embed_dropout
        self.energy_embed_kernel_size = energy_embed_kernel_size
        self.energy_predictor_channels = energy_predictor_channels
        self.energy_predictor_dropout = energy_predictor_dropout
        self.energy_predictor_kernel_size = energy_predictor_kernel_size
        self.energy_predictor_layers = energy_predictor_layers
        self.encoder_linear_units = encoder_linear_units
        self.lang_embs = lang_embs
        self.pitch_embed_dropout = pitch_embed_dropout
        self.pitch_embed_kernel_size = pitch_embed_kernel_size
        self.pitch_predictor_channels = pitch_predictor_channels
        self.pitch_predictor_dropout = pitch_predictor_dropout
        self.pitch_predictor_kernel_size = pitch_predictor_kernel_size
        self.pitch_predictor_layers = pitch_predictor_layers
        self.positionwise_conv_kernel_size = positionwise_conv_kernel_size
        self.speech_decoder_postnet_units = speech_decoder_postnet_units
        self.speech_decoder_postnet_dropout = speech_decoder_postnet_dropout
        self.speech_decoder_postnet_kernel = speech_decoder_postnet_kernel
        self.speech_decoder_postnet_layers = speech_decoder_postnet_layers
        self.should_postnet_compute_logits = False
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
        self.is_encoder_decoder = is_encoder_decoder

        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class FastSpeech2ConformerHifiGanConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastSpeech2ConformerHifiGanModel`]. It is used to
    instantiate a FastSpeech2Conformer HiFi-GAN vocoder model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2Conformer
    [connor-henderson/fastspeech2_conformer_hifigan](https://huggingface.co/connor-henderson/fastspeech2_conformer_hifigan)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 512):
            The number of input channels into the upsampling network.
        upsample_rates (`Tuple[int]` or `List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
            length of *upsample_rates* defines the number of convolutional layers and has to match the length of
            *upsample_kernel_sizes*.
        upsample_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[8, 8, 8, 8]`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
            length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match the length of
            *upsample_rates*.
        resblock_kernel_sizes (`Tuple[int]` or `List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
            fusion (MRF) module.
        resblock_dilation_sizes (`Tuple[Tuple[int]]` or `List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
            multi-receptive field fusion (MRF) module.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        normalize_before (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the spectrogram before vocoding using the vocoder's learned mean and variance.

    Example:

    ```python
    >>> from transformers import FastSpeech2ConformerHifiGan, FastSpeech2ConformerHifiGanConfig

    >>> # Initializing a FastSpeech2ConformerHifiGan configuration
    >>> configuration = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FastSpeech2ConformerHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "hifigan"

    def __init__(
        self,
        model_in_dim=80,
        upsample_initial_channel=512,
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        initializer_range=0.01,
        leaky_relu_slope=0.1,
        normalize_before=True,
        **kwargs,
    ):
        self.model_in_dim = model_in_dim
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.initializer_range = initializer_range
        self.leaky_relu_slope = leaky_relu_slope
        self.normalize_before = normalize_before
        super().__init__(**kwargs)
