# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
""" EnCodec model configuration"""

import functools
import operator

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Matthijs/encodec_24khz": "https://huggingface.co/Matthijs/encodec_24khz/resolve/main/config.json",
    "Matthijs/encodec_48khz": "https://huggingface.co/Matthijs/encodec_48khz/resolve/main/config.json",
}


class EnCodecConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`EnCodecModel`]. It is used to instantiate a
    EnCodec model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [Matthijs/encodec_24khz](https://huggingface.co/Matthijs/encodec_24khz) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        TODO!!!!

        target_bandwidths (`List[float]`, *optional*):
            TODO
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        audio_channels (`int`, *optional*, defaults to 1):
            Number of channels in the audio data. Either 1 for mono or 2 for stereo.

        audio_normalize=False,
            TODO
        segment=None,
            TODO
        overlap=0.01,
            TODO

        dimension (int):
            Intermediate representation dimension.
        num_filters (int):
            Base width for the model.
        num_residual_layers (int):
            Number of residual layers.
        ratios (Sequence[int]):
            Kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order.
        activation (str):
            Activation function.
        activation_params (dict):
            Parameters to provide to the activation function.
        norm (`str`, *optional*, defaults to `"weight_norm"`):
            Normalization method.
        norm_params (`Dict[str, Any]`, *optional*):
            Parameters to provide to the underlying normalization used along with the convolution.
        final_activation (str):
            Final activation function after all convolutions.
        final_activation_params (dict):
            Parameters to provide to the activation function.
        kernel_size (int):
            Kernel size for the initial convolution.
        last_kernel_size (int):
            Kernel size for the initial convolution.
        residual_kernel_size (int):
            Kernel size for the residual layers.
        dilation_base (int):
            How much to increase the dilation with each layer.
        causal (`bool`, *optional*, defaults to True):
            Whether to use fully causal convolution.
        pad_mode (str):
            Padding mode for the convolutions.
        true_skip (bool):
            Whether to use true skip connection or a simple (streamable) convolution as the skip connection
            in the residual network blocks.
        compress (int):
            Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int):
            Number of LSTM layers at the end of the encoder.
        trim_right_ratio (float):
            Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.


===OLD STUFF===
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        encoder_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer decoder.
        decoder_layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        positional_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the text position encoding layers.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for activations inside the fully connected layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        feat_extract_norm (`str`, *optional*, defaults to `"group"`):
            The norm to be applied to 1D convolutional layers in the speech encoder pre-net. One of `"group"` for group
            normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
            convolutional layers.
        feat_proj_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for output of the speech encoder pre-net.
        feat_extract_activation (`str, `optional`, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the 1D convolutional layers of the feature
            extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        conv_dim (`Tuple[int]` or `List[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
            A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            speech encoder pre-net. The length of *conv_dim* defines the number of 1D convolutional layers.
        conv_stride (`Tuple[int]` or `List[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
            A tuple of integers defining the stride of each 1D convolutional layer in the speech encoder pre-net. The
            length of *conv_stride* defines the number of convolutional layers and has to match the length of
            *conv_dim*.
        conv_kernel (`Tuple[int]` or `List[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 3, 3)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the speech encoder pre-net.
            The length of *conv_kernel* defines the number of convolutional layers and has to match the length of
            *conv_dim*.
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether the 1D convolutional layers have a bias.
        num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
        apply_spec_augment (`bool`, *optional*, defaults to `True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the speech encoder pre-net. For
            reference see [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://arxiv.org/abs/1904.08779).
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procecure generates ''mask_time_prob*len(time_axis)/mask_time_length'' independent masks over the axis. If
            reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2),:
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks''
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procecure generates ''mask_feature_prob*len(feature_axis)/mask_time_length'' independent masks over
            the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0),:
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            ''mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks''
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel features used per input features. Used by the speech decoder pre-net. Should correspond to
            the value used in the [`SpeechT5Processor`] class.
        speech_decoder_prenet_layers (`int`, *optional*, defaults to 2):
            Number of layers in the speech decoder pre-net.
        speech_decoder_prenet_units (`int`, *optional*, defaults to 256):
            Dimensionality of the layers in the speech decoder pre-net.
        speech_decoder_prenet_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability for the speech decoder pre-net layers.
        speaker_embedding_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the *XVector* embedding vectors.
        speech_decoder_postnet_layers (`int`, *optional*, defaults to 5):
            Number of layers in the speech decoder post-net.
        speech_decoder_postnet_units (`int`, *optional*, defaults to 256):
            Dimensionality of the layers in the speech decoder post-net.
        speech_decoder_postnet_kernel (`int`, *optional*, defaults to 5):
            Number of convolutional filter channels in the speech decoder post-net.
        speech_decoder_postnet_dropout (`float`, *optional*, defaults to 0.5):
            The dropout probability for the speech decoder post-net layers.
        reduction_factor (`int`, *optional*, defaults to 2):
            Spectrogram length reduction factor for the speech decoder inputs.
        max_speech_positions (`int`, *optional*, defaults to 4000):
            The maximum sequence length of speech features that this model might ever be used with.
        max_text_positions (`int`, *optional*, defaults to 450):
            The maximum sequence length of text features that this model might ever be used with.
        encoder_max_relative_position (`int`, *optional*, defaults to 160):
            Maximum distance for relative position embedding in the encoder.
        use_guided_attention_loss (`bool`, *optional*, defaults to `True`):
            Whether to apply guided attention loss while training the TTS model.
        guided_attention_loss_num_heads (`int`, *optional*, defaults to 2):
            Number of attention heads the guided attention loss will be applied to. Use -1 to apply this loss to all
            attention heads.
        guided_attention_loss_sigma (`float`, *optional*, defaults to 0.4):
            Standard deviation for guided attention loss.
        guided_attention_loss_scale (`float`, *optional*, defaults to 10.0):
            Scaling coefficient for guided attention loss (also known as lambda).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import EnCodecModel, EnCodecConfig

    >>> # Initializing a "Matthijs/encodec_24khz" style configuration
    >>> configuration = EnCodecConfig()

    >>> # Initializing a model (with random weights) from the "Matthijs/encodec_24khz" style configuration
    >>> model = EnCodecModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "encodec"
    # attribute_map = {"num_attention_heads": "encoder_attention_heads", "num_hidden_layers": "encoder_layers"}

    def __init__(
        self,

        target_bandwidths=[1.5, 3.0, 6.0, 12.0, 24.0],
        sampling_rate=24_000,
        audio_channels=1,

        audio_normalize=False,
        segment=None,
        overlap=0.01,

        dimension=128,
        num_filters=32,
        num_residual_layers=1,
        ratios=[8, 5, 4, 2],
        activation="ELU",
        activation_params={"alpha": 1.0},
        norm="weight_norm",
        norm_params={},
        final_activation=None,
        final_activation_params=None,
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_base=2,
        causal=True,
        pad_mode="reflect",
        true_skip=False,
        compress=2,
        lstm=2,
        trim_right_ratio=1.0,






        # vocab_size=81,
        # hidden_size=768,
        # encoder_layers=12,
        # encoder_attention_heads=12,
        # encoder_ffn_dim=3072,
        # encoder_layerdrop=0.1,
        # decoder_layers=6,
        # decoder_ffn_dim=3072,
        # decoder_attention_heads=12,
        # decoder_layerdrop=0.1,
        # hidden_act="gelu",
        # positional_dropout=0.1,
        # hidden_dropout=0.1,
        # attention_dropout=0.1,
        # activation_dropout=0.1,
        # initializer_range=0.02,
        # layer_norm_eps=1e-5,
        # scale_embedding=False,
        # feat_extract_norm="group",
        # feat_proj_dropout=0.0,
        # feat_extract_activation="gelu",
        # conv_dim=(512, 512, 512, 512, 512, 512, 512),
        # conv_stride=(5, 2, 2, 2, 2, 2, 2),
        # conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        # conv_bias=False,
        # num_conv_pos_embeddings=128,
        # num_conv_pos_embedding_groups=16,
        # apply_spec_augment=True,
        # mask_time_prob=0.05,
        # mask_time_length=10,
        # mask_time_min_masks=2,
        # mask_feature_prob=0.0,
        # mask_feature_length=10,
        # mask_feature_min_masks=0,
        # pad_token_id=1,
        # bos_token_id=0,
        # eos_token_id=2,
        # decoder_start_token_id=2,
        # num_mel_bins=80,
        # speech_decoder_prenet_layers=2,
        # speech_decoder_prenet_units=256,
        # speech_decoder_prenet_dropout=0.5,
        # speaker_embedding_dim=512,
        # speech_decoder_postnet_layers=5,
        # speech_decoder_postnet_units=256,
        # speech_decoder_postnet_kernel=5,
        # speech_decoder_postnet_dropout=0.5,
        # reduction_factor=2,
        # max_speech_positions=4000,
        # max_text_positions=450,
        # encoder_max_relative_position=160,
        # use_guided_attention_loss=True,
        # guided_attention_loss_num_heads=2,
        # guided_attention_loss_sigma=0.4,
        # guided_attention_loss_scale=10.0,
        # use_cache=True,
        is_encoder_decoder=True,
        **kwargs,
    ):
        self.target_bandwidths = target_bandwidths
        self.sampling_rate = sampling_rate
        self.audio_channels = audio_channels

        self.audio_normalize = audio_normalize
        self.segment = segment
        self.overlap = overlap

        self.dimension = dimension
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.ratios = ratios
        self.activation = activation
        self.activation_params = activation_params
        self.norm = norm
        self.norm_params = norm_params
        self.final_activation = final_activation
        self.final_activation_params = final_activation_params
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.dilation_base = dilation_base
        self.causal = causal
        self.pad_mode = pad_mode
        self.true_skip = true_skip
        self.compress = compress
        self.lstm = lstm
        self.trim_right_ratio = trim_right_ratio






        # self.vocab_size = vocab_size
        # self.hidden_size = hidden_size
        # self.encoder_layers = encoder_layers
        # self.encoder_ffn_dim = encoder_ffn_dim
        # self.encoder_attention_heads = encoder_attention_heads
        # self.encoder_layerdrop = encoder_layerdrop
        # self.decoder_layers = decoder_layers
        # self.decoder_ffn_dim = decoder_ffn_dim
        # self.decoder_attention_heads = decoder_attention_heads
        # self.decoder_layerdrop = decoder_layerdrop
        # self.hidden_act = hidden_act
        # self.positional_dropout = positional_dropout
        # self.hidden_dropout = hidden_dropout
        # self.attention_dropout = attention_dropout
        # self.activation_dropout = activation_dropout
        # self.initializer_range = initializer_range
        # self.layer_norm_eps = layer_norm_eps
        # self.scale_embedding = scale_embedding

        # self.feat_extract_norm = feat_extract_norm
        # self.feat_proj_dropout = feat_proj_dropout
        # self.feat_extract_activation = feat_extract_activation
        # self.conv_dim = list(conv_dim)
        # self.conv_stride = list(conv_stride)
        # self.conv_kernel = list(conv_kernel)
        # self.conv_bias = conv_bias
        # self.num_conv_pos_embeddings = num_conv_pos_embeddings
        # self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        # self.num_feat_extract_layers = len(self.conv_dim)

        # if (
        #     (len(self.conv_stride) != self.num_feat_extract_layers)
        #     or (len(self.conv_kernel) != self.num_feat_extract_layers)
        #     or (len(self.conv_dim) != self.num_feat_extract_layers)
        # ):
        #     raise ValueError(
        #         "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
        #         " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
        #         f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
        #         f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
        #     )

        # # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        # self.apply_spec_augment = apply_spec_augment
        # self.mask_time_prob = mask_time_prob
        # self.mask_time_length = mask_time_length
        # self.mask_time_min_masks = mask_time_min_masks
        # self.mask_feature_prob = mask_feature_prob
        # self.mask_feature_length = mask_feature_length
        # self.mask_feature_min_masks = mask_feature_min_masks

        # self.num_mel_bins = num_mel_bins
        # self.speech_decoder_prenet_layers = speech_decoder_prenet_layers
        # self.speech_decoder_prenet_units = speech_decoder_prenet_units
        # self.speech_decoder_prenet_dropout = speech_decoder_prenet_dropout
        # self.speaker_embedding_dim = speaker_embedding_dim

        # self.speech_decoder_postnet_layers = speech_decoder_postnet_layers
        # self.speech_decoder_postnet_units = speech_decoder_postnet_units
        # self.speech_decoder_postnet_kernel = speech_decoder_postnet_kernel
        # self.speech_decoder_postnet_dropout = speech_decoder_postnet_dropout
        # self.reduction_factor = reduction_factor

        # self.max_speech_positions = max_speech_positions
        # self.max_text_positions = max_text_positions
        # self.encoder_max_relative_position = encoder_max_relative_position

        # self.use_guided_attention_loss = use_guided_attention_loss
        # self.guided_attention_loss_num_heads = guided_attention_loss_num_heads
        # self.guided_attention_loss_sigma = guided_attention_loss_sigma
        # self.guided_attention_loss_scale = guided_attention_loss_scale

        # self.use_cache = use_cache
        self.is_encoder_decoder = is_encoder_decoder

        super().__init__(
            # pad_token_id=pad_token_id,
            # bos_token_id=bos_token_id,
            # eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            # decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
