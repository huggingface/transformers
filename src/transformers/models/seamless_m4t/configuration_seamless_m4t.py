# coding=utf-8
# Copyright 2022 ylacombe and The HuggingFace Inc. team. All rights reserved.
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
""" SeamlessM4T model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ylacombe/hf-seamless-m4t-medium": "https://huggingface.co/ylacombe/hf-seamless-m4t-medium/resolve/main/config.json",
    # See all SeamlessM4T models at https://huggingface.co/models?filter=seamless_m4t
}


# TODO: docstrings is a mix of wav2vec2_conformer, mBart, nllb
class SeamlessM4TConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~SeamlessM4TModel`]. It is used to instantiate an
    SeamlessM4T model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SeamlessM4T
    ["ylacombe/hf-seamless-m4t-medium"](https://huggingface.co/"ylacombe/hf-seamless-m4t-medium") architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 256102):
            Vocabulary size of the SeamlessM4T model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`~SeamlessM4TModel`], [`~SeamlessM4TForSpeechToSpeech`],
            [`~SeamlessM4TForSpeechToText`], [`~SeamlessM4TForTextToSpeech`] or [`~SeamlessM4TForTextToText`].
        unit_vocab_size (`int`, *optional*, defaults to 10082):
            Unit vocabulary size of the SeamlessM4T model. Defines the number of different unit tokens that can be
            represented by the `inputs_ids` passed when calling the Text-To-Units sub-model of [`~SeamlessM4TModel`],
            [`~SeamlessM4TForSpeechToSpeech`], [`~SeamlessM4TForSpeechToText`], [`~SeamlessM4TForTextToSpeech`] or
            [`~SeamlessM4TForTextToText`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the "intermediate" layers in the architecture.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model text encoder and decoder might ever be used with. Typically set
            this to something large just in case (e.g., 512 or 1024 or 2048).
        is_encoder_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as an encoder/decoder or not.
        encoder_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer text encoder.
        encoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text encoder.
        encoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text encoder.
        decoder_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer text decoder.
        decoder_ffn_dim (`int`, *optional*, defaults to 8192):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer text decoder.



        speech_encoder_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer speech encoder.
        speech_encoder_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer speech encoder.
        speech_encoder_intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer speech encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.

        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~SeamlessM4TModel`] or
            [`~TFSeamlessM4TModel`].



        model_in_dim (`int`, *optional*, defaults to 80):
            The number of frequency bins in the input log-mel spectrogram.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
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
            multi-receptive field fusion (MRF) module..
        leaky_relu_slope (`float`, *optional*, defaults to 0.1):
            The angle of the negative slope used by the leaky ReLU activation.
        Example:

    ```python
    >>> from transformers import SeamlessM4TModel, SeamlessM4TConfig

    >>> # Initializing a SeamlessM4T "ylacombe/hf-seamless-m4t-medium" style configuration
    >>> configuration = SeamlessM4TConfig()

    >>> # Initializing a model from the "ylacombe/hf-seamless-m4t-medium" style configuration
    >>> model = SeamlessM4TModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "seamless_m4t"

    def __init__(
        self,
        vocab_size=256102,
        unit_vocab_size=10082,
        # overall_config
        hidden_size=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        max_position_embeddings=1024,
        is_encoder_decoder=True,
        # left to add
        # text|unit encoder|decoder
        encoder_layers=24,
        encoder_ffn_dim=8192,
        encoder_attention_heads=16,
        decoder_layers=24,
        decoder_ffn_dim=8192,
        decoder_attention_heads=16,
        encoder_layerdrop=0.05,
        decoder_layerdrop=0.05,
        activation_function="relu",
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=3,
        scale_embedding=True,
        max_new_tokens=256,
        # speech_encoder
        speech_encoder_layers=24,
        speech_encoder_attention_heads=16,
        speech_encoder_intermediate_size=4096,
        speech_encoder_hidden_act="swish",
        speech_encoder_dropout=0.0,
        add_adapter=True,
        layerdrop=0.1,
        conv_dim=(512, 512, 512, 512, 512, 512, 160),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout=0.1,
        num_adapter_layers=1,
        output_hidden_size=None,
        position_embeddings_type="relative",
        rotary_embedding_base=10000,
        max_source_positions=4096,
        conv_depthwise_kernel_size=31,
        # t2u config
        t2u_bos_token_id=0,
        t2u_pad_token_id=1,
        t2u_eos_token_id=2,
        t2u_decoder_start_token_id=2,
        t2u_max_new_tokens=1024,
        t2u_encoder_layers=6,
        t2u_encoder_ffn_dim=8192,
        t2u_encoder_attention_heads=16,
        t2u_decoder_layers=6,
        t2u_decoder_ffn_dim=8192,
        t2u_decoder_attention_heads=16,
        t2u_num_langs=38,
        hidden_act="gelu",
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        # hifi-gan vocoder config
        model_in_dim=1792,
        sampling_rate=16000,
        upsample_initial_channel=512,
        upsample_rates=[5, 4, 4, 2, 2],
        upsample_kernel_sizes=[11, 8, 8, 4, 4],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        leaky_relu_slope=0.1,
        # specific to Code Hifi-Gan
        unit_hifi_gan_vocab_size=10000,
        unit_embed_dim=1280,
        lang_embed_dim=256,
        spkr_embed_dim=256,
        vocoder_num_langs=36,
        vocoder_num_spkrs=200,
        variance_predictor_kernel_size=3,
        var_pred_dropout=0.5,
        **kwargs,
    ):
        # overall_config
        self.vocab_size = vocab_size
        self.unit_vocab_size = unit_vocab_size
        self.hidden_size = hidden_size
        self.speech_encoder_intermediate_size = speech_encoder_intermediate_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.layerdrop = layerdrop
        self.max_new_tokens = max_new_tokens

        # text|unit encoder|decoder
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.init_std = init_std
        self.scale_embedding = scale_embedding

        # speech_encoder
        self.speech_encoder_layers = speech_encoder_layers
        self.speech_encoder_hidden_act = speech_encoder_hidden_act
        self.speech_encoder_dropout = speech_encoder_dropout
        self.speech_encoder_attention_heads = speech_encoder_attention_heads

        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.adaptor_kernel_size = adaptor_kernel_size
        self.adaptor_stride = adaptor_stride
        self.adaptor_layer_norm = adaptor_layer_norm
        self.adaptor_dropout = adaptor_dropout
        self.num_adapter_layers = num_adapter_layers
        self.output_hidden_size = output_hidden_size
        self.position_embeddings_type = position_embeddings_type
        self.rotary_embedding_base = rotary_embedding_base
        self.max_source_positions = max_source_positions
        self.conv_depthwise_kernel_size = conv_depthwise_kernel_size
        self.add_adapter = add_adapter

        # t2u config
        self.t2u_bos_token_id = t2u_bos_token_id
        self.t2u_pad_token_id = t2u_pad_token_id
        self.t2u_eos_token_id = t2u_eos_token_id
        self.t2u_decoder_start_token_id = t2u_decoder_start_token_id
        self.t2u_max_new_tokens = t2u_max_new_tokens
        self.hidden_act = hidden_act
        self.t2u_num_langs = t2u_num_langs
        # self.type_vocab_size = type_vocab_size
        self.t2u_encoder_layers = t2u_encoder_layers
        self.t2u_encoder_ffn_dim = t2u_encoder_ffn_dim
        self.t2u_encoder_attention_heads = t2u_encoder_attention_heads
        self.t2u_decoder_layers = t2u_decoder_layers
        self.t2u_decoder_ffn_dim = t2u_decoder_ffn_dim
        self.t2u_decoder_attention_heads = t2u_decoder_attention_heads

        # hifi-gan vocoder config
        # original parameters specific to Hifi-Gan
        self.model_in_dim = model_in_dim
        self.sampling_rate = sampling_rate
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.leaky_relu_slope = leaky_relu_slope

        # TODO: add to docstrings
        # specific to Code Hifi-Gan
        self.unit_hifi_gan_vocab_size = unit_hifi_gan_vocab_size
        self.unit_embed_dim = unit_embed_dim
        self.lang_embed_dim = lang_embed_dim
        self.spkr_embed_dim = spkr_embed_dim
        self.vocoder_num_langs = vocoder_num_langs
        self.vocoder_num_spkrs = vocoder_num_spkrs
        self.variance_predictor_kernel_size = variance_predictor_kernel_size
        self.var_pred_dropout = var_pred_dropout
        
        # for proper config init
        self.num_attention_heads = decoder_attention_heads
        self.num_hidden_layers = decoder_layers

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )
