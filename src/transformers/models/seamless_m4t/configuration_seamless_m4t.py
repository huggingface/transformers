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
"""SeamlessM4T model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/hf-seamless-m4t-medium")
@strict
class SeamlessM4TConfig(PreTrainedConfig):
    r"""
    t2u_vocab_size (`int`, *optional*, defaults to 10082):
        Unit vocabulary size of the SeamlessM4T model. Defines the number of different unit tokens that can be
        represented by the `inputs_ids` passed when calling the Text-To-Units sub-model of [`~SeamlessM4TModel`],
        [`~SeamlessM4TForSpeechToSpeech`] or [`~SeamlessM4TForTextToSpeech`].
    max_new_tokens (`int`, *optional*, defaults to 256):
        The maximum numbers of text tokens to generate, ignoring the number of tokens in the prompt.
    speech_encoder_layers (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer speech encoder.
    speech_encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer speech encoder.
    speech_encoder_intermediate_size (`int`, *optional*, defaults to 4096):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer speech encoder.
    speech_encoder_hidden_act (`str` or `function`, *optional*, defaults to `"swish"`):
        The non-linear activation function (function or string) in the speech encoder. If string, `"gelu"`,
        `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
    speech_encoder_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for all layers in the speech encoder.
    add_adapter (`bool`, *optional*, defaults to `True`):
        Add an adapter layer on top of the speech encoder.
    speech_encoder_layerdrop (`float`, *optional*, defaults to 0.1):
        The LayerDrop probability for the speech encoder. See the [LayerDrop paper](see
        https://huggingface.co/papers/1909.11556) for more details.
    feature_projection_input_dim (`int`, *optional*, defaults to 160):
        Input dimension of the input feature projection of the speech encoder, i.e the dimension after processing
        input audios with [`SeamlessM4TFeatureExtractor`].
    num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
        Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
        embeddings layer of the speech encoder.
    num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
        Number of groups of 1D convolutional positional embeddings layer of the speech encoder.
    adaptor_kernel_size (`int`, *optional*, defaults to 8):
        Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
    adaptor_stride (`int`, *optional*, defaults to 8):
        Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
    adaptor_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all layers in the speech adapter.
    num_adapter_layers (`int`, *optional*, defaults to 1):
        Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is
        True`.
    position_embeddings_type (`str`, *optional*, defaults to `"relative"`):
        Can be specified to `relative` or `rotary` for relative or rotary position embeddings respectively. If left
        `None` no relative position embedding is applied. Only applied to the speech encoder.
    rotary_embedding_base (`int`, *optional*, defaults to 10000):
        If `"rotary"` position embeddings are used, defines the size of the embedding base. Only applied to the
        speech encoder.
    max_source_positions (`int`, *optional*, defaults to 4096):
        if `"relative"` position embeddings are used, defines the maximum source input positions. Only applied to
        the speech encoder.
    conv_depthwise_kernel_size (`int`, *optional*, defaults to 31):
        Kernel size of convolutional depthwise 1D layer in Conformer blocks. Only applied to the speech encoder.
    t2u_bos_token_id (`int`, *optional*, defaults to 0):
        The id of the _beginning-of-stream_ unit token. Only applied to the text-to-unit seq2seq model.
    t2u_pad_token_id (`int`, *optional*, defaults to 1):
        The id of the _padding_ unit token. Only applied to the text-to-unit seq2seq model.
    t2u_eos_token_id (`int`, *optional*, defaults to 2):
        The id of the _end-of-stream_ unit token. Only applied to the text-to-unit seq2seq model.
    t2u_decoder_start_token_id (`int`, *optional*, defaults to 2):
        If an encoder-decoder model starts decoding with a different token than _bos_, the id of that token. Only
        applied to the text-to-unit seq2seq model.
    t2u_max_new_tokens (`int`, *optional*, defaults to 1024):
        The maximum numbers of unit tokens to generate, ignoring the number of tokens in the prompt. Only applied
        to the text-to-unit seq2seq model.
    t2u_encoder_layers (`int`, *optional*, defaults to 6):
        Number of hidden layers in the Transformer text-to-unit encoder.
    t2u_encoder_ffn_dim (`int`, *optional*, defaults to 8192):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text-to-unit encoder.
    t2u_encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer text-to-unit encoder.
    t2u_decoder_layers (`int`, *optional*, defaults to 6):
        Number of hidden layers in the Transformer text-to-unit decoder.
    t2u_decoder_ffn_dim (`int`, *optional*, defaults to 8192):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer text-to-unit decoder.
    t2u_decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer text-to-unit decoder.
    t2u_max_position_embeddings (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model text-to-unit component might ever be used with. Typically set
        this to something large just in case (e.g., 512 or 1024 or 2048).
    upsample_initial_channel (`int`, *optional*, defaults to 512):
        The number of input channels into the hifi-gan upsampling network. Applies to the vocoder only.
    upsample_rates (`tuple[int]` or `list[int]`, *optional*, defaults to `[5, 4, 4, 2, 2]`):
        A tuple of integers defining the stride of each 1D convolutional layer in the vocoder upsampling network.
        The length of *upsample_rates* defines the number of convolutional layers and has to match the length of
        *upsample_kernel_sizes*. Applies to the vocoder only.
    upsample_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[11, 8, 8, 4, 4]`):
        A tuple of integers defining the kernel size of each 1D convolutional layer in the vocoder upsampling
        network. The length of *upsample_kernel_sizes* defines the number of convolutional layers and has to match
        the length of *upsample_rates*. Applies to the vocoder only.
    resblock_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`):
        A tuple of integers defining the kernel sizes of the vocoder 1D convolutional layers in the multi-receptive
        field fusion (MRF) module. Applies to the vocoder only.
    resblock_dilation_sizes (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
        A nested tuple of integers defining the dilation rates of the vocoder dilated 1D convolutional layers in
        the multi-receptive field fusion (MRF) module. Applies to the vocoder only.
    leaky_relu_slope (`float`, *optional*, defaults to 0.1):
        The angle of the negative slope used by the leaky ReLU activation in the vocoder. Applies to the vocoder
        only.
    unit_hifi_gan_vocab_size (`int`, *optional*, defaults to 10000):
        Vocabulary size of the SeamlessM4T vocoder. Defines the number of different unit tokens that can be
        represented by the `inputs_ids` passed when calling the vocoder of [`~SeamlessM4TModel`],
        [`~SeamlessM4TForSpeechToSpeech`] or [`~SeamlessM4TForTextToSpeech`].
    unit_embed_dim (`int`, *optional*, defaults to 1280):
        The projection dimension of the input ids given to the hifi-gan vocoder. Applies to the vocoder only.
    lang_embed_dim (`int`, *optional*, defaults to 256):
        The projection dimension of the target language given to the hifi-gan vocoder. Applies to the vocoder only.
    spkr_embed_dim (`int`, *optional*, defaults to 256):
        The projection dimension of the speaker id given to the hifi-gan vocoder. Applies to the vocoder only.
    vocoder_num_langs (`int`, *optional*, defaults to 36):
        Number of langs supported by the vocoder. Might be different from `t2u_num_langs`.
    vocoder_num_spkrs (`int`, *optional*, defaults to 200):
        Number of speakers supported by the vocoder.
    variance_predictor_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size of the duration predictor. Applies to the vocoder only.
    var_pred_dropout (`float`, *optional*, defaults to 0.5):
        The dropout probability of the duration predictor. Applies to the vocoder only.
    vocoder_offset (`int`, *optional*, defaults to 4):
        Offset the unit token ids by this number to account for symbol tokens. Applies to the vocoder only.

    ```python
    >>> from transformers import SeamlessM4TModel, SeamlessM4TConfig

    >>> # Initializing a SeamlessM4T "facebook/hf-seamless-m4t-medium" style configuration
    >>> configuration = SeamlessM4TConfig()

    >>> # Initializing a model from the "facebook/hf-seamless-m4t-medium" style configuration
    >>> model = SeamlessM4TModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "seamless_m4t"
    attribute_map = {"num_hidden_layers": "decoder_layers", "num_attention_heads": "decoder_attention_heads"}

    vocab_size: int = 256102
    t2u_vocab_size: int = 10082
    hidden_size: int = 1024
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    max_position_embeddings: int = 1024
    is_encoder_decoder: bool = True
    encoder_layerdrop: float | int = 0.05
    decoder_layerdrop: float | int = 0.05
    activation_function: str = "relu"
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    scale_embedding: bool = True
    encoder_layers: int = 24
    encoder_ffn_dim: int = 8192
    encoder_attention_heads: int = 16
    decoder_layers: int = 24
    decoder_ffn_dim: int = 8192
    decoder_attention_heads: int = 16
    decoder_start_token_id: int = 3
    max_new_tokens: int | None = 256
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 3
    speech_encoder_layers: int = 24
    speech_encoder_attention_heads: int = 16
    speech_encoder_intermediate_size: int = 4096
    speech_encoder_hidden_act: str = "swish"
    speech_encoder_dropout: float | int = 0.0
    add_adapter: bool = True
    speech_encoder_layerdrop: float | int = 0.1
    feature_projection_input_dim: int = 160
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    adaptor_kernel_size: int = 8
    adaptor_stride: int = 8
    adaptor_dropout: float | int = 0.1
    num_adapter_layers: int = 1
    position_embeddings_type: str = "relative"
    rotary_embedding_base: int = 10000
    max_source_positions: int = 4096
    conv_depthwise_kernel_size: int = 31
    t2u_bos_token_id: int | None = 0
    t2u_pad_token_id: int | None = 1
    t2u_eos_token_id: int | list[int] | None = 2
    t2u_decoder_start_token_id: int = 2
    t2u_max_new_tokens: int = 1024
    t2u_encoder_layers: int = 6
    t2u_encoder_ffn_dim: int = 8192
    t2u_encoder_attention_heads: int = 16
    t2u_decoder_layers: int = 6
    t2u_decoder_ffn_dim: int = 8192
    t2u_decoder_attention_heads: int = 16
    t2u_max_position_embeddings: int = 2048
    sampling_rate: int = 16000
    upsample_initial_channel: int = 512
    upsample_rates: list[int] | tuple[int, ...] = (5, 4, 4, 2, 2)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (11, 8, 8, 4, 4)
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    leaky_relu_slope: float = 0.1
    unit_hifi_gan_vocab_size: int = 10000
    unit_embed_dim: int = 1280
    lang_embed_dim: int = 256
    spkr_embed_dim: int = 256
    vocoder_num_langs: int = 36
    vocoder_num_spkrs: int = 200
    variance_predictor_kernel_size: int = 3
    var_pred_dropout: float | int = 0.5
    vocoder_offset: int = 4
    tie_word_embeddings: bool = True


__all__ = ["SeamlessM4TConfig"]
