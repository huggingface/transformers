# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""M-CTC-T model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "speechbrain/m-ctc-t-large": "https://huggingface.co/speechbrain/m-ctc-t-large/resolve/main/config.json",
    # See all M-CTC-T models at https://huggingface.co/models?filter=mctct
}


class MCTCTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MCTCTModel`]. It is used to instantiate an
    M-CTC-T model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the M-CTC-T
    [speechbrain/m-ctc-t-large](https://huggingface.co/speechbrain/m-ctc-t-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 8065):
            Vocabulary size of the M-CTC-T model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MCTCTModel`].
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 36):
            Number of hidden layers in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 4):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_head_dim (`int`, *optional*, defaults to 384):
            Dimensions of each attention head for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 920):
            The maximum sequence length that this model might ever be used with (after log-mel spectrogram extraction).
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        layerdrop (`float`, *optional*, defaults to 0.3):
            The probability of dropping an encoder layer during training. The default 0.3 value is used in the original
            implementation.
        hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        pad_token_id (`int`, *optional*, defaults to 1):
            The tokenizer index of the pad token.
        bos_token_id (`int`, *optional*, defaults to 0):
            The tokenizer index of the bos token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The tokenizer index of the eos token.
        conv_glu_dim (`int`, *optional*, defaults to 1):
            The dimension of the output of the `Conv1dSubsampler` layer in which GLU is applied on. Though the original
            Flashlight code uses the value of 2, here it's adapted to 1 due to transposition differences.
        conv_dropout (`int`, *optional*, defaults to 0.3):
            The probability of randomly dropping the `Conv1dSubsampler` layer during training.
        num_conv_layers (`int`, *optional*, defaults to 1):
            Number of convolution layers before applying transformer encoder layers.
        conv_kernel (`List[int]`, *optional*, defaults to `[7]`):
            The kernel size of the 1D convolution applied before transformer layers. `len(conv_kernel)` must be equal
            to `num_conv_layers`.
        conv_stride (`List[int]`, *optional*, defaults to `[3]`):
            The stride length of the 1D convolution applied before transformer layers. `len(conv_stride)` must be equal
            to `num_conv_layers`.
        input_feat_per_channel (`int`, *optional*, defaults to 80):
            Feature dimensions of the channels of the input to the Conv1D layer.
        input_channels (`int`, *optional*, defaults to 1):
            Number of input channels of the input to the Conv1D layer.
        conv_channels (`List[int]`, *optional*, defaults to None):
            Channel sizes of intermediate Conv1D layers.
        ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`):
            Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
            instance of [`MCTCTForCTC`].
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
            occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
            of [`MCTCTForCTC`].

    Example:

    ```python
    >>> from transformers import MCTCTModel, MCTCTConfig

    >>> # Initializing a M-CTC-T mctct-large style configuration
    >>> configuration = MCTCTConfig()

    >>> # Initializing a model from the mctct-large style configuration
    >>> model = MCTCTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mctct"

    def __init__(
        self,
        vocab_size=8065,
        hidden_size=1536,
        num_hidden_layers=36,
        intermediate_size=6144,
        num_attention_heads=4,
        attention_head_dim=384,
        max_position_embeddings=920,
        layer_norm_eps=1e-5,
        layerdrop=0.3,
        hidden_act="relu",
        initializer_range=0.02,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        conv_glu_dim=1,
        conv_dropout=0.3,
        num_conv_layers=1,
        conv_kernel=(7,),
        conv_stride=(3,),
        input_feat_per_channel=80,
        input_channels=1,
        conv_channels=None,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        **kwargs
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.layerdrop = layerdrop
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.conv_glu_dim = conv_glu_dim
        self.conv_dropout = conv_dropout
        self.num_conv_layers = num_conv_layers
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        # prevents config testing fail with exporting to json
        self.conv_kernel = list(conv_kernel)
        self.conv_stride = list(conv_stride)

        if len(self.conv_kernel) != self.num_conv_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.conv_kernel)` == `config.num_conv_layers` "
                f"but is `len(config.conv_kernel) = {len(self.conv_kernel)}`, "
                f"`config.num_conv_layers = {self.num_conv_layers}`."
            )
