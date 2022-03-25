# coding=utf-8
# Copyright 2022 Chan Woo Kim and The HuggingFace Inc. team. All rights reserved.
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
""" mCTC model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MCTC_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mctc-large": "https://huggingface.co/mctc-large/resolve/main/config.json",
    # See all mCTC models at https://huggingface.co/models?filter=mctc
}


class MCTCConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~MCTCModel`].
    It is used to instantiate an mCTC model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the mCTC [mctc-large](https://huggingface.co/mctc-large) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 8065):
            Vocabulary size of the mCTC model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~MCTCModel`] or
            [`~TFMCTCModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~MCTCModel`] or
            [`~TFMCTCModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import MCTCModel, MCTCConfig

    >>> # Initializing a mCTC mctc-large style configuration
    >>> configuration = MCTCConfig()

    >>> # Initializing a model from the mctc-large style configuration
    >>> model = MCTCModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "mctc"
    
    '''
    From Flashlight docs:

    Transformer::Transformer(
        int32_t modelDim,
        int32_t headDim,
        int32_t mlpDim,
        int32_t nHeads,
        int32_t bptt,
        float pDropout,
        float pLayerdrop,
        bool useMask,
        bool preLN
    )
    
    wq_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
    wk_(std::make_shared<Linear>(
        transformerInitLinear(modelDim, headDim * nHeads))),
    wv_(std::make_shared<Linear>(
        transformerInitLinear(modelDim, headDim * nHeads))),
    wf_(std::make_shared<Linear>(
        transformerInitLinear(headDim * nHeads, modelDim))),),

    fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
        float std = std::sqrt(1.0 / float(inDim));
        return fl::uniform(outDim, inDim, -std, std, af::dtype::f32, true);
    }

    So for HF naming:
    
    hf.hidden_size == modelDim == headDim * nHeads == 768 (base) or 1536 (large)

    from original mCTC code:
    auto layer = std::make_shared<fl::Transformer>(
          // 768, 192, 3072, 4, 920, 0.3, 0.3, false, false); <- base model (same as slimIPL), whose weights aren't released.
          1536,
          384,
          6144,
          4,
          920,
          0.3,
          0.3,
          false,
          false);


    From Flashlight code:
    Conv2D::Conv2D(
        int nin,
        int nout,
        int wx,
        int wy,
        int sx,
        int sy,
        IntOrPadMode px,
        IntOrPadMode py,
        int dx,
        int dy,
        bool bias,
        int groups)
        : nIn_(nin),
        nOut_(nout),
        xFilter_(wx),
        yFilter_(wy),
        xStride_(sx),
        yStride_(sy),
        xPad_(px.padVal),
        yPad_(py.padVal),
        xDilation_(dx),
        yDilation_(dy),
        bias_(bias),
        groups_(groups) {
    initialize();
    }

    from mCTC original code:
    
    convFrontend_->add(
        std::make_shared<fl::View>(af::dim4(-1, 1, nFeature, 0)));
    // Time x 1 x nFeature x Batch
    std::vector<int> lnDims = {0, 1, 2};
    convFrontend_->add(std::make_shared<fl::LayerNorm>(lnDims));
    convFrontend_->add(
        // std::make_shared<fl::Conv2D>(nFeature, 1536, 7, 1, 3, 1, -1, 0, 1,
        // 1));
        std::make_shared<fl::Conv2D>(nFeature, 3072, 7, 1, 3, 1, -1, 0, 1, 1));
    convFrontend_->add(std::make_shared<fl::GatedLinearUnit>(2));
    convFrontend_->add(std::make_shared<fl::Dropout>(0.3));
    convFrontend_->add(std::make_shared<fl::Reorder>(2, 0, 3, 1));

    I0221 16:16:15.735110 10328 Test.cpp:90] [Network] Model myModel: Sequential [input -> (0) -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
	(0): View (-1 1 80 0)
	(1): LayerNorm ( axis : { 0 1 2 } , size : -1)
	(2): Conv2D (80->3072, 7x1, 3,1, SAME,0, 1, 1) (with bias)
	(3): GatedLinearUnit (2)
	(4): Dropout (0.300000)
	(5): Reorder (2,0,3,1)

    '''

    def __init__(
        self,
        vocab_size=8065,
        hidden_size=1536,
        num_hidden_layers=36,
        intermediate_size=6144,
        num_attention_heads=4,
        attention_head_dim=384,
        max_position_embeddings=920,
        layerdrop=0.3,
        hidden_act="relu",
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3,
        
        use_cache=True,
        is_encoder_decoder=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,

        conv_glu_dim=2,
        conv_dropout=0.3,
        num_conv_layers=1,
        conv_kernel_sizes=(7),
        conv_strides=(3),
        input_feat_per_channel=80,
        input_channels=1,
        conv_channels=None,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.bptt = bptt
        self.layerdrop = layerdrop
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_cache = use_cache
        self.is_encoder_decoder = is_encoder_decoder
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_conv_layers = num_conv_layers
        self.conv_kernel_size = conv_kernel_size)
        self.conv_stride = conv_stride)
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels
        self.conv_channels = conv_channels

        if len(self.conv_kernel_sizes) != self.num_conv_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.conv_kernel_sizes)` == `config.num_conv_layers` "
                f"but is `len(config.conv_kernel_sizes) = {len(self.conv_kernel_sizes)}`, "
                f"`config.num_conv_layers = {self.num_conv_layers}`."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

    