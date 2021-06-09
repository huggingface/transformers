# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" Wav2Vec2 model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
}


class Wav2Vec2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.Wav2Vec2Model`. It is used to
    instantiate an Wav2Vec2 model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the Wav2Vec2
    `facebook/wav2vec2-base-960h <https://huggingface.co/facebook/wav2vec2-base-960h>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 32):
            Vocabulary size of the Wav2Vec2 model. Defines the number of different tokens that can be represented by
            the :obj:`inputs_ids` passed when calling :class:`~transformers.Wav2Vec2Model` or
            :class:`~transformers.TFWav2Vec2Model`. Vocabulary size of the model. Defines the different tokens that can
            be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.Wav2Vec2Model`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        feat_extract_norm (:obj:`str`, `optional`, defaults to :obj:`"group"`):
            The norm to be applied to 1D convolutional layers in feature extractor. One of :obj:`"group"` for group
            normalization of only the first 1D convolutional layer or :obj:`"layer"` for layer normalization of all 1D
            convolutional layers.
        feat_extract_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for all 1D convolutional layers in feature extractor.
        feat_extract_activation (:obj:`str, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the 1D convolutional layers of the feature
            extractor. If string, :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        feat_quantizer_dropout (obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for quantized feature extractor states.
        conv_dim (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(512, 512, 512, 512, 512, 512, 512)`):
            A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            feature extractor. The length of `conv_dim` defines the number of 1D convolutional layers.
        conv_stride (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(5, 2, 2, 2, 2, 2, 2)`):
            A tuple of integers defining the stride of each 1D convolutional layer in the feature extractor. The length
            of `conv_stride` defines the number of convolutional layers and has to match the the length of `conv_dim`.
        conv_kernel (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(10, 3, 3, 3, 3, 3, 3)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the feature extractor. The
            length of `conv_kernel` defines the number of convolutional layers and has to match the the length of
            `conv_dim`.
        conv_bias (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the 1D convolutional layers have a bias.
        num_conv_pos_embeddings (:obj:`int`, `optional`, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (:obj:`int`, `optional`, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
        do_stable_layer_norm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether do apply `stable` layer norm architecture of the Transformer encoder. ``do_stable_layer_norm is
            True`` corresponds to applying layer norm before the attention layer, whereas ``do_stable_layer_norm is
            False`` corresponds to applying layer norm after the attention layer.
        apply_spec_augment (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature extractor. For reference see
            `SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
            <https://arxiv.org/abs/1904.08779>`__.
        mask_time_prob (:obj:`float`, `optional`, defaults to 0.05):
            Propability of each feature vector along the time axis to be chosen as the start of the vector span to be
            masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature vectors will be
            masked along the time axis. This is only relevant if ``apply_spec_augment is True``.
        mask_time_length (:obj:`int`, `optional`, defaults to 10):
            Length of vector span along the time axis.
        mask_feature_prob (:obj:`float`, `optional`, defaults to 0.0):
            Propability of each feature vector along the feature axis to be chosen as the start of the vector span to
            be masked. Approximately ``mask_time_prob * hidden_size // mask_time_length`` feature vectors will be
            masked along the time axis. This is only relevant if ``apply_spec_augment is True``.
        mask_feature_length (:obj:`int`, `optional`, defaults to 10):
            Length of vector span along the feature axis.
        num_codevectors_per_group (:obj:`int`, `optional`, defaults to 320):
            Number of entries in each quantization codebook (group).
        num_codevector_groups (:obj:`int`, `optional`, defaults to 2):
            Number of codevector groups for product codevector quantization.
        contrastive_logits_temperature (:obj:`float`, `optional`, defaults to 0.1):
            The temperature `kappa` in the contrastive loss.
        feat_quantizer_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probabilitiy for the output of the feature extractor that's used by the quantizer.
        num_negatives (:obj:`int`, `optional`, defaults to 100):
            Number of negative samples for the contrastive loss.
        codevector_dim (:obj:`int`, `optional`, defaults to 256):
            Dimensionality of the quantized feature vectors.
        proj_codevector_dim (:obj:`int`, `optional`, defaults to 256):
            Dimensionality of the final projection of both the quantized and the transformer features.
        diversity_loss_weight (:obj:`int`, `optional`, defaults to 0.1):
            The weight of the codebook diversity loss component.
        ctc_loss_reduction (:obj:`str`, `optional`, defaults to :obj:`"sum"`):
            Specifies the reduction to apply to the output of ``torch.nn.CTCLoss``. Only relevant when training an
            instance of :class:`~transformers.Wav2Vec2ForCTC`.
        ctc_zero_infinity (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to zero infinite losses and the associated gradients of ``torch.nn.CTCLoss``. Infinite losses
            mainly occur when the inputs are too short to be aligned to the targets. Only relevant when training an
            instance of :class:`~transformers.Wav2Vec2ForCTC`.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

    Example::

        >>> from transformers import Wav2Vec2Model, Wav2Vec2Config

        >>> # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
        >>> configuration = Wav2Vec2Config()

        >>> # Initializing a model from the facebook/wav2vec2-base-960h style configuration
        >>> model = Wav2Vec2Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "wav2vec2"

    def __init__(
        self,
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.1,
        feat_quantizer_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=False,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        num_codevectors_per_group=320,
        num_codevector_groups=2,
        contrastive_logits_temperature=0.1,
        num_negatives=100,
        codevector_dim=256,
        proj_codevector_dim=256,
        diversity_loss_weight=0.1,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        gradient_checkpointing=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_feat_extract_layers = len(self.conv_dim)
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.feat_proj_dropout = feat_proj_dropout
        self.final_dropout = final_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.gradient_checkpointing = gradient_checkpointing

        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect."
                "It is required that `len(config.conv_dim)` == `len(config.conv_stride)` == `len(config.conv_kernel)`,"
                f"but is `len(config.conv_dim) = {len(self.conv_dim)}`, `len(config.conv_stride)"
                f"= {len(self.conv_stride)}`, `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length

        # parameters for pretraining with codevector quantized representations
        self.num_codevectors_per_group = num_codevectors_per_group
        self.num_codevector_groups = num_codevector_groups
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.feat_quantizer_dropout = feat_quantizer_dropout
        self.num_negatives = num_negatives
        self.codevector_dim = codevector_dim
        self.proj_codevector_dim = proj_codevector_dim
        self.diversity_loss_weight = diversity_loss_weight

        # ctc loss
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
