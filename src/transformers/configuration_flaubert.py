# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
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
""" Flaubert configuration, based on XLM. """

from .configuration_xlm import XLMConfig
from .utils import logging


logger = logging.get_logger(__name__)

FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "flaubert/flaubert_small_cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_small_cased/config.json",
    "flaubert/flaubert_base_uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_uncased/config.json",
    "flaubert/flaubert_base_cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_cased/config.json",
    "flaubert/flaubert_large_cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_large_cased/config.json",
}


class FlaubertConfig(XLMConfig):
    """
    Configuration class to store the configuration of a `FlaubertModel`.
    This is the configuration class to store the configuration of a :class:`~transformers.XLMModel`.
    It is used to instantiate an XLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the `xlm-mlm-en-2048 <https://huggingface.co/xlm-mlm-en-2048>`__ architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.

    Args:
        pre_norm (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to apply the layer normalization before or after the feed forward layer following the
            attention in each layer (Vaswani et al., Tensor2Tensor for Neural Machine Translation. 2018)
        layerdrop (:obj:`float`, `optional`, defaults to 0.0):
            Probability to drop layers during training (Fan et al., Reducing Transformer Depth on Demand
            with Structured Dropout. ICLR 2020)
        vocab_size (:obj:`int`, optional, defaults to 30145):
            Vocabulary size of the Flaubert model. Defines the different tokens that
            can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.FlaubertModel`.
        emb_dim (:obj:`int`, optional, defaults to 2048):
            Dimensionality of the encoder layers and the pooler layer.
        n_layer (:obj:`int`, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, optional, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        dropout (:obj:`float`, optional, defaults to 0.1):
            The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, optional, defaults to 0.1):
            The dropout probability for the attention mechanism
        gelu_activation (:obj:`boolean`, optional, defaults to :obj:`True`):
            The non-linear activation function (function or string) in the
            encoder and pooler. If set to `True`, "gelu" will be used instead of "relu".
        sinusoidal_embeddings (:obj:`boolean`, optional, defaults to :obj:`False`):
            Whether to use sinusoidal positional embeddings instead of absolute positional embeddings.
        causal (:obj:`boolean`, optional, defaults to :obj:`False`):
            Set this to `True` for the model to behave in a causal manner.
            Causal models use a triangular attention mask in order to only attend to the left-side context instead
            if a bidirectional context.
        asm (:obj:`boolean`, optional, defaults to :obj:`False`):
            Whether to use an adaptive log softmax projection layer instead of a linear layer for the prediction
            layer.
        n_langs (:obj:`int`, optional, defaults to 1):
            The number of languages the model handles. Set to 1 for monolingual models.
        use_lang_emb (:obj:`boolean`, optional, defaults to :obj:`True`)
            Whether to use language embeddings. Some models use additional language embeddings, see
            `the multilingual models page <http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings>`__
            for information on how to use them.
        max_position_embeddings (:obj:`int`, optional, defaults to 512):
            The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        embed_init_std (:obj:`float`, optional, defaults to 2048^-0.5):
            The standard deviation of the truncated_normal_initializer for
            initializing the embedding matrices.
        init_std (:obj:`int`, optional, defaults to 50257):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices except the embedding matrices.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bos_index (:obj:`int`, optional, defaults to 0):
            The index of the beginning of sentence token in the vocabulary.
        eos_index (:obj:`int`, optional, defaults to 1):
            The index of the end of sentence token in the vocabulary.
        pad_index (:obj:`int`, optional, defaults to 2):
            The index of the padding token in the vocabulary.
        unk_index (:obj:`int`, optional, defaults to 3):
            The index of the unknown token in the vocabulary.
        mask_index (:obj:`int`, optional, defaults to 5):
            The index of the masking token in the vocabulary.
        is_encoder(:obj:`boolean`, optional, defaults to :obj:`True`):
            Whether the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
        summary_type (:obj:`string`, optional, defaults to "first"):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            :class:`~transformers.XLMForSequenceClassification`.
            Is one of the following options:

            - 'last' => take the last token hidden state (like XLNet)
            - 'first' => take the first token hidden state (like Bert)
            - 'mean' => take the mean of all tokens hidden states
            - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
            - 'attn' => Not implemented now, use multi-head attention
        summary_use_proj (:obj:`boolean`, optional, defaults to :obj:`True`):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            :class:`~transformers.XLMForSequenceClassification`.
            Add a projection after the vector extraction
        summary_activation (:obj:`string` or :obj:`None`, optional):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            :class:`~transformers.XLMForSequenceClassification`.
            'tanh' => add a tanh activation to the output, Other => no activation.
        summary_proj_to_labels (:obj:`boolean`, optional, defaults to :obj:`True`):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            :class:`~transformers.XLMForSequenceClassification`.
            If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
        summary_first_dropout (:obj:`float`, optional, defaults to 0.1):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            :class:`~transformers.XLMForSequenceClassification`.
            Add a dropout before the projection and activation
        start_n_top (:obj:`int`, optional, defaults to 5):
            Used in the SQuAD evaluation script for XLM and XLNet.
        end_n_top (:obj:`int`, optional, defaults to 5):
            Used in the SQuAD evaluation script for XLM and XLNet.
        mask_token_id (:obj:`int`, optional, defaults to 0):
            Model agnostic parameter to identify masked tokens when generating text in an MLM context.
        lang_id (:obj:`int`, optional, defaults to 1):
            The ID of the language used by the model. This parameter is used when generating
            text in a given language.
    """

    model_type = "flaubert"

    def __init__(self, layerdrop=0.0, pre_norm=False, pad_token_id=2, bos_token_id=0, **kwargs):
        """Constructs FlaubertConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, **kwargs)
        self.layerdrop = layerdrop
        self.pre_norm = pre_norm
