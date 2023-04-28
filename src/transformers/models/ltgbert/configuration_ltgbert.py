# coding=utf-8
# Copyright 2023 Language Technology Group from University of Oslo and The HuggingFace Inc. team.
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

""" LTG-BERT configutation """


from transformers.configuration_utils import PretrainedConfig


LTG_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bnc-bert-span": "https://huggingface.co/ltg/bnc-bert-span",
    "bnc-bert-span-2x": "https://huggingface.co/ltg/bnc-bert-span-2x",
    "bnc-bert-span-0.5x": "https://huggingface.co/ltg/bnc-bert-span-0.5x",
    "bnc-bert-span-0.25x": "https://huggingface.co/ltg/bnc-bert-span-0.25x",
    "bnc-bert-span-order": "https://huggingface.co/ltg/bnc-bert-span-order",
    "bnc-bert-span-document": "https://huggingface.co/ltg/bnc-bert-span-document",
    "bnc-bert-span-word": "https://huggingface.co/ltg/bnc-bert-span-word",
    "bnc-bert-span-subword": "https://huggingface.co/ltg/bnc-bert-span-subword",

    "norbert3-xs": "https://huggingface.co/ltg/norbert3-xs/config.json",
    "norbert3-small": "https://huggingface.co/ltg/norbert3-small/config.json",
    "norbert3-base": "https://huggingface.co/ltg/norbert3-base/config.json",
    "norbert3-large": "https://huggingface.co/ltg/norbert3-large/config.json",

    "norbert3-oversampled-base": "https://huggingface.co/ltg/norbert3-oversampled-base/config.json",
    "norbert3-ncc-base": "https://huggingface.co/ltg/norbert3-ncc-base/config.json",
    "norbert3-nak-base": "https://huggingface.co/ltg/norbert3-nak-base/config.json",
    "norbert3-nb-base": "https://huggingface.co/ltg/norbert3-nb-base/config.json",
    "norbert3-wiki-base": "https://huggingface.co/ltg/norbert3-wiki-base/config.json",
    "norbert3-c4-base": "https://huggingface.co/ltg/norbert3-c4-base/config.json"
}


class LtgBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LtgBertModel`]. It is used to
    instantiate an LTG-BERT model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 16384):
            Vocabulary size of the LTG-BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LtgBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
    """
    model_type = "bert"
    def __init__(
        self,
        vocab_size=16384,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        hidden_size=768,
        intermediate_size=2048,
        max_position_embeddings=512,
        position_bucket_size=32,
        num_attention_heads=12,
        num_hidden_layers=12,
        layer_norm_eps=1.0e-7,
        pad_token_id=4,
        output_all_encoded_layers=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.output_all_encoded_layers = output_all_encoded_layers
        self.position_bucket_size = position_bucket_size
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
