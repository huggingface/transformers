# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""RAG model implementation."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from ...configuration_utils import PretrainedConfig
from ...file_utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from ...generation_beam_search import BeamSearchScorer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RagConfig"


@dataclass
class RetrievAugLMMarginOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
            :obj:`question_encoder_last_hidden_state`.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see :obj:`past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs, hidden_size)`, `optional`, returned when `output_retrieved=True`):
            Embedded documents retrieved by the retriever. Is used with ``question_encoder_last_hidden_state`` to
            compute the ``doc_scores``.
        retrieved_doc_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, config.n_docs)`, `optional`, returned when `output_retrieved=True`):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
            retriever.
        question_encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RetrievAugLMOutput(ModelOutput):
    """
    Args:
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
            :obj:`question_encoder_last_hidden_state`.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see :obj:`past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs, hidden_size)`, `optional`, returned when `output_retrieved=True`):
            Embedded documents retrieved by the retriever. Is used with ``question_encoder_last_hidden_state`` to
            compute the ``doc_scores``.
        retrieved_doc_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, config.n_docs)`, `optional`, returned when `output_retrieved=True`):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
            retriever.
        question_encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
    """

    logits: torch.FloatTensor = None
    doc_scores: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    retrieved_doc_embeds: Optional[torch.FloatTensor] = None
    retrieved_doc_ids: Optional[torch.LongTensor] = None
    context_input_ids: Optional[torch.LongTensor] = None
    context_attention_mask: Optional[torch.LongTensor] = None
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = None
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = None


class RagPreTrainedModel(PreTrainedModel):
    r"""
    RAG models were released with the paper `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    <https://arxiv.org/abs/2005.11401>`_ by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.

    """
    config_class = RagConfig
    base_model_prefix = "rag"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: RagRetriever = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiates an question encoder and a generator from one or two base classes of the library from pretrained
        model checkpoints.

        The model is set in evaluation mode by default using :obj:`model.eval()` (Dropout modules are deactivated). To
        train the model, you need to first set it back in training mode with :obj:`model.train()`.

        Params:
            question_encoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the question encoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            generator_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the generator. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            retriever (:class:`~transformers.RagRetriever`, `optional`):
                The retriever to use.
            kwwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                ``output_attentions=True``).

                - To update the question_encoder configuration, use the prefix `question_encoder_` for each
                  configuration parameter.
                - To update the generator configuration, use the prefix `generator_` for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a :obj:`config` is provided or automatically loaded.

        Example::

            >>> from transformers import RagModel
            >>> # initialize a RAG from two pretrained models.
            >>> model = RagModel.from_question_encoder_generator_pretrained('facebook/dpr-question_encoder-single-nq-base', 't5-small')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./rag")
            >>> # load fine-tuned model
            >>> model = RagModel.from_pretrained("./rag")

        """

        kwargs_question_encoder = {
            argument[len("question_question_encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("question_encoder_")
        }

        kwargs_generator = {
            argument[len("generator_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("generator_")
        }

        # remove question_encoder, generator kwargs from kwargs
        for key in kwargs_question_encoder.keys():
            del kwargs["question_encoder_" + key]
        for key in kwargs_generator.keys():
            del kwargs["generator_" + key]

        # Load and initialize the question_encoder and generator
        # The distinction between question_encoder and generator at the model level is made
        # by the value of the flag `is_generator` that we need to set correctly.
        question_encoder = kwargs_question_encoder.pop("model", None)
        if question_encoder is None:
            assert (
                question_encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `question_encoder_pretrained_model_name_or_path` has to be defined"
            from ..auto.modeling_auto import AutoModel

            if "config" not in kwargs_question_encoder:
                from ..auto.configuration_auto import AutoConfig

                question_encoder_config = AutoConfig.from_pretrained(question_encoder_pretrained_model_name_or_path)
                kwargs_question_encoder["config"] = question_encoder_config

            question_encoder = AutoModel.from_pretrained(
                question_encoder_pretrained_model_name_or_path, *model_args, **kwargs_question_encoder
            )

        generator = kwargs_generator.pop("model", None)
        if generator is None:
            assert (
                generator_pretrained_model_name_or_path is not None
            ), "If `generator_model` is not defined as an argument, a `generator_pretrained_model_name_or_path` has to be defined"
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            if "config" not in kwargs_generator:
                from ..auto.configuration_auto import AutoConfig

                generator_config = AutoConfig.from_pretrained(generator_pretrained_model_name_or_path)
                kwargs_generator["config"] = generator_config

            generator = AutoModelForSeq2SeqLM.from_pretrained(
                generator_pretrained_model_name_or_path, **kwargs_generator
            )

        # instantiate config with corresponding kwargs
        config = kwargs.get("config", None)
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        return cls(question_encoder=question_encoder, generator=generator, config=config, retriever=retriever)


RAG_START_DOCSTRING = r"""

    RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward
    pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context
    documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

    The question encoder can be any `autoencoding` model, preferably :class:`~transformers.DPRQuestionEncoder`, and the
    generator can be any `seq2seq` model, preferably :class:`~transformers.BartForConditionalGeneration`.

    The model can be initialized with a :class:`~transformers.RagRetriever` for end-to-end generation or used in
    combination with the outputs of a retriever in multiple steps---see examples for more details. The model is
    compatible any `autoencoding` model as the ``question_encoder`` and any `seq2seq` model with language model head as
    the ``generator``. It has been tested with :class:`~transformers.DPRQuestionEncoder` as the ``question_encoder``
    and :class:`~transformers.BartForConditionalGeneration` or :class:`~transformers.T5ForConditionalGeneration` as the
    ``generator``.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Args:
        config (:class:`~transformers.RagConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
        question_encoder (:class:`transformers.PreTrainedModel`):
            An encoder model compatible with the faiss index encapsulated by the ``retriever``.
        generator (:class:`transformers.PreTrainedModel`):
            A seq2seq model used as the generator in the RAG architecture.
        retriever (:class:`~transformers.RagRetriever`):
            A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.
"""


RAG_FORWARD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. :class:`~transformers.RagConfig`, used to initialize
            the model, specifies which generator to use, it also specifies a compatible generator tokenizer. Use that
            tokenizer class to obtain the indices.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`)
            Tuple consists of (:obj:`generator_enc_last_hidden_state`, `optional`: :obj:`generator_enc_hidden_states`,
            `optional`: :obj:`generator_enc_attentions`). :obj:`generator_enc_last_hidden_state` of shape
            :obj:`(batch_size, n_docs * sequence_length, hidden_size)` is a sequence of hidden-states at the output of
            the last layer of the generator's encoder.

            Used by the (:class:`~transformers.RagModel`) model during decoding.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for generation tasks. `None` by default, construct as per instructions for the generator model
            you're using with your RAG instance.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size,  target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`):
            Tuple consists of two elements: :obj:`encoder_outputs` of the RAG model (see :obj:`encoder_outputs`) and
            :obj:`past_key_values` of the underlying generator. Can be used to speed up decoding.
            :obj:`past_key_values` are used in the (:class:`~transformers.RagTokenForGeneration`) model during
            decoding.
        doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
            :obj:`question_encoder_last_hidden_state`. If the model has is not initialized with a ``retriever``
            :obj:`doc_scores` has to be provided to the forward pass. :obj:`doc_scores` can be computed via
            :obj:`question_encoder_last_hidden_state` and :obj:`retrieved_doc_embeds`, see examples for more
            information.
        context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Input IDs post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
            retriever.

            If the model has is not initialized with a ``retriever`` :obj:`context_input_ids` has to be provided to the
            forward pass. :obj:`context_input_ids` are returned by :meth:`~transformers.RagRetriever.__call__`.
        context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
            retriever.

            If the model has is not initialized with a ``retriever`` :obj:`context_attention_mask` has to be provided
            to the forward pass. :obj:`context_attention_mask` are returned by
            :meth:`~transformers.RagRetriever.__call__`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        output_retrieved(:obj:`bool`, `optional`):
            Whether or not to return the :obj:`retrieved_doc_embeds`, :obj:`retrieved_doc_ids`,
            :obj:`context_input_ids` and :obj:`context_attention_mask`. See returned tensors for more detail.
        n_docs (:obj:`int`, `optional`, defaults to :obj:`config.n_docs`)
            Number of documents to retrieve and/or number of documents for which to generate an answer.
"""


@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class RagModel(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,  # or maybe just use a `set_retriever(...)` method
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an question_encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        super().__init__(config)
        if question_encoder is None:
            from ..auto.modeling_auto import AutoModel

            question_encoder = AutoModel.from_config(config.question_encoder)

        if generator is None:
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            generator = AutoModelForSeq2SeqLM.from_config(config.generator)

        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        doc_scores=None,
        context_input_ids=None,
        context_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_retrieved=None,
        n_docs=None,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagModel
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)

            >>> input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="pt")
            >>> input_ids = input_dict["input_ids"]
            >>> outputs = model(input_ids=input_ids)

        """
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved

        # whether retriever has to be used
        has_to_retrieve = (
            self.retriever is not None
            and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
            and encoder_outputs is None
        )
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:

            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )
                question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder

                retriever_outputs = self.retriever(
                    input_ids,
                    question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    return_tensors="pt",
                )
                context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
                    retriever_outputs["context_input_ids"],
                    retriever_outputs["context_attention_mask"],
                    retriever_outputs["retrieved_doc_embeds"],
                    retriever_outputs["doc_ids"],
                )

                # set to correct device
                retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
                context_input_ids = context_input_ids.to(input_ids)
                context_attention_mask = context_attention_mask.to(input_ids)

                # compute doc_scores
                doc_scores = torch.bmm(
                    question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                ).squeeze(1)
            else:
                assert (
                    context_input_ids is not None
                ), "Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
                assert (
                    context_attention_mask is not None
                ), "Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
                assert (
                    doc_scores is not None
                ), "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."

        assert (
            doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert (
            doc_scores.shape[1] % n_docs
        ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        gen_outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        if not has_to_retrieve:
            question_encoder_last_hidden_state = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            # don't output retrieved docs
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return RetrievAugLMOutput(
            logits=gen_outputs.logits,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state,
            generator_enc_hidden_states=gen_outputs.encoder_hidden_states,
            generator_enc_attentions=gen_outputs.encoder_attentions,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
        )


@add_start_docstrings_to_model_forward(
    """
    A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
    """,
    RAG_START_DOCSTRING,
)
class RagSequenceForGeneration(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        super().__init__(config)

        # instantiate model
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_retrieved=None,
        exclude_bos_score=None,
        reduce_loss=None,
        labels=None,
        n_docs=None,
        **kwargs  # needs kwargs for generation
    ):
        r"""
        exclude_bos_score (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the score of the BOS token is disregarded when
            computing the loss.
        reduce_loss (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the NLL loss is reduced using the
            ``torch.Tensor.sum`` operation.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
             Legacy dictionary, which is required so that model can use `generate()` function.

        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

            >>> input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="pt")
            >>> input_ids = input_dict["input_ids"]
            >>> outputs = model(input_ids=input_ids, labels=input_dict["labels"])

            >>> # or use retriever separately
            >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
            >>> # 1. Encode
            >>> question_hidden_states = model.question_encoder(input_ids)[0]
            >>> # 2. Retrieve
            >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
            >>> doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
            >>> # 3. Forward to generator
            >>> outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=input_dict["labels"])
        """
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        exclude_bos_score = exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        if labels is not None:
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                decoder_input_ids,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                exclude_bos_score=exclude_bos_score,
                n_docs=n_docs,
            )

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=outputs.logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
        )

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        context_input_ids=None,
        do_deduplication=None,  # defaults to True
        num_return_sequences=None,  # defaults to 1
        num_beams=None,  # defaults to 1
        n_docs=None,
        **model_kwargs
    ):
        """
        Implements RAG sequence "thorough" decoding. Read the :meth:`~transformers.PreTrainedModel.generate``
        documentation for more information on how to set other generate input parameters.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`input_ids` is not passed, then
                :obj:`context_input_ids` has to be provided.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
                retriever.
            do_deduplication (:obj:`bool`, `optional`):
                Whether or not to deduplicate the generations from different context documents for a given input. Has
                to be set to :obj:`False` if used while training with distributed backend.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the ``generator``'s `:func:`~transformers.PreTrainedModel.generate``
                function, where we set ``num_return_sequences`` to :obj:`num_beams`.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            n_docs (:obj:`int`, `optional`, defaults to :obj:`config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            kwargs:
                Additional kwargs will be passed to :meth:`~transformers.PreTrainedModel.generate`.

        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.
        """

        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        num_beams = num_beams if num_beams is not None else self.config.num_beams

        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            context_input_ids = self.retriever(
                input_ids,
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="pt",
            )["context_input_ids"]

            # set to correct device
            context_input_ids = context_input_ids.to(input_ids)

        hypos = []
        model_kwargs["num_beams"] = num_beams
        model_kwargs["num_return_sequences"] = num_beams
        model_kwargs["attention_mask"] = None

        for index in range(len(input_ids)):
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[index * n_docs : (index + 1) * n_docs]  # (n_docs, max_len)

            output_sequences = self.generator.generate(
                generator_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len
            if do_deduplication:
                # do_deduplication, max_output_len
                output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))

            # then, run model forwards to get nll scores:
            new_input_ids = input_ids[index : index + 1].repeat(len(output_sequences), 1)
            outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
            top_cand_inds = (-outputs["loss"]).topk(num_doc_return_sequences)[1]

            # add hypothesis
            hypos.append(output_sequences[top_cand_inds])

        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)

    def get_nll(
        self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None
    ):
        # shift tokens left
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # bos_token_id is None for T5
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x dim
        doc_logprobs = torch.nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        # calculate loss
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # sum over tokens, exclude bos while scoring
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        output = (
            tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
        )
        ind = 0
        for t in tensors:
            output[ind : ind + t.shape[0], : t.shape[1]] = t
            ind += t.shape[0]
        return output


@add_start_docstrings_to_model_forward(
    """
    A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
    """,
    RAG_START_DOCSTRING,
)
class RagTokenForGeneration(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super().__init__(config)

        # instantiate model
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        return self.rag.generator.adjust_logits_during_generation(logits, cur_len=cur_len, max_length=max_length)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        doc_scores=None,
        n_docs=None,
        **kwargs
    ):
        if past is not None:
            # if past is defined use only last decoder_input_ids
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "doc_scores": doc_scores,
            "context_attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past,
            "use_cache": use_cache,
            "do_marginalize": True,
            "n_docs": n_docs,
        }

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Reorders cache for generation. BART-inspired but we need to take care of the extra dimension for docs"""

        def _reorder_stacked(hidden_states, new_order):
            n_docs = hidden_states.shape[0] // new_order.shape[0]
            hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:])
            hidden_states = hidden_states.index_select(0, new_order)
            result = hidden_states.view(-1, *hidden_states.shape[2:])
            return result

        reordered_past = ()
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            reordered_past += (tuple(_reorder_stacked(past_state, beam_idx) for past_state in layer_past),)

        return reordered_past

    def marginalize(self, seq_logits, doc_scores, n_docs=None):

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # RAG-token marginalization
        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_retrieved=None,
        do_marginalize=None,
        reduce_loss=None,
        labels=None,
        n_docs=None,
        **kwargs  # needs kwargs for generation
    ):
        r"""
        do_marginalize (:obj:`bool`, `optional`):
            If :obj:`True`, the logits are marginalized over all documents by making use of
            ``torch.nn.functional.log_softmax``.
        reduce_loss (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the NLL loss is reduced using the
            ``torch.Tensor.sum`` operation.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Legacy dictionary, which is required so that model can use `generate()` function.

        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

            >>> input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="pt")
            >>> input_ids = input_dict["input_ids"]
            >>> outputs = model(input_ids=input_ids, labels=input_dict["labels"])

            >>> # or use retriever separately
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
            >>> # 1. Encode
            >>> question_hidden_states = model.question_encoder(input_ids)[0]
            >>> # 2. Retrieve
            >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
            >>> doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
            >>> # 3. Forward to generator
            >>> outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=input_dict["labels"])

            >>> # or directly generate
            >>> generated = model.generate(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores)
            >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        """
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
        )

        loss = None
        logits = outputs.logits
        if labels is not None:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=n_docs,
            )

        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, n_docs)

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        max_length=None,
        min_length=None,
        early_stopping=None,
        use_cache=None,
        num_beams=None,
        num_beam_groups=None,
        diversity_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        repetition_penalty=None,
        bad_words_ids=None,
        num_return_sequences=None,
        decoder_start_token_id=None,
        n_docs=None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
        **model_kwargs
    ):
        """
        Implements RAG token decoding.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`input_ids` is not passed, then
                :obj:`context_input_ids` has to be provided.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Input IDs post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
                retriever.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by
                the retriever.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
                :obj:`question_encoder_last_hidden_state`.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to stop the beam search when at least ``num_beams`` sentences are finished per batch or
                not.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
                beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
            diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
                enabled.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the ``generator``'s `:func:`~transformers.PreTrainedModel.generate`
                function, where we set ``num_return_sequences`` to :obj:`num_beams`.
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            n_docs (:obj:`int`, `optional`, defaults to :obj:`config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments :obj:`inputs_ids` and the batch ID
                :obj:`batch_id`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the previously generated tokens :obj:`inputs_ids` and the batch ID :obj:`batch_id`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.

        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.
        """
        # set default parameters
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.generator.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.generator.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.generator.pad_token_id
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.generator.decoder_start_token_id
        )

        # retrieve docs
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            out = self.retriever(
                input_ids,
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                prefix=self.generator.config.prefix,
                n_docs=n_docs,
                return_tensors="pt",
            )
            context_input_ids, context_attention_mask, retrieved_doc_embeds = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
            )

            # set to correct device
            retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
            context_input_ids = context_input_ids.to(input_ids)
            context_attention_mask = context_attention_mask.to(input_ids)

            # compute doc_scores
            doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(
                1
            )

        assert (
            context_input_ids.shape[0] % n_docs
        ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # batch_size
        batch_size = context_input_ids.shape[0] // n_docs

        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask)

        input_ids = torch.full(
            (batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            # split into `batch_size`, `num_beams`, `num_docs`
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            # repeat same last hidden states over `num_beams` dimension
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            # merge `batch_size`, `num_beams`, `num_docs` dims again
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])

        # correctly extend last_hidden_state and attention mask
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(last_hidden_state, num_beams=num_beams)

        doc_scores = doc_scores.repeat_interleave(num_beams, dim=0)

        # define start_len & additional parameters
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs

        pre_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )

        if num_beams == 1:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            return self.greedy_search(
                input_ids,
                pre_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        elif num_beams > 1:
            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                pre_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        else:
            raise ValueError(f"`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {num_beams}")

    def get_input_embeddings(self):
        return self.rag.generator.get_input_embeddings()

    def get_output_embeddings(self):
        return self.rag.generator.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.rag.generator.set_output_embeddings(new_embeddings)

    def shift_tokens_right(self, input_ids, start_token_id=None):
        """Shift input ids one token to the right, and pad with start_token_id"""
        if start_token_id is None:
            start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        return shifted_input_ids

    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        # shift tokens left
        target = torch.cat(
            [target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1
        )

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)  # sum over tokens
        smooth_obj = smooth_obj.sum(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss
