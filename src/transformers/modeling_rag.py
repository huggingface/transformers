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

import copy
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .configuration_auto import AutoConfig
from .configuration_dpr import DPRConfig
from .configuration_rag import RagConfig
from .configuration_utils import PretrainedConfig
from .file_utils import add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_auto import AutoModelForSeq2SeqLM
from .modeling_dpr import DPRQuestionEncoder
from .modeling_outputs import ModelOutput
from .modeling_t5 import T5ForConditionalGeneration
from .modeling_utils import PreTrainedModel
from .retrieval_rag import RagRetriever
from .utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RagConfig"


@dataclass
class BaseModelOutputWithDocs(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
            Scores of retrieved documents.
    """

    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    doc_scores: Optional[torch.FloatTensor] = None


@dataclass
class Seq2SeqLMOutputWithDocs(ModelOutput):
    """
    Outputs for sequence-to-sequence language models with retrieval in the loop.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)` if the ``logits are marginalized or :obj:`(batch_size * config.n_docs, sequence_length, config.vocab_size)` if they aren't):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``past_key_values`` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
            Scores of retrieved documents.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    doc_scores: Optional[torch.FloatTensor] = None


# Reshape from [batch_size, n_docs, dims] to [batch_size * n_docs, dims]
def _stack_ctxt(tensor):
    return tensor.view(-1, *tensor.shape[2:])


# Reshape from [batch_size * n_docs, dims] to [batch_size, n_docs, dims]
def _unstack_ctxt(tensor, n_docs):
    return tensor.view(-1, n_docs, *tensor.shape[1:])


class RagPreTrainedModel(PreTrainedModel):
    r"""
    RAG models encapsulate two trainable components - a question encoder and a generator, but as such they don't have any trainable parameters.
    We specialize `:func:`~transformers.PreTrainedModel.from_pretrained`` and `:func:`~transformers.PreTrainedModel.save_pretrained`` to reflect this.
    """
    config_class = RagConfig
    base_model_prefix = "rag"
    authorized_missing_keys = [r"position_ids"]



RAG_START_DOCSTRING = r"""
    RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator.
    During a forward pass, we encode the input with the question encoder and pass it
    to the retriever to extract relevant context documents. The documents are then prepended to the input.
    Such contextualized input is passed to the generator.

    The model is compatible with :class:`~transformers.DPRQuestionEncoder` as the ``question_encoder``. As for the ``generator``,
    two compatible architectures have been tested: :class:`~transformers.BartForConditionalGeneration`
    and :class:`~transformers.T5ForConditionalGeneration`.

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Args:
        config (:class:`~transformers.RagConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

RAG_CORE_DOCSTRING = r"""
    A base RAG model calculating raw sequence logits and document retrieval scores.
    The model takes a question encoder and a generator  as inputs to the constructor, so it can be a base
    for various RAG architectures encapsualting different retrievers and generators.

    Args:
        config (:class:`~transformers.RagConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
        question_encoder (:class:`transformers.PreTrainedModel`):
            An encoder model compatible with the faiss index encapsulated by the ``retriever``.
        generator (:class:`transformers.PreTrainedModel`):
            A seq2seq model used as the generator in the RAG architecture.
"""

RAG_FORWARD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            :class:`~transformers.RagConfig`, used to initialize the model, specifies which generator to use, it also specifies a compatible
            generator tokenizer. Use that tokenizer class to obtain the indices.
        retriever (:class:`~transformers.RagRetriever`):
            A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`, `doc_scores`)
            `last_hidden_state` of shape :obj:`(batch_size, n_docs * sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the encoder.
            `doc_scores` of shape :obj:`(batch_size, n_docs)` store retrieval scores of documents retrieved for each input in the batch.
            Used by the (:class:`~transformers.RagTokenForGeneration`) model during decoding.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for generation tasks. `None` by default, constuct as per instructions for the generator model you're using with your RAG instance.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`):
            Tuple consists of two elements: ``encoder_outputs`` of the RAG model (see ``encoder_outputs``) and ``past_key_values`` of the underlying generator.
            Can be used to speed up decoding. ``past_key_values`` are used in the (:class:`~transformers.RagTokenForGeneration`)
            model during decoding.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If `use_cache` is True, ``past_key_values`` are returned and can be used to speed up decoding (see
            ``past_key_values``).
        generator_kwargs (remaining dictionary of keyword arguments, `optional`):
            Additional keyword arguments will be passed to the generator forward pass.
"""

RAG_LOSS_INPUTS_DOCSTRING = r"""
        return_loss (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, computes the loss which is returned as part of the :class:`~transformers.file_utils.Seq2SeqLMOutputWithDocs`.
            Otherwise, loss defaults to :obj:`None`.
        reduce (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Only relevant if ``return_loss`` is set to :obj:`True`. If :obj:`True`, the NLL loss is reduced using the ``torch.Tensor.sum`` operation.
        label_smoothing (:obj:`float`, `optional`, defaults to ``0.0``):
            Only relevant if ``return_loss`` is set to :obj:`True`. Controls the ``epsilon`` parameter value for label smoothing in the loss calculation.
            If set to ``0.0``, no label smoothing is performed.
"""

@add_start_docstrings_to_callable(RAG_CORE_DOCSTRING)
class RagModel(RagPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,  # or maybe just use a `set_retriever(...)` method
        **kwargs,
    ):
        assert config is not None or (
            encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = RagConfig.from_encoder_generator_configs(encoder.config, generator.config, **kwargs)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        super().__init__(config)
        if encoder is None:
            from .modeling_auto import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from .modeling_auto import AutoModelForSeq2SeqLM

            generator = AutoModelForSeq2SeqLM.from_config(config.generator)

        if retriever is not None:
            assert isinstance(retriever, RagRetriever)
            self.retriever = retriever # or do all of this via a set method
        self.encoder = encoder
        self.generator = generator
        self.n_docs = self.config.n_docs

    @classmethod
    def from_pretrained_encoder_generator(cls, encoder_path, generator_path, retriever=None, **kwargs):
        # load correct encoder and generator models and config
        return cls(config=config, encoder=encoder, generator=generator, retriever=retriever, **kwargs)


    def from_pretrained(cls, 


    def contextualize(self, input_ids, retriever, print_docs=False):
        """
        Adds context to every input in the batch by querying the retriever.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes
                it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            retriever (:class:`~transformers.RagRetriever`):
                A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.
            print_docs  (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If :obj:`True`, documents retrieved during the forward pass will be printed out. Intended for debugging purposes.

        Return:
            :obj:`tuple(tuple(torch.FloatTensor)`: a tuple consisting od three elements: contextualized ``input_ids``,
                compatible ``attention_mask`` and scores of the retrieved documents.
        """
        question_encoder_input_ids, input_strings = retriever.preprocess_query(input_ids, self.generator.config.prefix)
        query_vectors = self.question_encoder(question_encoder_input_ids)[0]
        doc_vectors, docs = retriever.retrieve(query_vectors.cpu().detach().to(torch.float32), n_docs=self.n_docs)
        doc_vectors = doc_vectors.to(query_vectors)
        doc_scores = torch.bmm(query_vectors.unsqueeze(1), doc_vectors.transpose(1, 2)).squeeze(1)

        # T5 tokenizer doesn't add eos token by default even with add_special_tokens set to True
        add_eos = (input_ids == self.config.eos_token_id).any() and isinstance(
            self.generator, T5ForConditionalGeneration
        )
        input_ids, attention_mask = retriever.postprocess_docs(
            doc_scores, docs, input_strings, add_eos, self.generator.config.prefix, print_docs
        )
        return input_ids, attention_mask, doc_scores

    @add_start_docstrings_to_callable(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutputWithDocs, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        past_key_values=None,
        context_input_ids=None,  # NEW
        doc_vectors=None,  # NEW
        use_cache=None,
        print_docs=False,
        **generator_kwargs
    ):
        r"""
            print_docs  (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If :obj:`True`, documents retrieved during the forward pass will be logged. Intended for debugging purposes.

        Returns:

        """

        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:
            query_vectors = self.question_encoder(input_ids)[0]

        if self.retriever is not None and context_input_ids is None:
            context_input_ids, doc_vectors = self.retriever(query_vectors.cpu().detach().to(torch.float31), n_docs=self.n_docs)
            doc_vectors = doc_vectors.to(query_vectors)

        # T5 tokenizer doesn't add eos token by default even with add_special_tokens set to True
        # => handle following lines in RagRetriever
        # add_eos = (input_ids == self.config.eos_token_id).any() and isinstance(
        #    self.generator, T5ForConditionalGeneration
        # )
        # 
        # => handle the following all in `retriever.__call__`
        # input_ids, attention_mask = retriever.postprocess_docs(
        #     doc_scores, docs, input_strings, add_eos, self.generator.config.prefix, print_docs
        # )

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(self.n_docs, dim=0)

        outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **generator_kwargs,
        )

        # compute `doc_scores` here or maybe just in `ForTokenGeneration` and `ForSequenceGeneration`
        doc_scores = torch.bmm(outputs.query_vectors.unsqueeze(1), outputs.doc_vectors.transpose(1, 2)).squeeze(1)

        return Seq2SeqLMOutputWithDocs(
            loss=None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            doc_scores=doc_scores,
        )

@add_start_docstrings_to_callable(
    """A RAG-sequence model impementation. It performs RAG-sequence specific marginalization in the forward pass
    and specializes some of the functions of :class:`~transformers.PreTrainedModel` to enable RAG-sequence generation.
    """,
    RAG_START_DOCSTRING,
)
class RagSequenceForGeneration(RagPreTrainedModel):

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,  # or maybe just use a `set_retriever(...)` method
        **kwargs,
    ):
        assert config is not None or (
            encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."
        if config is None:
            config = RagConfig.from_encoder_generator_configs(encoder.config, generator.config, **kwargs)
        super().__init__(config)

        self.rag = RagModel(config=config, encoder=encoder, generator=generator, retriever=retriever)


    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: RagRetriever = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""Instantiates an question_encoder and a generator from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).
        To train the model, you need to first set it back in training mode with `model.train()`.

        Params:
            question_encoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the question_encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/question_encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            generator_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the generator. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/generator``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attentions=True``).
                - To update the question_encoder configuration, use the prefix `question_encoder_` for each configuration parameter
                - To update the generator configuration, use the prefix `generator_` for each configuration parameter
                - To update the parent model configuration, do not use a prefix for each configuration parameter
                Behave differently depending on whether a :obj:`config` is provided or automatically loaded.

        Examples::

            >>> from transformers import EncoderDecoderModel
            >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
            >>> model = EncoderDecoderModel.from_question_encoder_generator_pretrained('bert-base-uncased', 'bert-base-uncased')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./bert2bert")
            >>> # load fine-tuned model
            >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")

        """

        kwargs_question_encoder = {
            argument[len("question_question_encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("question_encoder_")
        }

        kwargs_generator = {
            argument[len("generator_") :]: value for argument, value in kwargs.items() if argument.startswith("generator_")
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
            from .modeling_auto import AutoModel

            if "config" not in kwargs_question_encoder:
                from .configuration_auto import AutoConfig

                question_encoder_config = AutoConfig.from_pretrained(question_encoder_pretrained_model_name_or_path)
                if question_encoder_config.is_generator is True or question_encoder_config.add_cross_attention is True:

                    logger.info(
                        f"Initializing {question_encoder_pretrained_model_name_or_path} as a question_encoder model from a generator model. Cross-attention and casual mask are disabled."
                    )
                    question_encoder_config.is_generator = False
                    question_encoder_config.add_cross_attention = False

                    kwargs_question_encoder["config"] = question_encoder_config

            question_encoder = AutoModel.from_pretrained(question_encoder_pretrained_model_name_or_path, *model_args, **kwargs_question_encoder)

        generator = kwargs_generator.pop("model", None)
        if generator is None:
            assert (
                generator_pretrained_model_name_or_path is not None
            ), "If `generator_model` is not defined as an argument, a `generator_pretrained_model_name_or_path` has to be defined"
            from .modeling_auto import AutoModelForCausalLM

            if "config" not in kwargs_generator:
                from .configuration_auto import AutoConfig

                generator_config = AutoConfig.from_pretrained(generator_pretrained_model_name_or_path)
                if generator_config.is_generator is False or generator_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {generator_pretrained_model_name_or_path} as a generator model. Cross attention layers are added to {generator_pretrained_model_name_or_path} and randomly initialized if {generator_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    generator_config.is_generator = True
                    generator_config.add_cross_attention = True

                kwargs_generator["config"] = generator_config

            if kwargs_generator["config"].is_generator is False or generator_config.add_cross_attention is False:
                logger.warning(
                    f"Decoder model {generator_pretrained_model_name_or_path} is not initialized as a generator. In order to initialize {generator_pretrained_model_name_or_path} as a generator, make sure that the attributes `is_generator` and `add_cross_attention` of `generator_config` passed to `.from_question_encoder_generator_pretrained(...)` are set to `True` or do not pass a `generator_config` to `.from_question_encoder_generator_pretrained(...)`"
                )

            generator = AutoModelForCausalLM.from_pretrained(generator_pretrained_model_name_or_path, **kwargs_generator)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_question_encoder_generator_configs(question_encoder.config, generator.config, **kwargs)
        return cls(question_encoder=question_encoder, generator=generator, config=config)

    @add_start_docstrings_to_callable(RAG_FORWARD_INPUTS_DOCSTRING, RAG_LOSS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutputWithDocs, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids,
        retriever,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        past_key_values=None,
        context_input_ids=None,  # NEW
        doc_vectors=None,  # NEW
        use_cache=None,
        return_loss=False,
        reduce=False,
        label_smoothing=0.0,
        score=False,
        **generator_kwargs
    ):
        r"""
            score (:obj:`bool`, `optional`, defaults to :obj:`False`):
                A flag passed as an argument to `:func:`~transformers.RagSequenceForGeneration.get_nll``. If :obj:`True`,
                we exclude the BOS token's score while scoring the sequence.

        Returns:
        """
        if return_loss:
            use_cache = False

        outputs = self.rag(
            input_ids,
            retriever,
            attention_mask,
            encoder_outputs,
            decoder_input_ids,
            context_input_ids,  # NEW
            doc_vectors,  # NEW
            past_key_values,
            use_cache,
            **generator_kwargs,
        )

        # compute doc scores or leave it at end of RagModel
        doc_scores = torch.bmm(outputs.query_vectors.unsqueeze(1), outputs.doc_vectors.transpose(1, 2)).squeeze(1)

        if return_loss:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits,
                doc_scores,
                decoder_input_ids,
                reduce=reduce,
                epsilon=label_smoothing,
                score=score,
            )
            return Seq2SeqLMOutputWithDocs(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
                doc_scores=outputs.doc_scores,
            )

        return Seq2SeqLMOutputWithDocs(
            loss=None,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            doc_scores=outputs.doc_scores,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        context_input_ids=None,  # NEW
        doc_vectors=None,  # NEW
        dedup=True,
        print_docs=False,
        num_return_sequences=1,
        num_beams=1,
        attention_mask=None,
        **kwargs
    ):
        """
        Implements RAG sequence "thorough" decoding.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes
                it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            retriever (:class:`~transformers.RagRetriever`):
                A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.
            dedup (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Controls whether we want to deduplicate the generations from different context documents for a given input.
                Has to be set to :obj:`False` if used while training with distributed backend.
            print_docs  (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If :obj:`True`, documents retrieved during the forward pass will be printed out. Intended for debugging purposes.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this is not the value
                we pass to the ``generator``'s  `:func:`~transformers.PreTrainedModel.generate`` function, where we set ``num_return_sequences``
                to `num_beams`.
            num_beams (:obj:`int`, `optional`, defaults to ``1``):
                Number of beams for beam search. ``1`` means no beam search.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`,  defaults to :obj:`None`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens.
            kwargs:
                Additional kwargs will be passed to the the ``generator``'s  `:func:`~transformers.PreTrainedModel.generate`` function call.

        Return:

            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        """

        def _get_unique_rows(_input_ids):
            return torch.stack(list({str(k.tolist()): k for k in _input_ids}.values()))

        ctxt_input_ids, _, _ = self.model.contextualize(input_ids, retriever, print_docs=print_docs)
        rag_num_return_sequences = num_return_sequences
        hypos = []

        for index in range(len(input_ids)):
            # first, generate beams from documents:
            generator_input_ids = ctxt_input_ids[index * self.n_docs : (index + 1) * self.n_docs]  # (n_docs, max_len)

            output_sequences = self.model.generator.generate(
                generator_input_ids, num_return_sequences=num_beams, num_beams=num_beams, attention_mask=None, **kwargs
            )  # n_docs * n_beam, tgt_len
            if dedup:
                output_sequences = _get_unique_rows(output_sequences)  # dedup, max_output_len

            # then, run model forwards to get nll scores:
            new_input_ids = input_ids[index : index + 1].repeat(len(output_sequences), 1)
            outputs = self.forward(
                new_input_ids, retriever=retriever, decoder_input_ids=output_sequences, return_loss=True, score=True
            )
            top_cand_inds = (-outputs["loss"]).topk(rag_num_return_sequences)[1]

            if logging.get_verbosity() == logging.DEBUG:
                output_strings = self.model.generator_tokenizer.batch_decode(output_sequences)
                logger.debug("Hypos with scores:")
                for score, hypo in zip(outputs.loss, output_strings):
                    logger.debug("\t{} {}".format(score, hypo))

            hypos.append(output_sequences[top_cand_inds])

        return self._cat_and_pad(hypos, pad_token_id=self.config.pad_token_id)

    def get_nll(self, seq_logits, doc_scores, target, reduce=False, epsilon=0.0, score=False):
        target = self.shift_tokens_left(target)
        # bos_token_id is None for T5
        use_bos = self.config.bos_token_id is not None and target[:, 0].eq(self.config.bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // self.n_docs, self.n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x dim
        doc_logprobs = torch.nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence marginaliation
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)

        # calcualate loss
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, self.n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # sum over tokens, exclude bos while scoring
        ll = ll[:, :, 1:].sum(2) if score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce:
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


@add_start_docstrings_to_callable(
    """A RAG-token model impementation. It performs RAG-token specific marginalization in the forward pass
    and specializes some of the functions of :class:`~transformers.PreTrainedModel` to enable RAG-token generation.
    """,
    RAG_START_DOCSTRING,
)
class RagTokenForGeneration(PreTrainedRagModel):

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,  # or maybe just use a `set_retriever(...)` method
        **kwargs,
    ):
        assert config is not None or (
            encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."
        if config is None:
            config = RagConfig.from_encoder_generator_configs(encoder.config, generator.config, **kwargs)
        super().__init__(config)
        self.rag = RagModel(config=config, encoder=encoder, generator=generator, retriever=retriever)


    @classmethod
    def from_pretrained_encoder_generator(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""Instantiates an encoder and a decoder from one or two base classes of the library from pre-trained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).
        To train the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the encoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/encoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                information necessary to initiate the decoder. Either:

                - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
                - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/decoder``.
                - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            kwargs: (`optional`) Remaining dictionary of keyword arguments.
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attentions=True``).
                - To update the encoder configuration, use the prefix `encoder_` for each configuration parameter
                - To update the decoder configuration, use the prefix `decoder_` for each configuration parameter
                - To update the parent model configuration, do not use a prefix for each configuration parameter
                Behave differently depending on whether a :obj:`config` is provided or automatically loaded.

        Examples::

            >>> from transformers import EncoderDecoderModel
            >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./bert2bert")
            >>> # load fine-tuned model
            >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")

        """

        kwargs_encoder = {
            argument[len("question_encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("generator_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert (
                encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"
            from .modeling_auto import AutoModel

            if "config" not in kwargs_encoder:
                from .configuration_auto import AutoConfig

                encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:

                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                    kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"
            from .modeling_auto import AutoModelForCausalLM

            if "config" not in kwargs_decoder:
                from .configuration_auto import AutoConfig

                decoder_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or decoder_config.add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config)

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        return self.model.generator.adjust_logits_during_generation(logits, cur_len, max_length)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        last_hidden_state = encoder_outputs["last_hidden_state"]
        doc_scores = encoder_outputs["doc_scores"]
        attention_mask = encoder_outputs["attentions"]

        beam_size = decoder_input_ids.shape[0] // doc_scores.shape[0]
        doc_scores = doc_scores.repeat_interleave(beam_size, dim=0)  # batch_size -> batch_size * beam_size
        attention_mask = attention_mask.repeat_interleave(beam_size, dim=0)  # batch_size -> batch_size * beam_size

        encoder_outputs = BaseModelOutputWithDocs(
            last_hidden_state=_stack_ctxt(last_hidden_state),
            hidden_states=encoder_outputs.hidden_states,
            attentions=attention_mask,
            doc_scores=doc_scores,
        )

        print_docs = getattr(kwargs, "print_docs", False)

        return {
            "input_ids": None,
            "retriever": kwargs["retriever"],
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past,
            "use_cache": use_cache,
            "marginalize": True,
            "print_docs": print_docs,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Reorders cache for generation. BART-inspired but we need to take care of the extra dimension for docs"""

        def _reorder_stacked(t):
            n_docs = t.shape[0] // beam_idx.shape[0]
            t = _unstack_ctxt(t, n_docs).index_select(0, beam_idx)
            return _stack_ctxt(t)

        def _reorder_buffer(attn_cache):
            for k, input_buffer_k in attn_cache.items():
                if input_buffer_k is not None:
                    attn_cache[k] = _reorder_stacked(input_buffer_k)
            return attn_cache

        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {attn_key: _reorder_buffer(attn_cache) for attn_key, attn_cache in layer_past.items()}
            reordered_past.append(layer_past_new)

        return reordered_past

    def marginalize(self, seq_logits, doc_scores):
        # RAG-token marginalization
        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // self.n_docs, self.n_docs, -1, seq_logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    @add_start_docstrings_to_callable(RAG_FORWARD_INPUTS_DOCSTRING, RAG_LOSS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutputWithDocs, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids,
        retriever,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        past_key_values=None,
        context_input_ids=None,  # NEW
        doc_vectors=None,  # NEW
        use_cache=None,
        return_loss=False,
        reduce=False,
        label_smoothing=0.0,
        marginalize=False,
        **generator_kwargs
    ):
        r"""
            marginalize (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`True`, `logits`, returned as part of :class:`~transformers.file_utils.Seq2SeqLMOutputWithDocs` are marginalized, yielding
                the shape of :obj:`(batch_size, sequence_length, hidden_size)`. Otherwise we return raw, non-marginalized logits of shape
                :obj:`(batch_size * n_docs, sequence_length, hidden_size)`. ``marginalize`` is set to :obj:`True` during generation. The parameter is
                ignored if ``return_loss`` is set to :obj:`True`.

        Returns:
        """

        if return_loss:
            use_cache = False

        outputs = self.rag(
            input_ids,
            retriever,
            attention_mask,
            encoder_outputs,
            decoder_input_ids,
            context_input_ids,  # NEW
            doc_vectors,  # NEW
            past_key_values,
            use_cache,
            **generator_kwargs,
        )

        # compute doc scores or leave it at end of RagModel
        doc_scores = torch.bmm(outputs.query_vectors.unsqueeze(1), outputs.doc_vectors.transpose(1, 2)).squeeze(1)

        if return_loss:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits, outputs.doc_scores, decoder_input_ids, reduce=reduce, epsilon=label_smoothing
            )
            return Seq2SeqLMOutputWithDocs(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
                doc_scores=doc_scores,
            )

        logits = self.marginalize(outputs.logits, outputs.doc_scores) if marginalize else outputs.logits

        return Seq2SeqLMOutputWithDocs(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            doc_scores=outputs.doc_scores,
        )

    def get_input_embeddings(self):
        return self.model.generator.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.generator.get_output_embeddings()

    def get_encoder(self):
        return RAGEncoder(self.model)

    def get_nll(self, seq_logits, doc_scores, target, reduce=False, epsilon=0.0):
        target = self.shift_tokens_left(target)

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        rag_logprobs = self.marginalize(seq_logits, doc_scores)

        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)  # sum over tokens
        smooth_obj = smooth_obj.sum(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss
