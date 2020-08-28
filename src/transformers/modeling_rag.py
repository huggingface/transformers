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
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .configuration_rag import RagConfig
from .configuration_utils import PretrainedConfig
from .file_utils import add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_auto import AutoModelForSeq2SeqLM
from .modeling_bart import BartForConditionalGeneration
from .modeling_dpr import DPRQuestionEncoder
from .modeling_outputs import ModelOutput
from .modeling_t5 import T5ForConditionalGeneration
from .modeling_utils import PreTrainedModel
from .retrieval_rag import RagRetriever
from .tokenization_auto import AutoTokenizer
from .tokenization_dpr import DPRQuestionEncoderTokenizer


logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "RagConfig"


@dataclass
class Seq2SeqLMOutputWithDocs(ModelOutput):
    """
    Outputs for sequence-to-sequence language models with retrieval in the loop.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Languaged modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)` if the ``logits are marginalized or :obj:`(batch_size, config.n_docs, sequence_length, config.vocab_size)` if they aren't):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        decoder_past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see ``decoder_past_key_values`` input) to speed up sequential decoding.
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
    decoder_past_key_values: Optional[List[torch.FloatTensor]] = None
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


RAG_START_DOCSTRING = r"""
    RAG is a seq2seq model which encapsulates three core components: a retriever, a question encoder and a generator,
    as well as respective tokenizers. During a forward pass, we encode the input with the question encoder and pass it
    to the retriever to extract relevant context documents. The documents are then prepended to the input.
    Such contextualized input is passed to the generator.

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Args:
        config (:class:`~transformers.RagConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

RAG_CORE_DOCSTRING = r"""
    A base RAG model returning raw sequence logits and document retrieval scores.
    The model takes a retriever, a question encoder (with a tokenizer) and a generator (with a tokenizer)
    as inputs to the constructor, so it can be a base for various RAG architectures
    encapsualting different retrievers and generators.

    Args:
        config (:class:`~transformers.RagConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
        retriever (:class:`~transformers.RagRetriever`):
            A retriever class encapsulating a faiss index queried to obtain context documents used in generation.
        question_encoder (:class:`transformers.PreTrainedModel`):
            An encoder model compatible with the faiss index encapsulated by the ``retriever``.
        question_encoder_tokenizer (:class:`transformers.PreTrainedTokenizer`):
            A tokenizer compatible with the ``question_encoder``.
        generator (:class:`transformers.PreTrainedModel`):
            A seq2seq model used as the generator in the RAG architecture.
        generator_tokenizer (:class:`transformers.PreTrainedTokenizer`):
            A tokenizer compatible with the ``generator``.
"""

RAG_FORWARD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            :class:`~transformers.RagConfig`, used to initialize the model, specifies which generator to use, it also specifies a compatible
            generator tokenizer. Use that tokenizer class to obtain the indices.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`, `doc_scores`)
            `last_hidden_state` of shape :obj:`(batch_size, n_docs * sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the encoder.
            `doc_scores` of shape :obj:`(batch_size, n_docs)` store retrieval scores of documents retrieved for each input in the batch.
            Used by the (:class:`~transformers.RagToken`) model during decoding.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for generation tasks. `None` by default, constuct as per instructions for the generator model you're using with your RAG instance.
        decoder_past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`):
            Tuple consists of two elements: ``encoder_outputs`` of the RAG model (see ``encoder_outputs``) and ``decoder_past_key_values`` of the underlying generator.
            Can be used to speed up decoding. ``decoder_past_key_values`` are used in the (:class:`~transformers.RagToken`)
            model during decoding.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If `use_cache` is True, ``decoder_past_key_values`` are returned and can be used to speed up decoding (see
            ``decoder_past_key_values``).
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
class RagModel(torch.nn.Module):
    def __init__(
        self,
        config,
        retriever,
        question_encoder,
        question_encoder_tokenizer,
        generator,
        generator_tokenizer,
    ):
        super().__init__()
        self.config = config
        self.retriever = retriever
        self.question_encoder = question_encoder
        self.question_encoder_tokenizer = question_encoder_tokenizer
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.n_docs = self.config.n_docs
        self._validate_configs_match(self.config, self.generator.config)

    def _validate_configs_match(self, rag_config, gen_config):
        assert rag_config.pad_token_id == gen_config.pad_token_id, "pad_token_id mismatch: {} vs. {}".format(
            rag_config.pad_token_id, gen_config.pad_token_id
        )
        assert rag_config.bos_token_id == gen_config.bos_token_id, "bos_token_id mismatch: {} vs. {}".format(
            rag_config.bos_token_id, gen_config.bos_token_id
        )
        assert rag_config.eos_token_id == gen_config.eos_token_id, "eos_token_id mismatch: {} vs. {}".format(
            rag_config.eos_token_id, gen_config.eos_token_id
        )
        assert (
            rag_config.decoder_start_token_id == gen_config.decoder_start_token_id
        ), "decoder_start_token_id mismatch: {} vs. {}".format(
            rag_config.decoder_start_token_id, gen_config.decoder_start_token_id
        )
        assert (
            rag_config.is_encoder_decoder == gen_config.is_encoder_decoder
        ), "pad_token_id mismatch: {} vs. {}".format(rag_config.is_encoder_decoder, gen_config.is_encoder_decoder)
        assert rag_config.vocab_size == gen_config.vocab_size, "vocab_size mismatch: {} vs. {}".format(
            rag_config.vocab_size, gen_config.vocab_size
        )

    @add_start_docstrings_to_callable(RAG_FORWARD_INPUTS_DOCSTRING, RAG_LOSS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutputWithDocs, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_past_key_values=None,
        use_cache=None,
        doc_scores=None,
        print_docs=False,
        **generator_kwargs
    ):
        r"""
            print_docs  (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If :obj:`True`, documents retrieved during the forward pass will be logged. Intended for debugging purposes.

        Returns:

        """
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is not None:
            doc_scores = encoder_outputs[-1]
        else:
            # Add context documents to input
            input_ids, attention_mask, doc_scores = self.contextualize(input_ids, print_docs=print_docs)

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(self.n_docs, dim=0)

        outputs = self.generator(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_past_key_values=decoder_past_key_values,
            use_cache=use_cache,
            **generator_kwargs,
        )
        return Seq2SeqLMOutputWithDocs(
            loss=None,
            logits=outputs.logits,
            decoder_past_key_values=outputs.decoder_past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            doc_scores=doc_scores,
        )

    def _cat_input_and_doc(self, doc_score, doc_title, doc_text, input_string, add_eos, print_docs=False):
        # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
        # TODO(piktus): handle truncation
        if doc_title.startswith('"'):
            doc_title = doc_title[1:]
        if doc_title.endswith('"'):
            doc_title = doc_title[:-1]
        prefix = self.generator.config.prefix if self.generator.config.prefix is not None else ""
        suffix = self.generator_tokenizer.eos_token if add_eos else ""
        out = (
            prefix + doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string + suffix
        ).replace("  ", " ")
        if print_docs:
            logger.info("{} {}".format(doc_score, out))
        return out

    def contextualize(self, input_ids, print_docs=False):
        r"""
        Retrieves context documents for every input in the batch.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            print_docs  (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If :obj:`True`, documents retrieved during the forward pass will be printed out. Intended for debugging purposes.

        Return:
            :obj:`tuple(tuple(torch.FloatTensor)`: a tuple consisting od three elements: contextualized ``input_ids``,
                compatible ``attention_mask`` and scores of the retrieved documents.
        """

        device = input_ids.device
        # T5 tokenizer doesn't add eos token by default evan with add_special_tokens set to True
        add_eos = (input_ids == self.generator_tokenizer.eos_token_id).any() and isinstance(
            self.generator, T5ForConditionalGeneration
        )

        input_strings = self.generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # handle prefix for T5
        if isinstance(self.generator, T5ForConditionalGeneration):
            for i, s in enumerate(input_strings):
                if not s.startswith(self.generator.config.prefix):
                    logger.warning("T5 prefix mismatch in {}".format(s))
                if len(input_strings[i]) <= len(self.generator.config.prefix):
                    input_strings[i] = ""
                else:
                    input_strings[i] = input_strings[i][len(self.generator.config.prefix) :]

        retriever_inputs = self.question_encoder_tokenizer.batch_encode_plus(
            input_strings,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        retriever_input_embs = self.question_encoder(retriever_inputs["input_ids"].to(device))[0]

        doc_scores, docs = self.retriever.retrieve(retriever_input_embs, n_docs=self.n_docs)

        rag_input_strings = [
            self._cat_input_and_doc(
                doc_scores[i][j], docs[i]["title"][j], docs[i]["text"][j], input_strings[i], add_eos, print_docs
            )
            for i in range(len(docs))
            for j in range(self.n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(device)

        return (
            contextualized_inputs["input_ids"],
            contextualized_inputs["attention_mask"],
            doc_scores.to(device),
        )


class RAGEncoder(torch.nn.Module):
    r"""
    RAG is an encoder-decoder model, however, we don't exaplicitly implement an encoder and a decoder layes,
    like it's done e.g. in BART and T5 implementations - for RAG these are encapsulated inside the generaotr instance.
    This is a dummy model simulating RAG encoder output, need for compatibility with transformers generation code
    """

    def __init__(self, rag_model: RagModel):
        super().__init__()
        self.rag_model = rag_model

    def forward(self, input_ids=None, attention_mask=None):
        ctxt_input_ids, ctxt_attention_mask, doc_scores = self.rag_model.contextualize(input_ids, print_docs=False)
        encoder = self.rag_model.generator.get_encoder()  # stacked
        encoder_outputs = encoder(input_ids=ctxt_input_ids, attention_mask=ctxt_attention_mask, return_dict=True)
        # needed to satisfy assertions in generation_utils
        unstacked_x = _unstack_ctxt(encoder_outputs.last_hidden_state, self.rag_model.n_docs)

        return (unstacked_x, encoder_outputs.hidden_states, ctxt_attention_mask, doc_scores)


class PreTrainedRagModel(PreTrainedModel):
    r"""
    RAG models encapsulate two trainable components - a question encoder and a generator, but as such they don't have any trainable parameters.
    We specialize `:func:`~transformers.PreTrainedModel.from_pretrained`` and `:func:`~transformers.PreTrainedModel.save_pretrained`` to reflect this.
    """
    config_class = RagConfig

    def __init__(
        self,
        config: RagConfig,
        retriever,
        question_encoder,
        question_encoder_tokenizer,
        generator,
        generator_tokenizer,
    ):
        super().__init__(config)
        self.model = RagModel(
            config,
            retriever,
            question_encoder,
            question_encoder_tokenizer,
            generator,
            generator_tokenizer,
        )
        self.n_docs = self.config.n_docs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, **kwargs):
        r"""
        Instantiates a pretrained RAG model from a pre-trained model configuration. Since RAG doesn't have any trainable parameters
        other than those encapsulated by the ``question _encoder`` and the ``generator``, we call `:func:`~transformers.PreTrainedModel.from_pretrained``
        for the ``question_encoder`` and the ``generator`` respectively. The method also uses the provided configuration to initialize the ``retriever``,
        the ``question_encoder_tokenizer`` and the ``generator_tokenizer``.

        Parameters:
            pretrained_model_name_or_path (:obj:`str`, `optional`):
                A string specifying the model to be loaded. See :func:`~transformers.PreTrainedModel.from_pretrained`` for details.
            config (:obj:`Union[PretrainedConfig, str]`, `optional`):
                Can be either:

                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                    - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

                See :func:`~transformers.PreTrainedModel.from_pretrained`` for more details.
            generator_config (:obj:`str`, `optional`):
                A string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`. Will be passed
                to the :func:`~transformers.PreTrainedModel.from_pretrained`` function initializing the ``generator`` model.
            question_encoder_config (:obj:`str`, `optional`):
                A string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`. Will be passed
                to the :func:`~transformers.PreTrainedModel.from_pretrained`` function initializing the ``question_encoder`` model.
            kwargs (remaining dictionary of keyword arguments, `optional`):
                `kwargs`` will be passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`).
                Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                attribute will be passed to the underlying model's ``__init__`` function.
        """
        config = kwargs.pop("config", None)
        generator_config = kwargs.pop("generator_config", None)
        question_encoder_config = kwargs.pop("question_encoder_config", None)

        assert pretrained_model_name_or_path is not None or config is not None
        if not isinstance(config, PretrainedConfig):
            config = cls.config_class.from_pretrained(
                config if config is not None else pretrained_model_name_or_path,
                **kwargs,
            )

        question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            config.pretrained_question_encoder_tokenizer_name_or_path
        )
        # TODO(piktus): To be replaced with AutoModel once it gets published (?)
        question_encoder = DPRQuestionEncoder.from_pretrained(
            config.pretrained_question_encoder_name_or_path, config=question_encoder_config
        )
        generator_tokenizer = AutoTokenizer.from_pretrained(config.pretrained_generator_tokenizer_name_or_path)
        generator_kwargs = {}
        if generator_config is not None:
            setattr(generator_config, "return_dict", True)
            generator_kwargs["config"] = generator_config
        else:
            generator_kwargs["return_dict"] = True
        generator = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_generator_name_or_path, **generator_kwargs)

        retriever = RagRetriever(config)

        model = cls(config, retriever, question_encoder, question_encoder_tokenizer, generator, generator_tokenizer)
        return model

    def save_pretrained(self, save_directory):
        r"""
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedRagModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str`):
                Base directory to which to save. Will be created if it doesn't exist. The generator model
                will be saved to save_directory/generator directory. The question encoder model will be saved
                to save_directory/genquestion_encoder directory.
        """
        if os.path.isfile(save_directory):
            logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
            return
        os.makedirs(save_directory, exist_ok=True)
        generator_output_dir = os.path.join(save_directory, "generator")
        self.model.generator.save_pretrained(generator_output_dir)
        qe_output_dir = os.path.join(save_directory, "question_encoder")
        self.model.question_encoder.save_pretrained(qe_output_dir)
        config = copy.deepcopy(self.config)
        config.pretrained_generator_name_or_path = generator_output_dir
        config.pretrained_question_encoder_name_or_path = qe_output_dir
        config.vocab_size = self.model.generator.config.vocab_size
        config.save_pretrained(save_directory)

    def shift_tokens_left(self, input_ids, pad_token_id=None):
        """Shift input ids one token to the left, and add a pad to right"""
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        return torch.cat([input_ids[:, 1:], input_ids.new(input_ids.shape[0], 1).fill_(pad_token_id)], 1)

    def shift_tokens_right(self, input_ids, start_token_id=None):
        """Shift input ids one token to the right, and pad with start_token_id"""
        if start_token_id is None:
            start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        return shifted_input_ids


@add_start_docstrings_to_callable(
    """A RAG-sequence model impementation. It performs RAG-sequence specific marginalization in the forward pass
    and specializes some of the functions of :class:`~transformers.PreTrainedModel` to enable RAG-sequence generation.

    It is compatible with :class:`~transformers.DPRQuestionEncoder` as the ``question_encoder``. As for the ``generator``,
    two compatible architectures have been tested: :class:`~transformers.BartForConditionalGeneration`
    and :class:`~transformers.T5ForConditionalGeneration`.
    """,
    RAG_START_DOCSTRING,
)
class RagSequence(PreTrainedRagModel):

    base_model_prefix = "rag_sequence"

    @add_start_docstrings_to_callable(RAG_FORWARD_INPUTS_DOCSTRING, RAG_LOSS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutputWithDocs, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_past_key_values=None,
        use_cache=None,
        return_loss=False,
        reduce=False,
        label_smoothing=0.0,
        score=False,
        **generator_kwargs
    ):
        r"""
        Returns:
        """
        if return_loss:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask,
            encoder_outputs,
            decoder_input_ids,
            decoder_past_key_values,
            use_cache,
            **generator_kwargs,
        )

        if return_loss:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                decoder_input_ids,
                reduce=reduce,
                epsilon=label_smoothing,
                score=score,
            )
            return Seq2SeqLMOutputWithDocs(
                loss=loss,
                logits=outputs.logits,
                decoder_past_key_values=outputs.decoder_past_key_values,
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
            decoder_past_key_values=outputs.decoder_past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            doc_scores=outputs.doc_scores,
        )

    def generate(
        self,
        input_ids,
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

        ctxt_input_ids, _, _ = self.model.contextualize(input_ids, print_docs=print_docs)
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
            outputs = self.forward(new_input_ids, decoder_input_ids=output_sequences, return_loss=True, score=True)
            top_cand_inds = (-outputs["loss"]).topk(rag_num_return_sequences)[1]

            if logger.level == logging.DEBUG:
                output_strings = self.model.generator_tokenizer.batch_decode(output_sequences)
                logger.debig("Hypos with scores:")
                for score, hypo in zip(outputs.loss, output_strings):
                    logger.debug("\t{} {}".format(score, hypo))

            hypos.append(output_sequences[top_cand_inds])

        return self._cat_and_pad(hypos, pad_token_id=self.model.generator_tokenizer.pad_token_id)

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

    It is compatible with :class:`~transformers.DPRQuestionEncoder` as the ``question_encoder``
    and :class:`~transformers.BartForConditionalGeneration` as ``generator``.
    """,
    RAG_START_DOCSTRING,
)
class RagToken(PreTrainedRagModel):

    base_model_prefix = "rag_token"

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        """BART-specific adjustment"""
        assert isinstance(self.model.generator, BartForConditionalGeneration)
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """BART-specific adjustment - force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # past cold start
        if past[1] is None:
            encoder_last_hidden_state, encoder_hidden_states, attention_mask, doc_scores = past[0]
            beam_size = encoder_last_hidden_state.shape[0] // doc_scores.shape[0]
            doc_scores = doc_scores.repeat_interleave(beam_size, dim=0)  # batch_size -> batch_size * beam_size
            attention_mask = attention_mask.repeat_interleave(beam_size, dim=0)  # batch_size -> batch_size * beam_size
            encoder_outputs = (_stack_ctxt(encoder_last_hidden_state), encoder_hidden_states, doc_scores)
            decoder_past_key_values = None
        # past warm start
        else:
            encoder_outputs, decoder_past_key_values = past
            attention_mask = encoder_outputs[-2]

        print_docs = getattr(kwargs, "print_docs", False)

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_past_key_values": decoder_past_key_values,
            "use_cache": use_cache,
            "marginalize": True,
            "print_docs": print_docs,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Reorders cache for generation."""

        (enc_out, enc_mask, attention_mask, doc_scores), decoder_past_key_values = past
        n_docs = doc_scores.shape[1]

        def _reorder_stacked(t):
            t = _unstack_ctxt(t, n_docs).index_select(0, beam_idx)
            return _stack_ctxt(t)

        def _reorder_buffer(attn_cache):
            for k, input_buffer_k in attn_cache.items():
                if input_buffer_k is not None:
                    attn_cache[k] = _reorder_stacked(input_buffer_k)
            return attn_cache

        reordered_decoder_past_key_values = []
        for layer_past in decoder_past_key_values:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {attn_key: _reorder_buffer(attn_cache) for attn_key, attn_cache in layer_past.items()}
            reordered_decoder_past_key_values.append(layer_past_new)

        enc_out = _reorder_stacked(enc_out) if enc_out is not None else None
        enc_mask = _reorder_stacked(enc_mask) if enc_mask is not None else None
        doc_scores = doc_scores.index_select(0, beam_idx)
        attention_mask = _reorder_stacked(attention_mask)

        return ((enc_out, enc_mask, attention_mask, doc_scores), reordered_decoder_past_key_values)

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
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_past_key_values=None,
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
                :obj:`(batch_size, n_docs, sequence_length, hidden_size)`. ``marginalize`` is set to :obj:`True` during generation. The parameter is
                ignored if ``return_loss`` is set to :obj:`True`.

        Returns:
        """

        if return_loss:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask,
            encoder_outputs,
            decoder_input_ids,
            decoder_past_key_values,
            use_cache,
            **generator_kwargs,
        )

        if outputs.decoder_past_key_values is not None:
            (enc_out, enc_mask), decoder_past_key_values = outputs.decoder_past_key_values
            decoder_past_key_values = (
                (enc_out, enc_mask, attention_mask, outputs.doc_scores),
                decoder_past_key_values,
            )
        else:
            decoder_past_key_values = None

        if return_loss:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits, outputs.doc_scores, decoder_input_ids, reduce=reduce, epsilon=label_smoothing
            )
            return Seq2SeqLMOutputWithDocs(
                loss=loss,
                logits=outputs.logits,
                decoder_past_key_values=decoder_past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
                doc_scores=outputs.doc_scores,
            )

        logits = self.marginalize(outputs.logits, outputs.doc_scores) if marginalize else outputs.logits

        return Seq2SeqLMOutputWithDocs(
            loss=None,
            logits=logits,
            decoder_past_key_values=decoder_past_key_values,
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
