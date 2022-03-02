# coding=utf-8
# Copyright 2020, The FID Authors and The HuggingFace Inc. team.
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
"""FID model implementation."""

from base64 import decode
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...configuration_utils import PretrainedConfig
from ...file_utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_fid import FiDConfig
from .retrieval_fid import FiDRetriever


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "FiDConfig"
FID_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/fid-nq-base",
    "facebook/fid-nq-large",
    "facebook/fid-tqa-base",
    "facebook/fid-tqa-large",
]


@dataclass
class FiDModelOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`.
        past_key_values (`List[torch.FloatTensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `torch.FloatTensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see `past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (`torch.FloatTensor` of shape `(batch_size, config.n_docs, hidden_size)`, *optional*, returned when *output_retrieved=True*):
            Embedded documents retrieved by the retriever. Is used with `question_encoder_last_hidden_state` to compute
            the `doc_scores`.
        retrieved_doc_ids (`torch.LongTensor` of shape `(batch_size, config.n_docs)`, *optional*, returned when *output_retrieved=True*):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.
        question_encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            avefide in the self-attention heads.
        generator_enc_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            avefide in the self-attention heads.
        generator_dec_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            avefide in the self-attention heads.
        generator_cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted avefide in the cross-attention heads.
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
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class FiDPreTrainedModel(PreTrainedModel):
    r"""
    FID models were released with the paper [Retrieval-Augmented Generation for Knowledge-Intensive NLP
    Tasks](https://arxiv.org/abs/2007.01282) by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    FID is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.

    """
    config_class = FiDConfig
    base_model_prefix = "fid"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        retriever: FiDRetriever = None,
        **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiates an question encoder and a generator from one or two base classes of the library from pretrained
        model checkpoints.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            question_encoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the question encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            generator_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the generator. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            retriever ([`FiDRetriever`], *optional*):
                The retriever to use.
            kwwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the question_encoder configuration, use the prefix *question_encoder_* for each
                  configuration parameter.
                - To update the generator configuration, use the prefix *generator_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import FiDModel

        >>> # initialize a FID from two pretrained models.
        >>> model = FiDModel.from_pretrained_question_encoder_generator(
        ...     "facebook/dpr-question_encoder-single-nq-base", "t5-small"
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./fid")
        >>> # load fine-tuned model
        >>> model = FiDModel.from_pretrained("./fid")
        ```"""

        kwargs_question_encoder = {
            argument[len("question_encoder_") :]: value
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

                question_encoder_config, kwargs_question_encoder = AutoConfig.from_pretrained(
                    question_encoder_pretrained_model_name_or_path,
                    **kwargs_question_encoder,
                    return_unused_kwargs=True,
                )
                kwargs_question_encoder["config"] = question_encoder_config

            question_encoder = AutoModel.from_pretrained(
                question_encoder_pretrained_model_name_or_path, **kwargs_question_encoder
            )

        generator = kwargs_generator.pop("model", None)
        if generator is None:
            assert (
                generator_pretrained_model_name_or_path is not None
            ), "If `generator_model` is not defined as an argument, a `generator_pretrained_model_name_or_path` has to be defined"
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM

            if "config" not in kwargs_generator:
                from ..auto.configuration_auto import AutoConfig

                generator_config, kwargs_generator = AutoConfig.from_pretrained(
                    generator_pretrained_model_name_or_path, **kwargs_generator, return_unused_kwargs=True
                )

                kwargs_generator["config"] = generator_config

            generator = AutoModelForSeq2SeqLM.from_pretrained(
                generator_pretrained_model_name_or_path, **kwargs_generator
            )

        # instantiate config with corresponding kwargs
        config = kwargs.get("config", None)
        if config is None:
            config = FiDConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        return cls(question_encoder=question_encoder, generator=generator, config=config, retriever=retriever)


FID_START_DOCSTRING = r"""

    FID is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward
    pass, we encode the input with the question encoder and pass it to the retriever to extract relevant context
    documents. The documents are then prepended to the input. Such contextualized inputs is passed to the generator.

    The question encoder can be any *autoencoding* model, preferably [`DPRQuestionEncoder`], and the generator can be
    any *seq2seq* model, preferably [`BartForConditionalGeneration`].

    The model can be initialized with a [`FiDRetriever`] for end-to-end generation or used in combination with the
    outputs of a retriever in multiple steps---see examples for more details. The model is compatible any
    *autoencoding* model as the `question_encoder` and any *seq2seq* model with language model head as the `generator`.
    It has been tested with [`DPRQuestionEncoder`] as the `question_encoder` and [`BartForConditionalGeneration`] or
    [`T5ForConditionalGeneration`] as the `generator`.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.


    Args:
        config ([`FiDConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        question_encoder ([`PreTrainedModel`]):
            An encoder model compatible with the faiss index encapsulated by the `retriever`.
        generator ([`PreTrainedModel`]):
            A seq2seq model used as the generator in the FID architecture.
        retriever ([`FiDRetriever`]):
            A retriever class encapsulating a faiss index queried to obtain context documents for current inputs.
"""


FID_FORWARD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [`FiDConfig`], used to initialize the model, specifies
            which generator to use, it also specifies a compatible generator tokenizer. Use that tokenizer class to
            obtain the indices.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*)
            Tuple consists of (`generator_enc_last_hidden_state`, *optional*: `generator_enc_hidden_states`,
            *optional*: `generator_enc_attentions`). `generator_enc_last_hidden_state` of shape `(batch_size, n_docs *
            sequence_length, hidden_size)` is a sequence of hidden-states at the output of the last layer of the
            generator's encoder.

            Used by the ([`FiDModel`]) model during decoding.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Provide for generation tasks. `None` by default, construct as per instructions for the generator model
            you're using with your FID instance.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size,  target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        past_key_values (`tuple(tuple(torch.FloatTensor))`):
            Tuple consists of two elements: `encoder_outputs` of the FID model (see `encoder_outputs`) and
            `past_key_values` of the underlying generator. Can be used to speed up decoding. `past_key_values` are used
            in the ([`FiDTokenForGeneration`]) model during decoding.
        doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
            `question_encoder_last_hidden_state`. If the model has is not initialized with a `retriever` `doc_scores`
            has to be provided to the forward pass. `doc_scores` can be computed via
            `question_encoder_last_hidden_state` and `retrieved_doc_embeds`, see examples for more information.
        context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
            Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
            retriever.

            If the model has is not initialized with a `retriever` ``context_input_ids` has to be provided to the
            forward pass. `context_input_ids` are returned by [`~FiDRetriever.__call__`]. context_attention_mask
            (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*,
            returned when *output_retrieved=True*): Attention mask post-processed from the retrieved documents and the
            question encoder `input_ids` by the retriever.

            If the model has is not initialized with a `retriever` `context_attention_mask` has to be provided to the
            forward pass. `context_attention_mask` are returned by [`~FiDRetriever.__call__`].
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_retrieved(`bool`, *optional*):
            Whether or not to return the `retrieved_doc_embeds`, `retrieved_doc_ids`, `context_input_ids` and
            `context_attention_mask`. See returned tensors for more detail.
        n_docs (`int`, *optional*, defaults to `config.n_docs``)
            Number of documents to retrieve and/or number of documents for which to generate an answer.
"""


@add_start_docstrings_to_model_forward(FID_START_DOCSTRING)
class FiDModel(FiDPreTrainedModel):
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
            config = FiDConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
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
                retriever, FiDRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `FiDRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

        self.ctx_encoder = None
        self.context_encoder_training = False

    @add_start_docstrings_to_model_forward(FID_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FiDModelOutput, config_class=_CONFIG_FOR_DOC)
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
        labels=None,
    ):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import RagTokenizer, FiDModel
        >>> import torch

        >>> tokenizer = RagTokenizer.from_pretrained("facebook/fid-token-base")
        >>> model = FiDModel.from_pretrained_question_encoder_generator(
        ...     question_encoder_pretrained_model_name_or_path="bert-base-uncased",
        ...     generator_pretrained_model_name_or_path="t5-small",
        ...     retriever=None,
        ... )

        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
        >>> outputs = model(input_ids=inputs["input_ids"])
        ```"""
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
        # TODO this if is skipped as we are directly passing `context_input_ids`, `context_attention_mask`, and `doc_scores`
        # encoder_outputs are pre-computed during FID-token generation
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
                if self.context_encoder_training:

                    (
                        context_input_ids,
                        context_attention_mask,
                        retrieved_doc_embeds,
                        retrived_doc_input_ids,
                        retrived_doc_attention_mask,
                        retrieved_doc_ids,
                    ) = (
                        retriever_outputs["context_input_ids"],
                        retriever_outputs["context_attention_mask"],
                        retriever_outputs["retrieved_doc_embeds"],
                        retriever_outputs["tokenized_doc_ids"],
                        retriever_outputs["tokenized_doc_attention_mask"],
                        retriever_outputs["doc_ids"],
                    )

                    context_input_ids = context_input_ids.to(input_ids)
                    context_attention_mask = context_attention_mask.to(input_ids)

                    retrived_doc_input_ids = retrived_doc_input_ids.to(input_ids)
                    retrived_doc_attention_mask = retrived_doc_attention_mask.to(input_ids)
                    retrieved_doc_embeds = self.ctx_encoder(
                        retrived_doc_input_ids, attention_mask=retrived_doc_attention_mask, return_dict=True
                    ).pooler_output
                    retrieved_doc_embeds = retrieved_doc_embeds.view(
                        -1, n_docs, question_encoder_last_hidden_state.shape[1]
                    )  # reshaping

                    # compute doc_scores involving ctx_encoder
                    doc_scores = torch.bmm(
                        question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)

                else:
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

        encoder = self.generator.get_encoder()
        decoder = self.generator.get_decoder()

        if context_input_ids.dim() == 3:
            encoder.n_passages = context_input_ids.size(1)

        # We need to resize as (B * N) x L here
        context_input_ids = context_input_ids.view(-1, context_input_ids.size(2))
        context_attention_mask = context_attention_mask.view(-1, context_attention_mask.size(2))

        encoder_outputs = encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        batch_size = encoder_outputs["last_hidden_state"].size(0) // n_docs
        max_length, hidden_size = encoder_outputs["last_hidden_state"].size(-2), encoder_outputs[
            "last_hidden_state"
        ].size(-1)
        encoder_outputs["last_hidden_state"] = encoder_outputs["last_hidden_state"].view(
            batch_size, n_docs * max_length, hidden_size
        )

        decoder_input_ids = self.shift_tokens_right(input_ids=labels)

        decoder_output = decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits = self.generator.lm_head(decoder_output[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.generator.config.vocab_size), labels.view(-1))

        # gen_outputs = self.generator(
        #     input_ids=context_input_ids,
        #     attention_mask=context_attention_mask,
        #     encoder_outputs=encoder_outputs,
        #     decoder_input_ids=decoder_input_ids,
        #     decoder_attention_mask=decoder_attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     return_dict=True,
        # )
        generator_enc_hidden_states = (
            encoder_outputs.hidden_states
            if encoder_outputs.hidden_states is not None
            else self.generator.config.output_hidden_states
        )
        generator_enc_attentions = (
            encoder_outputs.attentions
            if encoder_outputs.attentions is not None
            else self.generator.config.output_attentions
        )

        generator_dec_hidden_states = (
            decoder_output.hidden_states
            if decoder_output.hidden_states is not None
            else self.generator.config.output_hidden_states
        )
        generator_dec_attentions = (
            decoder_output.attentions
            if decoder_output.attentions is not None
            else self.generator.config.output_attentions
        )

        generator_cross_attentions = (
            decoder_output.cross_attentions
            if generator_dec_attentions is not None
            else self.generator.config.output_attentions
        )

        # TODO change lines 646-661 after adding retriever

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

        return FiDModelOutput(
            loss=loss,
            logits=logits,
            doc_scores=doc_scores,
            past_key_values=decoder_output.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=encoder_outputs.last_hidden_state,
            generator_enc_hidden_states=generator_enc_hidden_states,
            generator_enc_attentions=generator_enc_attentions,
            generator_dec_hidden_states=generator_dec_hidden_states,
            generator_dec_attentions=generator_dec_attentions,
            generator_cross_attentions=generator_cross_attentions,
        )

    def shift_tokens_right(self, input_ids, start_token_id=None):
        if start_token_id is None:
            start_token_id = self.generator.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id

        shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.generator.config.pad_token_id)
        return shifted_input_ids
