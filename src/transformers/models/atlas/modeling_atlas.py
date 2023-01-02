# coding=utf-8
# Copyright 2022, The ATLAS Authors and The HuggingFace Inc. team.
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
"""ATLAS model implementation."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from functools import reduce
import numpy as np

from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_atlas import AtlasConfig
from .tokenization_atlas import AtlasTokenizer

from .retriever import Contriever, UntiedDualEncoderRetriever, DualEncoderRetriever
from .fid import FiD
from .retrieval_atlas import AtlasRetrieverIndex

from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends

if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "AtlasConfig"

@dataclass
class AtlasModelOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        generator_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        retriever_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Retriever loss.
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
            average in the self-attention heads.
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
            average in the self-attention heads.
        generator_dec_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    generator_loss: Optional[torch.FloatTensor] = None
    retriever_loss: Optional[torch.FloatTensor] = None
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


class AtlasPreTrainedModel(PreTrainedModel):
    config_class = AtlasConfig
    base_model_prefix = "atlas"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)
    
    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        index: Dataset = None,
        **kwargs
    ) -> PreTrainedModel:
        pass
        

class AtlasModel(AtlasPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        retriever: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever_index: Optional[AtlasRetrieverIndex] = None,
        **kwargs,
    ):
        requires_backends(self, ["datasets", "faiss"])
        assert config is not None or (
            retriever is not None and generator is not None
        ), "Either a configuration or an retriever and a generator has to be provided."

        if config is None:
            config = AtlasConfig.from_pretrained_query_passage_generator(
                retriever.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        
        super().__init__(config)
        
        if retriever is None:
            from ..auto.modeling_auto import AutoModel
            contriever = Contriever(config.retriever)
            if config.query_side_retriever_training:
                retriever = UntiedDualEncoderRetriever(config, contriever)
            else:
                retriever = DualEncoderRetriever(config, contriever)

        if generator is None:
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM
            generator = FiD(config.generator)

        if retriever_index is None:
            raise ValueError("retriever_index is None, from_pretrained not implemented yet.")

        self.retriever_index = retriever_index
        self.retriever = retriever
        self.generator = generator

    # todo
    # - make it possible to separate the retrieval and generation steps
    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        query_input_ids,
        query_attention_mask,
        decoder_input_ids=None,
        top_k=5,

        # input_ids=None,
        # attention_mask=None,
        # encoder_hidden_states=None,
        # encoder_attention_mask=None,
        # inputs_embeds=None,
        # head_mask=None,
        # cross_attn_head_mask=None,
        # past_key_values=None,
        # use_cache=None,
        # output_attentions=None,
        # output_hidden_states=None,
        # return_dict=None,
    ):
        bsz = len(input_ids)

        query_hidden_states = self.retriever(input_ids=query_input_ids, attention_mask=query_attention_mask)
        query_hidden_states_numpy = query_hidden_states.cpu().detach().numpy()


        generator_tokens, retriever_tokens = self.retriever_index(query_hidden_states_numpy, input_ids, top_k)
        generator_input_ids = generator_tokens["input_ids"]
        generator_attention_mask = generator_tokens["attention_mask"].bool()

        n_context_training = min(top_k, generator_input_ids.size(1))
        cfg = self.generator.encoder.config
        cfg.bsz = generator_input_ids.size(0)
        cfg.n_context = n_context_training
    
        generator_input_ids_training = generator_input_ids[:, :n_context_training].contiguous()
        generator_attention_mask_training = generator_attention_mask[:, :n_context_training].contiguous()

        generator_input_ids_training = generator_input_ids_training.view(generator_input_ids.size(0), -1)
        generator_attention_mask_training = generator_attention_mask_training.view(generator_attention_mask.size(0), -1)

        generator_output = self.generator(
            input_ids=generator_input_ids_training,
            attention_mask=generator_attention_mask_training,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            use_cache=False,
        )

        reader_loss = generator_output[0]

        train_retriever = self.config.query_side_retriever_training and self.training

        retriever_loss = None
        if train_retriever:

            query_emb = self.retriever(input_ids=query_input_ids, attention_mask=query_attention_mask, is_passages=False)
            retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()}

            passage_emb = self.retriever(**retriever_tokens, is_passages=True).to(query_emb)
            passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))
            retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb])

            if decoder_input_ids is None:
                decoder_input_ids = self.generator._shift_right(labels)

            gold_score = self.perplexity_score(generator_input_ids, generator_attention_mask, decoder_input_ids, labels, cfg, bsz)
            retriever_score = retriever_score / np.sqrt(query_emb.size(-1))
            gold_score = gold_score.float()
            retriever_score = retriever_score.float()
            retriever_loss = self.kldivloss(retriever_score, gold_score)

            if self.training:
                self.generator.train()

        return AtlasModelOutput(
            generator_loss=reader_loss,
            retriever_loss=retriever_loss,
            logits=generator_output.logits,
            # doc_scores=outputs.doc_scores,
            # past_key_values=outputs.past_key_values,
            # context_input_ids=outputs.context_input_ids,
            # context_attention_mask=outputs.context_attention_mask,
            # retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            # retrieved_doc_ids=outputs.retrieved_doc_ids,
            # question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            # question_enc_hidden_states=outputs.question_enc_hidden_states,
            # question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=generator_output.encoder_last_hidden_state,
            generator_enc_hidden_states=generator_output.encoder_hidden_states,
            generator_enc_attentions=generator_output.encoder_attentions,
            generator_dec_hidden_states=generator_output.decoder_hidden_states,
            generator_dec_attentions=generator_output.decoder_attentions,
            generator_cross_attentions=generator_output.cross_attentions,
        )


    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score / self.config.temperature_gold, dim=-1)
        score = torch.nn.functional.log_softmax(score / self.config.temperature_score, dim=-1)
        return torch.nn.KLDivLoss()(score, gold_score)
    
    def generate(
        self,
        input_ids,
        attention_mask,
        query_input_ids,
        query_attention_mask,
        decoder_input_ids=None,
        top_k=5,
    ):
        query_hidden_states = self.retriever(input_ids=query_input_ids, attention_mask=query_attention_mask, is_passages=False)
        query_hidden_states_numpy = query_hidden_states.cpu().detach().numpy()

        generator_tokens, _ = self.retriever_index(query_hidden_states_numpy, input_ids, top_k)
        generator_input_ids = generator_tokens["input_ids"]
        generator_attention_mask = generator_tokens["attention_mask"].bool()

        n_context_training = min(top_k, generator_input_ids.size(1))
        cfg = self.generator.encoder.config
        cfg.bsz = generator_input_ids.size(0)
        cfg.n_context = n_context_training
    
        generator_input_ids_training = generator_input_ids[:, :n_context_training].contiguous()
        generator_attention_mask_training = generator_attention_mask[:, :n_context_training].contiguous()

        generator_input_ids_training = generator_input_ids_training.view(generator_input_ids.size(0), -1)
        generator_attention_mask_training = generator_attention_mask_training.view(generator_attention_mask.size(0), -1)

        generator_output = self.generator.generate(
            input_ids=generator_input_ids_training,
            attention_mask=generator_attention_mask_training,
            use_cache=False,
        )

        return generator_output


    # @torch.no_grad()
    # def generate(self, tokens, query, choices=None):
    #     cfg = self.generator.encoder.config
    #     cfg.bsz = tokens["input_ids"].size(0)
    #     cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

    #     tokens = {k: v.view(v.size(0), -1) for k, v in tokens.items()}

    #     bos_token_id = None

    #     prefix_allowed_tokens_fn = None
    #     if self.opt.decoder_prompt_format is not None:
    #         prefix_str = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
    #         prefix_allowed_tokens_fn = self.get_prefix_allowed_tokens_fn(prefix_str)

    #     outputs = self.generator.generate(
    #         input_ids=tokens["input_ids"],
    #         attention_mask=tokens["attention_mask"],
    #         num_return_sequences=1,
    #         max_length=self.opt.generation_max_length,
    #         min_length=self.opt.generation_min_length,
    #         num_beams=self.opt.generation_num_beams,
    #         length_penalty=self.opt.generation_length_penalty,
    #         forced_bos_token_id=bos_token_id,
    #         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #     )

    #     return outputs

    def perplexity_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        with torch.no_grad():
            self.generator.eval()
            total_context = reader_ids.size(1)
            cfg.n_context = 1
            cfg.bsz = bsz * total_context
            reader_ids_score = reader_ids.view(bsz * total_context, -1)
            reader_mask_score = reader_mask.view(bsz * total_context, -1)
            repeated_decoder_input_ids = torch.repeat_interleave(decoder_input_ids, total_context, dim=0)
            repeated_labels = torch.repeat_interleave(labels, total_context, dim=0)
            reader_output = self.generator(
                input_ids=reader_ids_score,
                attention_mask=reader_mask_score,
                decoder_input_ids=repeated_decoder_input_ids,
                labels=repeated_labels,
                use_cache=False,
            )
            token_loss = nn.functional.cross_entropy(
                reader_output.logits.view(-1, reader_output.logits.size(-1)),
                repeated_labels.flatten(),
                reduction="none",
            )
            gold_score = token_loss.view(bsz, total_context, -1)
            z = (repeated_labels.view(bsz, total_context, -1) > -1).sum(dim=-1)
            gold_score = -gold_score.sum(dim=-1) / z

            return gold_score
