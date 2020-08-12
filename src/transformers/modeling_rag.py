import logging
import os
import pprint
import string
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .configuration_rag import RagConfig
from .configuration_utils import PretrainedConfig
from .modeling_auto import AutoModel, AutoModelWithLMHead
from .modeling_bart import BartForConditionalGeneration
from .modeling_dpr import DPRContextEncoder, DPRQuestionEncoder
from .modeling_outputs import Seq2SeqLMOutputWithDocs
from .modeling_utils import PreTrainedModel
from .retrieval_rag import HFRetriever, MPIRetriever
from .tokenization_auto import AutoTokenizer
from .tokenization_bart import BartTokenizer
from .tokenization_dpr import DPRContextEncoderTokenizer
from .tokenization_rag import RagDefaultTokenizer
from .tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


# Helper functions.

# Reshape from [batch_size, n_docs, dims] to [batch_size * n_docs, dims]
def _stack_ctxt(tensor):
    return tensor.view(-1, *tensor.shape[2:])


# Reshape from [batch_size * n_docs, dims] to [batch_size, n_docs, dims]
def _unstack_ctxt(tensor, n_docs):
    return tensor.view(-1, n_docs, *tensor.shape[1:])


def _cat_and_pad(tensors, pad_token_id):
    output = tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
    ind = 0
    for t in tensors:
        output[ind : ind + t.shape[0], : t.shape[1]] = t
        ind += t.shape[0]
    return output


def _shift_tokens_left(input_ids, pad_token_id):
    """Shift input ids one token to the left, and add a pad to right"""
    return torch.cat([input_ids[:, 1:], input_ids.new(input_ids.shape[0], 1).fill_(pad_token_id)], 1)


class RagModel(torch.nn.Module):
    """This is a basic RAG model returning raw sequence and document logprobs.
    The model takes a retriever (with a tokenizer) and a generator (with a tokenizer)
    as input to the constructor, so it can be a base for various RAG architectures
    encapsualting different retrievers and generators.
    """

    def __init__(
        self, config, retriever, retriever_tokenizer, generator, generator_tokenizer, question_encoder,
    ):
        super().__init__()
        self.config = config
        self.retriever = retriever
        self.question_encoder = question_encoder
        self.retriever_tokenizer = retriever_tokenizer
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.n_docs = self.config.n_docs
        assert self.n_docs > 1  # dont support k > 1
        # TODO(piktus) validate if generator and retriever compatible / generator rag config compatible ??

    def forward(
        self, input_ids, decoder_input_ids=None, encoder_outputs: Optional[Tuple] = None, doc_scores=None, **kwargs,
    ):
        """
        returns a tuple ((generator seq logprobs, retriever_doc_scores), generator decoder outputs, generator encoder outputs)
        """
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is not None:
            doc_scores = encoder_outputs[-1]
        else:
            # Add context documents to input
            input_ids, attention_mask, doc_scores = self.contextualize(input_ids)
            kwargs["attention_mask"] = attention_mask

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(self.n_docs, dim=0)

        outputs = self.generator(
            input_ids, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, **kwargs,
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

    def _cat_input_and_doc(self, doc_score, doc_title, doc_text, input_string, print_docs=False):
        # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
        if doc_title.startswith('"'):
            doc_title = doc_title[1:]
        if doc_title.endswith('"'):
            doc_title = doc_title[:-1]
        out = (doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string).replace("  ", " ")
        if print_docs:
            print(doc_score, out)
            print()
        return out

    # TODO(piktus): handle truncation
    def contextualize(self, input_ids, print_docs=False):
        """
        Retrieve context documents for every sample in the batch
        input_ids dim: [batch_size, src_len]
        ctxt_input_ids dim [batch_size * num_docs, src_len]
        """
        device = input_ids.device
        input_strings = self.generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        retriever_inputs = self.retriever_tokenizer.batch_encode_plus(
            input_strings, return_tensors="pt", padding=True, truncation=True,
        )
        retriever_input_embs = self.question_encoder(retriever_inputs["input_ids"].to(device))[0]
        doc_scores, docs = self.retriever.retrieve(
            retriever_input_embs, n_docs=self.n_docs, query_strings=input_strings
        )
        # top_doc_scores = torch.bmm(query_vectors.unsqueeze(1), top_doc_vectors.transpose(1, 2)).squeeze(1)

        rag_input_strings = [
            self._cat_input_and_doc(
                doc_scores[i][j], docs[i]["title"][j], docs[i]["text"][j], input_strings[i], print_docs
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
            torch.FloatTensor(doc_scores).to(device),
        )


class RAGEncoder(torch.nn.Module):
    """RAG is an encoder-decoder model, however, we don't exaplicitly implement an encoder and a decoder layes,
    like it's done e.g. in BART and T5 implementations - for RAG these are encapsulated inside the generaotr instance.
    This is a dummy model simulating RAG encode output need in transformers generation code"""

    def __init__(self, rag_model: RagModel):
        super().__init__()
        self.rag_model = rag_model

    def forward(self, input_ids=None, attention_mask=None):
        ctxt_input_ids, ctxt_attention_mask, doc_scores = self.rag_model.contextualize(input_ids, print_docs=True)
        encoder = self.rag_model.generator.get_encoder()  # stacked
        encoder_outputs = encoder(input_ids=ctxt_input_ids, attention_mask=ctxt_attention_mask)
        # needed to satisfy assertions in generation_utils
        unstacked_x = _unstack_ctxt(encoder_outputs.last_hidden_state, self.rag_model.n_docs)

        return (unstacked_x, encoder_outputs.hidden_states, doc_scores)


class PreTrainedRagModel(PreTrainedModel):
    def __init__(
        self, config: RagConfig, retriever, retriever_tokenizer, generator, generator_tokenizer, question_encoder,
    ):
        super().__init__(config)
        self.model = RagModel(
            config, retriever, retriever_tokenizer, generator, generator_tokenizer, question_encoder,
        )
        self.n_docs = self.config.n_docs
        assert self.n_docs > 1  # dont support k > 1

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """RAG models wrap 2 core components: a retriever and a generator, but as such they don't have any trainable parameters
        other then those encapsulated in the components. We specialize from_pretrained function to reflect this.
        """
        config = (
            RagConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            if "config" not in kwargs
            else kwargs["config"]
        )
        retriever_tokenizer = AutoTokenizer.from_pretrained(config.pretrained_context_tokenizer_name_or_path)
        # TODO(piktus): To be replaced with AutoModel once it gets published (?)
        question_encoder = DPRQuestionEncoder.from_pretrained(config.pretrained_question_encoder_name_or_path)

        # TODO(piktus): handle multi-node scenarios for Retriever loading
        if config.retriever_type == "hf_retriever":
            retriever = HFRetriever(
                config.dataset,
                dataset_name=config.dataset_name,
                dataset_split=config.dataset_split,
                index_name=config.index_name,
                uncompressed=config.uncompressed,
                uncompressed_index_path=config.uncompressed_index_path,
            )
        elif config.retriever_type == "mpi_retriever":
            retriever = MPIRetriever(
                passages_path=config.passages_path,
                vector_size=config.retrieval_vector_size,
                index_path=config.index_path,
                n_docs=config.n_docs,
                batch_size=config.retrieval_batch_size,
                use_sampling=False,
            )

        generator_tokenizer = AutoTokenizer.from_pretrained(config.pretrained_generator_tokenizer_name_or_path)
        generator = AutoModelWithLMHead.from_pretrained(config.pretrained_generator_name_or_path)
        model = cls(config, retriever, retriever_tokenizer, generator, generator_tokenizer, question_encoder)
        return model


class RagSequenceModel(PreTrainedRagModel):
    """ This is a base class for RAG sequence model. It performs RAG-sequence specific marginalization in the forward pass
    and specialized some of the functions of PreTrainedModel to enable RAG-sequence generation.
    """

    config_class = RagConfig
    base_model_prefix = "rag_sequence"

    def forward(self, input_ids, decoder_input_ids=None, return_loss=False, **kwargs):
        if return_loss:
            kwargs["use_cache"] = False
        outputs = self.model(input_ids, decoder_input_ids, **kwargs)

        if return_loss:
            assert decoder_input_ids is not None
            outputs.loss = self.get_nll(outputs.logits, outputs.doc_scores, decoder_input_ids)

        return outputs

    def generate(self, input_ids, **kwargs):
        """implements RAG sequence "thorough" decoding"""

        def _get_unique_rows(_input_ids):
            return torch.stack(list({str(k.tolist()): k for k in _input_ids}.values()))

        ctxt_input_ids, _, _ = self.model.contextualize(input_ids, print_docs=True)
        nrs = kwargs.get("num_return_sequences", 1)
        kwargs["num_return_sequences"] = kwargs.get("num_beams", 1)
        hypos = []

        for index in range(len(input_ids)):
            # first, generate beams from documents:
            generator_input_ids = ctxt_input_ids[index * self.n_docs : (index + 1) * self.n_docs]  # (n_docs, max_len)

            output_sequences = self.model.generator.generate(
                generator_input_ids, **kwargs
            )  # n_docs * n_beam, max_output_len
            output_sequences = _get_unique_rows(output_sequences)  # dedup, max_output_len
            output_strings = self.model.generator_tokenizer.batch_decode(
                output_sequences
            )  # , skip_special_tokens=True)

            # then, run model forwards to get nll scores:
            new_input_ids = input_ids[index : index + 1].repeat(len(output_sequences), 1)

            outputs = self.forward(new_input_ids, decoder_input_ids=output_sequences, return_loss=True)
            top_cand_inds = (-outputs.loss).topk(nrs)[1]

            """
            print("hypos with scores")
            for score, hypo in zip(outputs.loss, output_strings):
                print("\t", score, hypo)
            """
            hypos.append(output_sequences[top_cand_inds])

        return _cat_and_pad(hypos, pad_token_id=self.model.generator_tokenizer.pad_token_id)

    def get_nll(self, seq_logits, doc_scores, target):
        """get negative log likelihood from rag log probs"""

        ignore_index = self.model.generator_tokenizer.pad_token_id
        target = _shift_tokens_left(target, ignore_index)

        def _mask_pads(ll):
            pad_mask = target.eq(ignore_index)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1)

        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // self.n_docs, self.n_docs, -1, seq_logits.size(-1)
        )
        print("doc scores", doc_scores[0])
        doc_logprobs = torch.nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        print("doc probs", torch.exp(doc_logprobs)[0].squeeze())

        # RAG-sequence marginaliation
        first_token_scores = seq_logprobs[:, :, :1, :]
        remainder = seq_logprobs[:, :, 1:, :]
        rag_logprobs = torch.cat([first_token_scores + doc_logprobs, remainder], dim=2,)

        # calcualate loss
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, self.n_docs, 1, 1)
        ll = rag_logprobs.gather(dim=-1, index=target)
        ll = _mask_pads(ll)
        ll = ll.sum(2)  # sum over tokens
        ll = ll.logsumexp(1)  # logsumexp over docs
        return -ll


class RagTokenModel(PreTrainedRagModel):
    """ This is a base class for RAG-token model. It performs RAG-token specific marginalization in the forward pass
    and specialized some of the functions of PreTrainedModel to enable RAG-token generation.
    """

    config_class = RagConfig
    base_model_prefix = "rag_token"

    # TODO(piktus) BART specific adjustment - figure out how to generalize to other architectures
    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        """Copied from BART
        """
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """Copied from BART.
        force one of token_ids to be generated by setting prob of all other tokens to 0"""
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
            encoder_last_hidden_state, encoder_hidden_states, doc_scores = past[0]
            beam_size = encoder_last_hidden_state.shape[0] // doc_scores.shape[0]
            # resize doc_scores: batch_size -> batch_size * beam_size
            doc_scores = doc_scores.repeat_interleave(beam_size, dim=0)
            encoder_outputs = _stack_ctxt(encoder_last_hidden_state), encoder_hidden_states, doc_scores
            decoder_past_key_values = None
        # past warm start
        else:
            encoder_outputs, decoder_past_key_values = past

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "decoder_past_key_values": decoder_past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask.repeat_interleave(self.n_docs, dim=0),
            "use_cache": use_cache,
            "marginalize": True,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Mostly copied from BART, but you need to handle extra dimensions"""

        (enc_out, enc_mask, doc_scores), decoder_past_key_values = past
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

        return ((enc_out, enc_mask, doc_scores), reordered_decoder_past_key_values)

    def marginalize(self, seq_logits, doc_scores):
        # RAG-token marginalization
        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // self.n_docs, self.n_docs, -1, seq_logits.size(-1)
        )
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    def forward(
        self,
        input_ids,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        return_loss=False,
        marginalize=False,
        **kwargs,
    ):
        if return_loss:
            kwargs["use_cache"] = False

        outputs = self.model(input_ids, decoder_input_ids, encoder_outputs, **kwargs)

        (enc_out, enc_mask), decoder_past_key_values = outputs.decoder_past_key_values
        outputs.decoder_past_key_values = (enc_out, enc_mask, outputs.doc_scores), decoder_past_key_values

        if return_loss:
            assert decoder_input_ids is not None
            outputs.loss = self.get_nll(rag_logprobs, decoder_input_ids)
            return outputs

        if marginalize:
            outputs.logits = self.marginalize(outputs.logits, outputs.doc_scores)
        return outputs

    def get_input_embeddings(self):
        return self.model.generator.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.generator.get_output_embeddings()

    def get_encoder(self):
        return RAGEncoder(self.model)

    def get_nll(self, rag_logprobs, target):
        """get negative log likelihood from rag log probs"""
        ignore_index = self.model.generator_tokenizer.pad_token_id
        target = _shift_tokens_left(target, ignore_index)

        def _mask_pads(ll):
            pad_mask = target.eq(ignore_index)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1)

        rag_logprobs = self.marginalize(seq_logits, doc_scores)

        target = target.unsqueeze(1)

        ll = rag_logprobs.gather(dim=-1, index=target)
        ll = _mask_pads(ll)
        ll = ll.sum(1)  # sum over tokens

        return -ll


# TODO(piktus): class RagForSequenceClassification(PreTrainedModel):
