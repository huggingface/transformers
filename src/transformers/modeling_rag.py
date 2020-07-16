import logging
import os
import pprint
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from nlp import load_dataset

from .configuration_rag import RagConfig
from .configuration_utils import PretrainedConfig
from .modeling_bart import BartForConditionalGeneration
from .modeling_dpr import DPRContextEncoder, DPRQuestionEncoder
from .modeling_utils import PreTrainedModel
from .tokenization_bart import BartTokenizer
from .tokenization_dpr import DPRContextEncoderTokenizer
from .tokenization_rag import RagDefaultTokenizer
from .tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)

logging.basicConfig(
    # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    # datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARNING  # if args.local_rank in [-1, 0] else logging.WARN,
)


class Retriever(torch.nn.Module):
    """
    Encapsulation of the retrieval index. We may not need a separate class
    for it given how Datasets are evolving but keeping
    it here for now as it may be easier to have it separate when implementing
    multi-node training.

    TODO(piktus): Implement handling of multi-node training.
    """

    def __init__(
        self,
        dataset,
        dataset_name=None,
        dataset_split=None,
        index_name="embeddings",
        doc_encoder: PreTrainedModel = None,
        doc_tokenizer: PreTrainedTokenizer = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.doc_encoder = doc_encoder
        self.doc_tokenizer = doc_tokenizer
        assert (self.doc_encoder is None) == (self.doc_tokenizer is None)
        self.device = torch.device("cpu")
        self.index = self.build_index() if self.doc_encoder is not None else self.load_index()

    # Build an index from scratch given a dataset. Higly inefficien.
    # Keeping this for now for testing - expects title and text columns
    def build_index(self):
        def _embed_ctx(examples):
            batch_inputs = self.doc_tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=list(zip(examples["title"], examples["text"])),
                return_tensors="pt",
                padding="max_length",
                max_length=None,
                truncation=True,
            )["input_ids"].to(self.device)
            return self.doc_encoder(batch_inputs)[0].cpu().numpy()

        dataset = load_dataset(self.dataset, self.dataset_name, split=self.dataset_split)
        with torch.no_grad():
            dataset_with_embeddings = dataset.map(
                lambda ex: {self.index_name: _embed_ctx(ex)}, batched=True, batch_size=16
            )
        # no device specified - index on CPU
        dataset_with_embeddings.add_faiss_index(column=self.index_name)
        return dataset_with_embeddings

    # Loading a dataset with a pre-computed index.
    def load_index(self):
        # any risk datasets will be silently updated? can I trust it'll remain build by a specific model? add version?
        return load_dataset(self.dataset, self.dataset_name, with_index=True, split=self.dataset_split)

    def retrieve(self, question_embs, k=5):
        return self.index.get_nearest_examples_batch(self.index_name, question_embs, k)


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


class RAGEncoder(torch.nn.Module):
    """RAG is an encoder-decoder model, however, we don't exaplicitly implement an encoder and a decoder layes,
    like it's done e.g. in BART and T5 implementations - for RAG these are encapsulated inside the generaotr instance.
    This is a dummy model simulating RAG encode output need in transformers generation code"""

    def __init__(self, RagModel):
        super().__init__()
        self.rag_model = RagModel

    def forward(self, input_ids=None, attention_mask=None):
        """
        returns a tuple (encoder_embeddings, encoder_hidden_states, encoder_attentions, doc_logprobs)
        """
        ctxt_input_ids, ctxt_attention_mask, doc_logprobs = self.rag_model.contextualize(input_ids)
        encoder = self.rag_model.generator.get_encoder()
        encoder_out = encoder(input_ids=ctxt_input_ids, attention_mask=ctxt_attention_mask)
        unstacked_x = _unstack_ctxt(encoder_out[0], self.rag_model.n_docs)

        return (unstacked_x,) + encoder_out[1:] + (doc_logprobs,)


class RagModel(PreTrainedModel):
    """This is a basic RAG model returning raw sequence and document logprobs.
    The model takes a retriever (with a tokenizer) and a generator (with a tokenizer)
    as input to the constructor, so it can be a base for various RAG architectures
    encapsualting different retrievers and generators.
    """

    def __init__(
        self, config, retriever, retriever_tokenizer, generator, generator_tokenizer, question_encoder,
    ):
        super().__init__(config)
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
        self, input_ids, decoder_input_ids=None, encoder_outputs: Optional[Tuple] = None, **kwargs,
    ):
        """
        returns a tuple ((generator seq logprobs, retriever_doc_logprobs), generator decoder outputs, generator encoder outputs)
        """
        if encoder_outputs is not None:
            # encoder_outputs[0] shape (batch_size, num_docs, ...)
            encoder_outputs = (_stack_ctxt(encoder_outputs[0]),) + encoder_outputs[1:]
            attention_mask = kwargs["attention_mask"].repeat_interleave(self.n_docs, dim=0)
            kwargs.pop("attention_mask", None)
            doc_logprobs = encoder_outputs[-1]
        else:
            # Add context documents to input
            input_ids, attention_mask, doc_logprobs = self.contextualize(input_ids)

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(self.n_docs, dim=0)

        outputs = self.generator(
            input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )
        # could add a parameter to return logits instead of logprobs
        seq_logits = outputs[0]
        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // self.n_docs, self.n_docs, -1, seq_logits.size(-1)
        )
        return ((doc_logprobs, seq_logprobs),) + outputs[1:]

    def _cat_input_and_doc(self, doc_title, doc_text, input_string):
        # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
        out = doc_title + self.config.title_sep + doc_text + self.config.doc_sep + input_string
        return out.replace("  ", " ")

    # TODO(piktus): handle truncation
    def contextualize(self, input_ids):
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
        retriever_input_embs = (
            self.question_encoder(retriever_inputs["input_ids"].to(device))[0].detach().cpu().numpy()
        )
        doc_scores, docs = self.retriever.retrieve(retriever_input_embs, self.n_docs)
        doc_logprobs = torch.log_softmax(torch.FloatTensor(doc_scores).to(device), dim=1)
        rag_input_strings = [
            self._cat_input_and_doc(docs[i]["title"][j], docs[i]["text"][j], input_strings[i])
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
            doc_logprobs,
        )


class RagSequenceModel(PreTrainedModel):
    """ This is a base class for RAG sequence model. It performs RAG-sequence specific marginalization in the forward pass
    and specialized some of the functions of PreTrainedModel to enable RAG-sequence generation.
    """

    config_class = RagConfig
    base_model_prefix = "rag_sequence"

    def __init__(
        self, config, retriever, retriever_tokenizer, generator, generator_tokenizer, question_encoder,
    ):
        super().__init__(config)
        self.model = RagModel(
            config, retriever, retriever_tokenizer, generator, generator_tokenizer, question_encoder,
        )
        self.n_docs = self.config.n_docs
        assert self.n_docs > 1  # dont support k > 1

    def forward(self, input_ids, decoder_input_ids=None, return_loss=False, **kwargs):
        if return_loss:
            kwargs["use_cache"] = False
        outputs = self.model(input_ids, decoder_input_ids, **kwargs)
        doc_logprobs, seq_logprobs = outputs[0]
        first_token_scores = seq_logprobs[:, :, :1, :]
        remainder = seq_logprobs[:, :, 1:, :]
        rag_logprobs = torch.cat([first_token_scores + doc_logprobs.unsqueeze(-1).unsqueeze(-1), remainder], dim=2,)
        if return_loss:
            assert decoder_input_ids is not None
            loss = self.get_nll(rag_logprobs, decoder_input_ids).sum()
            logger.info("loss", loss.shape, loss)
            return (loss, rag_logprobs,) + outputs[1:]
        return (rag_logprobs,) + outputs[1:]

    def generate(self, input_ids, **kwargs):
        """implements RAG sequence "thorough" decoding"""

        def _get_unique_rows(_input_ids):
            return torch.stack(list({str(k.tolist()): k for k in _input_ids}.values()))

        ctxt_input_ids, _, doc_scores = self.model.contextualize(input_ids)
        nrs = kwargs.get("num_return_sequences", 1)
        kwargs["num_return_sequences"] = kwargs.get("num_beams", 1)
        outputs = []

        for index in range(len(input_ids)):
            # first, generate beams from documents:
            generator_input_ids = ctxt_input_ids[index * self.n_docs : (index + 1) * self.n_docs]  # (k, max_len)
            generator_input_string = self.model.generator_tokenizer.batch_decode(
                generator_input_ids, skip_special_tokens=True
            )
            output_sequences = self.model.generator.generate(
                generator_input_ids, **kwargs
            )  # k * n_beam, max_output_len
            output_string = self.model.generator_tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            output_sequences = _get_unique_rows(output_sequences)  # dedup, max_output_len
            output_string = self.model.generator_tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            # then, run model forwards to get nll scores:
            new_input_ids = input_ids[index : index + 1].repeat(len(output_sequences), 1)
            new_input_str = self.model.generator_tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)

            rag_logprobs = self.forward(new_input_ids, decoder_input_ids=output_sequences, use_cache=False)[0]
            output_candidate_scores = self.get_nll(rag_logprobs, output_sequences)
            top_cand_inds = (-output_candidate_scores).topk(nrs)[1]
            outputs.append(output_sequences[top_cand_inds])

        return _cat_and_pad(outputs, pad_token_id=self.model.generator_tokenizer.pad_token_id)

    def get_nll(self, rag_logprobs, target):
        """get negative log likelihood from rag log probs"""

        ignore_index = self.model.generator_tokenizer.pad_token_id
        target = _shift_tokens_left(target, ignore_index)

        def _mask_pads(ll):
            pad_mask = target.eq(ignore_index)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1)

        target = target.unsqueeze(1).repeat(1, self.n_docs, 1)
        if target.dim() == rag_logprobs.dim() - 1:
            target = target.unsqueeze(-1)

        ll = rag_logprobs.gather(dim=-1, index=target)
        ll = _mask_pads(ll)
        ll = ll.sum(2)  # sum over tokens
        ll = ll.logsumexp(1)  # logsumexp v
        return -ll


class RagTokenModel(PreTrainedModel):
    """ This is a base class for RAG-token model. It performs RAG-token specific marginalization in the forward pass
    and specialized some of the functions of PreTrainedModel to enable RAG-token generation.
    """

    config_class = RagConfig
    base_model_prefix = "rag_token"

    def __init__(
        self, config, retriever, retriever_tokenizer, generator, generator_tokenizer, question_encoder,
    ):
        super().__init__(config)
        self.model = RagModel(
            config, retriever, retriever_tokenizer, generator, generator_tokenizer, question_encoder,
        )

        self.n_docs = self.config.n_docs
        assert self.n_docs > 1  # dont support k > 1

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        encoder_outputs, decoder_cached_states = past  # (past, None) if not past[1] else past
        batch_size = encoder_outputs[0].shape[0]
        doc_logprobs = encoder_outputs[-1]

        # repeat the doc scores to match the beam size - before the first iteration of generation
        if batch_size != doc_logprobs.shape[0]:
            # logger.info("reshaping doc scores", batch_size, doc_logprobs.shape[0])
            doc_logprobs = doc_logprobs.repeat_interleave(batch_size // doc_logprobs.shape[0], dim=0)
            encoder_outputs = encoder_outputs[:-1] + (doc_logprobs,)

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Mostly copied from BART, but you need to handle extra dimensions"""

        (enc_out, enc_mask, doc_logprobs), decoder_cached_states = past
        n_docs = doc_logprobs.shape[1]

        def _reorder_buffer(attn_cache, new_order):
            for k, input_buffer_k in attn_cache.items():
                if input_buffer_k is not None:
                    input_buffer_k = _unstack_ctxt(input_buffer_k, n_docs)
                    attn_cache[k] = _stack_ctxt(input_buffer_k.index_select(0, new_order))
            return attn_cache

        reordered_decoder_cached_states = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_decoder_cached_states.append(layer_past_new)

        enc_out = _unstack_ctxt(enc_out, n_docs).index_select(0, beam_idx) if enc_out is not None else None
        enc_mask = _unstack_ctxt(enc_mask, n_docs).index_select(0, beam_idx) if enc_mask is not None else None
        doc_logprobs = doc_logprobs.index_select(0, beam_idx)

        return ((enc_out, enc_mask, doc_logprobs), reordered_decoder_cached_states)

    def forward(
        self, input_ids, decoder_input_ids=None, encoder_outputs: Optional[Tuple] = None, return_loss=False, **kwargs,
    ):
        outputs = self.model(input_ids, decoder_input_ids, encoder_outputs, **kwargs)
        doc_logprobs, seq_logprobs = outputs[0]
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        rag_logprobs = torch.logsumexp(log_prob_sum, dim=1)
        (enc_out, enc_mask), decoder_cached_states = outputs[1]

        if return_loss:
            assert decoder_input_ids is not None
            loss = self.get_nll(rag_logprobs, decoder_input_ids).sum()
            return (loss, rag_logprobs, ((enc_out, enc_mask, doc_logprobs), decoder_cached_states),) + outputs[2:]
        return (rag_logprobs, ((enc_out, enc_mask, doc_logprobs), decoder_cached_states),) + outputs[2:]

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

        target = target.unsqueeze(1)
        if target.dim() == rag_logprobs.dim() - 1:
            target = target.unsqueeze(-1)

        ll = rag_logprobs.gather(dim=-1, index=target)
        ll = _mask_pads(ll)
        ll = ll.sum(1)  # sum over tokens

        return -ll


class RagDefaultMixin(object):
    """ A wrapper around an initializer of default RAG components - a DPR retriever and a BART generator.
    """

    def init_default_components(self, config: RagConfig):
        # models eval by default when loading from pretrained https://fburl.com/vvgl3h07
        dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained(config.pretrained_context_tokenizer_name_or_path)
        dpr_question_encoder = DPRQuestionEncoder.from_pretrained(config.pretrained_question_encoder_name_or_path)
        # TODO(piktus): handle multi-node scenarios for Retriever loading
        # no need to pass context tokenizer or encoder as context embeddings are pre-calculated in the dataset
        dpr_retriever = Retriever(
            config.dataset,
            dataset_name=config.dataset_name,
            dataset_split=config.dataset_split,
            index_name=config.index_name,
        )

        # this is the default BART tokenizer with extra tokens added for tile separator and doc deparator
        bart_tokenizer = RagDefaultTokenizer.from_pretrained(config.pretrained_generator_tokenizer_name_or_path)
        bart = BartForConditionalGeneration.from_pretrained(config.pretrained_generator_name_or_path)
        return dpr_retriever, dpr_tokenizer, bart, bart_tokenizer, dpr_question_encoder


class RagDefaultSequenceModel(RagDefaultMixin, RagSequenceModel):
    """ The default RAG-sequence architecture as proposed in the paper - with  a DPR retriever and a BART generaotr
    """

    def __init__(self, config):
        super().__init__(config, *self.init_default_components(config))


class RagDefaultTokenModel(RagDefaultMixin, RagTokenModel):
    """ The default RAG-token architecture as proposed in the paper - with  a DPR retriever and a BART generaotr
    """

    def __init__(self, config):
        super().__init__(config, *self.init_default_components(config))


class RagTestSequenceModel(RagSequenceModel):
    """ A specialization of a RagSequence re-calcualting index from scratch for a dataset using a DPR context encoder.
    For testing purposes.
    """

    def __init__(self, config):
        dpr_context_encoder = DPRContextEncoder.from_pretrained(config.pretrained_context_encoder_name_or_path)
        dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained(config.pretrained_context_tokenizer_name_or_path)
        dpr_question_encoder = DPRQuestionEncoder.from_pretrained(config.pretrained_question_encoder_name_or_path)

        # TODO(piktus): handle multi-node scenarios for Retriever loading
        dpr_retriever = Retriever(
            dataset=config.dataset,
            dataset_name=config.dataset_name,
            dataset_split=config.dataset_split,
            index_name=config.index_name,
            doc_encoder=dpr_context_encoder,
            doc_tokenizer=dpr_tokenizer,
        )

        # this is the default BART tokenizer with extra tokens added for tile separator and doc deparator
        bart_tokenizer = RagDefaultTokenizer.from_pretrained(config.pretrained_generator_tokenizer_name_or_path)
        bart = BartForConditionalGeneration.from_pretrained(config.pretrained_generator_name_or_path)
        super().__init__(config, dpr_retriever, dpr_tokenizer, bart, bart_tokenizer, dpr_question_encoder)


# TODO(piktus): class RagForSequenceClassification(PreTrainedModel):
