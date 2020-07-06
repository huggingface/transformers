import os

import numpy as np
import torch
from nlp import load_dataset

from transformers import (
    BartForConditionalGeneration,
    BartModel,
    BartTokenizer,
    DprConfig,
    DprContextEncoder,
    DprQuestionEncoder,
    DprTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)


torch.set_grad_enabled(False)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # fix libiomp5.dylib initialization


def shift_tokens_left(input_ids, pad_token_id):
    """Shift input ids one token to the left, and add a pad to right"""
    return torch.cat([input_ids[:, 1:], input_ids.new(input_ids.shape[0], 1).fill_(pad_token_id)], 1)


def _cat_and_pad(tensors, pad_token_id):
    output = tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
    ind = 0
    for t in tensors:
        output[ind : ind + t.shape[0], : t.shape[1]] = t
        ind += t.shape[0]
    return output


class Retriever(torch.nn.Module):
    def __init__(self, config: DprConfig, tokenizer: DprTokenizer):
        super().__init__()
        self.config = config
        self.ctx_encoder = DprContextEncoder(config)  # .eval()n
        self.device = torch.device("cpu")
        self.tokenizer = tokenizer
        self.index = None

    def to(self, *args, **kwargs):
        r = super().to(*args, **kwargs)
        self.device = kwargs["device"]
        return r

    # TODO: embedding is slow, can we load embeddings / index from file while loading dataset?
    def load_index(self):
        print("Loading dataset")
        wiki = load_dataset("wikipedia", "20200501.simple", split="train[:100]")
        wiki_passages = wiki.map(lambda ex: {"text": self._crop(ex["text"]), "title": ex["title"].strip()})
        print("Adding embeddings")
        with torch.no_grad():
            wiki_passages = wiki_passages.add_embeddings(self._embed_ctx, batched=True, batch_size=64)
        print("Indexing")
        wiki_passages.init_index(device=0)  # faiss index with device=0 for gpu
        self.index = wiki_passages

    def _crop(self, text: str, max_spaces=100):
        """Dummy function to create a snippet out of a full article"""
        if text.count(" ") > max_spaces:
            text = " ".join(text.split(" ", max_spaces + 1)[:-1])
        return text.replace("\n", " ")

    def _embed_ctx(self, examples):
        tokenized_examples = torch.cat(
            [
                self.tokenizer.encode(
                    title.strip(),
                    text_pair=text.replace("\n", " "),
                    return_tensors="pt",
                    pad_to_max_length=True,
                    model_max_length=512,
                )[:, :512]
                for title, text in zip(examples["title"], examples["text"])
            ],
            dim=0,
        ).to(self.device)
        return self.ctx_encoder(tokenized_examples).cpu().numpy()

    def retrieve(self, question_embs, k=5):
        return self.index.get_nearest_batch(question_embs, k)


def _stack_enc(tensor, k):
    return tensor.view(-1, *tensor.shape[2:])


def _unstack_enc(tensor, k):
    return tensor.view(-1, k, *tensor.shape[1:])


class RAGEnc(torch.nn.Module):
    """Dummy model needed for running rag seq2seq encoder in transformers generation code"""

    def __init__(self, RagModel):
        super().__init__()
        self.rag_model = RagModel

    def forward(
        self, input_ids=None, attention_mask=None,
    ):
        ctxt_input_ids, ctxt_attention_mask, doc_scores = self.rag_model.contextualize(input_ids)
        enc = self.rag_model.generator.get_encoder()
        encoder_out = enc(input_ids=ctxt_input_ids, attention_mask=ctxt_attention_mask,)
        stacked_enc_out = _unstack_enc(encoder_out[0], self.rag_model.k)
        return (stacked_enc_out,) + encoder_out[1:] + (doc_scores,)


class RagModel(PreTrainedModel):
    def __init__(
        self, config, retriever, retriever_tokenizer, generator, generator_tokenizer,
    ):
        super().__init__(config)
        self.config = config
        self.retriever = retriever
        self.question_encoder = DprQuestionEncoder(self.retriever.config)
        self.retriever_tokenizer = retriever_tokenizer

        self.generator = generator
        self.generator_tokenizer = generator_tokenizer

        self.k = self.config.k
        assert self.k > 1  # dont support k > 1

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        """Mostly copied from BART"""
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        encoder_outputs, decoder_cached_states = (past, None) if not past[1] else past

        enc_outs, doc_scores = encoder_outputs[0], encoder_outputs[-1]

        if (len(enc_outs.shape) == 4) and (enc_outs.shape[0] != doc_scores.shape[0]):
            new_doc_scores = doc_scores.repeat_interleave(enc_outs.shape[0] // doc_scores.shape[0], dim=0)
            encoder_outputs = encoder_outputs[:-1] + (new_doc_scores,)

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

        def _reorder_buffer(attn_cache, new_order, n_docs):
            for k, input_buffer_k in attn_cache.items():
                if input_buffer_k is not None:
                    input_buffer_k = _unstack_enc(input_buffer_k, n_docs)
                    attn_cache[k] = _stack_enc(input_buffer_k.index_select(0, new_order), n_docs)
            return attn_cache

        (enc_out, enc_mask, doc_scores), decoder_cached_states = past
        n_docs = doc_scores.shape[1]
        enc_out = _unstack_enc(enc_out, n_docs)
        enc_mask = _unstack_enc(enc_mask, n_docs)

        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx, n_docs) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else _stack_enc(enc_out.index_select(0, beam_idx), n_docs)
        new_enc_mask = enc_mask if enc_mask is None else _stack_enc(enc_mask.index_select(0, beam_idx), n_docs)
        new_doc_scores = doc_scores.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask, new_doc_scores), reordered_past)
        return past

    def _forward_rag_token_generate(self, input_ids, decoder_input_ids, encoder_outputs, **kwargs):
        """sequence transition function for rag token (rag sequence can just reply on bart)"""
        if len(encoder_outputs[0].shape) == 4:
            encoder_outputs = (_stack_enc(encoder_outputs[0], self.k),) + encoder_outputs[1:]

        kwargs["attention_mask"] = kwargs["attention_mask"].repeat_interleave(self.k, dim=0)
        decoder_input_ids = decoder_input_ids.repeat_interleave(self.k, dim=0)
        outputs = self.generator(
            input_ids, decoder_input_ids=decoder_input_ids, encoder_outputs=encoder_outputs, **kwargs
        )

        seq_logits = outputs[0]
        (enc_out, enc_mask), decoder_cached_states = outputs[1]
        doc_scores = encoder_outputs[-1]

        rag_logprobs, _, _ = self._get_rag_logprobs(doc_scores, seq_logits)
        return (rag_logprobs, ((enc_out, enc_mask, doc_scores), decoder_cached_states), *outputs[2:])

    def _get_rag_logprobs(self, doc_logits, seq_logits):
        """ Marginalize over document predictions to get rag output logprobs"""
        doc_logprobs = torch.log_softmax(doc_logits, dim=1)
        seq_logprobs = torch.nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // self.k, self.k, -1, seq_logits.size(-1)
        )
        if self.config.rag_model_type == "rag_token":
            log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
            rag_logprobs = torch.logsumexp(log_prob_sum, dim=1)
        elif self.config.rag_model_type == "rag_sequence":
            first_token_scores = seq_logprobs[:, :, :1, :]
            remainder = seq_logprobs[:, :, 1:, :]
            rag_logprobs = torch.cat([first_token_scores + doc_logprobs.unsqueeze(-1).unsqueeze(-1), remainder], dim=2)
        else:
            raise Exception("Unrecognized RAG model")
        return rag_logprobs, seq_logprobs, doc_logprobs

    def forward(self, input_ids, decoder_input_ids=None, **kwargs):
        """forward function, if input_ids is none, its called in rag_token generation"""
        if input_ids is None:
            assert self.config.rag_model_type == "rag_token"
            return self._forward_rag_token_generate(input_ids, decoder_input_ids, **kwargs)

        # Add context documents to encoder input
        ctxt_input_ids, ctxt_attention_mask, doc_scores = self.contextualize(input_ids)

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(self.k, dim=0)

        seq_logits = self.generator(
            input_ids=ctxt_input_ids, attention_mask=ctxt_attention_mask, decoder_input_ids=decoder_input_ids
        )[0]
        return self._get_rag_logprobs(doc_scores, seq_logits)

    def _cat_input_and_doc(self, doc, input_string):
        # out = input_string + self.config.title_sep + doc["title"] + self.config.doc_sep + doc["text"] TODO Patrick: if we train more RAG models, I want to put the input first to take advantage of effortless truncation
        out = doc["title"] + self.config.title_sep + doc["text"] + self.config.doc_sep + input_string
        return out.replace("  ", " ")

    def contextualize(self, input_ids, max_combined_length=300):
        """
        Retrieve context documents for every sample in the batch
        input_ids dim: [batch_size, src_len]
        ctxt_input_ids dim [batch_size * num_docs, src_len]
        """
        device = input_ids.device
        input_strings = self.generator_tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        dpr_inputs = self.retriever_tokenizer.batch_encode_plus(
            input_strings, return_tensors="pt", pad_to_max_length=True
        )
        dpr_input_embs = self.question_encoder(dpr_inputs["input_ids"].to(device=device))

        doc_scores, docs = self.retriever.retrieve(np.asarray(dpr_input_embs.cpu(), order="C"), self.k)

        rag_input_strings = [
            self._cat_input_and_doc(docs[i][j], input_strings[i])
            for i in range(len(docs))
            for j in range(len(docs[i]))
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings, max_length=max_combined_length, return_tensors="pt", pad_to_max_length=True
        ).to(device)

        return (
            contextualized_inputs["input_ids"],
            contextualized_inputs["attention_mask"],
            torch.FloatTensor(doc_scores).to(device),
        )

    def get_output_embeddings(self):
        return self.generator.get_output_embeddings()

    def get_encoder(self):
        return RAGEnc(self)

    def _generate_rag_token(self, input_ids, **kwargs):
        return super().generate(input_ids, **kwargs)

    def _generate_rag_sequence(self, input_ids, **kwargs):
        """implements RAG sequence "thorough" decoding"""

        def _get_unique_rows(_input_ids):
            return torch.stack(list({str(k.tolist()): k for k in _input_ids}.values()))

        ctxt_input_ids, _, doc_scores = self.contextualize(input_ids)
        nrs = kwargs.get("num_return_sequences", 1)
        kwargs["num_return_sequences"] = kwargs.get("num_beams", 1)
        ctxt_input_ids, ctxt_attention_mask, doc_scores = self.contextualize(input_ids)

        outputs = []
        for index in range(len(input_ids)):
            # first, generate beams from documents:
            generator_input_ids = ctxt_input_ids[index * self.k : (index + 1) * self.k]  # (k, max_len)
            output_sequences = self.generator.generate(generator_input_ids, **kwargs)
            output_sequences = _get_unique_rows(output_sequences)

            # then, run model forwards to get nll scores:
            rag_logprobs, seq_scores, doc_scores = self.forward(
                input_ids[index : index + 1].repeat(len(output_sequences), 1), decoder_input_ids=output_sequences
            )
            output_candidate_scores = self.get_nll(rag_logprobs, output_sequences)
            top_cand_inds = (-output_candidate_scores).topk(nrs)[1]
            outputs.append(output_sequences[top_cand_inds])

        return _cat_and_pad(outputs, pad_token_id=self.generator_tokenizer.pad_token_id)

    def generate(self, *args, **kwargs):
        if self.config.rag_model_type == "rag_token":
            return self._generate_rag_token(*args, **kwargs)
        elif self.config.rag_model_type == "rag_sequence":
            return self._generate_rag_sequence(*args, **kwargs)
        else:
            raise Exception("Unrecognized RAG model")

    def get_nll(self, rag_log_probs, target):
        """get negative log likelihood from rag log probs"""
        ignore_index = self.generator_tokenizer.pad_token_id
        target = shift_tokens_left(target, ignore_index)

        def _mask_pads(ll):
            pad_mask = target.eq(ignore_index)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1)

        if self.config.rag_model_type == "rag_token":
            target = target.unsqueeze(1)
            if target.dim() == rag_log_probs.dim() - 1:
                target = target.unsqueeze(-1)

            ll = rag_log_probs.gather(dim=-1, index=target)
            ll = _mask_pads(ll)
            ll = ll.sum(1)  # sum over tokens

        elif self.config.rag_model_type == "rag_sequence":
            target = target.unsqueeze(1).repeat(1, self.k, 1)
            if target.dim() == rag_log_probs.dim() - 1:
                target = target.unsqueeze(-1)

            ll = rag_log_probs.gather(dim=-1, index=target)
            ll = _mask_pads(ll)
            ll = ll.sum(2)  # sum over tokens
            ll = ll.logsumexp(1)  # logsumexp v
        else:
            raise Exception("Unrecognized RAG model")
        return -ll
