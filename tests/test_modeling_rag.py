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


import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from transformers.file_utils import (
    cached_property,
    is_datasets_available,
    is_faiss_available,
    is_psutil_available,
    is_torch_available,
)
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.tokenization_bart import BartTokenizer
from transformers.tokenization_bert import VOCAB_FILES_NAMES as DPR_VOCAB_FILES_NAMES
from transformers.tokenization_dpr import DPRQuestionEncoderTokenizer
from transformers.tokenization_roberta import VOCAB_FILES_NAMES as BART_VOCAB_FILES_NAMES

from .test_modeling_bart import ModelTester as BartModelTester
from .test_modeling_common import ids_tensor
from .test_modeling_dpr import DPRModelTester


TOLERANCE = 1e-4


if is_torch_available() and is_datasets_available() and is_faiss_available() and is_psutil_available():
    import faiss
    import torch
    from datasets import Dataset

    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSeq2SeqLM,
        BartConfig,
        DPRConfig,
        RagConfig,
        RagModel,
        RagRetriever,
        RagSequenceForGeneration,
        RagTokenForGeneration,
    )


def _assert_tensors_equal(a, b, atol=1e-12, prefix=""):
    """If tensors not close, or a and b arent both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        msg = "{} != {}".format(a, b)
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def require_retrieval(test_case):
    """
    Decorator marking a test that requires a set of dependencies necessary for pefrorm retrieval with
    :class:`~transformers.RagRetriever`.

    These tests are skipped when respective libraries are not installed.

    """
    if not (is_torch_available() and is_datasets_available() and is_faiss_available() and is_psutil_available()):
        test_case = unittest.skip("test requires PyTorch")(test_case)
    return test_case


class RagModelTester:
    def __init__(
        self,
        parent,
    ):
        # Global params
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7

        # RAG params
        self.n_docs = 3
        self.vocab_size = 50265
        self.bos_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.decoder_start_token_id = 2
        self.max_combined_length = 123
        self.retrieval_vector_size = 768
        self.retrieval_batch_size = 8

        self.rag_config = RagConfig(
            n_docs=self.n_docs,
            vocab_size=self.vocab_size,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            max_combined_length=self.max_combined_length,
            retrieval_vector_size=self.retrieval_vector_size,
            retrieval_batch_size=self.retrieval_batch_size,
        )

        # BART params
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 4
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 20

        self.bart_config = BartConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            return_dict=True,
        )

        # DPR params
        self.dpr_vocab_size = 51
        self.hidden_size = 20
        self.num_hidden_layers = 3
        self.num_attention_heads = 5
        self.intermediate_size = 5
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2
        self.max_position_embeddings = 19
        self.type_vocab_size = 17
        self.initializer_range = 0.02
        self.projection_dim = 0

        self.dpr_config = DPRConfig(
            projection_dim=self.projection_dim,
            vocab_size=self.dpr_vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            return_dict=True,
        )

    def prepare_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id
        attention_mask = input_ids.ne(self.pad_token_id)

        return input_ids, attention_mask


class RagTestMixin:

    all_model_classes = (RagModel, RagTokenForGeneration, RagSequenceForGeneration)
    retrieval_vector_size = 32
    n_docs = 2
    max_combined_length = 16

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        # DPR tok
        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        dpr_tokenizer_path = os.path.join(self.tmpdirname, "dpr_tokenizer")
        os.makedirs(dpr_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(dpr_tokenizer_path, DPR_VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        # BART tok
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        bart_tokenizer_path = os.path.join(self.tmpdirname, "bart_tokenizer")
        os.makedirs(bart_tokenizer_path, exist_ok=True)
        self.vocab_file = os.path.join(bart_tokenizer_path, BART_VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(bart_tokenizer_path, BART_VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    @cached_property
    def dpr_tokenizer(self) -> DPRQuestionEncoderTokenizer:
        return DPRQuestionEncoderTokenizer.from_pretrained(os.path.join(self.tmpdirname, "dpr_tokenizer"))

    @cached_property
    def bart_tokenizer(self) -> BartTokenizer:
        return BartTokenizer.from_pretrained(os.path.join(self.tmpdirname, "bart_tokenizer"))

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_retriever(self, config) -> RagRetriever:
        dataset = Dataset.from_dict(
            {
                "id": ["0", "1"],
                "text": ["foo", "bar"],
                "title": ["Foo", "Bar"],
                "embeddings": [np.ones(self.retrieval_vector_size), 2 * np.ones(self.retrieval_vector_size)],
            }
        )
        dataset.add_faiss_index("embeddings", string_factory="Flat", metric_type=faiss.METRIC_INNER_PRODUCT)
        with patch("transformers.retrieval_rag.load_dataset") as mock_load_dataset:
            mock_load_dataset.return_value = dataset
            retriever = RagRetriever(
                config,
                question_encoder_tokenizer=self.dpr_tokenizer,
                generator_tokenizer=self.bart_tokenizer,
            )
            retriever.init_retrieval(0)
        return retriever

    def check_model_with_retriever(
        self, config, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, **kwargs
    ):
        self.assertIsNotNone(config.question_encoder)
        self.assertIsNotNone(config.generator)

        for model_class in self.all_model_classes:
            model = model_class(config, retriever=self.get_retriever(config)).to(torch_device)
            model.eval()

            self.assertTrue(model.config.is_encoder_decoder)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )

            # logits
            self.assertEqual(
                outputs[0].shape,
                (self.n_docs * decoder_input_ids.shape[0], decoder_input_ids.shape[1], config.generator.vocab_size),
            )
            # generator encoder last hidden states
            self.assertEqual(
                outputs[1].shape,
                (self.n_docs * decoder_input_ids.shape[0], self.max_combined_length, config.generator.hidden_size),
            )
            # doc scores
            self.assertEqual(outputs[2].shape, (input_ids.shape[0], self.n_docs))

    #    def check_model_without_retriever(
    #        self,
    #        config,
    #        input_ids,
    #        attention_mask,
    #        encoder_hidden_states,
    #        decoder_config,
    #        decoder_input_ids,
    #        decoder_attention_mask,
    #        **kwargs
    #    ):
    #        self.assertIsNotNone(config.question_encoder)
    #        self.assertIsNotNone(config.generator)
    #
    #        for model_class in self.all_model_classes:
    #            model = model_class(config).to(torch_device)
    #            model.eval()
    #
    #            self.assertTrue(model.config.is_encoder_decoder)
    #
    #            outputs = model(context_input_ids, context_attention_mask, retrieved_doc_embeds, decoder_input_ids, decoder_attention_mask)
    #
    #            self.assertEqual(outputs[0].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
    #            self.assertEqual(outputs[1].shape, (input_ids.shape + (config.hidden_size,)))

    #    def check_model_with_encoder_outputs(
    #        self,
    #        config,
    #        input_ids,
    #        attention_mask,
    #        encoder_hidden_states,
    #        decoder_config,
    #        decoder_input_ids,
    #        decoder_attention_mask,
    #        **kwargs
    #    ):
    #        self.assertIsNotNone(config.question_encoder)
    #        self.assertIsNotNone(config.generator)
    #
    #        for model_class in self.all_model_classes:
    #            model = model_class(config).to(torch_device)
    #            model.eval()
    #
    #            self.assertTrue(model.config.is_encoder_decoder)
    #
    #            outputs = model(encoder_outputs, decoder_input_ids, decoder_attention_mask)
    #
    #            self.assertEqual(outputs[0].shape, (decoder_input_ids.shape + (decoder_config.vocab_size,)))
    #            self.assertEqual(outputs[1].shape, (input_ids.shape + (config.hidden_size,)))

    def test_model_with_retriever(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_model_with_retriever(**inputs_dict)


@require_torch
@require_retrieval
class RagDPRBartTest(RagTestMixin, unittest.TestCase):
    def prepare_config_and_inputs(self):
        question_encoder_tester = DPRModelTester(self)
        dpr_config_and_inputs = question_encoder_tester.prepare_config_and_inputs()
        generator_tester = BartModelTester(self)
        bart_config_and_inputs = generator_tester.prepare_config_and_inputs_for_common()

        (question_encoder_config, input_ids, _, input_mask, _, _, _) = dpr_config_and_inputs
        (generator_config, bart_inputs_dict) = bart_config_and_inputs
        decoder_input_ids, decoder_attention_mask = bart_inputs_dict["input_ids"], bart_inputs_dict["attention_mask"]

        config = RagConfig.from_question_encoder_generator_configs(
            question_encoder_config,
            generator_config,
            n_docs=self.n_docs,
            retrieval_vector_size=self.retrieval_vector_size,
            max_combined_length=self.max_combined_length,
            use_cache=False,
        )

        return {
            "config": config,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }


# def mock_contextualize(*args, **kwargs):
#    input_ids = torch.tensor([[0, 31414, 232, 328, 2]] * 3 * 13)
#    attention_mask = torch.tensor([[1, 1, 1, 1, 1]] * 3 * 13)
#    doc_scores = torch.tensor([[0.111, 0.222, 0.333]] * 13)
#    return input_ids, attention_mask, doc_scores


# @patch("transformers.RagModel.contextualize", mock_contextualize)
# def test_forward_pass(self):
#    input_ids, attention_mask = self.model_tester.prepare_inputs()
#    decoder_input_ids = torch.tensor([[0, 31414, 232, 328, 2]] * self.model_tester.batch_size)
#    tgt_len = decoder_input_ids.shape[1]
#    for model_class in self.all_model_classes:
#        model = model_class(
#            config=self.model_tester.rag_config,
#            question_encoder=DPRQuestionEncoder(self.model_tester.dpr_config),
#            generator=BartForConditionalGeneration(self.model_tester.bart_config),
#        )
#        model.to(torch_device)
#        model.eval()
#
# use cache
#        result = model(
#            input_ids,
#            retriever=None,
#            decoder_input_ids=decoder_input_ids,
#            attention_mask=attention_mask,
#            marginalize=False,
#            use_cache=True,
#        )
#        self.assertEqual(
#            result.logits.shape,
#            (self.model_tester.rag_config.n_docs * self.model_tester.batch_size, 1, self.model_tester.vocab_size),
#        )
#        self.assertEqual(
#            result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
#        )
#        self.assertIsNone(result.loss)
#
# no cache
#        result = model(
#            input_ids,
#            retriever=None,
#            decoder_input_ids=decoder_input_ids,
#            attention_mask=attention_mask,
#            marginalize=False,
#            use_cache=False,
#        )
#        self.assertEqual(
#            result.logits.shape,
#            (
#                self.model_tester.rag_config.n_docs * self.model_tester.batch_size,
#                tgt_len,
#                self.model_tester.vocab_size,
#            ),
#        )
#        self.assertEqual(
#            result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
#        )
#        self.assertIsNone(result.loss)
#
# marginalization in RagTokenForGeneration + no cache
#        if isinstance(model_class, RagTokenForGeneration):
#            result = model(
#                input_ids,
#                decoder_input_ids=decoder_input_ids,
#                attention_mask=attention_mask,
#                marginalize=True,
#                use_cache=False,
#            )
#            self.assertEqual(
#                result.logits.shape, (self.model_tester.batch_size, tgt_len, self.model_tester.vocab_size)
#            )
#            self.assertEqual(
#                result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
#            )
#            self.assertIsNone(result.loss)
#
# return_loss, no reduce
#        result = model(
#            input_ids,
#            retriever=None,
#            decoder_input_ids=decoder_input_ids,
#            attention_mask=attention_mask,
#            return_loss=True,
#        )
#        self.assertEqual(
#            result.logits.shape,
#            (
#                self.model_tester.rag_config.n_docs * self.model_tester.batch_size,
#                tgt_len,
#                self.model_tester.vocab_size,
#            ),
#        )
#        self.assertEqual(
#            result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
#        )
#        self.assertEqual(result.loss.shape, (self.model_tester.batch_size,))
#
# return_loss, reduce
#        result = model(
#            input_ids,
#            retriever=None,
#            decoder_input_ids=decoder_input_ids,
#            attention_mask=attention_mask,
#            return_loss=True,
#            reduce=True,
#        )
#        self.assertEqual(
#            result.logits.shape,
#            (
#                self.model_tester.rag_config.n_docs * self.model_tester.batch_size,
#                tgt_len,
#                self.model_tester.vocab_size,
#            ),
#        )
#        self.assertEqual(
#            result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
#        )
#        self.assertEqual(result.loss.shape, torch.Size([]))


@require_torch
@require_retrieval
class RagModelIntegrationTests(unittest.TestCase):
    @cached_property
    def sequence_model(self):
        return RagSequenceForGeneration.from_pretrained_question_encoder_generator(
            "facebook/dpr-question_encoder-single-nq-base", "facebook/bart-large-cnn"
        ).to(torch_device)

    @cached_property
    def token_model(self):
        return RagTokenForGeneration.from_pretrained_question_encoder_generator(
            "facebook/dpr-question_encoder-single-nq-base", "facebook/bart-large-cnn"
        ).to(torch_device)

    def get_rag_config(self):
        question_encoder_config = AutoConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        generator_config = AutoConfig.from_pretrained("facebook/bart-large-cnn")
        return RagConfig.from_question_encoder_generator_configs(
            question_encoder_config,
            generator_config,
            bos_token_id=0,
            decoder_start_token_id=2,
            eos_token_id=2,
            is_encoder_decoder=True,
            pad_token_id=1,
            vocab_size=50264,
            title_sep=" / ",
            doc_sep=" // ",
            n_docs=5,
            max_combined_length=300,
            dataset="wiki_dpr",
            dataset_split="train",
            index_name="exact",
            index_path=None,
            use_dummy_dataset=True,
            retrieval_vector_size=768,
            retrieval_batch_size=8,
        )

    @slow
    def test_rag_sequence_inference(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_sequence = self.sequence_model
        rag_sequence.set_retriever(rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="pt").input_ids

        input_ids = input_ids.to(torch_device)
        decoder_input_ids = decoder_input_ids.to(torch_device)

        with torch.no_grad():
            output = rag_sequence(
                input_ids,
                labels=decoder_input_ids,
            )

        expected_shape = torch.Size([5, 5, 50264])
        self.assertEqual(output.logits.shape, expected_shape)

        expected_doc_scores = torch.tensor([[75.0286, 74.4998, 74.0804, 74.0306, 73.9504]])
        _assert_tensors_equal(expected_doc_scores, output.doc_scores, atol=TOLERANCE)

        expected_loss = torch.tensor([38.7446])
        _assert_tensors_equal(expected_loss, output.loss, atol=TOLERANCE)

    @slow
    def test_rag_token_inference(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_token = self.token_model
        rag_token.set_retriever(rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="pt").input_ids

        input_ids = input_ids.to(torch_device)
        decoder_input_ids = decoder_input_ids.to(torch_device)

        with torch.no_grad():
            output = rag_token(
                input_ids,
                labels=decoder_input_ids,
            )

        expected_shape = torch.Size([5, 5, 50264])
        self.assertEqual(output.logits.shape, expected_shape)

        expected_doc_scores = torch.tensor([[75.0286, 74.4998, 74.0804, 74.0306, 73.9504]])
        _assert_tensors_equal(expected_doc_scores, output.doc_scores, atol=TOLERANCE)

        expected_loss = torch.tensor([38.7045])
        _assert_tensors_equal(expected_loss, output.loss, atol=TOLERANCE)

    @slow
    def test_rag_sequence_generate(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_sequence = self.sequence_model
        rag_sequence.set_retriever(rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids

        input_ids = input_ids.to(torch_device)

        output_ids = rag_sequence.generate(
            input_ids,
        )
        # sequence generate test
        output_text = rag_decoder_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        EXPECTED_OUTPUT_TEXT = """The album showed a songwriting maturity and depth of feeling distinctly lacking from their earlier recordings. The album\'s title track refers to secret meetings held against the approval of totalitarian governments in Soviet-dominated states. The only major single release, "One of Us", proved to be the last of ABBA\'s nine number-one singles in Germany."""
        self.assertEqual(output_text, EXPECTED_OUTPUT_TEXT)

    @slow
    def test_rag_token_generate(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        rag_token = self.token_model
        rag_token.set_retriever(rag_retriever)

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids

        input_ids = input_ids.to(torch_device)

        output_ids = rag_token.generate(
            input_ids, decoder_start_token_id=rag_token.generator.config.decoder_start_token_id
        )
        # sequence generate test
        output_text = rag_decoder_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        EXPECTED_OUTPUT_TEXT = """. The song peaked at"""
        self.assertEqual(output_text, EXPECTED_OUTPUT_TEXT)

    @slow
    def test_rag_sequence_from_pretrained(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="pt").input_ids

        input_ids = input_ids.to(torch_device)
        decoder_input_ids = decoder_input_ids.to(torch_device)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            rag_sequence = RagSequenceForGeneration.from_pretrained_question_encoder_generator(
                "facebook/dpr-question_encoder-single-nq-base",
                "facebook/bart-large-cnn",
                retriever=rag_retriever,
                config=rag_config,
            )
            # check that the from pretrained methods work
            rag_sequence.save_pretrained(tmp_dirname)
            rag_sequence.from_pretrained(tmp_dirname, retriever=rag_retriever)

            with torch.no_grad():
                output = rag_sequence(
                    input_ids,
                    labels=decoder_input_ids,
                )

            loss_pretrained = output.loss
            del rag_sequence

        question_encoder = AutoModel.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        generator = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        rag_sequence = RagSequenceForGeneration(
            config=rag_config, question_encoder=question_encoder, generator=generator, retriever=rag_retriever
        )

        with torch.no_grad():
            output = rag_sequence(
                input_ids,
                labels=decoder_input_ids,
            )

        loss_init = output.loss

        self.assertAlmostEqual(loss_pretrained.item(), loss_init.item(), places=4)

    @slow
    def test_rag_token_from_pretrained(self):
        rag_config = self.get_rag_config()
        rag_decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        rag_question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        rag_retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=rag_question_encoder_tokenizer,
            generator_tokenizer=rag_decoder_tokenizer,
        )

        input_ids = rag_question_encoder_tokenizer(
            "who sings does he love me with reba", return_tensors="pt"
        ).input_ids
        decoder_input_ids = rag_decoder_tokenizer("Linda Davis", return_tensors="pt").input_ids

        input_ids = input_ids.to(torch_device)
        decoder_input_ids = decoder_input_ids.to(torch_device)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            rag_token = RagTokenForGeneration.from_pretrained_question_encoder_generator(
                "facebook/dpr-question_encoder-single-nq-base",
                "facebook/bart-large-cnn",
                retriever=rag_retriever,
                config=rag_config,
            )
            # check that the from pretrained methods work
            rag_token.save_pretrained(tmp_dirname)
            rag_token.from_pretrained(tmp_dirname, retriever=rag_retriever)

            with torch.no_grad():
                output = rag_token(
                    input_ids,
                    labels=decoder_input_ids,
                )

            loss_pretrained = output.loss
            del rag_token

        question_encoder = AutoModel.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        generator = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        rag_token = RagTokenForGeneration(
            config=rag_config, question_encoder=question_encoder, generator=generator, retriever=rag_retriever
        )

        with torch.no_grad():
            output = rag_token(
                input_ids,
                labels=decoder_input_ids,
            )

        loss_init = output.loss

        self.assertAlmostEqual(loss_pretrained.item(), loss_init.item(), places=4)
