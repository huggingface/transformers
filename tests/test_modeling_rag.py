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


import copy
import unittest
from unittest.mock import patch

from transformers.file_utils import is_faiss_available, is_nlp_available, is_psutil_available, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ids_tensor


if is_torch_available() and is_nlp_available() and is_faiss_available() and is_psutil_available():
    import torch

    from transformers import (
        BartConfig,
        BartForConditionalGeneration,
        DPRConfig,
        DPRQuestionEncoder,
        RagConfig,
        RagSequence,
        RagToken,
    )


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


@require_torch
class RagModelTest(unittest.TestCase):
    all_model_classes = (
        (RagSequence, RagToken)
        if is_torch_available() and is_nlp_available() and is_faiss_available() and is_psutil_available()
        else ()
    )

    def setUp(self):
        if is_torch_available() and is_nlp_available() and is_faiss_available() and is_psutil_available():
            self.model_tester = RagModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RagConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()

    def test_constructor_from_config(self):
        for model_class in self.all_model_classes:
            model = model_class(config=self.model_tester.rag_config)
            self.assertEqual(model.n_docs, self.model_tester.rag_config.n_docs)
            self.assertEqual(model.model.n_docs, self.model_tester.rag_config.n_docs)
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.model.question_encoder)
            self.assertIsNotNone(model.model.generator)
            self.assertTrue(model.config.is_encoder_decoder)

    def test_constructor_from_object(self):
        for model_class in self.all_model_classes:
            model = model_class(
                config=self.model_tester.rag_config,
                question_encoder=DPRQuestionEncoder(self.model_tester.dpr_config),
                generator=BartForConditionalGeneration(self.model_tester.bart_config),
            )
            self.assertEqual(model.n_docs, self.model_tester.rag_config.n_docs)
            self.assertEqual(model.model.n_docs, self.model_tester.rag_config.n_docs)
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.model.question_encoder)
            self.assertIsNotNone(model.model.generator)
            self.assertTrue(model.config.is_encoder_decoder)

    def test_constructor_from_pretrained(self):
        for model_class in self.all_model_classes:
            model = model_class.from_pretrained(config=self.model_tester.rag_config)
            self.assertEqual(model.n_docs, self.model_tester.rag_config.n_docs)
            self.assertEqual(model.model.n_docs, self.model_tester.rag_config.n_docs)
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.model.question_encoder)
            self.assertIsNotNone(model.model.generator)
            self.assertTrue(model.config.is_encoder_decoder)

    def test_constructor_mismatch(self):
        mismatched_bart_config = copy.deepcopy(self.model_tester.bart_config)

        def test_mismatch():
            for model_class in self.all_model_classes:
                with self.assertRaises(
                    AssertionError,
                ):
                    model_class(
                        config=self.model_tester.rag_config,
                        question_encoder=DPRQuestionEncoder(self.model_tester.dpr_config),
                        generator=BartForConditionalGeneration(mismatched_bart_config),
                    )

        mismatched_bart_config.eos_token_id = self.model_tester.bart_config.eos_token_id + 1
        test_mismatch()
        mismatched_bart_config.eos_token_id = self.model_tester.bart_config.eos_token_id
        mismatched_bart_config.bos_token_id = self.model_tester.bart_config.bos_token_id + 1
        test_mismatch()
        mismatched_bart_config.bos_token_id = self.model_tester.bart_config.bos_token_id
        mismatched_bart_config.pad_token_id = self.model_tester.bart_config.pad_token_id + 1
        test_mismatch()
        mismatched_bart_config.pad_token_id = self.model_tester.bart_config.pad_token_id
        mismatched_bart_config.decoder_start_token_id = self.model_tester.bart_config.decoder_start_token_id + 1
        test_mismatch()
        mismatched_bart_config.decoder_start_token_id = self.model_tester.bart_config.decoder_start_token_id
        mismatched_bart_config.is_encoder_decoder = not self.model_tester.bart_config.is_encoder_decoder
        test_mismatch()
        mismatched_bart_config.is_encoder_decoder = self.model_tester.bart_config.is_encoder_decoder
        mismatched_bart_config.vocab_size = not self.model_tester.bart_config.vocab_size + 1
        test_mismatch()

    def mock_contextualize(*args, **kwargs):
        input_ids = torch.tensor([[0, 31414, 232, 328, 2]] * 3 * 13)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]] * 3 * 13)
        doc_scores = torch.tensor([[0.111, 0.222, 0.333]] * 13)
        return input_ids, attention_mask, doc_scores

    @patch("transformers.RagModel.contextualize", mock_contextualize)
    def test_forward_pass(self):
        input_ids, attention_mask = self.model_tester.prepare_inputs()
        decoder_input_ids = torch.tensor([[0, 31414, 232, 328, 2]] * self.model_tester.batch_size)
        tgt_len = decoder_input_ids.shape[1]
        for model_class in self.all_model_classes:
            model = model_class(
                config=self.model_tester.rag_config,
                question_encoder=DPRQuestionEncoder(self.model_tester.dpr_config),
                generator=BartForConditionalGeneration(self.model_tester.bart_config),
            )
            model.to(torch_device)
            model.eval()

            # use cache
            result = model(
                input_ids,
                retriever=None,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                marginalize=False,
                use_cache=True,
            )
            self.assertEqual(
                result.logits.shape,
                (self.model_tester.rag_config.n_docs * self.model_tester.batch_size, 1, self.model_tester.vocab_size),
            )
            self.assertEqual(
                result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
            )
            self.assertIsNone(result.loss)

            # no cache
            result = model(
                input_ids,
                retriever=None,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                marginalize=False,
                use_cache=False,
            )
            self.assertEqual(
                result.logits.shape,
                (
                    self.model_tester.rag_config.n_docs * self.model_tester.batch_size,
                    tgt_len,
                    self.model_tester.vocab_size,
                ),
            )
            self.assertEqual(
                result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
            )
            self.assertIsNone(result.loss)

            # marginalization in RagToken + no cache
            if isinstance(model_class, RagToken):
                result = model(
                    input_ids,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    marginalize=True,
                    use_cache=False,
                )
                self.assertEqual(
                    result.logits.shape, (self.model_tester.batch_size, tgt_len, self.model_tester.vocab_size)
                )
                self.assertEqual(
                    result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
                )
                self.assertIsNone(result.loss)

            # return_loss, no reduce
            result = model(
                input_ids,
                retriever=None,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                return_loss=True,
            )
            self.assertEqual(
                result.logits.shape,
                (
                    self.model_tester.rag_config.n_docs * self.model_tester.batch_size,
                    tgt_len,
                    self.model_tester.vocab_size,
                ),
            )
            self.assertEqual(
                result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
            )
            self.assertEqual(result.loss.shape, (self.model_tester.batch_size,))

            # return_loss, reduce
            result = model(
                input_ids,
                retriever=None,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                return_loss=True,
                reduce=True,
            )
            self.assertEqual(
                result.logits.shape,
                (
                    self.model_tester.rag_config.n_docs * self.model_tester.batch_size,
                    tgt_len,
                    self.model_tester.vocab_size,
                ),
            )
            self.assertEqual(
                result.doc_scores.shape, (self.model_tester.batch_size, self.model_tester.rag_config.n_docs)
            )
            self.assertEqual(result.loss.shape, torch.Size([]))
