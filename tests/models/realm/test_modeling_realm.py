# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch REALM model. """

import copy
import unittest

import numpy as np

from transformers import RealmConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        RealmEmbedder,
        RealmForOpenQA,
        RealmKnowledgeAugEncoder,
        RealmReader,
        RealmRetriever,
        RealmScorer,
        RealmTokenizer,
    )


class RealmModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        retriever_proj_size=128,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        span_hidden_size=50,
        max_span_width=10,
        reader_layer_norm_eps=1e-3,
        reader_beam_size=4,
        reader_seq_len=288 + 32,
        num_block_records=13353718,
        searcher_beam_size=8,
        searcher_seq_len=64,
        num_labels=3,
        num_choices=4,
        num_candidates=10,
        scope=None,
    ):
        # General config
        self.parent = parent
        self.batch_size = batch_size
        self.retriever_proj_size = retriever_proj_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # Reader config
        self.span_hidden_size = span_hidden_size
        self.max_span_width = max_span_width
        self.reader_layer_norm_eps = reader_layer_norm_eps
        self.reader_beam_size = reader_beam_size
        self.reader_seq_len = reader_seq_len

        # Searcher config
        self.num_block_records = num_block_records
        self.searcher_beam_size = searcher_beam_size
        self.searcher_seq_len = searcher_seq_len

        self.num_labels = num_labels
        self.num_choices = num_choices
        self.num_candidates = num_candidates
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        candiate_input_ids = ids_tensor([self.batch_size, self.num_candidates, self.seq_length], self.vocab_size)
        reader_input_ids = ids_tensor([self.reader_beam_size, self.reader_seq_len], self.vocab_size)

        input_mask = None
        candiate_input_mask = None
        reader_input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
            candiate_input_mask = random_attention_mask([self.batch_size, self.num_candidates, self.seq_length])
            reader_input_mask = random_attention_mask([self.reader_beam_size, self.reader_seq_len])

        token_type_ids = None
        candidate_token_type_ids = None
        reader_token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
            candidate_token_type_ids = ids_tensor(
                [self.batch_size, self.num_candidates, self.seq_length], self.type_vocab_size
            )
            reader_token_type_ids = ids_tensor([self.reader_beam_size, self.reader_seq_len], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        # inputs with additional num_candidates axis.
        scorer_encoder_inputs = (candiate_input_ids, candiate_input_mask, candidate_token_type_ids)
        # reader inputs
        reader_inputs = (reader_input_ids, reader_input_mask, reader_token_type_ids)

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            scorer_encoder_inputs,
            reader_inputs,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return RealmConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            retriever_proj_size=self.retriever_proj_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_candidates=self.num_candidates,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
        )

    def create_and_check_embedder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        scorer_encoder_inputs,
        reader_inputs,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RealmEmbedder(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.projected_score.shape, (self.batch_size, self.retriever_proj_size))

    def create_and_check_encoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        scorer_encoder_inputs,
        reader_inputs,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RealmKnowledgeAugEncoder(config=config)
        model.to(torch_device)
        model.eval()
        relevance_score = floats_tensor([self.batch_size, self.num_candidates])
        result = model(
            scorer_encoder_inputs[0],
            attention_mask=scorer_encoder_inputs[1],
            token_type_ids=scorer_encoder_inputs[2],
            relevance_score=relevance_score,
            labels=token_labels,
        )
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size * self.num_candidates, self.seq_length, self.vocab_size)
        )

    def create_and_check_reader(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        scorer_encoder_inputs,
        reader_inputs,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RealmReader(config=config)
        model.to(torch_device)
        model.eval()
        relevance_score = floats_tensor([self.reader_beam_size])
        result = model(
            reader_inputs[0],
            attention_mask=reader_inputs[1],
            token_type_ids=reader_inputs[2],
            relevance_score=relevance_score,
        )
        self.parent.assertEqual(result.block_idx.shape, ())
        self.parent.assertEqual(result.candidate.shape, ())
        self.parent.assertEqual(result.start_pos.shape, ())
        self.parent.assertEqual(result.end_pos.shape, ())

    def create_and_check_scorer(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        scorer_encoder_inputs,
        reader_inputs,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RealmScorer(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            candidate_input_ids=scorer_encoder_inputs[0],
            candidate_attention_mask=scorer_encoder_inputs[1],
            candidate_token_type_ids=scorer_encoder_inputs[2],
        )
        self.parent.assertEqual(result.relevance_score.shape, (self.batch_size, self.num_candidates))
        self.parent.assertEqual(result.query_score.shape, (self.batch_size, self.retriever_proj_size))
        self.parent.assertEqual(
            result.candidate_score.shape, (self.batch_size, self.num_candidates, self.retriever_proj_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            scorer_encoder_inputs,
            reader_inputs,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class RealmModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            RealmEmbedder,
            RealmKnowledgeAugEncoder,
            # RealmScorer is excluded from common tests as it is a container model
            # consisting of two RealmEmbedders & a simple inner product calculation.
            # RealmScorer
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = ()

    # disable these tests because there is no base_model in Realm
    test_save_load_fast_init_from_base = False
    test_save_load_fast_init_to_base = False

    def setUp(self):
        self.test_pruning = False
        self.model_tester = RealmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RealmConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_embedder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_embedder(*config_and_inputs)

    def test_encoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encoder(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_embedder(*config_and_inputs)
            self.model_tester.create_and_check_encoder(*config_and_inputs)

    def test_scorer(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_scorer(*config_and_inputs)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        config, *inputs = self.model_tester.prepare_config_and_inputs()
        input_ids, token_type_ids, input_mask, scorer_encoder_inputs = inputs[0:4]
        config.return_dict = True

        tokenizer = RealmTokenizer.from_pretrained("google/realm-orqa-nq-openqa")

        # RealmKnowledgeAugEncoder training
        model = RealmKnowledgeAugEncoder(config)
        model.to(torch_device)
        model.train()

        inputs_dict = {
            "input_ids": scorer_encoder_inputs[0].to(torch_device),
            "attention_mask": scorer_encoder_inputs[1].to(torch_device),
            "token_type_ids": scorer_encoder_inputs[2].to(torch_device),
            "relevance_score": floats_tensor([self.model_tester.batch_size, self.model_tester.num_candidates]),
        }
        inputs_dict["labels"] = torch.zeros(
            (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
        )
        inputs = inputs_dict
        loss = model(**inputs).loss
        loss.backward()

        # RealmForOpenQA training
        openqa_config = copy.deepcopy(config)
        openqa_config.vocab_size = 30522  # the retrieved texts will inevitably have more than 99 vocabs.
        openqa_config.num_block_records = 5
        openqa_config.searcher_beam_size = 2

        block_records = np.array(
            [
                b"This is the first record.",
                b"This is the second record.",
                b"This is the third record.",
                b"This is the fourth record.",
                b"This is the fifth record.",
            ],
            dtype=np.object,
        )
        retriever = RealmRetriever(block_records, tokenizer)
        model = RealmForOpenQA(openqa_config, retriever)
        model.to(torch_device)
        model.train()

        inputs_dict = {
            "input_ids": input_ids[:1].to(torch_device),
            "attention_mask": input_mask[:1].to(torch_device),
            "token_type_ids": token_type_ids[:1].to(torch_device),
            "answer_ids": input_ids[:1].tolist(),
        }
        inputs = self._prepare_for_class(inputs_dict, RealmForOpenQA)
        loss = model(**inputs).reader_output.loss
        loss.backward()

        # Test model.block_embedding_to
        device = torch.device("cpu")
        model.block_embedding_to(device)
        loss = model(**inputs).reader_output.loss
        loss.backward()
        self.assertEqual(model.block_emb.device.type, device.type)

    @slow
    def test_embedder_from_pretrained(self):
        model = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")
        self.assertIsNotNone(model)

    @slow
    def test_encoder_from_pretrained(self):
        model = RealmKnowledgeAugEncoder.from_pretrained("google/realm-cc-news-pretrained-encoder")
        self.assertIsNotNone(model)

    @slow
    def test_open_qa_from_pretrained(self):
        model = RealmForOpenQA.from_pretrained("google/realm-orqa-nq-openqa")
        self.assertIsNotNone(model)

    @slow
    def test_reader_from_pretrained(self):
        model = RealmReader.from_pretrained("google/realm-orqa-nq-reader")
        self.assertIsNotNone(model)

    @slow
    def test_scorer_from_pretrained(self):
        model = RealmScorer.from_pretrained("google/realm-cc-news-pretrained-scorer")
        self.assertIsNotNone(model)


@require_torch
class RealmModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_embedder(self):
        retriever_projected_size = 128

        model = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        expected_shape = torch.Size((1, retriever_projected_size))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor([[-0.0714, -0.0837, -0.1314]])
        self.assertTrue(torch.allclose(output[:, :3], expected_slice, atol=1e-4))

    @slow
    def test_inference_encoder(self):
        num_candidates = 2
        vocab_size = 30522

        model = RealmKnowledgeAugEncoder.from_pretrained(
            "google/realm-cc-news-pretrained-encoder", num_candidates=num_candidates
        )
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        relevance_score = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
        output = model(input_ids, relevance_score=relevance_score)[0]

        expected_shape = torch.Size((2, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor([[[-11.0888, -11.2544], [-10.2170, -10.3874]]])

        self.assertTrue(torch.allclose(output[1, :2, :2], expected_slice, atol=1e-4))

    @slow
    def test_inference_open_qa(self):
        from transformers.models.realm.retrieval_realm import RealmRetriever

        tokenizer = RealmTokenizer.from_pretrained("google/realm-orqa-nq-openqa")
        retriever = RealmRetriever.from_pretrained("google/realm-orqa-nq-openqa")

        model = RealmForOpenQA.from_pretrained(
            "google/realm-orqa-nq-openqa",
            retriever=retriever,
        )

        question = "Who is the pioneer in modern computer science?"

        question = tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=model.config.searcher_seq_len,
            return_tensors="pt",
        ).to(model.device)

        predicted_answer_ids = model(**question).predicted_answer_ids

        predicted_answer = tokenizer.decode(predicted_answer_ids)
        self.assertEqual(predicted_answer, "alan mathison turing")

    @slow
    def test_inference_reader(self):
        config = RealmConfig(reader_beam_size=2, max_span_width=3)
        model = RealmReader.from_pretrained("google/realm-orqa-nq-reader", config=config)

        concat_input_ids = torch.arange(10).view((2, 5))
        concat_token_type_ids = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.int64)
        concat_block_mask = torch.tensor([[0, 0, 1, 1, 0], [0, 0, 1, 1, 0]], dtype=torch.int64)
        relevance_score = torch.tensor([0.3, 0.7], dtype=torch.float32)

        output = model(
            concat_input_ids,
            token_type_ids=concat_token_type_ids,
            relevance_score=relevance_score,
            block_mask=concat_block_mask,
            return_dict=True,
        )

        block_idx_expected_shape = torch.Size(())
        start_pos_expected_shape = torch.Size((1,))
        end_pos_expected_shape = torch.Size((1,))
        self.assertEqual(output.block_idx.shape, block_idx_expected_shape)
        self.assertEqual(output.start_pos.shape, start_pos_expected_shape)
        self.assertEqual(output.end_pos.shape, end_pos_expected_shape)

        expected_block_idx = torch.tensor(1)
        expected_start_pos = torch.tensor(3)
        expected_end_pos = torch.tensor(3)

        self.assertTrue(torch.allclose(output.block_idx, expected_block_idx, atol=1e-4))
        self.assertTrue(torch.allclose(output.start_pos, expected_start_pos, atol=1e-4))
        self.assertTrue(torch.allclose(output.end_pos, expected_end_pos, atol=1e-4))

    @slow
    def test_inference_scorer(self):
        num_candidates = 2

        model = RealmScorer.from_pretrained("google/realm-cc-news-pretrained-scorer", num_candidates=num_candidates)

        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        candidate_input_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        output = model(input_ids, candidate_input_ids=candidate_input_ids)[0]

        expected_shape = torch.Size((1, 2))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor([[0.7410, 0.7170]])
        self.assertTrue(torch.allclose(output, expected_slice, atol=1e-4))
