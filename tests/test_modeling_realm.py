# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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


import unittest

from tests.test_modeling_common import floats_tensor
from transformers import RealmConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import RealmEmbedder, RealmEncoder, RealmRetriever
    from transformers.models.realm.modeling_realm import REALM_PRETRAINED_MODEL_ARCHIVE_LIST


class RealmModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        retriever_proj_size=128,
        seq_length=7,
        is_training=False,
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
        num_labels=3,
        num_choices=4,
        num_candidates=10,
        scope=None,
    ):
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
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.num_candidates = num_candidates
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        candiate_input_ids = ids_tensor([self.batch_size, self.num_candidates, self.seq_length], self.vocab_size)

        input_mask = None
        candiate_input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
            candiate_input_mask = random_attention_mask([self.batch_size, self.num_candidates, self.seq_length])

        token_type_ids = None
        candidate_token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
            candidate_token_type_ids = ids_tensor(
                [self.batch_size, self.num_candidates, self.seq_length], self.type_vocab_size
            )

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        # inputs with additional num_candidates axis.
        candidate_inputs = (candiate_input_ids, candiate_input_mask, candidate_token_type_ids)

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            candidate_inputs,
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
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def create_and_check_embedder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        candidate_inputs,
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
        candidate_inputs,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RealmEncoder(config=config)
        model.to(torch_device)
        model.eval()
        relevance_score = floats_tensor([self.batch_size, self.num_candidates])
        result = model(
            candidate_inputs[0],
            attention_mask=candidate_inputs[1],
            token_type_ids=candidate_inputs[2],
            relevance_score=relevance_score,
            labels=token_labels,
        )
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size * self.num_candidates, self.seq_length, self.vocab_size)
        )

    def create_and_check_retriever(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        candidate_inputs,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RealmRetriever(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            candidate_input_ids=candidate_inputs[0],
            candidate_attention_mask=candidate_inputs[1],
            candidate_token_type_ids=candidate_inputs[2],
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
            candidate_inputs,
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
            RealmEncoder,
            # RealmRetriever is excluded from common tests as it is a container model
            # consisting of two RealmEmbedders & simple inner product calculation.
            # RealmRetriever
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

    def test_retriever(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_retriever(*config_and_inputs)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in [RealmEncoder]:
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    @slow
    def test_encoder_from_pretrained(self):
        for model_name in REALM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = RealmEncoder.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_retriever_from_pretrained(self):
        for model_name in REALM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = RealmRetriever.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class RealmModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_embedder(self):
        retriever_projected_size = 128

        model = RealmEmbedder.from_pretrained("qqaatw/realm-cc-news-pretrained-embedder")
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

        model = RealmEncoder.from_pretrained("qqaatw/realm-cc-news-pretrained-bert", num_candidates=num_candidates)
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        relevance_score = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
        output = model(input_ids, relevance_score=relevance_score)[0]

        expected_shape = torch.Size((2, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor([[[-11.0888, -11.2544], [-10.2170, -10.3874]]])

        self.assertTrue(torch.allclose(output[1, :2, :2], expected_slice, atol=1e-4))

    @slow
    def test_inference_retriever(self):
        num_candidates = 2

        model = RealmRetriever.from_pretrained(
            "qqaatw/realm-cc-news-pretrained-retriever", num_candidates=num_candidates
        )

        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        candidate_input_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        output = model(input_ids, candidate_input_ids=candidate_input_ids)[0]

        expected_shape = torch.Size((1, 2))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor([[0.7410, 0.7170]])
        self.assertTrue(torch.allclose(output, expected_slice, atol=1e-4))
