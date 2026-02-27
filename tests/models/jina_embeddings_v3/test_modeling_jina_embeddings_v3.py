# Copyright 2026 the HuggingFace Team. All rights reserved.
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
import unittest

from transformers import is_torch_available
from transformers.models.jina_embeddings_v3 import JinaEmbeddingsV3Config
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        JinaEmbeddingsV3ForMaskedLM,
        JinaEmbeddingsV3ForQuestionAnswering,
        JinaEmbeddingsV3ForSequenceClassification,
        JinaEmbeddingsV3ForTokenClassification,
        JinaEmbeddingsV3Model,
    )


class JinaEmbeddingsV3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=20,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=8,
        type_vocab_size=1,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
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
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels

    def get_config(self):
        return JinaEmbeddingsV3Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels):
        model = JinaEmbeddingsV3Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

        result = model(input_ids, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        model = JinaEmbeddingsV3ForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        model = JinaEmbeddingsV3ForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = JinaEmbeddingsV3ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
    ):
        config.num_labels = self.num_labels
        model = JinaEmbeddingsV3ForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_torch
class JinaEmbeddingsV3ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            JinaEmbeddingsV3Model,
            JinaEmbeddingsV3ForMaskedLM,
            JinaEmbeddingsV3ForQuestionAnswering,
            JinaEmbeddingsV3ForSequenceClassification,
            JinaEmbeddingsV3ForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": JinaEmbeddingsV3Model,
            "fill-mask": JinaEmbeddingsV3ForMaskedLM,
            "question-answering": JinaEmbeddingsV3ForQuestionAnswering,
            "text-classification": JinaEmbeddingsV3ForSequenceClassification,
            "token-classification": JinaEmbeddingsV3ForTokenClassification,
            "zero-shot": JinaEmbeddingsV3ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = JinaEmbeddingsV3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=JinaEmbeddingsV3Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "jinaai/jina-embeddings-v3"
        model = JinaEmbeddingsV3Model.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class JinaEmbeddingsV3ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = JinaEmbeddingsV3Model.from_pretrained("jinaai/jina-embeddings-v3", revision = "refs/pr/137", dtype=torch.float32)
        model.eval()
        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-2.1561, 1.8075, -0.1943], [-2.0813, 1.9855, -0.2760], [-2.1181, 2.0043, -0.1322]]]
        )
        torch.testing.assert_close(output[:, 1:4, 1:4], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_retrieval_query_adapter(self):
        model_id = "jinaai/jina-embeddings-v3"
        revision = "refs/pr/137"
        task = "retrieval_query"

        model = JinaEmbeddingsV3Model.from_pretrained(model_id, revision=revision, dtype=torch.float32)
        model.load_adapter(model_id, adapter_name=task, adapter_kwargs={"subfolder": task, "revision": revision})
        model.set_adapter(task)
        model.eval()

        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]

        self.assertEqual(output.shape, torch.Size((1, 11, 1024)))
        expected_slice = torch.tensor(
            [[[-1.5202, 2.4074, -0.9706], [-1.5279, 2.4025, -1.1237], [-1.5252, 2.6143, -0.9416]]]
        )
        torch.testing.assert_close(output[:, 1:4, 1:4], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_retrieval_passage_adapter(self):
        model_id = "jinaai/jina-embeddings-v3"
        revision = "refs/pr/137"
        task = "retrieval_passage"

        model = JinaEmbeddingsV3Model.from_pretrained(model_id, revision=revision, dtype=torch.float32)
        model.load_adapter(model_id, adapter_name=task, adapter_kwargs={"subfolder": task, "revision": revision})
        model.set_adapter(task)
        model.eval()

        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]

        expected_shape = torch.Size((1, 11, 1024))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [[[-1.0778, 1.8124, -0.8223], [-1.3796, 1.8108, -0.9868], [-1.1073, 1.9459, -0.8104]]]
        )
        torch.testing.assert_close(output[:, 1:4, 1:4], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_separation_adapter(self):
        model_id = "jinaai/jina-embeddings-v3"
        revision = "refs/pr/137"
        task = "separation"

        model = JinaEmbeddingsV3Model.from_pretrained(model_id, revision=revision, dtype=torch.float32)
        model.load_adapter(model_id, adapter_name=task, adapter_kwargs={"subfolder": task, "revision": revision})
        model.set_adapter(task)
        model.eval()

        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]

        self.assertEqual(output.shape, torch.Size((1, 11, 1024)))
        expected_slice = torch.tensor(
            [[[-2.3134, 1.9034, 0.1885], [-2.1804, 1.9923, 0.1184], [-2.2325, 2.0251, 0.2533]]]
        )
        torch.testing.assert_close(output[:, 1:4, 1:4], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_classification_adapter(self):
        model_id = "jinaai/jina-embeddings-v3"
        revision = "refs/pr/137"
        task = "classification"

        model = JinaEmbeddingsV3Model.from_pretrained(model_id, revision=revision, dtype=torch.float32)
        model.load_adapter(model_id, adapter_name=task, adapter_kwargs={"subfolder": task, "revision": revision})
        model.set_adapter(task)
        model.eval()

        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]

        self.assertEqual(output.shape, torch.Size((1, 11, 1024)))
        expected_slice = torch.tensor(
            [
                    [
                        [-1.9793e00, 2.2602e00, -1.7995e-01],
                        [-1.8780e00, 2.2660e00, -2.1822e-01],
                        [-1.9202e00, 2.4457e00, -2.2832e-04],
                    ]
                ]
        )
        torch.testing.assert_close(output[:, 1:4, 1:4], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_text_matching_adapter(self):
        model_id = "jinaai/jina-embeddings-v3"
        revision = "refs/pr/137"
        task = "text_matching"

        model = JinaEmbeddingsV3Model.from_pretrained(model_id, revision=revision, dtype=torch.float32)
        model.load_adapter(model_id, adapter_name=task, adapter_kwargs={"subfolder": task, "revision": revision})
        model.set_adapter(task)
        model.eval()

        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]

        self.assertEqual(output.shape, torch.Size((1, 11, 1024)))
        expected_slice = torch.tensor(
            [[[-1.6952, 1.8568, -0.2258], [-1.6239, 1.9056, -0.2681], [-1.5218, 1.8185, 0.0062]]]
        )
        torch.testing.assert_close(output[:, 1:4, 1:4], expected_slice, rtol=1e-4, atol=1e-4)
