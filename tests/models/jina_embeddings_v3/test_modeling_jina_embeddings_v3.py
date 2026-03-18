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

from transformers import AutoModel, AutoTokenizer, is_torch_available
from transformers.models.jina_embeddings_v3 import JinaEmbeddingsV3Config
from transformers.testing_utils import (
    cleanup,
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


@require_torch
class JinaEmbeddingsV3ModelIntegrationTest(unittest.TestCase):
    model_id = "jinaai/jina-embeddings-v3-hf"
    prompt = "Jina Embeddings V3 is great for semantic search."

    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _prepare_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.prompt, return_tensors="pt", padding=True)
        return inputs

    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = AutoModel.from_pretrained(self.model_id, dtype=torch.float32)
        model.eval()
        inputs = self._prepare_inputs()

        with torch.no_grad():
            output = model(**inputs)[0]

        expected_shape = torch.Size((1, 17, 1024))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [
                    [-3.1011, 0.8560, -0.2491, 0.9427, 1.4015, -1.1527, 1.3804, -0.5453, -1.8164],
                    [-3.1108, 1.0107, -0.2097, 1.3495, 0.9984, -0.9518, 1.3189, -0.6295, -2.1128],
                    [-2.7095, 0.6469, -0.4475, 1.1364, 1.5975, -0.7545, 1.0803, 0.5199, -2.3569],
                ]
            ]
        )

        torch.testing.assert_close(output[:, 1:4, 1:10], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_retrieval_query_adapter(self):
        task = "retrieval_query"
        model = AutoModel.from_pretrained(self.model_id, dtype=torch.float32)
        model.load_adapter(self.model_id, adapter_name=task, adapter_kwargs={"subfolder": task})
        model.set_adapter(task)
        model.eval()
        inputs = self._prepare_inputs()

        with torch.no_grad():
            output = model(**inputs)[0]

        self.assertEqual(output.shape, torch.Size((1, 17, 1024)))
        expected_slice = torch.tensor(
            [
                [
                    [-1.9765, 0.7356, -0.4414, 0.5823, 2.1507, -0.8906, 0.0233, -0.2389, -1.5708],
                    [-2.0078, 0.9562, -0.3315, 1.0080, 1.8247, -0.6678, -0.2505, -0.3441, -1.9328],
                    [-1.9107, 0.7120, -0.4675, 0.9436, 2.1607, -0.4170, -0.1513, 1.0063, -2.0103],
                ]
            ]
        )

        torch.testing.assert_close(output[:, 1:4, 1:10], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_retrieval_passage_adapter(self):
        task = "retrieval_passage"
        model = AutoModel.from_pretrained(self.model_id, dtype=torch.float32)
        model.load_adapter(self.model_id, adapter_name=task, adapter_kwargs={"subfolder": task})
        model.set_adapter(task)
        model.eval()
        inputs = self._prepare_inputs()

        with torch.no_grad():
            output = model(**inputs)[0]

        expected_shape = torch.Size((1, 17, 1024))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [
                    [-1.7028, 0.5688, -0.8541, 0.4696, 2.5396, -0.8374, -0.1404, -0.3123, -1.4636],
                    [-1.6631, 0.6571, -0.8641, 0.9177, 2.3502, -0.6578, -0.3763, -0.3975, -1.7684],
                    [-1.4739, 0.4739, -0.8745, 0.8812, 2.6848, -0.4496, -0.4964, 0.6403, -2.0821],
                ]
            ]
        )

        torch.testing.assert_close(output[:, 1:4, 1:10], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_separation_adapter(self):
        task = "separation"
        model = AutoModel.from_pretrained(self.model_id, dtype=torch.float32)
        model.load_adapter(self.model_id, adapter_name=task, adapter_kwargs={"subfolder": task})
        model.set_adapter(task)
        model.eval()

        inputs = self._prepare_inputs()

        with torch.no_grad():
            output = model(**inputs)[0]

        self.assertEqual(output.shape, torch.Size((1, 17, 1024)))
        expected_slice = torch.tensor(
            [
                [
                    [-3.0336, 1.4392, 0.2875, 0.7660, 0.7054, -1.1701, 1.6121, -0.6325, -1.5177],
                    [-3.0875, 1.5134, 0.3620, 1.0281, 0.4895, -1.0484, 1.6574, -0.7636, -1.6736],
                    [-2.7605, 1.2920, 0.2223, 0.9895, 0.8515, -0.9050, 1.5558, 0.1410, -1.8531],
                ]
            ]
        )

        torch.testing.assert_close(output[:, 1:4, 1:10], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_classification_adapter(self):
        task = "classification"
        model = AutoModel.from_pretrained(self.model_id, dtype=torch.float32)
        model.load_adapter(self.model_id, adapter_name=task, adapter_kwargs={"subfolder": task})
        model.set_adapter(task)
        model.eval()

        inputs = self._prepare_inputs()

        with torch.no_grad():
            output = model(**inputs)[0]

        self.assertEqual(output.shape, torch.Size((1, 17, 1024)))
        expected_slice = torch.tensor(
            [
                [
                    [-2.7150, 0.2485, 1.2297, 0.6988, 0.9804, -1.2831, 1.3446, -0.1663, -0.6874],
                    [-2.8101, 0.1711, 1.2010, 0.9873, 0.5092, -1.3312, 1.4633, -0.2467, -0.7835],
                    [-2.6067, 0.2362, 0.6945, 1.0134, 0.7105, -1.3767, 0.9999, 0.4427, -1.1153],
                ]
            ]
        )

        torch.testing.assert_close(output[:, 1:4, 1:10], expected_slice, rtol=1e-4, atol=1e-4)

    @slow
    def test_inference_text_matching_adapter(self):
        task = "text_matching"
        model = AutoModel.from_pretrained(self.model_id, dtype=torch.float32)
        model.load_adapter(self.model_id, adapter_name=task, adapter_kwargs={"subfolder": task})
        model.set_adapter(task)
        model.eval()

        inputs = self._prepare_inputs()

        with torch.no_grad():
            output = model(**inputs)[0]

        self.assertEqual(output.shape, torch.Size((1, 17, 1024)))
        expected_slice = torch.tensor(
            [
                [
                    [-1.5888, 1.0527, 0.1237, -0.0822, 1.6507, -1.0371, -0.8815, -0.8082, -0.6564],
                    [-1.6529, 1.3143, 0.1957, 0.2914, 1.4897, -0.8735, -1.0067, -0.7544, -1.0513],
                    [-1.5308, 1.4805, -0.1393, 0.3879, 1.4373, -0.6064, -1.6436, 0.4793, -1.3388],
                ]
            ]
        )

        torch.testing.assert_close(output[:, 1:4, 1:10], expected_slice, rtol=1e-4, atol=1e-4)
