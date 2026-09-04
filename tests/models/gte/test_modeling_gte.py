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

from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GteConfig,
    is_torch_available,
)
from transformers.testing_utils import Expectations, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        GteForMaskedLM,
        GteForSequenceClassification,
        GteForTokenClassification,
        GteModel,
    )


class GteModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
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
        self.num_choices = num_choices
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
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        """
        Returns a tiny configuration by default.
        """
        return GteConfig(
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

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = GteModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output, None)

    def create_and_check_model_without_token_types(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.type_vocab_size = 0
        model = GteModel(config=config)
        model.to(torch_device)
        model.eval()
        self.parent.assertIsNone(model.embeddings.token_type_embeddings)
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = GteForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = GteForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = GteForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, token_type_ids, input_mask, _, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class GteModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            GteModel,
            GteForMaskedLM,
            GteForSequenceClassification,
            GteForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GteModel,
            "fill-mask": GteForMaskedLM,
            "text-classification": GteForSequenceClassification,
            "token-classification": GteForTokenClassification,
            "zero-shot": GteForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = GteModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GteConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_without_token_types(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_without_token_types(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)


@require_torch
class GteModelIntegrationTest(unittest.TestCase):
    sentences = ["Plants create oxygen.", "Photosynthesis is a process where plants create oxygen."]

    # TODO: Remove revision

    @slow
    def test_inference_no_head_multilingual(self):
        model = AutoModel.from_pretrained(
            "Alibaba-NLP/gte-multilingual-base", revision="refs/pr/31", dtype=torch.float32
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", revision="refs/pr/31")

        inputs = tokenizer(self.sentences, return_tensors="pt", padding=True, truncation=True).to(torch_device)

        with torch.no_grad():
            output = model(**inputs)[0]

        expected_shape = torch.Size((2, 15, 768))
        self.assertEqual(output.shape, expected_shape)

        # fmt: off
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [[ 0.2660, -0.2507,  0.3888], [ 0.5936, -0.4777,  0.5725], [ 0.5867, -0.4302,  0.5467]],
                        [[ 0.4506, -0.2547,  0.4193], [ 0.3973, -0.2549,  0.3836], [ 0.3309, -0.3932,  0.4126]],
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(output[:, 1:4, 1:4].cpu().detach(), expected_slice, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_no_head_english_v1_5(self):
        model = AutoModel.from_pretrained(
            "Alibaba-NLP/gte-base-en-v1.5", revision="refs/pr/17", dtype=torch.float32
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", revision="refs/pr/17")

        inputs = tokenizer(self.sentences, return_tensors="pt", padding=True, truncation=True).to(torch_device)

        with torch.no_grad():
            output = model(**inputs)[0]

        expected_shape = torch.Size((2, 13, 768))
        self.assertEqual(output.shape, expected_shape)

        # fmt: off
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [[-0.0137, -0.4422,  0.5301], [-0.2834,  0.4779,  0.6164], [-0.2570,  0.3076,  0.7647]],
                        [[-0.1446, -0.0088, -0.1134], [-0.1478,  0.1888,  0.3145], [-0.1605,  0.1750,  0.1975]],
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(output[:, 1:4, 1:4].cpu().detach(), expected_slice, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_no_head_snowflake_arctic_embed(self):
        model = AutoModel.from_pretrained("Snowflake/snowflake-arctic-embed-m-v2.0", dtype=torch.float32).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m-v2.0")

        inputs = tokenizer(self.sentences, return_tensors="pt", padding=True, truncation=True).to(torch_device)

        with torch.no_grad():
            output = model(**inputs)[0]

        expected_shape = torch.Size((2, 15, 768))
        self.assertEqual(output.shape, expected_shape)

        # fmt: off
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [[ 1.2827,  0.5775, -0.4335], [ 1.3125,  0.2771, -0.4684], [ 1.4864,  0.1338, -0.3023]],
                        [[ 1.6184, -0.2137, -0.8186], [ 1.6171, -0.2500, -0.5360], [ 1.4568, -0.3333, -0.5992]],
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(output[:, 1:4, 1:4].cpu().detach(), expected_slice, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_reranker(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            "Alibaba-NLP/gte-multilingual-reranker-base", revision="refs/pr/23", dtype=torch.float32
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-reranker-base", revision="refs/pr/23")

        inputs = tokenizer(self.sentences, return_tensors="pt", padding=True, truncation=True).to(torch_device)

        with torch.no_grad():
            output = model(**inputs).logits

        expected_shape = torch.Size((2, 1))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = Expectations({(None, None): torch.tensor([[0.2428], [0.5710]])}).get_expectation()

        torch.testing.assert_close(output.cpu().detach(), expected_slice, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_masked_lm(self):
        model = AutoModelForMaskedLM.from_pretrained(
            "Alibaba-NLP/gte-multilingual-mlm-base", revision="refs/pr/2", dtype=torch.float32
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-mlm-base", revision="refs/pr/2")

        inputs = tokenizer(self.sentences, return_tensors="pt", padding=True, truncation=True).to(torch_device)

        with torch.no_grad():
            output = model(**inputs).logits

        expected_shape = torch.Size((2, 15, 250048))
        self.assertEqual(output.shape, expected_shape)

        # fmt: off
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [[-1.9263,  5.8634,  3.9742], [-1.2569,  9.1684,  6.1088], [-1.1027,  6.3202,  8.6400]],
                        [[-2.3521,  4.1811,  9.3579], [-2.2531, -0.8155,  5.1775], [-2.1518,  4.4150,  7.3230]],
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(output[:, 1:4, 1:4].cpu().detach(), expected_slice, rtol=1e-3, atol=1e-3)
