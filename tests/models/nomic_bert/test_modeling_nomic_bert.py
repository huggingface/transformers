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

from transformers import AutoModel, AutoTokenizer, NomicBertConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
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
        NomicBertForMaskedLM,
        NomicBertForSequenceClassification,
        NomicBertForTokenClassification,
        NomicBertModel,
    )


class NomicBertModelTester:
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
        max_position_embeddings=2048,
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
        next_sentence_label = None

        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)
            next_sentence_label = ids_tensor([self.batch_size], 2)

        config = self.get_config()

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            next_sentence_label,
        )

    def get_config(self):
        """
        Returns a tiny configuration by default.
        """
        return NomicBertConfig(
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
            is_decoder=False,
            use_cache=False,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        next_sentence_label,
    ):
        model = NomicBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output, None)

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        next_sentence_label,
    ):
        model = NomicBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        next_sentence_label,
    ):
        config.num_labels = self.num_labels
        model = NomicBertForSequenceClassification(config)
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
        choice_labels,
        next_sentence_label,
    ):
        config.num_labels = self.num_labels
        model = NomicBertForTokenClassification(config=config)
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
            choice_labels,
            next_sentence_label,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class NomicBertModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            NomicBertModel,
            NomicBertForMaskedLM,
            NomicBertForSequenceClassification,
            NomicBertForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": NomicBertModel,
            "fill-mask": NomicBertForMaskedLM,
            "text-classification": NomicBertForSequenceClassification,
            "token-classification": NomicBertForTokenClassification,
            "zero-shot": NomicBertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = NomicBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NomicBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

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
class NomicBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding_v1_5(self):
        # TODO: remove revision
        model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", revision="refs/pr/57").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", revision="refs/pr/57")

        sentences = ["Plants create oxygen.", "Photosynthesis is a process where plants create oxygen."]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(torch_device)

        with torch.no_grad():
            output = model(**inputs)[0]

        expected_shape = torch.Size((2, 13, 768))
        self.assertEqual(output.shape, expected_shape)

        # fmt: off
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [
                            [1.7039e00, -4.5610e00, 1.5236e00],
                            [1.8685e00, -3.6936e00, 1.6641e00],
                            [5.3303e-01, -4.2081e00, 2.3375e00],
                        ],
                        [
                            [2.6867e-03, -3.7496e00, 9.0820e-01],
                            [1.8297e-02, -3.3884e00, 3.5300e-01],
                            [-1.4282e-01, -3.6776e00, -3.5079e-01],
                        ],
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(output[:, 1:4, 1:4].cpu().detach(), expected_slice, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_no_head_absolute_embedding_v1(self):
        # TODO: remove revision
        model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1", revision="refs/pr/34").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1", revision="refs/pr/34")

        sentences = ["Plants create oxygen.", "Photosynthesis is a process where plants create oxygen."]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(torch_device)

        with torch.no_grad():
            output = model(**inputs)[0]

        expected_shape = torch.Size((2, 13, 768))
        self.assertEqual(output.shape, expected_shape)

        # fmt: off
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [
                            [ 1.2961, -1.1757,  1.2094],
                            [ 1.1350,  0.5400,  1.4580],
                            [-0.2897, -0.5351,  2.0092],
                        ],
                        [
                            [-0.2866, -0.9786,  0.8613],
                            [-0.3104, -0.3421,  0.4867],
                            [-0.4336, -0.8528, -0.2509],
                        ]
                    ]
                ),
            }
        ).get_expectation()
        # fmt: on

        torch.testing.assert_close(output[:, 1:4, 1:4].cpu().detach(), expected_slice, rtol=1e-3, atol=1e-3)
