# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ESMC model."""

import unittest

from transformers import AutoTokenizer, EsmcConfig, is_torch_available
from transformers.testing_utils import Expectations, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        EsmcForMaskedLM,
        EsmcForSequenceClassification,
        EsmcForTokenClassification,
        EsmcModel,
    )


class EsmcModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        vocab_size=33,
        d_model=32,
        n_layers=2,
        n_heads=4,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        # aliases consumed by ModelTesterMixin
        self.hidden_size = d_model
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.num_labels)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()
        return config, input_ids, input_mask, sequence_labels, token_labels

    def get_config(self):
        return EsmcConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            pad_token_id=1,
            initializer_range=self.initializer_range,
            num_labels=self.num_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels):
        model = EsmcModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

    def create_and_check_for_masked_lm(self, config, input_ids, input_mask, sequence_labels, token_labels):
        model = EsmcForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = EsmcForSequenceClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(self, config, input_ids, input_mask, sequence_labels, token_labels):
        config.num_labels = self.num_labels
        model = EsmcForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, sequence_labels, token_labels = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class EsmcModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_mismatched_shapes = False
    test_resize_embeddings = False  # ESMC's lm_head decoder is untied (tie_word_embeddings=False)

    all_model_classes = (
        (
            EsmcModel,
            EsmcForMaskedLM,
            EsmcForSequenceClassification,
            EsmcForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": EsmcModel,
            "fill-mask": EsmcForMaskedLM,
            "text-classification": EsmcForSequenceClassification,
            "token-classification": EsmcForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_sequence_classification_problem_types = True

    def setUp(self):
        self.model_tester = EsmcModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EsmcConfig, common_properties=["d_model", "n_heads"])

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


@slow
@require_torch
class EsmcModelIntegrationTest(unittest.TestCase):
    checkpoint = "biohub/ESMC-300M"
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"

    def test_inference_masked_lm(self):
        model = EsmcForMaskedLM.from_pretrained(self.checkpoint, dtype=torch.bfloat16).to(torch_device).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        inputs = tokenizer([self.sequence], return_tensors="pt").to(torch_device)

        with torch.no_grad():
            logits = model(**inputs).logits

        self.assertEqual(logits.shape, (1, inputs["input_ids"].shape[1], model.config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

        # fmt: off
        expected_slice = Expectations(
            {
                (None, None): torch.tensor([
                    [-36.000, -36.000, -36.000, 14.250, 21.250, 20.125],
                    [-29.750, -29.750, -29.875, 22.500, 28.125, 27.750],
                    [-31.250, -31.250, -31.250, 21.250, 27.500, 27.125],
                ]),
                ("cpu", None): torch.tensor([
                    [-36.000, -36.000, -36.250, 14.250, 21.375, 20.125],
                    [-29.875, -29.875, -29.875, 22.500, 28.125, 27.750],
                    [-31.250, -31.250, -31.250, 21.125, 27.500, 27.125],
                ]),
            }
        ).get_expectation()
        # fmt: on
        torch.testing.assert_close(logits[0, 1:4, :6].float().cpu(), expected_slice, rtol=1e-2, atol=0.5)

    def test_inference_last_hidden_state(self):
        model = EsmcModel.from_pretrained(self.checkpoint, dtype=torch.bfloat16).to(torch_device).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        inputs = tokenizer([self.sequence], return_tensors="pt").to(torch_device)

        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state

        self.assertEqual(last_hidden_state.shape, (1, inputs["input_ids"].shape[1], model.config.hidden_size))
        self.assertTrue(torch.isfinite(last_hidden_state).all())

        # fmt: off
        expected_slice = Expectations(
            {
                (None, None): torch.tensor([
                    [ 0.006805, -0.008179, 0.038574, 0.038330, 0.011841, 0.039307],
                    [-0.016113, -0.017090, 0.008972, 0.027832, 0.003937, 0.071777],
                    [-0.003204, -0.026367, 0.002411, 0.024170, 0.025024, 0.047852],
                ]),
                ("cpu", None): torch.tensor([
                    [ 0.007080, -0.008179, 0.038574, 0.038574, 0.011597, 0.039307],
                    [-0.015991, -0.016968, 0.008911, 0.027710, 0.003784, 0.071777],
                    [-0.003159, -0.026489, 0.002502, 0.024170, 0.024902, 0.047852],
                ]),
            }
        ).get_expectation()
        # fmt: on
        torch.testing.assert_close(last_hidden_state[0, 1:4, :6].float().cpu(), expected_slice, rtol=1e-2, atol=1e-2)
