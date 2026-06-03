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

from transformers import ESMCConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        ESMCForMaskedLM,
        ESMCForSequenceClassification,
        ESMCForTokenClassification,
        ESMCModel,
    )


class ESMCModelTester:
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
        return ESMCConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            pad_token_id=1,
            initializer_range=self.initializer_range,
            num_labels=self.num_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels):
        model = ESMCModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

    def create_and_check_for_masked_lm(self, config, input_ids, input_mask, sequence_labels, token_labels):
        model = ESMCForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = ESMCForSequenceClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = ESMCForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, sequence_labels, token_labels = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class ESMCModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_mismatched_shapes = False
    test_resize_embeddings = False  # ESMC's lm_head is an nn.Sequential, not a tied decoder

    all_model_classes = (
        (
            ESMCModel,
            ESMCForMaskedLM,
            ESMCForSequenceClassification,
            ESMCForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ESMCModel,
            "fill-mask": ESMCForMaskedLM,
            "text-classification": ESMCForSequenceClassification,
            "token-classification": ESMCForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_sequence_classification_problem_types = True

    def setUp(self):
        self.model_tester = ESMCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ESMCConfig, common_properties=["d_model", "n_heads"])

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

    @unittest.skip(
        reason="ESMC returns `hidden_states` as a single stacked tensor (used by the SAE feature), "
        "not the live per-layer tensors, so grad does not flow back to the returned copy."
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass


@slow
@require_torch
class ESMCModelIntegrationTest(unittest.TestCase):
    def test_inference_masked_lm(self):
        model = ESMCForMaskedLM.from_pretrained("biohub/ESMC-300M").to(torch_device).eval()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-300M")
        inputs = tokenizer(["MKTAYIAKQR"], return_tensors="pt").to(torch_device)
        with torch.no_grad():
            logits = model(**inputs).logits
        self.assertEqual(logits.shape, (1, inputs["input_ids"].shape[1], model.config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())
