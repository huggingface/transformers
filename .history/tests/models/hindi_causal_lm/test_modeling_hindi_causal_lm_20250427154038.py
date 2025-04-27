# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the Hindi Causal LM model."""

import unittest

from transformers import is_torch_available
from transformers.models.hindi_causal_lm.configuration_hindi_causal_lm import HindiCausalLMConfig
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device
from transformers.generation.utils import GenerationMixin

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...generation.test_utils import GenerationTesterMixin


if is_torch_available():
    import torch

    from transformers.models.hindi_causal_lm.modeling_hindi_causal_lm import (
        HindiCausalLMForCausalLM,
        HindiCausalLMModel,
    )
else:
    from transformers.models.hindi_causal_lm.dummy_pt_objects import (
        HindiCausalLMForCausalLM,
        HindiCausalLMModel,
    )


class HindiCausalLMModelTester:
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
        intermediate_size=64,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        scope=None,
        normalization_layer="rmsnorm",
        positional_encoding_type="rope",
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
        self.scope = scope
        self.normalization_layer = normalization_layer
        self.positional_encoding_type = positional_encoding_type
        # ensure hidden_size is divisible by num_attention_heads
        if hidden_size % num_attention_heads != 0:
            self.hidden_size = (
                hidden_size + num_attention_heads - (hidden_size % num_attention_heads)
            )  # make divisible

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
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            choice_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return HindiCausalLMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            normalization_layer=self.normalization_layer,
            positional_encoding_type=self.positional_encoding_type,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = HindiCausalLMModel(config=config)
        model.to(torch_device)
        model.eval()
        
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = HindiCausalLMForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, token_type_ids, input_mask, _, _, _ = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class HindiCausalLMModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (HindiCausalLMModel, HindiCausalLMForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (HindiCausalLMForCausalLM,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    is_encoder_decoder = False

    def setUp(self):
        self.model_tester = HindiCausalLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HindiCausalLMConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "convaiinnovations/hindi-foundational-model-base"
        model = HindiCausalLMModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_generate(self):
        """Test that the model can generate text from a prompt."""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        
        model = HindiCausalLMForCausalLM(config)
        model.to(torch_device)
        model.eval()
        
        # Generate using greedy decoding
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device=torch_device)
        
        # Check that generate works and produces longer output than input
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=10,
            do_sample=False,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
        )
        
        self.assertIsNotNone(generated_ids)
        self.assertTrue(generated_ids.shape[1] > input_ids.shape[1])

    def test_weight_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if is_torch_available():
            # Initialize model with default weights
            model = HindiCausalLMForCausalLM(config)
            
            # Check that weights are properly tied
            if config.tie_word_embeddings:
                self.assertTrue(
                    torch.allclose(
                        model.hindi_causal_lm.token_embeddings.weight,
                        model.lm_head.weight
                    )
                )
            
            # Check that lm_head weight has the correct shape
            self.assertEqual(
                model.lm_head.weight.shape,
                (config.vocab_size, config.hidden_size)
            )


@require_sentencepiece
@require_tokenizers
@require_torch
class HindiCausalLMIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_causal_lm(self):
        model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-foundational-model-base")
        
        # Hindi text: "हिंदी भाषा"
        input_ids = torch.tensor([[1, 47, 5096, 4329, 3697, 2567, 956]])
        
        output = model(input_ids)
        expected_shape = torch.Size([1, 7, 16000])  # [batch_size, seq_length, vocab_size]
        self.assertEqual(output.logits.shape, expected_shape)