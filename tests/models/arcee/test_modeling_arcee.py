# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Arcee model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        ArceeConfig,
        ArceeForCausalLM,
        ArceeForQuestionAnswering,
        ArceeForSequenceClassification,
        ArceeForTokenClassification,
        ArceeModel,
    )
    from transformers.models.arcee.modeling_arcee import ArceeRotaryEmbedding


class ArceeModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = ArceeConfig
        base_model_class = ArceeModel
        causal_lm_class = ArceeForCausalLM
        sequence_class = ArceeForSequenceClassification
        token_class = ArceeForTokenClassification


@require_torch
class ArceeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            ArceeModel,
            ArceeForCausalLM,
            ArceeForSequenceClassification,
            ArceeForQuestionAnswering,
            ArceeForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ArceeModel,
            "text-classification": ArceeForSequenceClassification,
            "text-generation": ArceeForCausalLM,
            "zero-shot": ArceeForSequenceClassification,
            "question-answering": ArceeForQuestionAnswering,
            "token-classification": ArceeForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    model_tester_class = ArceeModelTester
    rotary_embedding_layer = ArceeRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = ArceeForCausalLM if is_torch_available() else None

    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for scaling_type in ["linear", "dynamic"]:
            config.rope_scaling = {"type": scaling_type, "factor": 2.0}
            model = ArceeModel(config)
            model.to(torch_device)
            model.eval()
            input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(torch_device)
            with torch.no_grad():
                model(input_ids)

    def test_model_rope_scaling_yarn(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.rope_scaling = {
            "type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
        }
        model = ArceeModel(config)
        model.to(torch_device)
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(torch_device)
        with torch.no_grad():
            model(input_ids)

    def test_arcee_mlp_uses_relu_squared(self):
        """Test that ArceeMLP uses ReLU² activation instead of SiLU."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.hidden_act = "relu2"  # Ensure we're using relu2 activation
        model = ArceeModel(config)

        # Check that the MLP layers use the correct activation
        for layer in model.layers:
            mlp = layer.mlp
            # Test with a simple input
            x = torch.randn(1, 10, config.hidden_size)
            up_output = mlp.up_proj(x)

            # Verify ReLU² activation: x * relu(x)
            expected_activation = up_output * torch.relu(up_output)
            actual_activation = mlp.act_fn(up_output)

            self.assertTrue(torch.allclose(expected_activation, actual_activation, atol=1e-5))
            break  # Only test the first layer


@require_torch_accelerator
class ArceeIntegrationTest(unittest.TestCase):
    def tearDown(self):
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    @slow
    def test_model_from_pretrained(self):
        # This test would be enabled once a pretrained model is available
        # For now, we just test that the model can be instantiated
        config = ArceeConfig()
        model = ArceeForCausalLM(config)
        self.assertIsInstance(model, ArceeForCausalLM)
