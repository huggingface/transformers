# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        ModernBertDecoderForCausalLM,
        ModernBertDecoderForSequenceClassification,
        ModernBertDecoderModel,
    )


class ModernBertDecoderModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = ModernBertDecoderModel


@require_torch
class ModernBertDecoderModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = ModernBertDecoderModelTester

    def test_model_rope_scaling_frequencies(self):
        """Tests the frequency properties of the different RoPE scaling types on the model RoPE layer."""
        # ModernBertDecoder has different RoPE configs per layer type
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # Retrieves the RoPE layer class from the base model class. Uses `.named_modules()` to avoid hardcoding the
        # named location of the RoPE layer class.
        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",  # most common case
            "global_rotary_emb",
            "local_rotary_emb",
        ]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        rope_params = {"rope_type": "default", "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short, layer_type="sliding_attention")
        original_cos_long, original_sin_long = original_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        rope_params = {"rope_type": "linear", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        rope_params = {"rope_type": "dynamic", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue(
            (ntk_scaling_rope.sliding_attention_inv_freq <= original_rope.sliding_attention_inv_freq).all()
        )

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        rope_params = {"rope_type": "yarn", "factor": scaling_factor, "rope_theta": 10_000.0}
        config.rope_parameters = {"sliding_attention": rope_params, "full_attention": rope_params}
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short, layer_type="sliding_attention")
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long, layer_type="sliding_attention")
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)


@slow
@require_torch
class ModernBertDecoderIntegrationTest(unittest.TestCase):
    def test_inference_causal_lm(self):
        model = ModernBertDecoderForCausalLM.from_pretrained("blab-jhu/test-32m-dec", attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained("blab-jhu/test-32m-dec")

        inputs = tokenizer("Paris is the capital of", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 6, model.config.vocab_size))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[-8.0183, -7.1578, -0.4453], [-6.2909, -6.1557, 4.9063], [-6.7689, -5.8068, 6.1078]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    def test_inference_no_head(self):
        model = ModernBertDecoderModel.from_pretrained("blab-jhu/test-32m-dec", attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained("blab-jhu/test-32m-dec")

        inputs = tokenizer("Paris is the capital of", return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs)[0]
        expected_shape = torch.Size((1, 6, model.config.hidden_size))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[-0.0306, -0.0115, 0.0007], [-0.2485, -0.1381, 0.0872], [0.3133, -0.1777, 0.1667]]]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)

    def test_generation(self):
        model = ModernBertDecoderForCausalLM.from_pretrained("blab-jhu/test-32m-dec", attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained("blab-jhu/test-32m-dec")

        inputs = tokenizer("The weather today is", return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Check that we got some reasonable output
        self.assertEqual(len(output_text), 1)
        self.assertTrue(len(output_text[0]) > len("The weather today is"))

    def test_sliding_window_long_context(self):
        """
        Test that ModernBertDecoder works with sliding window attention for longer sequences.
        """

        model = ModernBertDecoderForCausalLM.from_pretrained("blab-jhu/test-32m-dec", attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained("blab-jhu/test-32m-dec")

        # Create a longer input to test sliding window attention
        long_input = "This is a test. " * 50  # Repeat to make it longer
        inputs = tokenizer(long_input, return_tensors="pt", truncation=True, max_length=512)

        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        # Check that generation worked with longer context
        self.assertEqual(outputs.shape[0], 1)
        self.assertGreater(outputs.shape[1], inputs["input_ids"].shape[1])

    def test_sequence_classification(self):
        """
        Test that ModernBertDecoderForSequenceClassification works correctly.
        """

        model = ModernBertDecoderForSequenceClassification.from_pretrained(
            "blab-jhu/test-32m-dec", num_labels=2, attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained("blab-jhu/test-32m-dec")

        # Test with sample input
        inputs = tokenizer("This is a positive example.", return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Check output shape
        expected_shape = (1, 2)  # batch_size=1, num_labels=2
        self.assertEqual(outputs.logits.shape, expected_shape)

        # Test with labels
        labels = torch.tensor([1])
        outputs_with_loss = model(**inputs, labels=labels)

        # Check that loss is computed
        self.assertIsNotNone(outputs_with_loss.loss)
        self.assertTrue(isinstance(outputs_with_loss.loss.item(), float))
