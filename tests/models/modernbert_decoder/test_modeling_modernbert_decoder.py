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

from packaging import version

from transformers import AutoTokenizer, ModernBertDecoderConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import _config_zero_init


if is_torch_available():
    import torch

    from transformers import (
        ModernBertDecoderForCausalLM,
        ModernBertDecoderForSequenceClassification,
        ModernBertDecoderModel,
    )


class ModernBertDecoderModelTester(CausalLMModelTester):
    config_class = ModernBertDecoderConfig
    if is_torch_available():
        base_model_class = ModernBertDecoderModel
        causal_lm_class = ModernBertDecoderForCausalLM


@require_torch
class ModernBertDecoderModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (ModernBertDecoderModel, ModernBertDecoderForCausalLM, ModernBertDecoderForSequenceClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": ModernBertDecoderModel,
            "text-generation": ModernBertDecoderForCausalLM,
            "text-classification": ModernBertDecoderForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    test_head_masking = False
    test_pruning = False
    model_tester_class = ModernBertDecoderModelTester

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                # The classifier.weight from ModernBertDecoderForSequenceClassification
                # is initialized without `initializer_range`, so it's not set to ~0 via the _config_zero_init
                if param.requires_grad and not (
                    name == "classifier.weight" and model_class in [ModernBertDecoderForSequenceClassification]
                ):
                    data = torch.flatten(param.data)
                    n_elements = torch.numel(data)
                    # skip 2.5% of elements on each side to avoid issues caused by `nn.init.trunc_normal_` described in
                    # https://github.com/huggingface/transformers/pull/27906#issuecomment-1846951332
                    n_elements_to_skip_on_each_side = int(n_elements * 0.025)
                    data_to_check = torch.sort(data).values
                    if n_elements_to_skip_on_each_side > 0:
                        data_to_check = data_to_check[n_elements_to_skip_on_each_side:-n_elements_to_skip_on_each_side]
                    self.assertIn(
                        ((data_to_check.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


@slow
@require_torch
class ModernBertDecoderIntegrationTest(unittest.TestCase):
    def test_inference_causal_lm(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

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
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

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
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

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
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

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
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

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
