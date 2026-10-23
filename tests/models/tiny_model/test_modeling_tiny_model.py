# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch TinyModel model."""

import unittest
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download

from transformers import TinyModelConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import TinyModel, TinyModelForCausalLM
    from transformers.models.tiny_model.convert_tiny_model_weights_to_hf import convert_tiny_model_checkpoint


class TinyModelModelTester(CausalLMModelTester):
    config_class = TinyModelConfig
    if is_torch_available():
        base_model_class = TinyModel
        causal_lm_class = TinyModelForCausalLM

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            hidden_act="relu",
            intermediate_size=64,
            max_position_embeddings=64,
        )


@require_torch
class TinyModelModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = TinyModelModelTester
    training_overfit_learning_rate = 2e-3

    def test_cache_inputs_are_rejected(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TinyModelForCausalLM(config).to(torch_device)

        with self.assertRaisesRegex(ValueError, "does not support key/value caching"):
            model(**inputs_dict, use_cache=True)
        with self.assertRaisesRegex(ValueError, "does not support `past_key_values`"):
            model(**inputs_dict, past_key_values=object())

    def test_generation_from_input_embeddings_is_rejected(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = TinyModelForCausalLM(config).to(torch_device)
        input_ids = inputs_dict["input_ids"]

        with self.assertRaisesRegex(ValueError, "cannot generate from `inputs_embeds`"):
            model.prepare_inputs_for_generation(
                input_ids,
                inputs_embeds=model.get_input_embeddings()(input_ids),
                use_cache=False,
            )

    def test_config_return_dict_false(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = False
        model = TinyModelForCausalLM(config).to(torch_device).eval()

        with torch.no_grad():
            tuple_outputs = model(**inputs_dict)
            structured_outputs = model(**inputs_dict, return_dict=True)

        self.assertIsInstance(tuple_outputs, tuple)
        torch.testing.assert_close(tuple_outputs[0], structured_outputs.logits)

    def _greedy_generate(self, *args, use_cache=False, **kwargs):
        return super()._greedy_generate(*args, use_cache=use_cache, **kwargs)

    def _sample_generate(self, *args, use_cache=False, **kwargs):
        return super()._sample_generate(*args, use_cache=use_cache, **kwargs)

    def _beam_search_generate(self, *args, use_cache=False, **kwargs):
        return super()._beam_search_generate(*args, use_cache=use_cache, **kwargs)

    def _beam_sample_generate(self, *args, use_cache=False, **kwargs):
        return super()._beam_sample_generate(*args, use_cache=use_cache, **kwargs)

    @unittest.skip("TinyModel does not support caching, but this common test explicitly enables it")
    def test_generate_methods_with_logits_to_keep(self):
        pass

    @unittest.skip("TinyModel does not support caching, but this common test explicitly enables it")
    def test_generate_with_and_without_position_ids(self):
        pass


@slow
@require_torch
class TinyModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint_dir = TemporaryDirectory()
        cls.addClassCleanup(cls.checkpoint_dir.cleanup)
        checkpoint_path = hf_hub_download(
            repo_id="noanabeshima/tiny_model",
            filename="tiny_model.pt",
            revision="502a1f2453f61260c937f7807a1270a167faba07",
        )
        # The author publishes raw .pt weights, so prepare a native checkpoint for from_pretrained.
        convert_tiny_model_checkpoint(checkpoint_path, cls.checkpoint_dir.name, expected_num_hidden_layers=4)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_logits(self):
        model = TinyModelForCausalLM.from_pretrained(self.checkpoint_dir.name, dtype=torch.float16).to(torch_device)
        model.eval()
        input_ids = torch.tensor([[9_996, 51, 56, 4, 36]], device=torch_device)

        with torch.no_grad():
            logits = model(input_ids).logits.float()

        # Reference: the author's FP16 model at revision ae0c9031c467d05e12914bd7be19aaa6262dc7bf.
        expected_logits = torch.tensor(
            [13.625, 27.046875, 14.78125, 15.2109375, 19.171875, 13.296875, 15.9375, 9.046875, 14.515625, 8.8984375],
            device=torch_device,
        )
        expected_mean = torch.tensor(3.2073133, device=torch_device)

        torch.testing.assert_close(logits[0, -1, :10], expected_logits, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(logits[0, -1].mean(), expected_mean, rtol=1e-3, atol=1e-3)

    def test_model_generation(self):
        model = TinyModelForCausalLM.from_pretrained(self.checkpoint_dir.name, dtype=torch.float16).to(torch_device)
        model.eval()
        input_ids = torch.tensor([[9_996, 51, 56, 4, 36]], device=torch_device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False)

        expected_ids = [[9_996, 51, 56, 4, 36, 1, 38, 6, 4, 31, 66, 39, 49, 0, 21]]
        self.assertEqual(generated_ids.tolist(), expected_ids)
