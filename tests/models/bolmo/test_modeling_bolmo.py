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
"""Testing suite for the PyTorch Bolmo model."""

import unittest

from parameterized import parameterized

from transformers import BolmoConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION, ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, BolmoForCausalLM, BolmoModel
    from transformers.models.bolmo.modeling_bolmo import BolmoCache


class BolmoModelTester:
    def __init__(self, parent, batch_size=2, seq_length=16):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = True
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.vocab_size = 520

    def get_config(self):
        text_config = {
            "model_type": "olmo3",
            "vocab_size": 99,
            "hidden_size": self.hidden_size,
            "intermediate_size": 64,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
        }
        return BolmoConfig(
            text_config=text_config,
            hidden_size=self.hidden_size,
            local_intermediate_size=48,
            num_local_heads=4,
            num_local_decoder_layers=2,
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size // 2 - 4) + 4
        inputs_dict = {
            "input_ids": input_ids,
            "expanded_input_ids": ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size),
            "attention_mask": torch.ones_like(input_ids),
            # one patch per byte, so per-patch outputs have `seq_length` and batches have no patch padding
            "boundary_mask": torch.ones_like(input_ids, dtype=torch.bool),
        }
        return config, inputs_dict


class _DummyExpander:
    """Stands in for `BolmoTokenizer`: maps every byte to subword id 0."""

    def expand_byte_ids(self, byte_ids, n_last=None):
        return [0] * (n_last or len(byte_ids))


@require_torch
class BolmoModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BolmoModel, BolmoForCausalLM) if is_torch_available() else ()
    # `generate` needs `tokenizer=` to expand every new byte into a subword id, covered by `test_decode_matches_full_forward`
    all_generative_model_classes = ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = BolmoModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BolmoConfig, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_decode_matches_full_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = BolmoForCausalLM(config).to(torch_device).eval()
        input_ids, expanded_input_ids = inputs["input_ids"][:1], inputs["expanded_input_ids"][:1]
        with torch.no_grad():
            boundary_mask = model.model.predict_boundaries(input_ids, expanded_input_ids)
            full_logits = model(input_ids, expanded_input_ids, boundary_mask=boundary_mask).logits
            cache = BolmoCache(config)
            model(
                input_ids[:, :4], expanded_input_ids[:, :4], boundary_mask=boundary_mask[:, :4], past_key_values=cache
            )
            for i in range(4, input_ids.shape[1]):
                # a byte shifted by `vocab_size // 2` closes a patch
                step_ids = input_ids[:, i : i + 1] + config.vocab_size // 2 * boundary_mask[:, i : i + 1]
                step_logits = model(step_ids, expanded_input_ids[:, i : i + 1], past_key_values=cache).logits
                torch.testing.assert_close(step_logits[:, -1], full_logits[:, i])

    def test_beam_search_reorders_local_states(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        inputs.pop("boundary_mask")  # `generate` predicts it
        model = BolmoForCausalLM(config).to(torch_device).eval()
        tokenizer = _DummyExpander()
        greedy = model.generate(**inputs, tokenizer=tokenizer, max_new_tokens=6, do_sample=False)
        beams = model.generate(
            **inputs, tokenizer=tokenizer, max_new_tokens=6, do_sample=False, num_beams=2, num_return_sequences=2
        )
        self.assertEqual(beams.shape[0], 2 * greedy.shape[0])

    def test_reverse_loading_mapping(self):
        # the renames are anchored at `BolmoForCausalLM`'s `model.`, `base_model_prefix` handles `BolmoModel` (as llava)
        super().test_reverse_loading_mapping(skip_base_model=True)

    @unittest.skip("Bolmo embeds bytes and subwords, `inputs_embeds` alone is not enough")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("Calls the model without `expanded_input_ids`; RoPE is Olmo3's, tested there")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("Calls the model without `expanded_input_ids`; attention is Olmo3's, tested there")
    def test_eager_matches_sdpa_inference(self, *args):
        pass

    @unittest.skip("The boundary predictor is thresholded, it is trained by an auxiliary loss outside of this forward")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip("The boundary predictor is thresholded, it is trained by an auxiliary loss outside of this forward")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip("The boundary predictor is thresholded, it is trained by an auxiliary loss outside of this forward")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass


@slow
@require_torch
class BolmoIntegrationTest(unittest.TestCase):
    def test_model_1b_greedy_generation(self):
        # the original implementation (`trust_remote_code=True`, fp32) generates these texts and continues past them
        prompts_and_texts = [
            ("Language modeling is ", "Language modeling is a subfield of natural language processing that "),
            ("The capital of France is", "The capital of France is Paris, France.\n\nThe capital of France is Paris"),
            ("def fibonacci(n):", "def fibonacci(n):\n    if n == 1:\n        return 1\n    elif n == "),
        ]
        tokenizer = AutoTokenizer.from_pretrained("allenai/Bolmo-1B")
        model = BolmoForCausalLM.from_pretrained("allenai/Bolmo-1B", dtype=torch.float32)
        for prompt, expected_text in prompts_and_texts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, tokenizer=tokenizer, max_new_tokens=48, do_sample=False)
            self.assertEqual(tokenizer.decode(output[0], skip_special_tokens=True), expected_text)
