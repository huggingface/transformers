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
"""Testing suite for the PyTorch Dory model."""

import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import AutoTokenizer, DoryConfig, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import DoryForCausalLM, DoryModel


class DoryModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DoryModel

    # One loop keeps the common tests' one-cache-slot-per-layer assumption; loops are tested below
    small_config_overrides = {"n_input_layers": 1, "n_output_layers": 0, "n_recurrent_loops": 1}


@require_torch
class DoryModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = DoryModelTester
    # The nGPT normalized updates start small (`ngpt_alpha_init`), so overfitting takes a few more steps
    training_overfit_steps = 400

    @unittest.skip("New model: `cache` only, no `past_key_values` BC")
    def test_bc_cache_name_forward(self):
        pass

    def test_reverse_loading_mapping(self):
        # Checkpoints are saved with the `backbone.` prefix, which only exists on the head model
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_chunked_prefill_and_decode_match_full_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_hidden_layers, config.layer_types, config.rope_parameters = 4, None, None
        config.n_input_layers, config.n_recurrent_layers, config.n_output_layers, config.n_recurrent_loops = 1, 2, 1, 3
        config.__post_init__()
        model = DoryForCausalLM(config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        with torch.no_grad():
            full_logits = model(input_ids).logits
            outputs = model(input_ids[:, :4], use_cache=True)
            logits = [outputs.logits]
            for position in range(4, input_ids.shape[1]):
                outputs = model(input_ids[:, position : position + 1], cache=outputs.cache)
                logits.append(outputs.logits)
        torch.testing.assert_close(torch.cat(logits, dim=1), full_logits, rtol=1e-4, atol=1e-4)

    def test_prompt_lookup_past_sliding_window_with_loops(self):
        # Assisted decoding rolls the cache back: the later loops' sliding layers must record their past too
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_hidden_layers, config.layer_types, config.rope_parameters = 4, None, None
        config.n_input_layers, config.n_recurrent_layers, config.n_output_layers, config.n_recurrent_loops = 1, 2, 1, 3
        config.sliding_window = 8
        config.__post_init__()
        model = DoryForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5] * 3], device=torch_device)
        greedy = model.generate(input_ids, max_new_tokens=24, do_sample=False)
        prompt_lookup = model.generate(input_ids, max_new_tokens=24, do_sample=False, prompt_lookup_num_tokens=4)
        self.assertTrue(torch.equal(greedy, prompt_lookup))

    def test_at_least_one_recurrent_loop(self):
        with self.assertRaises(StrictDataclassClassValidationError):
            DoryConfig(n_recurrent_loops=0)

    def test_loop_count_override_is_validated_on_load(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        with tempfile.TemporaryDirectory() as directory:
            DoryForCausalLM(config).save_pretrained(directory)
            for cls in (DoryConfig, DoryForCausalLM):
                for loops in (0, -1):
                    with self.subTest(cls=cls.__name__, loops=loops):
                        with self.assertRaises(StrictDataclassClassValidationError):
                            cls.from_pretrained(directory, n_recurrent_loops=loops)
            loaded = DoryForCausalLM.from_pretrained(directory, n_recurrent_loops=2)
            self.assertEqual(loaded.config.n_recurrent_loops, 2)

    @torch.no_grad()
    def test_resize_preserves_learned_logit_scale(self):
        for mean_resizing in (False, True):
            with self.subTest(mean_resizing=mean_resizing):
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                input_ids = inputs_dict["input_ids"] % (config.vocab_size - 2)
                model = DoryForCausalLM(config).to(torch_device).eval()
                original_size = model.config.vocab_size
                scale = torch.linspace(0.5, 1.5, original_size, device=torch_device) * config.initializer_range
                model.lm_head.logit_scale.copy_(scale)
                expected = model(input_ids, use_cache=False).logits
                for size in (original_size + 8, original_size - 2):
                    model.resize_token_embeddings(size, mean_resizing=mean_resizing)
                    retained = min(size, original_size)
                    torch.testing.assert_close(model.lm_head.logit_scale[:retained], scale[:retained])
                    torch.testing.assert_close(
                        model(input_ids, use_cache=False).logits[..., :retained], expected[..., :retained]
                    )
                    if size > original_size:
                        torch.testing.assert_close(
                            model.lm_head.logit_scale[original_size:],
                            scale.new_full((size - original_size,), config.initializer_range),
                        )
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    restored = DoryForCausalLM.from_pretrained(directory).to(torch_device).eval()
                torch.testing.assert_close(restored(input_ids, use_cache=False).logits, expected[..., :retained])

    def _small_config(self, **kwargs):
        # The tester's widths with the default 3-sliding-1-full layer pattern: full-width defaults use ~5 GB
        sizes = (
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
        return DoryConfig(**{name: getattr(self.model_tester, name) for name in sizes}, **kwargs)

    def test_recurrent_cache_layout(self):
        # Plain cache layers: `per_loop` appends one copy of the recurrent layers per extra loop, `last_loop` adds none
        config = self._small_config(num_hidden_layers=4, n_input_layers=1, n_output_layers=1, n_recurrent_loops=3)
        input_ids = torch.tensor([[1, 2, 3]])
        for mode, expected in [
            ("per_loop", ["DynamicSlidingWindowLayer"] * 3 + ["DynamicLayer"] + ["DynamicSlidingWindowLayer"] * 4),
            ("last_loop", ["DynamicSlidingWindowLayer"] * 3 + ["DynamicLayer"]),
        ]:
            config.recurrent_kv_cache_mode = mode
            cache = DoryModel(config).eval()(input_ids, use_cache=True).cache
            self.assertEqual([type(layer).__name__ for layer in cache.layers], expected)
            self.assertTrue(all(layer.get_seq_length() == 3 for layer in cache.layers))

    def test_recurrent_layers_use_the_mask_and_rope_of_their_attention_type(self):
        config = self._small_config(num_hidden_layers=4, n_input_layers=1, n_output_layers=1, n_recurrent_loops=2)
        model = DoryModel(config)
        self.assertEqual(
            [layer.attention_type for layer in model.layers],
            ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        )
        self.assertTrue(model.layers[1].self_attn.is_local_attention)
        self.assertEqual(model.layers[1].self_attn.sliding_window, config.sliding_window)


@slow
@require_torch_accelerator
class DoryIntegrationTest(unittest.TestCase):
    checkpoint = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"

    def test_bf16_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model = DoryForCausalLM.from_pretrained(self.checkpoint, dtype=torch.bfloat16, device_map=torch_device)
        inputs = tokenizer("The capital of France is", return_tensors="pt").to(torch_device)
        output = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        self.assertEqual(
            tokenizer.decode(output[0], skip_special_tokens=True),
            "The capital of France is Paris. The capital of Germany is Berlin. "
            "The capital of Italy is Rome. The capital of Spain is Madrid. The",
        )
