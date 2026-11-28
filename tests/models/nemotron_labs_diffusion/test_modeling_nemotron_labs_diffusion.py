# Copyright 2026 NVIDIA Corporation and the HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest
from pathlib import Path

from safetensors.torch import save_file

from transformers import AutoModel, AutoModelForCausalLM, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import NemotronLabsDiffusionModel


class NemotronLabsDiffusionModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = NemotronLabsDiffusionModel

    def get_config(self):
        config = super().get_config()
        config.mask_token_id = config.vocab_size - 1
        return config


@require_torch
class NemotronLabsDiffusionModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = NemotronLabsDiffusionModelTester

    def test_reverse_loading_mapping(self):
        super().test_reverse_loading_mapping(skip_base_model=True)

    @torch.inference_mode()
    def test_diffusion_canvas_reads_the_causal_prefix_without_growing_it(self):
        prefix = torch.tensor([[1, 2, 3]], device=torch_device)
        canvas = torch.tensor([[4, 5, 6]], device=torch_device)
        changed_canvas = torch.tensor([[4, 5, 7]], device=torch_device)
        block_ids = torch.tensor([[-1, -1, -1, 0, 0, 0]], device=torch_device)
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                config = self.model_tester.get_config()
                model = AutoModelForCausalLM.from_config(config, attn_implementation=backend).to(torch_device).eval()
                prefill = model(prefix, use_cache=True)
                cache = prefill.cache
                original_layers = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
                denoise = model(canvas, cache=cache, use_cache=False, attention_mode="bidirectional").logits
                changed = model(changed_canvas, cache=cache, use_cache=False, attention_mode="bidirectional").logits
                self.assertFalse(torch.allclose(denoise[:, 0], changed[:, 0]))
                self.assertEqual(cache.get_seq_length(), prefix.shape[-1])
                for layer, (keys, values) in zip(cache.layers, original_layers):
                    torch.testing.assert_close(layer.keys, keys, atol=0, rtol=0)
                    torch.testing.assert_close(layer.values, values, atol=0, rtol=0)

                full_ids = torch.cat((prefix, canvas), dim=-1)
                dense = model(full_ids, attention_mode="block", block_sequence_ids=block_ids).logits
                torch.testing.assert_close(dense[:, :3], prefill.logits)
                torch.testing.assert_close(dense[:, 3:], denoise)
                committed = model(canvas, cache=cache, use_cache=True)
                torch.testing.assert_close(committed.logits, model(full_ids).logits[:, 3:])
                self.assertEqual(committed.cache.get_seq_length(), full_ids.shape[-1])

    @torch.inference_mode()
    def test_two_stream_block_visibility(self):
        config = self.model_tester.get_config()
        config.block_size = 2
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").to(torch_device).eval()
        tokens = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], device=torch_device)
        output = model(tokens, attention_mode="block_diff", output_attentions=True, use_cache=False)
        visible_keys = [
            [0, 1],
            [0, 1],
            [2, 3, 4, 5],
            [2, 3, 4, 5],
            [4],
            [4, 5],
            [4, 5, 6],
            [4, 5, 6, 7],
        ]
        for query_idx, expected in enumerate(visible_keys):
            self.assertEqual(output.attentions[0][0, 0, query_idx].nonzero().flatten().tolist(), expected)
        with self.assertRaisesRegex(ValueError, "Diffusion training"):
            model(tokens, labels=tokens, attention_mode="block_diff")

    @torch.inference_mode()
    def test_released_checkpoint_key_layout_loads_an_untied_head(self):
        config = self.model_tester.get_config()
        config.auto_map = {"AutoModel": "modeling_nemotron_labs_diffusion.NemotronLabsDiffusionModel"}
        model = AutoModelForCausalLM.from_config(config).eval()
        tokens = torch.tensor([[1, 2, 3]])
        state_dict = {
            name.replace("model.", "encoder.", 1).replace("lm_head.", "diffusion_head.", 1): weight
            for name, weight in model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            save_file(state_dict, Path(directory) / "model.safetensors")
            loaded = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=False)
            backbone = AutoModel.from_pretrained(directory, trust_remote_code=False)
            self.assertIsInstance(backbone, NemotronLabsDiffusionModel)
            torch.testing.assert_close(loaded(tokens).logits, model(tokens).logits)
            self.assertNotEqual(loaded.lm_head.weight.data_ptr(), loaded.model.embed_tokens.weight.data_ptr())
            hidden_states = backbone(tokens).last_hidden_state
            torch.testing.assert_close(hidden_states, model.model(tokens).last_hidden_state)
            changed_hidden_states = backbone(torch.tensor([[1, 2, 4]])).last_hidden_state
            self.assertFalse(torch.allclose(hidden_states[:, 0], changed_hidden_states[:, 0]))

    @torch.inference_mode()
    def test_linear_speculation_keeps_only_verified_cache_positions(self):
        config = self.model_tester.get_config()
        config.mask_token_id = config.vocab_size - 1
        model = AutoModelForCausalLM.from_config(config).to(torch_device).eval()
        prompt = torch.tensor([[1, 2, 3]], device=torch_device)
        expected_calls = iter(
            [
                ("causal", True, 3, [8]),
                ("bidirectional", False, 3, [8, 9, 10, 11]),
                ("causal", True, 7, [9, 20, 30, 40]),
                ("bidirectional", False, 5, [20, 21, 22, 23]),
                ("causal", True, 9, [21, 22, 23, 2]),
            ]
        )

        def force_predictions(module, args, kwargs, output):
            mode, use_cache, cache_len, token_ids = next(expected_calls)
            self.assertEqual(kwargs.get("attention_mode", "causal"), mode)
            self.assertEqual(kwargs["use_cache"], use_cache)
            cache = output.cache if use_cache else kwargs["cache"]
            self.assertEqual(cache.get_seq_length(), cache_len)
            output.logits.fill_(-1)
            prediction_ids = torch.tensor(token_ids, device=output.logits.device).view(1, -1, 1)
            output.logits.scatter_(-1, prediction_ids, 1)
            return output

        handle = model.register_forward_hook(force_predictions, with_kwargs=True)
        try:
            generated, nfe = model.linear_spec_generate(prompt, max_new_tokens=12, block_length=4, eos_token_id=2)
        finally:
            handle.remove()
        self.assertEqual(generated.tolist(), [[1, 2, 3, 8, 9, 20, 21, 22, 23, 2]])
        self.assertEqual(nfe, 5)
        self.assertEqual(list(expected_calls), [])

    @torch.inference_mode()
    def test_diffusion_generate_unmasks_blocks(self):
        config = self.model_tester.get_config()
        config.mask_token_id = config.vocab_size - 1
        model = AutoModelForCausalLM.from_config(config).to(torch_device).eval()
        prompt = torch.tensor([[1, 2, 3]], device=torch_device)
        for causal_context, threshold in ((True, None), (True, 0.9), (False, None)):
            with self.subTest(causal_context=causal_context, threshold=threshold):
                output_ids, nfe = model.diffusion_generate(
                    prompt, max_new_tokens=6, block_length=3, threshold=threshold, causal_context=causal_context
                )
                self.assertEqual(output_ids.shape[0], 1)
                self.assertGreater(nfe, 0)
                self.assertFalse((output_ids[:, prompt.shape[1] :] == config.mask_token_id).any())
