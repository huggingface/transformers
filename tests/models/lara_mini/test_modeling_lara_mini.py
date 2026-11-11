# Copyright 2026 Translated SRL and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LaraMini model."""

import unittest

from transformers import AutoTokenizer, DataCollatorWithFlattening, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_causal_conv1d,
    require_flash_linear_attention,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import DynamicCache, LaraMiniForCausalLM, LaraMiniModel


class LaraMiniModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = LaraMiniModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.hidden_act = "silu"
        self.layer_types = ["linear_attention", "sliding_attention", "full_attention"]
        self.num_hidden_layers = len(self.layer_types)
        # Grouped-query attention, with as many key/value heads as tensor-parallel test ranks
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 2
        self.linear_num_value_heads = 4


@require_torch
class LaraMiniModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = LaraMiniModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        conv_dim = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        return (batch_size, conv_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim)

    def test_attention_outputs(self):
        """LaraMini records attentions for the sliding-window and global attention layers, not for the GDN layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)
        num_attention_layers = sum(layer_type != "linear_attention" for layer_type in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            config = model.config
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(len(outputs.attentions), num_attention_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(outputs.attentions), num_attention_layers)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len]
            )

    def test_linear_attention_multi_token_cached_forward_matches_single_token(self):
        """
        The GDN layers must give the same output for a token whether it is fed alone to a populated cache or as the
        first token of a multi-token chunk (chunked-prefill continuation, speculative verification).
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = LaraMiniModel._from_config(config, attn_implementation="eager").to(torch_device).eval()

        prefill_len = 8
        prompt = ids_tensor((1, prefill_len), config.vocab_size).to(torch_device)
        next_token = ids_tensor((1, 1), config.vocab_size).to(torch_device)

        # Reference: prefill, then forward the next token alone with the populated cache.
        cache_single = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, cache=cache_single, use_cache=True)
            single_out = model(input_ids=next_token, cache=cache_single, use_cache=True)

        # Under test: prefill, then forward [next_token, *distractors] in one call.
        distractors = ids_tensor((1, 7), config.vocab_size).to(torch_device)
        cache_multi = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, cache=cache_multi, use_cache=True)
            multi_out = model(input_ids=torch.cat([next_token, distractors], dim=1), cache=cache_multi, use_cache=True)

        # The single-token step runs the recurrent GDN kernel and the multi-token step the chunked one: with the fla
        # kernels they differ by up to ~3e-4 in float32, while a lost GDN cache state typically changes this by ~4e-2.
        torch.testing.assert_close(
            multi_out.last_hidden_state[:, 0], single_out.last_hidden_state[:, 0], rtol=1e-3, atol=1e-3
        )

    @require_causal_conv1d
    @require_flash_linear_attention
    @require_torch_gpu
    def test_padding_free_matches_padded_fast_path(self):
        """Packed sequences through the GDN kernels match the padded batch."""
        torch.manual_seed(0)
        config = self.model_tester.get_config()
        model = LaraMiniForCausalLM(config).to(torch_device).eval()

        data_collator = DataCollatorWithFlattening(
            return_tensors="pt", return_seq_idx=True, return_flash_attn_kwargs=True
        )
        test_cases = [
            (
                torch.tensor([[0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 4, 5]], device=torch_device),
                torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]], dtype=torch.long, device=torch_device),
                [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}],
            ),
            # A one-token sequence in the pack
            (
                torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 6]], device=torch_device),
                torch.tensor([[0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]], dtype=torch.long, device=torch_device),
                [{"input_ids": [1, 2, 3, 4, 5]}, {"input_ids": [6]}],
            ),
        ]
        for padded_input_ids, attention_mask, features in test_cases:
            position_ids = ((attention_mask == 1).long().cumsum(dim=1) - 1) * (attention_mask == 1).long()
            padding_free_batch = {
                key: value.to(torch_device) if torch.is_tensor(value) else value
                for key, value in data_collator(features).items()
            }
            with torch.no_grad():
                res_padded = model(
                    input_ids=padded_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                res_padfree = model(**padding_free_batch, use_cache=False)

            torch.testing.assert_close(
                res_padded.logits[attention_mask.bool()], res_padfree.logits[0], atol=1e-5, rtol=1e-5
            )

    def test_full_attention_reaches_beyond_the_window_for_any_layer_order(self):
        """A sliding layer ahead of the first global layer must not shrink the global attention mask."""
        for layer_types in (
            ["sliding_attention", "full_attention", "sliding_attention"],
            ["sliding_attention", "sliding_attention", "full_attention"],
        ):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            # a window smaller than the sequence, so that it is actually applied
            config.layer_types, config.sliding_window = layer_types, 3
            # eager materializes the masks; sdpa skips an unpadded causal mask and would hide a wrongly sized one
            model = LaraMiniForCausalLM._from_config(config, attn_implementation="eager").to(torch_device).eval()
            input_ids = inputs_dict["input_ids"][:1]
            altered = input_ids.clone()
            altered[0, 0] = (input_ids[0, 0] + 1) % config.vocab_size
            with torch.no_grad():
                last = model(input_ids=input_ids, use_cache=False).logits[:, -1]
                altered_last = model(input_ids=altered, use_cache=False).logits[:, -1]
                cached = model(
                    input_ids=input_ids[:, -1:],
                    cache=model(input_ids=input_ids[:, :-1], use_cache=True).cache,
                ).logits[:, -1]
            # the first token is out of reach of the two sliding windows: only the global layer can see it
            self.assertFalse(torch.allclose(last, altered_last, rtol=1e-4, atol=1e-4), layer_types)
            torch.testing.assert_close(cached, last, rtol=1e-4, atol=1e-4)

    def test_full_attention_uses_rope_only_when_configured(self):
        """Global layers ignore positions unless `full_attention_uses_rope` is set."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        input_ids = inputs_dict["input_ids"][:1]
        position_ids = torch.arange(input_ids.shape[1], device=torch_device)[None]
        mask = torch.ones_like(input_ids)

        for full_attention_uses_rope in (False, True):
            config.full_attention_uses_rope = full_attention_uses_rope
            model = LaraMiniModel(config).to(torch_device).eval()
            with torch.no_grad():
                # An explicit mask, so that the spread positions aren't read as packed sequences
                out = model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids).last_hidden_state
                spread_out = model(
                    input_ids=input_ids, attention_mask=mask, position_ids=3 * position_ids
                ).last_hidden_state
            self.assertEqual(
                torch.allclose(out, spread_out, rtol=1e-5, atol=1e-5),
                not full_attention_uses_rope,
                f"full_attention_uses_rope={full_attention_uses_rope}",
            )

    def test_static_cache_generation_matches_dynamic_for_any_layer_order(self):
        """Static-cache masks are sized by layer type, also for left-padded rows."""
        for layer_types in (
            ["linear_attention", "sliding_attention", "full_attention"],
            ["sliding_attention", "linear_attention", "full_attention"],
        ):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.layer_types, config.sliding_window = layer_types, 3
            model = LaraMiniForCausalLM(config).to(torch_device).eval()
            input_ids = inputs_dict["input_ids"][:2].clone()
            input_ids[1, :3] = config.pad_token_id
            attention_mask = torch.ones_like(input_ids)
            attention_mask[1, :3] = 0
            with torch.no_grad():
                dynamic = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=6, do_sample=False)
                static = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=6,
                    do_sample=False,
                    cache_implementation="static",
                    disable_compile=True,
                )
            self.assertTrue(torch.equal(dynamic, static), layer_types)


@require_torch_accelerator
@slow
class LaraMiniIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "translated/Lara-4-Mini-Base"

    def test_model_logits(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer("La capitale d'Italia è", return_tensors="pt").to(torch_device)
        model = LaraMiniForCausalLM.from_pretrained(
            self.model_id, device_map=torch_device, dtype=torch.bfloat16, experts_implementation="eager"
        )
        with torch.no_grad():
            logits = model(**inputs).logits.float()

        # Recorded on an H200 with the PyTorch gated delta net path (no flash-linear-attention or causal-conv1d)
        expectations = Expectations(
            {(None, None): [-11.6003, -19.6072, -14.6934, -16.1877, -24.367, -15.3516, -18.2544]}
        )
        expected_mean = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0].mean(-1), expected_mean, rtol=1e-2, atol=1e-2)
        expectations = Expectations(
            {
                (None, None): [
                    -6.9688, -8.4375, -11.3125, -12.4375, -14.125, -11.625, -10.0, -13.0,
                    -9.3125, -10.0625, -13.375, -0.8555, -7.8125, -5.375, -11.4375, -10.375,
                ]
            }
        )  # fmt: skip
        expected_slice = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0, -1, :16], expected_slice, rtol=1e-2, atol=1e-2)

    def test_model_greedy_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        prompt = "The capital of Italy is Rome. The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)
        model = LaraMiniForCausalLM.from_pretrained(
            self.model_id, device_map=torch_device, dtype=torch.bfloat16, experts_implementation="eager"
        )
        generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(generated_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        self.assertEqual(text, " Paris. The capital of Germany is Berlin. The capital of Spain is Madrid.")

    def test_model_retrieves_beyond_the_sliding_window(self):
        """The pass key is ~740 tokens before the question, out of reach of the 512-token sliding window, and
        generation runs with a cache that crosses the window: it takes the global attention layers to recall the key."""
        filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * 30
        prompt = (
            "There is a pass key hidden in this text. Remember it. The pass key is 71432. "
            f"{filler}What is the pass key? The pass key is"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(prompt, return_tensors="pt").to(torch_device)
        model = LaraMiniForCausalLM.from_pretrained(
            self.model_id, device_map=torch_device, dtype=torch.bfloat16, experts_implementation="eager"
        )
        self.assertGreater(inputs.input_ids.shape[1], model.config.sliding_window + 200)

        generated_ids = model.generate(**inputs, max_new_tokens=6, do_sample=False)
        text = tokenizer.decode(generated_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        self.assertEqual(text, " 71432")
