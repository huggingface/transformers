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
"""Testing suite for the PyTorch LFM3 model."""

import ast
import inspect
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, DynamicCache, Lfm3Config, Lfm3ForCausalLM, Lfm3Model


class Lfm3ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Lfm3Model

    def __init__(self, parent):
        super().__init__(parent)
        self.layer_types = ["full_attention", "sliding_attention", "conv"]
        self.num_hidden_layers = len(self.layer_types)
        self.conv_L_cache = 7
        # large enough that the common generation tests stay within the window, like the other sliding-window
        # models; `test_generation_beyond_sliding_window` covers the windowed path
        self.sliding_window = 4096
        self.rope_parameters = {
            "full_attention": {"rope_type": "proportional", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        }


@require_torch
class Lfm3ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Lfm3ModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Lfm3ForCausalLM if is_torch_available() else None

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, config.hidden_size, config.conv_L_cache)

    def test_attention_outputs(self):
        """LFM3 records attentions for the full and sliding attention layers, not for the conv layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)
        num_attention_layers = sum(layer_type != "conv" for layer_type in config.layer_types)

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

    def test_default_layer_types_follow_the_lfm3_pattern(self):
        config = Lfm3Config(num_hidden_layers=8)
        self.assertEqual(
            config.layer_types,
            ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"],
        )
        self.assertEqual(config.head_dim, config.hidden_size // config.num_attention_heads)
        self.assertEqual(set(config.rope_parameters), {"full_attention", "sliding_attention"})

    def test_config_rejects_inconsistent_layer_types(self):
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "sliding_window"):
            Lfm3Config(num_hidden_layers=2, layer_types=["full_attention", "sliding_attention"], sliding_window=None)
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "rope_parameters"):
            Lfm3Config(
                num_hidden_layers=2,
                layer_types=["full_attention", "sliding_attention"],
                sliding_window=4,
                rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
            )
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "num_hidden_layers"):
            Lfm3Config(num_hidden_layers=3, layer_types=["conv", "full_attention"])
        with self.assertRaisesRegex(StrictDataclassClassValidationError, "layer_types"):
            Lfm3Config(num_hidden_layers=1, layer_types=["linear_attention"])

    def test_sandwich_norm_can_be_disabled(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        self.assertTrue(hasattr(Lfm3Model(config).layers[0], "post_operator_norm"))
        config.use_sandwich_norm = False
        model = Lfm3Model(config)
        self.assertFalse(hasattr(model.layers[0], "post_operator_norm"))
        self.assertFalse(hasattr(model.layers[0], "post_ffn_norm"))

    def test_rope_buffers_survive_meta_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        reference = Lfm3Model(config)
        with torch.device("meta"):
            model = Lfm3Model(config)
        model.to_empty(device="cpu")
        model.init_weights()
        torch.testing.assert_close(
            model.rotary_emb.full_attention_inv_freq, reference.rotary_emb.full_attention_inv_freq
        )
        torch.testing.assert_close(
            model.rotary_emb.sliding_attention_inv_freq, reference.rotary_emb.sliding_attention_inv_freq
        )

    def test_left_padding_does_not_change_outputs(self):
        """The short-conv layers zero the padded positions, so a left-padded row must match the unpadded forward."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Lfm3ForCausalLM(config).to(torch_device).eval()
        input_ids = ids_tensor((1, 9), config.vocab_size).to(torch_device)
        padded_input_ids = torch.cat([torch.full((1, 4), config.pad_token_id, device=torch_device), input_ids], dim=1)
        attention_mask = torch.cat([torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 9, dtype=torch.long)], dim=1)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0).to(torch_device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
            padded_logits = model(
                input_ids=padded_input_ids,
                attention_mask=attention_mask.to(torch_device),
                position_ids=position_ids,
                use_cache=False,
            ).logits

        torch.testing.assert_close(padded_logits[:, 4:], logits, rtol=1e-4, atol=1e-4)

    def test_cached_decoding_matches_full_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Lfm3ForCausalLM(config).to(torch_device).eval()
        input_ids = ids_tensor((2, 12), config.vocab_size).to(torch_device)

        with torch.no_grad():
            full_logits = model(input_ids=input_ids, use_cache=False).logits
            prefill = model(input_ids=input_ids[:, :8], use_cache=True)
            decoded = []
            for step in range(8, 12):
                outputs = model(
                    input_ids=input_ids[:, step : step + 1], past_key_values=prefill.past_key_values, use_cache=True
                )
                decoded.append(outputs.logits)

        torch.testing.assert_close(torch.cat(decoded, dim=1), full_logits[:, 8:], rtol=1e-4, atol=1e-4)

    def test_chunked_prefill_matches_full_forward_around_the_window(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.sliding_window = 4
        model = Lfm3ForCausalLM(config).to(torch_device).eval()
        for chunks in ([3, 1, 4, 3], [1] * 11, [5, 5, 1], [9, 2]):
            input_ids = ids_tensor((2, sum(chunks)), config.vocab_size).to(torch_device)
            with torch.no_grad():
                expected = model(input_ids=input_ids, use_cache=False).logits
                cache, logits, start = DynamicCache(config=config), [], 0
                for size in chunks:
                    outputs = model(
                        input_ids=input_ids[:, start : start + size], past_key_values=cache, use_cache=True
                    )
                    cache, start = outputs.past_key_values, start + size
                    logits.append(outputs.logits)
            torch.testing.assert_close(torch.cat(logits, dim=1), expected, rtol=1e-4, atol=1e-4)

    def test_full_attention_reaches_beyond_the_window_for_any_layer_order(self):
        """The cache sizes a mask from the layer it is asked about; a sliding layer ahead of the first full-attention
        layer must not shrink the full-attention mask to the window."""
        for layer_types in (
            ["conv", "sliding_attention", "full_attention"],
            ["sliding_attention", "conv", "full_attention"],
        ):
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            config.layer_types, config.sliding_window, config.conv_L_cache = layer_types, 4, 3
            # eager materializes the masks; sdpa skips an unpadded causal mask and would hide a wrongly sized one
            model = Lfm3ForCausalLM._from_config(config, attn_implementation="eager").to(torch_device).eval()
            input_ids = ids_tensor((1, 17), config.vocab_size).to(torch_device)
            altered = input_ids.clone()
            altered[0, 0] = (input_ids[0, 0] + 1) % config.vocab_size
            with torch.no_grad():
                last = model(input_ids=input_ids, use_cache=False).logits[:, -1]
                altered_last = model(input_ids=altered, use_cache=False).logits[:, -1]
                cached = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=model(input_ids=input_ids[:, :-1], use_cache=True).past_key_values,
                ).logits[:, -1]
            # the first token is far outside the sliding window and the conv taps: only full attention can see it
            self.assertFalse(torch.allclose(last, altered_last, rtol=1e-4, atol=1e-4), layer_types)
            torch.testing.assert_close(cached, last, rtol=1e-4, atol=1e-4)

    def test_generation_beyond_sliding_window(self):
        """Generation with an input longer than `sliding_window`, so the hybrid cache and the mask slicing of the
        sliding-attention layers are covered."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.sliding_window = 4
        model = Lfm3ForCausalLM(config).to(torch_device).eval()

        input_ids = ids_tensor((2, 10), config.vocab_size).to(torch_device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            cached = model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=5, do_sample=False, use_cache=True
            )
            uncached = model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=5, do_sample=False, use_cache=False
            )
        self.assertEqual(cached.shape, (2, 15))
        self.assertTrue(torch.equal(cached, uncached))

    def test_static_cache_generation_matches_dynamic_for_any_layer_order(self):
        """`generate` builds the static-cache masks itself: they must be sized by layer type like the forward's, and
        the left-padded rows of the batch must not be taken for packed documents."""
        for layer_types in (
            ["conv", "sliding_attention", "full_attention"],
            ["sliding_attention", "conv", "full_attention"],
        ):
            config, _ = self.model_tester.prepare_config_and_inputs_for_common()
            config.layer_types, config.sliding_window, config.conv_L_cache = layer_types, 4, 3
            model = Lfm3ForCausalLM(config).to(torch_device).eval()
            input_ids = ids_tensor((2, 9), config.vocab_size).to(torch_device)
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

    def test_chunked_prefill_accepts_a_left_padded_batch(self):
        """A row whose padding reaches into a later chunk opens that chunk at position 0 on a filled cache, which is
        padding, not packing, under the dynamic cache's 2D mask and under the mask dict of a static cache alike."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Lfm3ForCausalLM(config).to(torch_device).eval()
        for pad_length, cache_implementation in ((4, None), (5, None), (4, "static"), (5, "static")):
            input_ids = ids_tensor((2, 10), config.vocab_size).to(torch_device)
            input_ids[1, :pad_length] = config.pad_token_id
            attention_mask = torch.ones_like(input_ids)
            attention_mask[1, :pad_length] = 0
            kwargs = {"max_new_tokens": 3, "do_sample": False, "disable_compile": True}
            with torch.no_grad():
                expected = model.generate(
                    input_ids, attention_mask=attention_mask, cache_implementation=cache_implementation, **kwargs
                )
                chunked = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    cache_implementation=cache_implementation,
                    prefill_chunk_size=4,
                    **kwargs,
                )
            self.assertTrue(torch.equal(chunked, expected), (pad_length, cache_implementation))

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_from_config_0_linear(self):
        pass

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_from_config_1_dynamic(self):
        pass

    @unittest.skip("LFM3 keys `rope_parameters` by attention layer type, unsupported by this test")
    def test_model_rope_scaling_from_config_2_yarn(self):
        pass

    def test_no_hub_kernel_replaces_the_fp32_rotary(self):
        from transformers.models.lfm3 import modeling_lfm3

        self.assertFalse(hasattr(modeling_lfm3.apply_rotary_pos_emb, "kernel_layer_name"))
        # `use_kernelized_func` refuses unregistered functions at import time once `kernels` is installed.
        tree = ast.parse(inspect.getsource(modeling_lfm3))
        attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Lfm3Attention")
        self.assertEqual(attention.decorator_list, [])


@require_torch_accelerator
@slow
class Lfm3IntegrationTest(unittest.TestCase):
    model_id = "LiquidAI/LFM3-1.7B-Preview"
    revision = "92fc473ab9c9042e26c62f1417c9d609b1ddbc1b"

    def test_model_logits(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model = Lfm3ForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, device_map=torch_device, dtype=torch.float32
        )
        input_ids = tokenizer("Simply put, the theory of relativity states that ", return_tensors="pt").to(
            torch_device
        )
        with torch.no_grad():
            logits = model(**input_ids).logits.float()

        expectations = Expectations({(None, None): LFM3_EXPECTED_LOGITS_MEAN})
        expected_mean = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0].mean(-1), expected_mean, rtol=1e-3, atol=1e-3)
        expectations = Expectations({(None, None): LFM3_EXPECTED_LOGITS_SLICE})
        expected_slice = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0, 0, :16], expected_slice, rtol=1e-3, atol=1e-3)

    def test_model_greedy_generation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model = Lfm3ForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, device_map=torch_device, dtype=torch.bfloat16
        )
        messages = [{"role": "user", "content": "Simply put, what does the theory of relativity state?"}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(torch_device)
        generated_ids = model.generate(**input_ids, max_new_tokens=32, do_sample=False)
        text = tokenizer.decode(generated_ids[0, input_ids["input_ids"].shape[1] :], skip_special_tokens=True)

        expectations = Expectations({(None, None): LFM3_EXPECTED_GENERATION})
        self.assertEqual(text, expectations.get_expectation())


LFM3_EXPECTED_LOGITS_MEAN = [
    -1.9214,
    -1.6632,
    -1.0323,
    -1.0694,
    0.0741,
    0.2619,
    0.3455,
    0.2205,
    0.2545,
    0.1654,
    -1.9167,
]
LFM3_EXPECTED_LOGITS_SLICE = [
    -0.0203,
    -2.5232,
    -0.4502,
    -1.087,
    1.0517,
    0.6892,
    0.7451,
    2.0436,
    9.0642,
    5.4245,
    7.058,
    3.5806,
    5.2226,
    2.1968,
    0.0952,
    2.6452,
]
LFM3_EXPECTED_GENERATION = (
    'We need to answer the question: "Simply put, what does the theory of relativity state?" This is a '
    "straightforward question about the theory of relativity. The"
)
