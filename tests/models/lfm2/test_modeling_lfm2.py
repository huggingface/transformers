# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LLaMA model."""

import unittest
from unittest.mock import patch

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import Lfm2ForCausalLM, Lfm2Model
    from transformers.models.lfm2 import modeling_lfm2


class Lfm2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Lfm2Model

    def __init__(
        self,
        parent,
        layer_types=["full_attention", "conv"],
    ):
        super().__init__(parent)
        self.layer_types = layer_types


@require_torch
class Lfm2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Lfm2ModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Lfm2ForCausalLM if is_torch_available() else None

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, config.hidden_size, config.conv_L_cache)

    def test_packed_documents_match_separate_forwards(self):
        """`seq_idx` (as `DataCollatorWithFlattening(return_seq_idx=True)` produces it) keeps the short-conv window
        inside each packed document, so a packed forward must match the per-document forwards."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.layer_types = ["conv"] * config.num_hidden_layers
        model = Lfm2Model(config).to(torch_device).eval()

        doc_1 = torch.arange(1, 9, device=torch_device).unsqueeze(0)
        doc_2 = torch.arange(9, 14, device=torch_device).unsqueeze(0)
        packed_input_ids = torch.cat([doc_1, doc_2], dim=1)
        seq_idx = torch.tensor([[0] * doc_1.shape[1] + [1] * doc_2.shape[1]], dtype=torch.int32, device=torch_device)

        with torch.no_grad():
            packed_out = model(input_ids=packed_input_ids, seq_idx=seq_idx, use_cache=False).last_hidden_state
            unpacked_out = model(input_ids=packed_input_ids, use_cache=False).last_hidden_state
            doc_1_out = model(input_ids=doc_1, use_cache=False).last_hidden_state
            doc_2_out = model(input_ids=doc_2, use_cache=False).last_hidden_state

        torch.testing.assert_close(packed_out[:, : doc_1.shape[1]], doc_1_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(packed_out[:, doc_1.shape[1] :], doc_2_out, rtol=1e-4, atol=1e-4)
        # without the boundaries the second document reads into the first
        self.assertFalse(torch.allclose(unpacked_out[:, doc_1.shape[1] :], doc_2_out, rtol=1e-4, atol=1e-4))

    def _config_with_two_conv_layers(self, **overrides):
        """An attention layer (the cache needs one to count tokens) followed by two conv layers, so a leak through
        the first conv window also reaches the second."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.layer_types = ["full_attention", "conv", "conv"]
        config.num_hidden_layers = len(config.layer_types)
        for name, value in overrides.items():
            setattr(config, name, value)
        return config

    def _cached_forward(self, model, input_ids, num_decode_steps=2):
        """Prefill everything but the last `num_decode_steps` tokens, then decode them one at a time."""
        prefill_len = input_ids.shape[1] - num_decode_steps
        outputs = model(input_ids=input_ids[:, :prefill_len], use_cache=True)
        hidden_states = [outputs.last_hidden_state]
        for step in range(prefill_len, input_ids.shape[1]):
            outputs = model(
                input_ids=input_ids[:, step : step + 1], past_key_values=outputs.past_key_values, use_cache=True
            )
            hidden_states.append(outputs.last_hidden_state)
        return torch.cat(hidden_states, dim=1)

    def test_wide_conv_never_reaches_the_kernels(self):
        """The `causal_conv1d` kernels only serve widths 2-4, so a width-7 conv (LFM3) must stay on the torch
        reference for both the prefill and the single-token update, while width 3 keeps using the kernels."""

        def forbidden(*args, **kwargs):
            raise AssertionError("the causal_conv1d kernel was called for a width it does not serve")

        config = self._config_with_two_conv_layers(conv_L_cache=7)
        model = Lfm2Model(config).to(torch_device).eval()
        input_ids = ids_tensor((2, 10), config.vocab_size).to(torch_device)
        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).last_hidden_state
            with (
                patch.object(modeling_lfm2, "causal_conv1d_fn", forbidden),
                patch.object(modeling_lfm2, "causal_conv1d_update", forbidden),
            ):
                cached = self._cached_forward(model, input_ids)
        torch.testing.assert_close(cached, expected, rtol=1e-4, atol=1e-4)

        config = self._config_with_two_conv_layers(conv_L_cache=3)
        model = Lfm2Model(config).to(torch_device).eval()
        with (
            torch.no_grad(),
            patch.object(modeling_lfm2, "causal_conv1d_fn", wraps=modeling_lfm2.causal_conv1d_fn) as conv_fn,
            patch.object(modeling_lfm2, "causal_conv1d_update", wraps=modeling_lfm2.causal_conv1d_update) as update,
        ):
            self._cached_forward(model, input_ids)
        self.assertTrue(conv_fn.called)
        self.assertTrue(update.called)

    def test_left_padding_with_conv_bias(self):
        """With `conv_bias`, the `in_proj` bias re-fills the masked padding positions, so they must be masked again
        before they enter the conv window: a left-padded prompt has to match its unpadded forward."""
        config = self._config_with_two_conv_layers(conv_bias=True)
        model = Lfm2ForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            # `_init_weights` zeroes the biases, which hides the leak that a trained checkpoint shows
            for layer in model.model.layers:
                if not layer.is_attention_layer:
                    layer.conv.in_proj.bias.uniform_(-1, 1)
        prompts = [ids_tensor((1, 8), config.vocab_size), ids_tensor((1, 5), config.vocab_size)]
        pad_lens = [8 - prompt.shape[1] for prompt in prompts]
        padded_input_ids = torch.cat(
            [torch.nn.functional.pad(prompt, (pad, 0), value=0) for prompt, pad in zip(prompts, pad_lens)]
        ).to(torch_device)
        attention_mask = torch.cat(
            [
                torch.nn.functional.pad(torch.ones_like(prompt), (pad, 0), value=0)
                for prompt, pad in zip(prompts, pad_lens)
            ]
        ).to(torch_device)

        with torch.no_grad():
            padded = model(input_ids=padded_input_ids, attention_mask=attention_mask, use_cache=True)
            for row, (prompt, pad) in enumerate(zip(prompts, pad_lens)):
                unpadded = model(input_ids=prompt.to(torch_device), use_cache=True)
                torch.testing.assert_close(padded.logits[row : row + 1, pad:], unpadded.logits, rtol=1e-5, atol=1e-5)
                for layer_idx, layer_type in enumerate(config.layer_types):
                    if layer_type != "conv":
                        continue
                    torch.testing.assert_close(
                        padded.past_key_values.get_conv_state(layer_idx)[row : row + 1],
                        unpadded.past_key_values.get_conv_state(layer_idx),
                        rtol=1e-5,
                        atol=1e-5,
                    )

    def test_attention_outputs(self):
        """Lfm2Moe alternates between attention and short-conv layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            config = model.config
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])


@require_torch_accelerator
@slow
class Lfm2IntegrationTest(unittest.TestCase):
    pass
