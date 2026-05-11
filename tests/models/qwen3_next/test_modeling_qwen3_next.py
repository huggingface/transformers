# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_multi_gpu, slow, torch_device


if is_torch_available():
    import torch

    from transformers import (
        DynamicCache,
        Qwen3NextModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_sdpa_inference,
    ids_tensor,
)


class Qwen3NextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Qwen3NextModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        # NOTE(3outeille): must be 0.0 for TP backward tests. In train mode, non-zero dropout causes
        # different RNG states between the non-TP and TP model forward passes (they run sequentially),
        # leading to different dropout masks and mismatched losses.
        self.attention_probs_dropout_prob = 0.0
        self.layer_types = ["linear_attention", "full_attention"]
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 8


@require_torch
class Qwen3NextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Qwen3NextModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        intermediate_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

        return (batch_size, intermediate_size, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim

        return (batch_size, num_v_heads, head_k_dim, head_v_dim)

    @unittest.skip("Qwen3-Next hybrid linear-attention cache is not compatible with quantized cache yet.")
    def test_generate_with_quant_cache(self):
        pass

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3-Next alternates between attention layers and gated deltanet layers."
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    def test_linear_attention_multi_token_cached_forward_matches_single_token(self):
        """
        Qwen3-Next's gated-delta-net layers must produce the same output for a token regardless of
        whether it's fed as a single-token cached forward or as the first token of a multi-token chunk
        after the cache has been populated (chunked-prefill continuation / speculative verification).
        A causal LM's logits at position `i` cannot depend on tokens at positions > `i`, even across
        separate forward calls with a shared cache.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        # GatedDeltaNet's fused norm-gate kernel only supports silu/swish/sigmoid; the shared tester
        # default `gelu` would raise before exercising the cache path.
        config.hidden_act = "silu"
        model = Qwen3NextModel._from_config(config)
        model.to(torch_device)
        model.eval()

        prefill_len = 8
        prompt = ids_tensor((1, prefill_len), config.vocab_size).to(torch_device)
        next_token = ids_tensor((1, 1), config.vocab_size).to(torch_device)

        # Reference: prefill, then forward the next token alone with the populated cache.
        cache_single = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, past_key_values=cache_single, use_cache=True)
            single_out = model(input_ids=next_token, past_key_values=cache_single, use_cache=True)
        ref_first = single_out.last_hidden_state[:, 0, :]

        # Under test: prefill, then forward [next_token, *distractors] in one call. The first
        # position must match the single-token forward exactly (causal attention).
        distractors = ids_tensor((1, 7), config.vocab_size).to(torch_device)
        multi_input = torch.cat([next_token, distractors], dim=1)
        cache_multi = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, past_key_values=cache_multi, use_cache=True)
            multi_out = model(input_ids=multi_input, past_key_values=cache_multi, use_cache=True)
        under_test_first = multi_out.last_hidden_state[:, 0, :]

        torch.testing.assert_close(under_test_first, ref_first, rtol=1e-4, atol=1e-4)

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        """
        We need to overwrite this without the fp16 part of the dtype, because the slow path `torch_chunk_gated_delta_rule`
        is not robust enough (flaky test) in fp16 due to upscaling in fp32 and then downscaling to fp16 at the end
        """
        if dtype == "fp16":
            self.skipTest("Not robust in fp16")
        _test_eager_matches_sdpa_inference(
            self,
            name,
            dtype,
            padding_side,
            use_attention_mask,
            output_attentions,
            enable_kernels,
        )

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @require_torch_multi_gpu
    def test_can_use_device_map(self):
        """
        Test that this model can be dispatched on multiple gpus. It's not obvious as the Cache is not standard,
        ant each layer need to use the correct device on which it reside (i.e. it needs to be lazy initialized).
        """
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            inputs_dict = {k: v.to(0) if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}
            # We want the linear attention layer to reside on device 1 with the device map (i.e. not the first/default device),
            # to check if cache initialization is on the correct device
            config.layer_types = ["full_attention", "linear_attention"]
            model = model_class(config).eval()

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                del model
                model = model_class.from_pretrained(
                    tmpdirname,
                    device_map={
                        "lm_head": 0,
                        "model.embed_tokens": 0,
                        "model.norm": 0,
                        "model.layers.0": 0,
                        "model.layers.1": 1,
                    },
                )

                # Check that we indeed use 2 different devices for each layer
                self.assertTrue({param.device for param in model.model.layers[0].parameters()} == {torch.device(0)})
                self.assertTrue({param.device for param in model.model.layers[1].parameters()} == {torch.device(1)})

                # This should not crash
                _ = model.generate(**inputs_dict, max_new_tokens=5, min_new_tokens=5)


@slow
class Qwen3NextIntegrationTest(unittest.TestCase):
    pass
