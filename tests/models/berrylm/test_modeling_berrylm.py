# Copyright 2026 The RWB AI Assist team and The HuggingFace Inc. team. All rights reserved.
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

import copy
import itertools
import tempfile
import unittest

from parameterized import parameterized

from transformers import DataCollatorWithFlattening, is_torch_available
from transformers.testing_utils import (
    require_causal_conv1d,
    require_flash_linear_attention,
    require_torch,
    require_torch_gpu,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_flash_linear_attention_available


if is_torch_available():
    import torch

    from transformers import (
        BerryLMForCausalLM,
        BerryLMModel,
        DynamicCache,
    )
    from transformers.generation.utils import ALL_CACHE_NAMES
    from transformers.models.berrylm.modeling_berrylm import (
        torch_chunk_kda,
        torch_recurrent_kda,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_sdpa_inference,
    ids_tensor,
)


class BerryLMModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = BerryLMModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        # Must be 0.0 for the TP backward tests: non-zero dropout gives the sequential non-TP / TP forward passes
        # different RNG states and mismatched losses.
        self.attention_probs_dropout_prob = 0.0
        # The linear-attention layers gate their output with the fused norm-gate kernel (silu / swish / sigmoid only).
        self.hidden_act = "silu"
        self.layer_types = ["linear_attention", "full_attention"]
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 8
        # Block size 1: every layer commits a block, so the AttnRes mixer of layer 1 already mixes two streams.
        self.attn_res_block_size = 1
        self.kda_gate_bottleneck = 8


@require_torch
class BerryLMModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = BerryLMModelTester

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

    @unittest.skip("The BerryLM hybrid linear-attention cache is not compatible with quantized cache yet.")
    def test_generate_with_quant_cache(self):
        pass

    def test_attention_outputs(self):
        "Needs to be overwritten as BerryLM alternates between attention layers and linear-attention layers."
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
        The KDA layers must produce the same output for a token regardless of whether it's fed as a single-token
        cached forward or as the first token of a multi-token chunk after the cache has been populated
        (chunked-prefill continuation / speculative verification). A causal LM's logits at position `i` cannot
        depend on tokens at positions > `i`, even across separate forward calls with a shared cache.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        model = BerryLMModel._from_config(config)
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

        # With flash-linear-attention installed the multi-token chunk runs the chunked kernel (tensor-core
        # accumulation) and the single token the recurrent one: they agree to ~2e-4 on O(0.05) hidden states.
        rtol, atol = (1e-2, 5e-4) if is_flash_linear_attention_available() else (1e-4, 1e-4)
        torch.testing.assert_close(under_test_first, ref_first, rtol=rtol, atol=atol)

    def test_recurrent_layers_mask_padding_on_continued_forward(self):
        """
        Overwritten: the scenario of the shared test (a left-padded continuation on top of the turn-1 cache must match
        the single full forward, i.e. the recurrent layers mask padding out of their state on continued forwards too),
        with the tolerance opened to the precision of the fused chunked KDA kernel when flash-linear-attention is
        installed: its two chunkings of the padded batch agree to ~2e-5 (tensor-core accumulation) while the
        unmasked-padding bug this guards against diverges by ~1e-3. Without the kernels the fp-floor tolerance applies.
        """
        rtol, atol = (1e-3, 5e-5) if is_flash_linear_attention_available() else (1e-4, 1e-5)
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config).to(torch_device).eval()
            input_ids = inputs_dict["input_ids"][:2].to(torch_device)
            # An ordinary token id: padding is defined by the attention mask, not the token value.
            pad_token_id = 7

            seq_len = input_ids.shape[1]
            turn1_len = seq_len // 2 + 1
            pad_len = max(1, seq_len - turn1_len - 1)
            turn1, turn2 = input_ids[:, :turn1_len], input_ids[:, turn1_len:].clone()
            turn2[1, :pad_len] = pad_token_id
            attention_mask = torch.ones_like(input_ids)
            attention_mask[1, turn1_len : turn1_len + pad_len] = 0
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

            with torch.no_grad():
                single = model(
                    input_ids=torch.cat([turn1, turn2], dim=-1),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                ).logits
                out1 = model(
                    input_ids=turn1,
                    attention_mask=attention_mask[:, :turn1_len],
                    position_ids=position_ids[:, :turn1_len],
                    use_cache=True,
                )
                cache_kwarg, cache = next(
                    ((name, getattr(out1, name)) for name in ALL_CACHE_NAMES if getattr(out1, name, None) is not None),
                    ("past_key_values", None),
                )
                out2 = model(
                    input_ids=turn2,
                    attention_mask=attention_mask,
                    position_ids=position_ids[:, turn1_len:],
                    use_cache=True,
                    **{cache_kwarg: cache},
                )
            torch.testing.assert_close(out2.logits[:, -1], single[:, -1], rtol=rtol, atol=atol)

    def test_attn_res_zero_gate_is_identity(self):
        """
        The Gated Block AttnRes mixer is initialized at the exact identity (`gate = 0`): a model with the mixer
        must match the same weights without the mixer, and a non-zero gate must change the output.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        config.attn_res_block_size = 1
        config.attn_res_gated = True
        model = BerryLMModel._from_config(config).to(torch_device).eval()
        self.assertTrue(all(torch.all(layer.attn_res.gate == 0) for layer in model.layers))

        plain_config = copy.deepcopy(config)
        plain_config.attn_res_block_size = 0
        plain = BerryLMModel._from_config(plain_config).to(torch_device).eval()
        missing, unexpected = plain.load_state_dict(model.state_dict(), strict=False)
        self.assertEqual(missing, [])
        self.assertTrue(all("attn_res" in key for key in unexpected))
        self.assertTrue(all(layer.attn_res is None for layer in plain.layers))

        inputs = {k: v.to(torch_device) for k, v in inputs_dict.items() if v is not None}
        with torch.no_grad():
            mixed = model(**inputs).last_hidden_state
            reference = plain(**inputs).last_hidden_state
        torch.testing.assert_close(mixed, reference)

        # Open the gates: the layer input becomes a mixture of the committed block streams, the output changes.
        with torch.no_grad():
            for layer in model.layers:
                layer.attn_res.gate.fill_(1.0)
                layer.attn_res.pseudo_query.normal_()
            opened = model(**inputs).last_hidden_state
        self.assertFalse(torch.allclose(opened, reference, rtol=1e-4, atol=1e-4))

    @require_causal_conv1d
    @require_flash_linear_attention
    @require_torch_gpu
    def test_padding_free_matches_padded_fast_path_regression(self):
        torch.manual_seed(0)
        config = self.model_tester.get_config()
        model = BerryLMForCausalLM(config).to(torch_device).eval()

        data_collator = DataCollatorWithFlattening(
            return_tensors="pt", return_seq_idx=True, return_flash_attn_kwargs=True
        )
        test_cases = [
            (
                torch.tensor([[0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 4, 5]], device=torch_device),
                torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]], dtype=torch.long, device=torch_device),
                [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}],
            ),
            (
                torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 6]], device=torch_device),
                torch.tensor([[0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]], dtype=torch.long, device=torch_device),
                [{"input_ids": [1, 2, 3, 4, 5]}, {"input_ids": [6]}],
            ),
        ]

        for padded_input_ids, attention_mask, features in test_cases:
            position_ids = ((attention_mask == 1).long().cumsum(dim=1) - 1) * (attention_mask == 1).long()
            padding_free_batch = data_collator(features)
            padding_free_batch = {
                key: value.to(torch_device) if torch.is_tensor(value) else value
                for key, value in padding_free_batch.items()
            }

            with torch.no_grad():
                res_padded = model(
                    input_ids=padded_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                res_padfree = model(**padding_free_batch, use_cache=False)

            logits_padded = res_padded.logits[attention_mask.bool()]
            logits_padfree = res_padfree.logits[0]

            torch.testing.assert_close(logits_padded, logits_padfree, atol=1e-5, rtol=1e-5)

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
        We need to overwrite this without the fp16 part of the dtype, because the slow path `torch_recurrent_kda`
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

    @require_torch_multi_accelerator
    def test_can_use_device_map(self):
        """
        Test that this model can be dispatched on multiple accelerators. It's not obvious as the Cache is not standard,
        and each layer need to use the correct device on which it reside (i.e. it needs to be lazy initialized).
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

    # seq_length 3 fits in one padded chunk, 12 spans exactly 3 chunks, 13 spans 4 chunks with padding
    @parameterized.expand(itertools.product([3, 12, 13], [False, True]))
    def test_kda_chunked_matches_recurrent(self, seq_length: int, with_initial_state: bool):
        """Ensures that the chunked KDA implementation matches the token-by-token recurrence (with the fused kernels
        installed both sides run the flash-linear-attention kernels; without them both run the torch reference, and the
        torch chunked path is the recurrence itself)."""
        torch.manual_seed(0)
        batch_size, num_heads, k_head_dim, v_head_dim = 2, 3, 8, 16
        query = torch.randn(batch_size, seq_length, num_heads, k_head_dim, device=torch_device)
        key = torch.randn(batch_size, seq_length, num_heads, k_head_dim, device=torch_device)
        value = torch.randn(batch_size, seq_length, num_heads, v_head_dim, device=torch_device)
        # per-channel log-decays, must be <= 0
        g = -torch.rand(batch_size, seq_length, num_heads, k_head_dim, device=torch_device)
        beta = torch.rand(batch_size, seq_length, num_heads, device=torch_device)
        initial_state = None
        if with_initial_state:
            initial_state = torch.randn(batch_size, num_heads, k_head_dim, v_head_dim, device=torch_device)

        chunk_out, chunk_state = torch_chunk_kda(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent_out, recurrent_state = torch_recurrent_kda(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # With flash-linear-attention installed both sides run the fused kernels; the chunked one accumulates the
        # intra-chunk products on bf16 tensor cores, so the agreement is bf16-level (~5e-4 absolute on O(1) outputs).
        torch.testing.assert_close(chunk_out, recurrent_out, rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(chunk_state, recurrent_state, rtol=2e-2, atol=2e-3)


@slow
class BerryLMIntegrationTest(unittest.TestCase):
    pass
