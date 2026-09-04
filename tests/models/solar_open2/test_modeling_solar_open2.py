# Copyright 2026 Upstage AI and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch SolarOpen2 model."""

import tempfile
import unittest
from unittest.mock import patch

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    require_torch_bf16,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_flash_linear_attention_available

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, SolarOpen2Config, SolarOpen2ForCausalLM, SolarOpen2Model


class SolarOpen2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = SolarOpen2Model

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=0,
        use_rope=False,
        use_qk_norm=False,
        use_gqa_gate=True,
        kda_allow_neg_eigval=True,
        hidden_act="silu",
    ):
        super().__init__(parent=parent, num_experts_per_tok=num_experts_per_tok)
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.first_k_dense_replace = first_k_dense_replace
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_gqa_gate = use_gqa_gate
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.hidden_act = hidden_act
        # Interleave a full-attention (GQA) layer (0) and a Kimi-Delta linear-attention layer (1).
        self.gqa_layers = [0]
        self.linear_attn_config = {
            "short_conv_kernel_size": 4,
            "head_dim": self.head_dim,
            "num_heads": self.num_attention_heads,
            "num_kv_heads": None,
        }


@require_torch
class SolarOpen2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = SolarOpen2ModelTester
    model_split_percents = [0.5, 0.85, 0.9]  # it tries to offload everything with the default value

    @unittest.skip("SolarOpen2 hybrid linear-attention cache is not compatible with quantized cache yet.")
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip("The recurrent linear-attention cache cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        self.skipTest(
            "SolarOpen2 defaults to NoPE (use_rope=False); RoPE scaling has no effect on the shipped config."
        )

    def _get_conv_state_shape(self, batch_size: int, config):
        linear_attn_config = config.linear_attn_config
        num_heads = linear_attn_config["num_heads"]
        head_dim = linear_attn_config["head_dim"]
        num_kv_heads = linear_attn_config.get("num_kv_heads") or num_heads
        conv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        return (batch_size, conv_dim, linear_attn_config["short_conv_kernel_size"])

    def _get_recurrent_state_shape(self, batch_size: int, config):
        linear_attn_config = config.linear_attn_config
        return (
            batch_size,
            linear_attn_config["num_heads"],
            linear_attn_config["head_dim"],
            linear_attn_config["head_dim"],
        )

    def test_attention_outputs(self):
        """Overwritten: SolarOpen2 alternates full-attention (GQA) and Kimi-Delta linear-attention layers,
        so only the full-attention layers emit attention weights."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)
        n_full = sum(layer == "full_attention" for layer in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), n_full)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

            # check that output_attentions also works using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), n_full)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # check attention outputs coexist with hidden states and stay last in the output tuple
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(out_len + 1, len(outputs))
            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), n_full)
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    def test_rope_parameters_partially_initialized(self):
        """SolarOpen2Config overrides the parent's default partial_rotary_factor to 1.0."""
        config = SolarOpen2Config(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 65536,
            }
        )
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 1.0)

    def test_use_rope_forward(self):
        """`use_rope=True` applies rotary embeddings on the full-attention layers (non-default path).

        Both models are built with the same seed so their weights are identical; the logits must stay
        finite on both paths and must differ, proving the flag actually changes the computation.
        """
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.assertFalse(config.use_rope)
        torch.manual_seed(0)
        nope_model = SolarOpen2ForCausalLM(config).to(torch_device).eval()
        config.use_rope = True
        torch.manual_seed(0)
        rope_model = SolarOpen2ForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            nope_logits = nope_model(inputs["input_ids"]).logits
            rope_logits = rope_model(inputs["input_ids"]).logits
        self.assertTrue(torch.isfinite(nope_logits).all())
        self.assertTrue(torch.isfinite(rope_logits).all())
        # the effect is small on a tiny random model (~6e-4 max) but well above fp32 noise
        self.assertFalse(torch.allclose(nope_logits, rope_logits, rtol=0.0, atol=1e-5))

    def test_kda_cached_decode_matches_full_forward(self):
        """The hybrid (KDA conv+recurrent state and GQA KV) cache must match a full forward, both for
        single-token decode steps and for a multi-token chunk fed on top of a populated cache
        (chunked-prefill continuation / speculative verification)."""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs["input_ids"][:1]
        length = input_ids.shape[1]
        model = SolarOpen2ForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            full = model(input_ids, use_cache=False).logits
            out = model(input_ids[:, :3], use_cache=True)
            cache = out.past_key_values
            collected = [out.logits[:, -1:]]
            # a single-token decode step, then a multi-token chunk covering the rest
            boundaries = [3, 4, length]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                out = model(input_ids[:, start:end], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                collected.append(out.logits)
        incremental = torch.cat(collected, dim=1)
        torch.testing.assert_close(incremental, full[:, 2:], rtol=1e-4, atol=1e-4)

    def test_layer_types_derivation(self):
        """gqa_layers (or gqa_interval) determines the per-layer attention pattern."""
        expected = [
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ]
        config = SolarOpen2Config(num_hidden_layers=6, gqa_layers=[0, 4])
        self.assertEqual(config.layer_types, expected)

        # gqa_interval=N alone: one full-attention layer followed by N linear-attention layers, starting at layer 0.
        config = SolarOpen2Config(num_hidden_layers=6, gqa_interval=3)
        self.assertEqual(config.layer_types, expected)
        config = SolarOpen2Config(num_hidden_layers=6, gqa_interval=5)
        self.assertEqual(config.layer_types.count("full_attention"), 1)

        # When both are given, gqa_layers wins over gqa_interval.
        config = SolarOpen2Config(num_hidden_layers=6, gqa_layers=[1], gqa_interval=3)
        self.assertEqual(config.layer_types.count("full_attention"), 1)
        self.assertEqual(config.layer_types[1], "full_attention")

    def test_public_250b_config_defaults(self):
        config = SolarOpen2Config()
        self.assertEqual(config.max_position_embeddings, 1_048_576)
        self.assertEqual(config.n_routed_experts, 320)
        self.assertTrue(config.kda_allow_neg_eigval)
        self.assertFalse(any(key.startswith("hc_") for key in config.to_dict()))
        # The default (interval-derived) layer pattern reproduces the public upstage/Solar-Open2-250B checkpoint.
        self.assertIsNone(config.gqa_layers)
        self.assertEqual(config.gqa_interval, 3)
        self.assertEqual(
            config.layer_types,
            ["full_attention" if i % 4 == 0 else "linear_attention" for i in range(48)],
        )
        self.assertEqual((config.bos_token_id, config.eos_token_id, config.pad_token_id), (1, 2, 2))
        self.assertEqual(config.rope_parameters["rope_theta"], 10000.0)

    def test_hyper_connections_are_rejected(self):
        config = SolarOpen2Config(hc_rate=1)
        with self.assertRaisesRegex(NotImplementedError, "Hyper-Connections"):
            SolarOpen2Model(config)

    def test_config_validation(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            SolarOpen2Config(num_hidden_layers=6, gqa_interval=0)
        with self.assertRaisesRegex(ValueError, "valid layer indices"):
            SolarOpen2Config(num_hidden_layers=6, gqa_layers=[0, 99])
        with self.assertRaisesRegex(ValueError, "at least one full-attention layer"):
            SolarOpen2Config(num_hidden_layers=6, gqa_layers=[])
        with self.assertRaisesRegex(ValueError, "divisible"):
            SolarOpen2Config(
                num_hidden_layers=6,
                linear_attn_config={"short_conv_kernel_size": 4, "head_dim": 64, "num_heads": 8, "num_kv_heads": 3},
            )

    @require_torch_multi_accelerator
    def test_can_use_device_map(self):
        """
        Test that this model can be dispatched on multiple accelerators. It's not obvious as the Cache is not
        standard, and each layer needs to use the correct device on which it resides (i.e. it needs to be lazy
        initialized).
        """
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            inputs_dict = {k: v.to(0) if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}
            # We want the linear attention layer to reside on device 1 with the device map (i.e. not the
            # first/default device), to check if cache initialization is on the correct device
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


@require_torch_accelerator
@slow
class SolarOpen2FlaParityTest(unittest.TestCase):
    def test_fla_kernels_match_torch_fallback(self):
        """The fla kernels (chunk prefill and fused recurrent decode) must match the pure-PyTorch fallback.

        Runs a tiny model in float32 through a prefill and two cached decode steps on both paths. The
        measured max abs logits delta on B200 with fla-core 0.5.0 is 6.6e-4 (prefill) / 1.6e-4 (decode);
        the tolerance leaves ~8x margin.
        """
        if not is_flash_linear_attention_available("0.5.0"):
            self.skipTest("fla-core >= 0.5.0 is not available")
        import transformers.models.solar_open2.modeling_solar_open2 as modeling

        config = SolarOpen2Config(
            vocab_size=512,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            intermediate_size=128,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            max_position_embeddings=512,
        )
        torch.manual_seed(0)
        model = SolarOpen2ForCausalLM(config).to(torch_device, torch.float32).eval()
        prompt = torch.randint(0, config.vocab_size, (2, 16), device=torch_device)
        steps = [torch.randint(0, config.vocab_size, (2, 1), device=torch_device) for _ in range(2)]

        def run():
            logits = []
            with torch.no_grad():
                out = model(prompt, use_cache=True)
                cache = out.past_key_values
                logits.append(out.logits[:, -1])
                for step in steps:
                    out = model(step, past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    logits.append(out.logits[:, -1])
            return logits

        fla_logits = run()
        with (
            patch.object(modeling, "chunk_kda", None),
            patch.object(modeling, "fused_recurrent_kda", None),
            patch.object(modeling, "fused_kda_gate", None),
        ):
            fallback_logits = run()

        for got, expected in zip(fla_logits, fallback_logits):
            torch.testing.assert_close(got, expected, rtol=1e-4, atol=5e-3)


@require_torch_accelerator
@require_torch_bf16
@slow
class SolarOpen2IntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_batch_forward_logits_dummy_bf16(self):
        """Original model is 250B, hence using a dummy model on our CI to sanity check against.

        Random weights make greedy text brittle across devices and KDA kernel paths, so this checks
        last-position logits with a tolerance instead. Expectations were generated on B200 with
        fla-core 0.5.0; the measured fla vs torch-fallback delta on these 40-dim slices is 0.17 max /
        0.026 mean, so the same expectations hold on both kernel paths within the tolerances below.
        """
        model_id = "SSON9/solar-open2-tiny-dummy"
        prompts = [
            "Orange is the new black",
            "Lorem ipsum dolor sit amet",
        ]
        # fmt: off
        EXPECTED_LOGIT_SLICES = torch.tensor([
            [
                -0.1514, -0.0542, 0.2793, -0.5430, -1.2109, 0.3047, -0.2207, 0.6953, 0.2021, -0.1768,
                0.5781, -0.5039, -0.8828, 0.7305, 0.4922, 0.5078, -0.2402, 0.3105, -0.1592, 0.0679,
                0.2520, -0.1982, 1.1797, 0.1021, -0.4805, -0.2539, -0.2324, -1.0859, 1.3203, -0.4766,
                -0.6680, -0.2090, 0.1797, 0.5000, 0.4258, 0.9648, -0.7031, 0.4238, 0.1074, 0.4062,
            ],
            [
                -0.5039, -0.4824, 0.0845, -0.6367, -0.9648, 0.2285, -0.2266, -1.2812, 0.2422, -0.3867,
                -0.1289, -0.6094, 0.2393, 0.1416, 0.5312, -0.6797, 0.1406, -0.3125, -0.3730, -0.8438,
                -0.9883, -0.4570, 0.5938, -0.5742, 1.1953, 0.7266, -0.8086, 0.1504, -0.6250, 0.1045,
                0.5312, 0.0047, -0.4648, 0.0752, 0.9727, -0.0425, -1.0312, 0.6523, -0.0048, 0.2930,
            ],
        ])
        # fmt: on

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = SolarOpen2ForCausalLM.from_pretrained(
            model_id, experts_implementation="eager", device_map=torch_device, dtype=torch.bfloat16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :].float().cpu()
        # max-based bound covers the cross-kernel-path outliers; the per-prompt mean-based bound
        # catches broad regressions well below the outlier tolerance.
        torch.testing.assert_close(logits[:, :40], EXPECTED_LOGIT_SLICES, rtol=0.0, atol=0.25)
        per_prompt_mean = (logits[:, :40] - EXPECTED_LOGIT_SLICES).abs().mean(dim=-1)
        self.assertLess(per_prompt_mean.max().item(), 0.06)

        # generation smoke: the hybrid cache must sustain a full greedy decode
        generated_ids = model.generate(**inputs, max_new_tokens=20, min_new_tokens=20, do_sample=False)
        self.assertEqual(generated_ids.shape[1], inputs["input_ids"].shape[1] + 20)
