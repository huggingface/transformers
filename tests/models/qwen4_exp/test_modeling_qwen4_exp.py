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
"""Testing suite for the PyTorch Qwen4-Exp model."""

import os
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        DynamicCache,
        Qwen4ExpConfig,
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
        Qwen4ExpModel,
        Qwen4ExpTextConfig,
        Qwen4ExpTextModel,
        Qwen4ExpVisionModel,
        StaticCache,
    )
    from transformers.distributed.fsdp import verify_fsdp_plan
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )


def get_qwen4_exp_text_config(ple_layer_ids=None, ple_conv_kernel_size=2, layer_types=None, use_qsa=False):
    layer_types = ["full_attention", "linear_attention"] if layer_types is None else layer_types
    qsa_kwargs = (
        {
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 8,
            "indexer_budget": 4,
            "indexer_compress_ratio": 2,
        }
        if use_qsa
        else {}
    )
    return Qwen4ExpTextConfig(
        vocab_size=99,
        hidden_size=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=2,
        num_experts=4,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=512,
        layer_types=layer_types,
        rope_parameters={"rope_type": "default", "mrope_section": [16, 8, 8], "mrope_interleaved": True},
        linear_conv_kernel_dim=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        output_gate_type="sigmoid",
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[] if ple_layer_ids is None else ple_layer_ids,
        ple_embed_dim=16,
        ple_conv_kernel_size=ple_conv_kernel_size,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        **qsa_kwargs,
    )


class Qwen4ExpTextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Qwen4ExpTextModel
        causal_lm_class = Qwen4ExpForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.hidden_act = "silu"
        self.layer_types = ["full_attention", "linear_attention"]
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 8
        self.hc_count = 2
        self.hc_lowrank = 8
        self.ple_layer_ids = []
        self.ple_embed_dim = 16
        self.ple_conv_kernel_size = 2
        self.ngram_size = 3
        self.heads_per_ngram = 2
        self.ngram_vocab_size_base = 31
        self.make_ngram_vocab_size_divisible_by = 8
        self.split_ngram_parts = 4
        self.moe_intermediate_size = 8
        self.shared_expert_intermediate_size = 8
        self.num_experts_per_tok = 2
        self.num_experts = 4


@require_torch
class Qwen4ExpTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Qwen4ExpTextModelTester
    model_split_percents = [0.5, 0.8, 0.9]

    def _get_conv_state_shape(self, batch_size: int, config):
        intermediate_size = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        return (batch_size, intermediate_size, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (
            batch_size,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        )

    @staticmethod
    def _get_parallel_config():
        return get_qwen4_exp_text_config(ple_layer_ids=[1], use_qsa=True)

    def _get_tp_config(self, tie_word_embeddings=None):
        config = self._get_parallel_config()
        remainder = config.vocab_size % self.tensor_parallel_size
        if remainder:
            config.vocab_size += self.tensor_parallel_size - remainder
        if tie_word_embeddings is not None:
            config.tie_word_embeddings = tie_word_embeddings
        return config

    def _get_tiny_config(self):
        config = self._get_parallel_config()
        config.vocab_size = 256
        return type(config), config.to_diff_dict()

    def _assert_cached_matches_full(self, model, input_ids, test_static=False):
        split_idx = (input_ids.shape[1] + 1) // 2
        with torch.no_grad():
            expected = model(input_ids, use_cache=False).last_hidden_state
            chunk_cache = DynamicCache(config=model.config)
            actual_outputs = [
                torch.cat(
                    [
                        model(input_ids[:, :split_idx], past_key_values=chunk_cache, use_cache=True).last_hidden_state,
                        model(input_ids[:, split_idx:], past_key_values=chunk_cache, use_cache=True).last_hidden_state,
                    ],
                    dim=1,
                )
            ]
            decode_caches = [DynamicCache(config=model.config)]
            if test_static:
                decode_caches.append(StaticCache(config=model.config, max_cache_len=input_ids.shape[1]))
            for cache in decode_caches:
                actual_outputs.append(
                    torch.cat(
                        [
                            model(
                                input_ids[:, token_idx : token_idx + 1],
                                past_key_values=cache,
                                use_cache=True,
                            ).last_hidden_state
                            for token_idx in range(input_ids.shape[1])
                        ],
                        dim=1,
                    )
                )

        for actual in actual_outputs:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        config.return_dict = True
        seq_len = self.model_tester.seq_length

        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(
                    **self._prepare_for_class(inputs_dict, model_class),
                    output_attentions=True,
                    output_hidden_states=True,
                )

            self.assertEqual(
                len(outputs.attentions),
                sum(layer == "full_attention" for layer in config.layers_block_type),
            )
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]),
                [config.num_attention_heads, seq_len, seq_len],
            )
            self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
            self.assertTrue(all(state.shape[-1] == config.hidden_size for state in outputs.hidden_states))

    def test_tp_plan_matches_params(self):
        self.model_tester.ple_layer_ids = [1]
        self.model_tester.indexer_n_heads = 2
        self.model_tester.indexer_kv_heads = 1
        self.model_tester.indexer_head_dim = 8
        self.model_tester.indexer_budget = 4
        self.model_tester.indexer_compress_ratio = 2
        try:
            super().test_tp_plan_matches_params()
        finally:
            self.model_tester.ple_layer_ids = []
            self.model_tester.indexer_n_heads = None
            self.model_tester.indexer_kv_heads = None
            self.model_tester.indexer_head_dim = None
            self.model_tester.indexer_budget = None
            self.model_tester.indexer_compress_ratio = None

    def test_qsa_matches_dense_attention_when_budget_covers_context(self):
        torch.manual_seed(0)
        dense_config = get_qwen4_exp_text_config(layer_types=["full_attention"])
        qsa_config = get_qwen4_exp_text_config(layer_types=["full_attention"], use_qsa=True)
        dense_config._attn_implementation = "eager"
        qsa_config._attn_implementation = "eager"
        qsa_model = Qwen4ExpTextModel(qsa_config).to(torch_device).eval()
        dense_model = Qwen4ExpTextModel(dense_config).to(torch_device).eval()
        dense_model.load_state_dict(qsa_model.state_dict(), strict=False)

        input_ids = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]], device=torch_device)
        with torch.no_grad():
            expected = dense_model(input_ids).last_hidden_state
            actual = qsa_model(input_ids).last_hidden_state

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        qsa_model.config._attn_implementation = "sdpa"
        with torch.no_grad():
            actual_sdpa = qsa_model(input_ids).last_hidden_state
        torch.testing.assert_close(actual_sdpa, actual, rtol=1e-5, atol=1e-5)

    def test_qsa_left_padding_matches_unpadded_sequence(self):
        config = get_qwen4_exp_text_config(layer_types=["full_attention"], use_qsa=True)
        config._attn_implementation = "eager"
        model = Qwen4ExpTextModel(config).to(torch_device).eval()
        padded_ids = torch.tensor([[config.pad_token_id, config.pad_token_id, 5, 6, 7]], device=torch_device)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], device=torch_device)
        unpadded_ids = padded_ids[:, -3:]

        with torch.no_grad():
            padded = model(padded_ids, attention_mask=attention_mask).last_hidden_state[:, -3:]
            unpadded = model(unpadded_ids).last_hidden_state

        torch.testing.assert_close(padded, unpadded, rtol=1e-5, atol=1e-5)

    def test_ple_padding_matches_unpadded_sequence(self):
        torch.manual_seed(0)
        config = get_qwen4_exp_text_config(ple_layer_ids=[1], layer_types=["full_attention"])
        config._attn_implementation = "eager"
        model = Qwen4ExpTextModel(config).to(torch_device).eval()
        with torch.no_grad():
            model.layers[0].ple.norm_conv.weight.fill_(1)
            model.layers[0].ple.conv1d.weight.normal_(mean=0.0, std=0.2)

        padded_ids = torch.tensor([[config.pad_token_id, config.pad_token_id, 5, 6, 7]], device=torch_device)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], device=torch_device)

        with torch.no_grad():
            padded = model(padded_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state[:, -3:]
            unpadded = model(padded_ids[:, -3:], use_cache=False).last_hidden_state

        torch.testing.assert_close(padded, unpadded, rtol=1e-5, atol=1e-5)

    def test_ple_static_cache_preserves_padding_mask(self):
        torch.manual_seed(0)
        config = get_qwen4_exp_text_config(ple_layer_ids=[1], layer_types=["full_attention"])
        config._attn_implementation = "eager"
        model = Qwen4ExpForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            model.model.layers[0].ple.norm_conv.weight.fill_(1)
            model.model.layers[0].ple.conv1d.weight.normal_(mean=0.0, std=0.2)

        input_ids = torch.tensor([[config.pad_token_id, config.pad_token_id, 5, 6, 7]], device=torch_device)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], device=torch_device)
        static_cache = StaticCache(config=config, max_cache_len=input_ids.shape[1])
        static_inputs = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=static_cache,
            attention_mask=attention_mask,
            is_first_iteration=True,
        )

        with torch.no_grad():
            static_logits = model(**static_inputs, use_cache=True).logits[:, -3:]
            dynamic_logits = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=DynamicCache(config=config),
                use_cache=True,
            ).logits[:, -3:]

        torch.testing.assert_close(static_logits, dynamic_logits, rtol=1e-5, atol=1e-5)

    def test_qsa_static_generation(self):
        config = get_qwen4_exp_text_config(layer_types=["full_attention"], use_qsa=True)
        config._attn_implementation = "eager"
        model = Qwen4ExpForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[5, 6, 7]], device=torch_device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=2,
                cache_implementation="static",
                disable_compile=True,
                do_sample=False,
                eos_token_id=None,
            )

        self.assertEqual(output.shape, (1, input_ids.shape[1] + 2))

    def test_qsa_cached_forward_matches_full_forward_with_ple(self):
        torch.manual_seed(0)
        config = get_qwen4_exp_text_config(
            ple_layer_ids=[1],
            layer_types=["full_attention", "full_attention"],
            use_qsa=True,
        )
        config._attn_implementation = "eager"
        model = Qwen4ExpTextModel(config).to(torch_device).eval()
        with torch.no_grad():
            model.layers[0].ple.conv1d.weight.normal_(mean=0.0, std=0.02)

        input_ids = torch.tensor(
            [[5, 6, config.eos_token_id, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17]],
            device=torch_device,
        )
        self._assert_cached_matches_full(model, input_ids, test_static=True)

    def test_qsa_ple_beam_generation(self):
        config = get_qwen4_exp_text_config(
            ple_layer_ids=[1],
            layer_types=["full_attention"],
            use_qsa=True,
        )
        config._attn_implementation = "eager"
        model = Qwen4ExpForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[5, 6, 7]], device=torch_device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=3,
                num_beams=2,
                do_sample=False,
                eos_token_id=None,
            )

        self.assertEqual(output.shape, (1, input_ids.shape[1] + 3))

    def test_sharded_checkpoint_layout_loads_and_forwards(self):
        config = get_qwen4_exp_text_config(ple_layer_ids=[1], layer_types=["full_attention"], use_qsa=True)
        config.split_ngram_parts = 12
        model = Qwen4ExpForCausalLM(config)
        state_dict = model.state_dict()
        checkpoint_state_dict = {}

        def add_checkpoint_tensor(key, tensor):
            if key.startswith("model."):
                key = key.replace("model.", "model.language_model.", 1)
            checkpoint_state_dict[key] = tensor.detach().clone()

        for key, tensor in state_dict.items():
            if key.endswith("mlp.experts.gate_up_proj"):
                gate_proj, up_proj = tensor.chunk(2, dim=1)
                for expert_idx in range(config.num_experts):
                    add_checkpoint_tensor(
                        key.replace("experts.gate_up_proj", f"experts.{expert_idx}.gate_proj.weight"),
                        gate_proj[expert_idx],
                    )
                    add_checkpoint_tensor(
                        key.replace("experts.gate_up_proj", f"experts.{expert_idx}.up_proj.weight"),
                        up_proj[expert_idx],
                    )
            elif key.endswith("mlp.experts.down_proj"):
                for expert_idx in range(config.num_experts):
                    add_checkpoint_tensor(
                        key.replace("experts.down_proj", f"experts.{expert_idx}.down_proj.weight"),
                        tensor[expert_idx],
                    )
            elif key.endswith("ple.ple_embedding.ngram_embedding.weight"):
                shards = torch.chunk(tensor, config.split_ngram_parts, dim=0)
                for shard_idx, shard in enumerate(shards):
                    add_checkpoint_tensor(
                        key.replace("ngram_embedding.weight", f"ngram_embedding.shard_{shard_idx}.weight"),
                        shard,
                    )
            else:
                add_checkpoint_tensor(key, tensor)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            save_file(checkpoint_state_dict, os.path.join(tmpdirname, "model.safetensors"))
            loaded_model, loading_info = Qwen4ExpForCausalLM.from_pretrained(
                tmpdirname,
                output_loading_info=True,
            )

        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        loaded_state_dict = loaded_model.state_dict()
        for key, expected_weight in state_dict.items():
            torch.testing.assert_close(loaded_state_dict[key], expected_weight)
        input_ids = torch.tensor([[5, 6, 7, 8]])
        with torch.no_grad():
            expected = model.eval()(input_ids, use_cache=False).logits
            actual = loaded_model.eval()(input_ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected)

    def test_sparse_moe_forward(self):
        config = get_qwen4_exp_text_config()
        model = Qwen4ExpForCausalLM(config).to(torch_device).eval()

        input_ids = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]], device=torch_device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)

        self.assertEqual(len(outputs.router_logits), config.num_hidden_layers)
        self.assertTrue(
            all(
                router_logits.shape == (input_ids.numel(), config.num_experts)
                for router_logits in outputs.router_logits
            )
        )
        self.assertIsNotNone(outputs.aux_loss)
        self.assertIsNotNone(outputs.loss)

    def test_ple_cached_forward_matches_full_forward(self):
        torch.manual_seed(0)
        config = get_qwen4_exp_text_config(ple_layer_ids=[1, 2])
        config._attn_implementation = "eager"
        model = Qwen4ExpTextModel(config).to(torch_device).eval()
        # Use non-zero convolution weights to exercise cached PLE state.
        with torch.no_grad():
            for layer in model.layers:
                if layer.ple is not None:
                    layer.ple.conv1d.weight.normal_(mean=0.0, std=0.02)

        input_ids = torch.tensor(
            [[5, 6, config.eos_token_id, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17]],
            device=torch_device,
        )
        self._assert_cached_matches_full(model, input_ids)

    def test_ple_and_qsa_reject_offloaded_cache(self):
        cases = (
            ("ple", [1], False),
            ("qsa", [], True),
        )
        input_ids = torch.tensor([[5, 6, 7]], device=torch_device)

        for name, ple_layer_ids, use_qsa in cases:
            with self.subTest(name=name):
                config = get_qwen4_exp_text_config(
                    layer_types=["full_attention"],
                    ple_layer_ids=ple_layer_ids,
                    use_qsa=use_qsa,
                )
                config._attn_implementation = "eager"
                model = Qwen4ExpTextModel(config).to(torch_device).eval()
                cache = DynamicCache(config=config, offloading=True)

                with self.assertRaisesRegex(ValueError, "does not support offloaded caches"):
                    model(input_ids, past_key_values=cache, use_cache=True)

    def test_gdn_cached_forward_with_ple_on_another_layer(self):
        torch.manual_seed(0)
        config = get_qwen4_exp_text_config(ple_layer_ids=[1])
        config._attn_implementation = "eager"
        model = Qwen4ExpTextModel(config).to(torch_device).eval()
        model.layers[1].linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        model.layers[1].linear_attn.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
        input_ids = torch.tensor([[5, 6, 7, 8, 9]], device=torch_device)

        with torch.no_grad():
            expected = model(input_ids, use_cache=False).last_hidden_state[:, -1:]
            cache = DynamicCache(config=config)
            model(input_ids[:, :-1], past_key_values=cache, use_cache=True)

            decoded = model(
                input_ids[:, -1:],
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state

        torch.testing.assert_close(decoded, expected, rtol=1e-5, atol=1e-5)

    def test_ple_with_inputs_embeds(self):
        config = get_qwen4_exp_text_config(ple_layer_ids=[1], ple_conv_kernel_size=1)
        model = Qwen4ExpTextModel(config).to(torch_device).eval()
        input_ids = torch.tensor([[5, 6, 7]], device=torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        with self.assertRaisesRegex(ValueError, "ple_input_ids must be provided"):
            model(inputs_embeds=inputs_embeds)
        with torch.no_grad():
            outputs = model(inputs_embeds=inputs_embeds, ple_input_ids=input_ids)
        self.assertEqual(outputs.last_hidden_state.shape, (1, 3, config.hidden_size))

        causal_model = Qwen4ExpForCausalLM(config).to(torch_device).eval()
        causal_inputs_embeds = causal_model.get_input_embeddings()(input_ids)
        with torch.no_grad():
            causal_outputs = causal_model(
                inputs_embeds=causal_inputs_embeds,
                ple_input_ids=input_ids,
                use_cache=False,
            )
        self.assertEqual(causal_outputs.logits.shape, (1, 3, config.vocab_size))

    def test_generate_with_inputs_embeds_and_ple_input_ids(self):
        config = get_qwen4_exp_text_config(ple_layer_ids=[1])
        model = Qwen4ExpForCausalLM(config).to(torch_device).eval()
        model.generation_config.eos_token_id = None
        input_ids = torch.tensor([[5, 6, 7]], device=torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        with torch.no_grad():
            output = model.generate(
                inputs_embeds=inputs_embeds,
                ple_input_ids=input_ids,
                max_new_tokens=2,
                do_sample=False,
            )
        self.assertEqual(output.shape, (1, 2))

    def test_auto_classes(self):
        config = get_qwen4_exp_text_config()
        self.assertIsInstance(AutoConfig.for_model("qwen4_exp_text"), Qwen4ExpTextConfig)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        self.assertIsInstance(model, Qwen4ExpForCausalLM)

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        self.model_tester.ple_layer_ids = [1]
        self.model_tester.split_ngram_parts = 12
        try:
            super().test_reverse_loading_mapping(check_keys_were_modified)
        finally:
            self.model_tester.ple_layer_ids = []
            self.model_tester.split_ngram_parts = 4


@require_torch
class Qwen4ExpCompositeModelTest(unittest.TestCase):
    def get_config(self, use_qsa=False, layer_types=None):
        text_config = get_qwen4_exp_text_config(
            ple_layer_ids=[1],
            layer_types=layer_types,
            use_qsa=use_qsa,
        )
        vision_config = {
            "depth": 1,
            "in_channels": 3,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 32,
            "out_hidden_size": text_config.hidden_size,
            "hidden_size": 32,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        }
        return Qwen4ExpConfig(
            text_config=text_config.to_dict(),
            vision_config=vision_config,
            image_token_id=3,
            video_token_id=4,
            vision_start_token_id=5,
            vision_end_token_id=6,
        )

    def test_multimodal_model_with_inputs_embeds_and_ple_input_ids(self):
        config = self.get_config()
        model = Qwen4ExpForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.tensor([[7, 8, 9, 10]], device=torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)

        with self.assertRaisesRegex(ValueError, "ple_input_ids must be provided"):
            model(inputs_embeds=inputs_embeds, use_cache=False)
        with self.assertRaisesRegex(ValueError, "same batch size and sequence length"):
            model(inputs_embeds=inputs_embeds, ple_input_ids=input_ids[:, :-1], use_cache=False)

        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).logits
            actual = model(inputs_embeds=inputs_embeds, ple_input_ids=input_ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_input_ids_take_priority_over_stale_ple_input_ids(self):
        config = self.get_config()
        model = Qwen4ExpForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.tensor([[7]], device=torch_device)
        stale_ple_input_ids = torch.tensor([[5, 6, 7, 8]], device=torch_device)
        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).logits
            actual = model(input_ids=input_ids, ple_input_ids=stale_ple_input_ids, use_cache=False).logits

        torch.testing.assert_close(actual, expected)

    def test_qsa_unsupported_attention_backend_matches_eager(self):
        fallback_config = self.get_config(use_qsa=True, layer_types=["full_attention"])
        fallback_config._attn_implementation = "flash_attention_2"
        fallback_model = Qwen4ExpForConditionalGeneration(fallback_config).to(torch_device).eval()

        eager_config = self.get_config(use_qsa=True, layer_types=["full_attention"])
        eager_config._attn_implementation = "eager"
        eager_model = Qwen4ExpForConditionalGeneration(eager_config).to(torch_device).eval()
        eager_model.load_state_dict(fallback_model.state_dict())

        input_ids = torch.tensor([[7, 8, 9, 10]], device=torch_device)
        with torch.no_grad():
            expected = eager_model(input_ids, use_cache=False).logits
            actual = fallback_model(input_ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected)

    def test_fsdp_plan_has_no_unused_rules(self):
        with torch.device("meta"):
            model = Qwen4ExpForConditionalGeneration(self.get_config())
        with self.assertNoLogs("transformers.distributed.fsdp", level="WARNING"):
            verify_fsdp_plan([name for name, _ in model.named_modules()], model._fsdp_plan)

    def test_generate_with_inputs_embeds_and_ple_input_ids(self):
        config = self.get_config()
        model = Qwen4ExpForConditionalGeneration(config).to(torch_device).eval()
        model.generation_config.eos_token_id = None
        input_ids = torch.tensor([[7, 8, 9, 10]], device=torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        with torch.no_grad():
            output = model.generate(
                inputs_embeds=inputs_embeds,
                ple_input_ids=input_ids,
                max_new_tokens=2,
                do_sample=False,
            )
        self.assertEqual(output.shape, (1, 2))

    def test_qsa_image_forward_matches_dense_attention(self):
        torch.manual_seed(0)
        layer_types = ["full_attention", "full_attention"]
        dense_config = self.get_config(layer_types=layer_types)
        qsa_config = self.get_config(use_qsa=True, layer_types=layer_types)
        qsa_config.text_config.indexer_budget = 8
        dense_config.text_config._attn_implementation = "eager"
        qsa_config.text_config._attn_implementation = "eager"

        qsa_model = Qwen4ExpForConditionalGeneration(qsa_config).to(torch_device).eval()
        dense_model = Qwen4ExpForConditionalGeneration(dense_config).to(torch_device).eval()
        dense_model.load_state_dict(qsa_model.state_dict(), strict=False)

        input_ids = torch.tensor([[5, 3, 3, 3, 3, 6, 7, 8]], device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[:, 1:5] = 1
        image_grid_thw = torch.tensor([[1, 2, 2]], device=torch_device)
        flattened_patch_size = (
            qsa_config.vision_config.in_channels
            * qsa_config.vision_config.temporal_patch_size
            * qsa_config.vision_config.patch_size**2
        )
        pixel_values = torch.randn((4, flattened_patch_size), device=torch_device)
        with torch.no_grad():
            expected = dense_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            ).logits
            actual = qsa_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            ).logits

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_ple_embedding_shards_are_merged_on_load(self):
        config = self.get_config()
        model = Qwen4ExpForConditionalGeneration(config).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            reloaded = AutoModelForImageTextToText.from_pretrained(tmpdirname).eval()

        self.assertIsInstance(reloaded, Qwen4ExpForConditionalGeneration)
        input_ids = torch.tensor([[7, 8, 9, 10]])
        with torch.no_grad():
            expected = model(input_ids=input_ids, use_cache=False).logits
            actual = reloaded(input_ids=input_ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected)

    def test_composite_checkpoint_loads_as_causal_lm(self):
        config = self.get_config(use_qsa=True)
        composite_model = Qwen4ExpForConditionalGeneration(config).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            composite_model.save_pretrained(tmpdirname)
            causal_model, loading_info = AutoModelForCausalLM.from_pretrained(
                tmpdirname,
                output_loading_info=True,
            )

        self.assertIsInstance(causal_model, Qwen4ExpForCausalLM)
        self.assertFalse(loading_info["missing_keys"])
        self.assertFalse(loading_info["unexpected_keys"])
        input_ids = torch.tensor([[7, 8, 9, 10]])
        with torch.no_grad():
            expected = composite_model(input_ids=input_ids, use_cache=False).logits
            actual = causal_model(input_ids=input_ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected)

    def test_base_model_checkpoint_loads_as_conditional_generation(self):
        config = self.get_config(use_qsa=True)
        base_model = Qwen4ExpModel(config).eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            base_model.save_pretrained(tmpdirname)
            conditional_model, loading_info = Qwen4ExpForConditionalGeneration.from_pretrained(
                tmpdirname,
                output_loading_info=True,
            )

        self.assertEqual(loading_info["missing_keys"], {"lm_head.weight"})
        self.assertFalse(loading_info["unexpected_keys"])
        input_ids = torch.tensor([[7, 8, 9, 10]])
        with torch.no_grad():
            expected = base_model(input_ids=input_ids, use_cache=False).last_hidden_state
            actual = conditional_model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected)

    def test_vision(self):
        config = self.get_config()
        config.num_labels = 3

        vision_model = Qwen4ExpVisionModel(config.vision_config).to(torch_device).eval()
        flattened_patch_size = (
            config.vision_config.in_channels
            * config.vision_config.temporal_patch_size
            * config.vision_config.patch_size**2
        )
        pixel_values = torch.randn((1, flattened_patch_size), device=torch_device)
        grid_thw = torch.tensor([[1, 1, 1]], device=torch_device)

        with torch.no_grad():
            vision_outputs = vision_model(pixel_values, grid_thw=grid_thw)

        self.assertEqual(vision_outputs.pooler_output.shape, (1, config.vision_config.out_hidden_size))

    def test_composite_auto_classes(self):
        config = self.get_config()
        self.assertIsInstance(AutoConfig.for_model("qwen4_exp"), Qwen4ExpConfig)
        with torch.device("meta"):
            conditional_model = AutoModelForImageTextToText.from_config(config)
            causal_model = AutoModelForCausalLM.from_config(config)
        self.assertIsInstance(conditional_model, Qwen4ExpForConditionalGeneration)
        self.assertIsInstance(causal_model, Qwen4ExpForCausalLM)
        self.assertIsInstance(causal_model.config, Qwen4ExpTextConfig)
