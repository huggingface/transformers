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

import copy
import os
import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import is_fsdp_test, require_torch, require_torch_greater_or_equal, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import (
        AutoModelForCausalLM,
        DynamicCache,
        Qwen4ExpConfig,
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
        Qwen4ExpModel,
        Qwen4ExpTextConfig,
        Qwen4ExpTextModel,
        Qwen4ExpVisionConfig,
        StaticCache,
    )
    from transformers.distributed.fsdp import verify_fsdp_plan


class Qwen4ExpTextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Qwen4ExpTextModel
        causal_lm_class = Qwen4ExpForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.hidden_act = "silu"
        self.rope_parameters = {"rope_type": "default", "partial_rotary_factor": 0.25}
        self.layer_types = ["linear_attention", "qwen_sparse_attention"]
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
        self.indexer_n_heads = 2
        self.indexer_kv_heads = 1
        self.indexer_head_dim = 8
        self.indexer_budget = 4
        self.indexer_compress_ratio = 2
        self.moe_intermediate_size = 8
        self.shared_expert_intermediate_size = 8
        self.num_experts = 4

    def get_config(self, **overrides):
        config = super().get_config()
        if not overrides:
            return config

        if "layer_types" in overrides:
            overrides.setdefault("num_hidden_layers", len(overrides["layer_types"]))
        config_dict = config.to_dict()
        config_dict.pop("number_of_conv_states", None)
        config_dict.update(overrides)
        return self.config_class.from_dict(config_dict)


@require_torch
class Qwen4ExpTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Qwen4ExpTextModelTester
    model_split_percents = [0.5, 0.8, 0.9]

    # QSA indexer parameters are trained through a separate objective rather than the causal-LM loss.
    test_all_params_have_gradient = False
    test_torch_exportable = False  # QSA index selection has data-dependent control flow

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

    def _get_all_layer_types_config(self):
        return self.model_tester.get_config(
            ple_layer_ids=[2],
            layer_types=[
                "linear_attention",
                "linear_attention",
                "qwen_sparse_attention",
            ],
        )

    def _get_tp_config(self, tie_word_embeddings=None):
        config = self._get_all_layer_types_config()
        remainder = config.vocab_size % self.tensor_parallel_size
        if remainder:
            config.vocab_size += self.tensor_parallel_size - remainder
        if tie_word_embeddings is not None:
            config.tie_word_embeddings = tie_word_embeddings
        return config

    def _get_tiny_config(self):
        config = self._get_all_layer_types_config()
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
                )

            self.assertEqual(
                len(outputs.attentions),
                sum(layer != "linear_attention" for layer in config.layer_types),
            )
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]),
                [config.num_attention_heads, seq_len, seq_len],
            )

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        batch_size, seq_len = inputs_dict["input_ids"].shape
        expected_shapes = [(batch_size, seq_len, config.hc_count * config.hidden_size)] * config.num_hidden_layers
        expected_shapes.append((batch_size, seq_len, config.hidden_size))

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                hidden_states = model(**inputs_dict).hidden_states
            self.assertListEqual([state.shape for state in hidden_states], expected_shapes)

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), output_length - prompt_length)
        hidden_sizes = [config.hc_count * config.hidden_size] * config.num_hidden_layers + [config.hidden_size]

        for generated_length, iteration_hidden_states in enumerate(hidden_states):
            seq_len = 1 if use_cache and generated_length > 0 else prompt_length + generated_length
            expected_shapes = [(batch_size, seq_len, hidden_size) for hidden_size in hidden_sizes]
            self.assertListEqual([state.shape for state in iteration_hidden_states], expected_shapes)

    @unittest.skip("QSA index selection has data-dependent control flow")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("QSA index selection has data-dependent control flow")
    def test_generate_compilation_all_outputs(self):
        pass

    def test_tp_plan_matches_params(self):
        self.model_tester.ple_layer_ids = [1]
        try:
            super().test_tp_plan_matches_params()
        finally:
            self.model_tester.ple_layer_ids = []

    def test_ple_layers_must_use_linear_attention(self):
        with self.assertRaisesRegex(
            StrictDataclassClassValidationError, "PLE is only supported on linear_attention layers"
        ):
            self.model_tester.get_config(
                ple_layer_ids=[2],
                layer_types=["linear_attention", "qwen_sparse_attention"],
            )

    def test_ple_padding_and_static_cache_match_unpadded_sequence(self):
        torch.manual_seed(0)
        config = self.model_tester.get_config(
            ple_layer_ids=[2],
            layer_types=["qwen_sparse_attention", "linear_attention"],
        )
        config._attn_implementation = "sdpa"
        model = Qwen4ExpForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            model.model.layers[1].ple.norm_conv.weight.fill_(1)
            model.model.layers[1].ple.conv1d.weight.normal_(mean=0.0, std=0.2)

        padded_ids = torch.tensor([[config.pad_token_id, config.pad_token_id, 5, 6, 7]], device=torch_device)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]], device=torch_device)
        static_cache = StaticCache(config=config, max_cache_len=padded_ids.shape[1])
        static_inputs = model.prepare_inputs_for_generation(
            padded_ids,
            past_key_values=static_cache,
            attention_mask=attention_mask,
            is_first_iteration=True,
        )

        with torch.no_grad():
            expected = model(padded_ids[:, -3:], use_cache=False).logits
            outputs = [model(padded_ids, attention_mask=attention_mask, use_cache=False).logits[:, -3:]]
            for cache in (DynamicCache(config=config), static_cache):
                inputs = (
                    static_inputs
                    if cache is static_cache
                    else {
                        "input_ids": padded_ids,
                        "attention_mask": attention_mask,
                        "past_key_values": cache,
                    }
                )
                outputs.append(model(**inputs, use_cache=True).logits[:, -3:])

        for actual in outputs:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_all_layer_types_cached_forward_match_full_forward(self):
        torch.manual_seed(0)
        config = self._get_all_layer_types_config()
        config._attn_implementation = "eager"
        model = Qwen4ExpTextModel(config).to(torch_device).eval()
        with torch.no_grad():
            for layer in model.layers:
                if layer.ple is not None:
                    layer.ple.conv1d.weight.normal_(mean=0.0, std=0.02)

        input_ids = torch.tensor(
            [[5, 6, config.eos_token_id, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17]],
            device=torch_device,
        )
        self._assert_cached_matches_full(model, input_ids, test_static=True)

    def test_ple_beam_generation(self):
        config = self.model_tester.get_config(
            ple_layer_ids=[1],
            layer_types=["linear_attention", "qwen_sparse_attention"],
        )
        config._attn_implementation = "eager"
        model = Qwen4ExpForCausalLM(config).to(torch_device).eval()
        input_ids = torch.tensor([[5, 6, 7]], device=torch_device)

        for cache_implementation in (None, "static"):
            with self.subTest(cache_implementation=cache_implementation), torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=3,
                    num_beams=2,
                    cache_implementation=cache_implementation,
                    disable_compile=True,
                    do_sample=False,
                    eos_token_id=None,
                )

            self.assertEqual(output.shape, (1, input_ids.shape[1] + 3))

    def test_ple_sharded_checkpoint_loads_and_forwards(self):
        config = self.model_tester.get_config(ple_layer_ids=[1], layer_types=["linear_attention"])
        config.split_ngram_parts = 12
        model = Qwen4ExpForCausalLM(config)
        state_dict = model.state_dict()
        checkpoint_state_dict = {key: tensor.detach().clone() for key, tensor in state_dict.items()}
        embedding_key = next(key for key in state_dict if key.endswith("ngram_embedding.weight"))
        embedding_weight = checkpoint_state_dict.pop(embedding_key)
        for shard_idx, shard in enumerate(torch.chunk(embedding_weight, config.split_ngram_parts, dim=0)):
            shard_key = embedding_key.replace("ngram_embedding.weight", f"ngram_embedding.shard_{shard_idx}.weight")
            checkpoint_state_dict[shard_key] = shard.contiguous()

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

    def test_generate_with_ple_and_inputs_embeds(self):
        config = self.model_tester.get_config(ple_layer_ids=[1])
        model = Qwen4ExpForCausalLM(config).to(torch_device).eval()
        model.generation_config.eos_token_id = None
        input_ids = torch.tensor([[5, 6, 7]], device=torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        with torch.no_grad():
            expected = model.generate(input_ids, max_new_tokens=2, do_sample=False)
            actual = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=2,
                do_sample=False,
            )
        torch.testing.assert_close(actual, expected[:, input_ids.shape[1] :])

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("Qwen4-Exp hybrid linear-attention cache is not compatible with quantized cache yet.")
    def test_generate_with_quant_cache(self):
        pass

    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        self.model_tester.ple_layer_ids = [1]
        self.model_tester.split_ngram_parts = 12
        try:
            super().test_reverse_loading_mapping(check_keys_were_modified)
        finally:
            self.model_tester.ple_layer_ids = []
            self.model_tester.split_ngram_parts = 4

    @require_torch_greater_or_equal("2.7")
    @is_fsdp_test
    @unittest.skip(reason="FIXME: Cyril or Ferdinand")
    def test_fsdp2_save_load(self):
        pass

    @require_torch_greater_or_equal("2.7")
    @is_fsdp_test
    @unittest.skip(reason="FIXME: Cyril or Ferdinand")
    def test_fsdp2_save_load_dcp(self):
        pass

    @parameterized.expand(["untied", "tied"])
    @require_torch_greater_or_equal("2.7")
    @is_fsdp_test
    @unittest.skip(reason="FIXME: Cyril or Ferdinand")
    def test_fsdp2_plan_vs_ddp(self, label):
        pass


class Qwen4ExpVisionText2TextModelTester(VLMModelTester):
    base_model_class = Qwen4ExpModel
    config_class = Qwen4ExpConfig
    text_config_class = Qwen4ExpTextConfig
    vision_config_class = Qwen4ExpVisionConfig
    conditional_generation_class = Qwen4ExpForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("num_key_value_heads", 1)
        kwargs.setdefault("head_dim", 24)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("layer_types", ["linear_attention", "qwen_sparse_attention"])
        kwargs.setdefault("linear_conv_kernel_dim", 2)
        kwargs.setdefault("linear_key_head_dim", 16)
        kwargs.setdefault("linear_value_head_dim", 16)
        kwargs.setdefault("linear_num_key_heads", 4)
        kwargs.setdefault("linear_num_value_heads", 8)
        kwargs.setdefault("hc_count", 2)
        kwargs.setdefault("hc_lowrank", 8)
        kwargs.setdefault("ple_layer_ids", [])
        kwargs.setdefault("ple_embed_dim", 16)
        kwargs.setdefault("ple_conv_kernel_size", 2)
        kwargs.setdefault("ngram_size", 3)
        kwargs.setdefault("heads_per_ngram", 2)
        kwargs.setdefault("ngram_vocab_size_base", 31)
        kwargs.setdefault("make_ngram_vocab_size_divisible_by", 8)
        kwargs.setdefault("split_ngram_parts", 4)
        kwargs.setdefault("indexer_n_heads", 2)
        kwargs.setdefault("indexer_kv_heads", 1)
        kwargs.setdefault("indexer_head_dim", 8)
        kwargs.setdefault("indexer_budget", 4)
        kwargs.setdefault("indexer_compress_ratio", 2)
        kwargs.setdefault("moe_intermediate_size", 8)
        kwargs.setdefault("shared_expert_intermediate_size", 8)
        kwargs.setdefault("num_experts", 4)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 1)
        kwargs.setdefault("pad_token_id", 2)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("vision_start_token_id", 5)
        kwargs.setdefault("vision_end_token_id", 6)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("depth", 1)
        kwargs.setdefault("vision_hidden_act", "gelu_pytorch_tanh")
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("num_position_embeddings", 16)
        kwargs.setdefault(
            "rope_parameters",
            {
                "rope_type": "default",
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 1, 1],
                "mrope_interleaved": True,
            },
        )
        super().__init__(parent, **kwargs)
        self.in_channels = self.num_channels
        self.out_hidden_size = self.hidden_size
        self.vision_hidden_size = self.hidden_size
        self.vision_intermediate_size = self.hidden_size
        self.expected_num_hidden_layers = self.depth + 1

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size * (self.image_size // self.patch_size) ** 2,
                self.num_channels * self.patch_size**2 * self.temporal_patch_size,
            ]
        )

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        }

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, -1] = self.pad_token_id
        input_ids[:, 0] = self.vision_start_token_id
        input_ids[:, 1] = self.image_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "mm_token_type_ids": mm_token_type_ids,
        }

    def get_config(self, ple_layer_ids=None):
        text_config = self.get_text_config().to_dict()
        text_config.pop("number_of_conv_states", None)
        if ple_layer_ids is not None:
            text_config["ple_layer_ids"] = ple_layer_ids
        return Qwen4ExpConfig(
            text_config=text_config,
            vision_config=self.get_vision_config().to_dict(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
        )


@require_torch
class Qwen4ExpCompositeModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Qwen4ExpVisionText2TextModelTester
    test_all_params_have_gradient = False
    test_torch_exportable = False  # QSA index selection has data-dependent control flow

    def get_config(self):
        return self.model_tester.get_config(ple_layer_ids=[1])

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

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), output_length - prompt_length)
        hidden_sizes = [config.hc_count * config.hidden_size] * config.num_hidden_layers + [config.hidden_size]

        for generated_length, iteration_hidden_states in enumerate(hidden_states):
            seq_length = 1 if use_cache and generated_length > 0 else prompt_length + generated_length
            expected_shapes = [(batch_size, seq_length, hidden_size) for hidden_size in hidden_sizes]
            self.assertListEqual([state.shape for state in iteration_hidden_states], expected_shapes)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"
        expected_num_attentions = sum(
            layer_type != "linear_attention" for layer_type in config.text_config.layer_types
        )

        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(
                    **self._prepare_for_class(inputs_dict, model_class),
                    output_attentions=True,
                )

            self.assertEqual(len(outputs.attentions), expected_num_attentions)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]),
                [config.text_config.num_attention_heads, self.model_tester.seq_length, self.model_tester.seq_length],
            )

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        batch_size, seq_length = inputs_dict["input_ids"].shape
        expected_shapes = [
            (batch_size, seq_length, config.text_config.hc_count * config.text_config.hidden_size)
        ] * config.text_config.num_hidden_layers
        expected_shapes.append((batch_size, seq_length, config.text_config.hidden_size))

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config)).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(
                    **self._prepare_for_class(inputs_dict, model_class),
                    output_hidden_states=True,
                )
            self.assertListEqual([hidden_state.shape for hidden_state in outputs.hidden_states], expected_shapes)

    def test_mismatching_num_image_tokens(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                model(**inputs_dict)

            mismatched_inputs = copy.deepcopy(inputs_dict)
            mismatched_inputs["pixel_values"] = mismatched_inputs["pixel_values"][-1:]
            mismatched_inputs["image_grid_thw"] = mismatched_inputs["image_grid_thw"][-1:]
            with self.assertRaises(ValueError):
                model(**mismatched_inputs)

            single_inputs = {
                key: value[:1] if isinstance(value, torch.Tensor) else value for key, value in inputs_dict.items()
            }
            two_prompt_inputs = {
                key: torch.cat([value, value]) if isinstance(value, torch.Tensor) else value
                for key, value in single_inputs.items()
            }
            two_prompt_inputs["pixel_values"] = single_inputs["pixel_values"]
            two_prompt_inputs["image_grid_thw"] = single_inputs["image_grid_thw"]
            with self.assertRaises(ValueError):
                model(**two_prompt_inputs)

            two_prompt_inputs["pixel_values"] = torch.cat(
                [single_inputs["pixel_values"], single_inputs["pixel_values"]]
            )
            two_prompt_inputs["image_grid_thw"] = torch.cat(
                [single_inputs["image_grid_thw"], single_inputs["image_grid_thw"]]
            )
            with torch.no_grad():
                model(**two_prompt_inputs)

    @unittest.skip("QSA index selection has data-dependent control flow")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("QSA index selection has data-dependent control flow")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("Qwen4-Exp hybrid linear-attention cache is not compatible with quantized cache yet.")
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_tp_plan_matches_params(self):
        self.model_tester.ple_layer_ids = [1]
        try:
            super().test_tp_plan_matches_params()
        finally:
            self.model_tester.ple_layer_ids = []

    def test_fsdp_plan_has_no_unused_rules(self):
        with torch.device("meta"):
            model = Qwen4ExpForConditionalGeneration(self.get_config())
        with self.assertNoLogs("transformers.distributed.fsdp", level="WARNING"):
            verify_fsdp_plan([name for name, _ in model.named_modules()], model._fsdp_plan)

    def test_generate_with_ple_and_inputs_embeds(self):
        config = self.get_config()
        model = Qwen4ExpForConditionalGeneration(config).to(torch_device).eval()
        model.generation_config.eos_token_id = None
        input_ids = torch.tensor([[7, 8, 9, 10]], device=torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        with torch.no_grad():
            expected = model.generate(input_ids, max_new_tokens=2, do_sample=False)
            actual = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=2,
                do_sample=False,
            )
        torch.testing.assert_close(actual, expected[:, input_ids.shape[1] :])

    def test_video_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        inputs["input_ids"][inputs["input_ids"] == config.image_token_id] = config.video_token_id
        inputs["mm_token_type_ids"][inputs["mm_token_type_ids"] == 1] = 2
        inputs["pixel_values_videos"] = inputs.pop("pixel_values")
        inputs["video_grid_thw"] = inputs.pop("image_grid_thw")

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                model(**inputs)

    def test_composite_checkpoint_loads_as_causal_lm(self):
        config = self.get_config()
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
        config = self.get_config()
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
