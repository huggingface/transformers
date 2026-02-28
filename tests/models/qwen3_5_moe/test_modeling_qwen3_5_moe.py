# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen3.5 model."""

import copy
import os
import re
import tempfile
import unittest

from safetensors.torch import load_file

from transformers import is_torch_available
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightRenaming, process_target_pattern
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    compare_state_dicts,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeModel,
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeTextModel,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDynamicCache


class Qwen3_5MoeTextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Qwen3_5MoeTextModel
        causal_lm_class = Qwen3_5MoeForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["full_attention", "linear_attention"]
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 8


@require_torch
class Qwen3_5MoeTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Qwen3_5MoeTextModelTester
    config_class = Qwen3_5MoeTextConfig

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        "Qwen3.5 Moe has a special Cache as it alternates with gated deltanet layers"
        self.assertIsInstance(past_key_values, Qwen3_5MoeDynamicCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        expected_shape = (batch_size, num_heads, seq_length, head_dim)

        attention_layer_indices = past_key_values.transformer_layers
        self.assertListEqual(
            [past_key_values.key_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )
        self.assertListEqual(
            [past_key_values.value_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )

    def _check_caches_are_equal(self, cache1, cache2):
        "Qwen3.5 Moe has a special Cache as it alternates with gated deltanet layers"
        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            if cache1.key_cache[idx] is not None:
                torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
                torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3.5 Moe alternates between attention layers and gated deltanet layers."
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

    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        """
        Overwritten to check for the moe portion but ignore the prefix as it results into a noop
        (except we have a VLM struct initially)
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        #  Some MoE models alternate between a classic MLP and a MoE layer, in which case we want to have at
        # lest one MoE layer here to check the mapping
        config_to_set = config.get_text_config(decoder=True)
        config_to_set.first_k_dense_replace = 1  # means that the first layer (idx 0) will be MLP, then MoE
        config_to_set.moe_layer_start_index = 1  # same as above but for Ernie 4.5...
        config_to_set.mlp_only_layers = [0]  # same but for qwens
        config_to_set.num_dense_layers = 1  # lfm2_moe

        for model_class in self.all_model_classes:
            # Each individual model is a subtest
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config))
                # Skip if no conversions
                conversions = get_model_conversion_mapping(model, add_legacy=False)
                if len(conversions) == 0:
                    self.skipTest("No conversion found for this model")

                # Find the model keys, so the targets according to the conversions
                model_keys = list(model.state_dict().keys())

                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Serialize with reverse mapping
                    model.save_pretrained(tmpdirname)
                    state_dict = load_file(os.path.join(tmpdirname, "model.safetensors"))
                    # Get all the serialized keys that we just saved according to the reverse mapping
                    serialized_keys = list(state_dict.keys())

                if check_keys_were_modified:
                    # They should be different, otherwise we did not perform any mapping
                    self.assertNotEqual(sorted(serialized_keys), sorted(model_keys), "No key mapping was performed!")

                # Check that for each conversion entry, we at least map to one key
                for conversion in conversions:
                    for source_pattern in conversion.source_patterns:
                        # Sometimes the mappings specify keys that are tied, so absent from the saved state dict
                        if isinstance(conversion, WeightRenaming):
                            # We need to revert the target pattern to make it compatible with regex search
                            target_pattern_reversed = conversion.target_patterns[0]
                            captured_group = process_target_pattern(source_pattern)[1]
                            if captured_group:
                                target_pattern_reversed = target_pattern_reversed.replace(r"\1", captured_group)
                            if any(re.search(target_pattern_reversed, k) for k in model.all_tied_weights_keys.keys()):
                                continue
                        num_matches = sum(re.search(source_pattern, key) is not None for key in serialized_keys)

                        # Key change: special case to load causal lm within vlm
                        if source_pattern == "^model.language_model":
                            continue

                        self.assertTrue(
                            num_matches > 0,
                            f"`{source_pattern}` in `{conversion}` did not match any of the source keys. "
                            "This indicates whether that the pattern is not properly written, ot that it could not be reversed correctly",
                        )

                # If everything is still good at this point, let's test that we perform the same operations both when
                # reverting ops from `from_pretrained` and from `__init__`
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # The model was instantiated from __init__ before being saved
                    model.save_pretrained(tmpdirname)
                    state_dict_saved_from_init = load_file(os.path.join(tmpdirname, "model.safetensors"))

                    # Now reload it
                    model_reloaded = model_class.from_pretrained(tmpdirname)

                    # Make sure both loaded state_dict are identical
                    self.assertTrue(compare_state_dicts(model_reloaded.state_dict(), model.state_dict()))

                    # The model was instantiated from `from_pretrained` before being saved
                    model_reloaded.save_pretrained(tmpdirname)
                    state_dict_saved_from_pretrained = load_file(os.path.join(tmpdirname, "model.safetensors"))

                    # Make sure both saved state_dict are identical
                    self.assertTrue(compare_state_dicts(state_dict_saved_from_init, state_dict_saved_from_pretrained))

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


class Qwen3_5MoeVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=16,
        text_config={
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "head_dim": 8,
            "hidden_size": 32,
            "vocab_size": 99,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "model_type": "qwen3_vl",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "linear_attention"],
            "num_key_value_heads": 2,
            "rope_theta": 10000,
            "tie_word_embeddings": True,
            "rope_parameters": {"rope_type": "default", "mrope_section": [16, 8, 8], "mrope_interleaved": True},
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 36,
            "num_experts_per_tok": 2,
            "num_experts": 8,
        },
        vision_config={
            "depth": 2,
            "in_chans": 3,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "hidden_size": 32,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        tie_word_embeddings=True,
        is_training=True,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.is_training = is_training

        self.vision_config = vision_config
        self.text_config = text_config

        self.vocab_size = text_config["vocab_size"]
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.head_dim = text_config["head_dim"]
        self.hidden_size = text_config["hidden_size"]
        self.intermediate_size = text_config["intermediate_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_key_value_heads = text_config["num_key_value_heads"]
        self.rope_theta = text_config["rope_theta"]
        self.rope_parameters = text_config["rope_parameters"]
        self.hidden_act = text_config["hidden_act"]
        self.max_position_embeddings = text_config["max_position_embeddings"]
        self.model_type = text_config["model_type"]

        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return Qwen3_5MoeConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Qwen3_5MoeModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen3_5MoeForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Qwen3_5MoeModel,
            Qwen3_5MoeForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = Qwen3_5MoeVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3_5MoeConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        "Qwen3.5 Moe has a special Cache as it alternates with gated deltanet layers"
        self.assertIsInstance(past_key_values, Qwen3_5MoeDynamicCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        expected_shape = (batch_size, num_heads, seq_length, head_dim)

        attention_layer_indices = past_key_values.transformer_layers
        self.assertListEqual(
            [past_key_values.key_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )
        self.assertListEqual(
            [past_key_values.value_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )

    def _check_caches_are_equal(self, cache1, cache2):
        "Qwen3.5 Moe has a special Cache as it alternates with gated deltanet layers"
        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            if cache1.key_cache[idx] is not None:
                torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
                torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3.5 Moe alternates between attention layers and gated deltanet layers."
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
            self.assertEqual(
                len(attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.text_config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(
                len(attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )
            self.assertListEqual(
                list(attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )
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
            self.assertEqual(
                len(self_attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass
