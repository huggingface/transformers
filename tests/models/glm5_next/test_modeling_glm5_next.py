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
"""Testing suite for the PyTorch Glm5Next model."""

import copy
import unittest

import pytest

from transformers import (
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
    Glm5NextVisionConfig,
    is_torch_available,
    logging,
)
from transformers.cache_utils import DynamicCache
from transformers.generation import CompileConfig
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.testing_utils import (
    CaptureLogger,
    require_torch,
    require_torch_accelerator,
    require_torch_greater_or_equal,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)

from ...generation.test_utils import (
    assert_similar_generate_outputs,
    is_moe_model,
)
from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Glm5NextVisionText2TextModelTester(VLMModelTester):
    base_model_class = Glm5NextModel
    config_class = Glm5NextConfig
    text_config_class = Glm5NextTextConfig
    vision_config_class = Glm5NextVisionConfig
    conditional_generation_class = Glm5NextForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("video_start_token_id", 3)
        kwargs.setdefault("video_end_token_id", 4)
        kwargs.setdefault("image_start_token_id", 5)
        kwargs.setdefault("image_end_token_id", 6)
        kwargs.setdefault("image_token_id", 7)
        kwargs.setdefault("video_token_id", 8)
        kwargs.setdefault("image_size", 112)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("projection_intermediate_size", 48 * 3)
        kwargs.setdefault("num_image_tokens", 64)
        kwargs.setdefault("seq_length", 64 + 7)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("num_experts_per_tok", 4)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("num_local_experts", 8)
        kwargs.setdefault("linear_num_heads", 2)
        kwargs.setdefault("linear_head_dim", 16)
        kwargs.setdefault("linear_conv_kernel_dim", 2)
        kwargs.setdefault("v_head_dim", 16)
        kwargs.setdefault("qk_rope_head_dim", 0)
        kwargs.setdefault("qk_nope_head_dim", 64)
        kwargs.setdefault("q_lora_rank", 32)
        kwargs.setdefault("kv_lora_rank", 16)
        kwargs.setdefault("index_head_dim", 16)
        kwargs.setdefault("index_n_heads", 2)
        kwargs.setdefault("index_topk", 48)
        kwargs.setdefault("index_kpool", 3)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("hidden_size", 48)
        kwargs.setdefault("intermediate_size", 16)
        kwargs.setdefault("mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("layer_types", ["linear_attention", "deepseek_sparse_attention"])
        super().__init__(parent, **kwargs)

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        # Clear any accidental special tokens first
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_end_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_end_token_id] = self.pad_token_id
        # Place image tokens with image start/end prefix/suffix
        input_ids[:, 0] = self.image_start_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 1 + self.num_image_tokens] = self.image_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[:, 1 : 1 + self.num_image_tokens] = 1
        patches_per_side = self.image_size // self.patch_size
        return {
            "image_grid_thw": torch.tensor(
                [[1, patches_per_side, patches_per_side]] * self.batch_size, device=torch_device
            ),
            "mm_token_type_ids": mm_token_type_ids,
        }

    def get_vision_config(self):
        return self.vision_config_class(
            depth=self.depth,
            hidden_act=self.hidden_act,
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            out_hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            projection_intermediate_size=self.projection_intermediate_size,
            patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
            temporal_patch_size=self.temporal_patch_size,
        )

    def get_config(self):
        return self.config_class(
            text_config=self.get_text_config(),
            vision_config=self.get_vision_config(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
        )


@require_torch
class Glm5NextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Glm5NextVisionText2TextModelTester
    test_all_params_have_gradient = False  # MoE
    model_split_percents = [0.5, 0.8, 0.9]
    # FIXME: export is very sensitive to any shape changes
    test_torch_exportable = False

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        """Override similar to GLM4V: images shaped as (bs*patch_len, dim) so we can't slice to batches in generate"""
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size)
        _, full_inputs = self.model_tester.prepare_config_and_inputs_for_common()

        num_patches = int(inputs_dict["image_grid_thw"].prod(-1).sum().item())
        inputs_dict["pixel_values"] = full_inputs["pixel_values"][:num_patches]

        return config, inputs_dict

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, 3 * config.linear_num_heads * config.linear_head_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.linear_num_heads, config.linear_head_dim, config.linear_head_dim)

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        """Override to account for the difference in MHC and the final state shapes"""
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (output_length - prompt_length))

        # When `output_hidden_states=True`, each iteration of generate appends the hidden states corresponding to the
        # new token(s)
        # NOTE: `StaticCache` may have different lengths on different layers, if this test starts failing add more
        # elaborate checks
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length

            # We have raw MHC shapes until the final one which is collapsed
            mhc_shape = (batch_size, model_input_length, config.hc_mult, config.hidden_size)
            final_shape = (batch_size, model_input_length, config.hidden_size)
            expected_shapes = [mhc_shape] * (len(iter_hidden_states) - 1)
            expected_shapes.append(final_shape)

            # check hidden size
            self.assertListEqual(
                [state.shape for state in iter_hidden_states],
                expected_shapes,
            )

    def test_image_and_video_placeholder_masks_are_disjoint(self):
        config = self.model_tester.get_config()
        model = Glm5NextModel(config).to(torch_device).eval()
        input_ids = torch.tensor(
            [
                [
                    config.image_token_id,
                    config.video_start_token_id,
                    config.image_token_id,
                    config.image_token_id,
                    config.video_end_token_id,
                    config.text_config.pad_token_id,
                ]
            ],
            device=torch_device,
        )
        inputs_embeds = model.get_input_embeddings()(input_ids)
        hidden_size = inputs_embeds.shape[-1]
        image_features = torch.zeros(1, hidden_size, device=torch_device)
        video_features = torch.zeros(2, hidden_size, device=torch_device)

        in_video_span = (input_ids == config.video_start_token_id).cumsum(-1) > (
            input_ids == config.video_end_token_id
        ).cumsum(-1)
        expected_image_mask = (input_ids == config.image_token_id) & ~in_video_span
        expected_video_mask = (input_ids == config.image_token_id) & in_video_span
        for ids in (input_ids, None):
            image_mask, video_mask = model.get_placeholder_mask(
                ids,
                inputs_embeds,
                image_features=image_features,
                video_features=video_features,
            )
            self.assertTrue(torch.equal(image_mask.squeeze(-1), expected_image_mask))
            self.assertTrue(torch.equal(video_mask.squeeze(-1), expected_video_mask))
            self.assertFalse(torch.logical_and(image_mask, video_mask).any())

    def test_attention_outputs(self):
        """Needs to be overwritten as GLM5 Next VL alternates between attention layers and KDA layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        config.return_dict = True
        text_config = config.get_text_config()

        # Force eager attention to support output attentions.
        text_config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True

            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            text_config = config.get_text_config()

            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.attentions
            self.assertEqual(
                len(attentions),
                sum(layer == "deepseek_sparse_attention" for layer in text_config.layer_types),
            )

            # Check that output_attentions also works through config.
            del inputs_dict["output_attentions"]
            text_config.output_attentions = True

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.attentions
            self.assertEqual(
                len(attentions),
                sum(layer == "deepseek_sparse_attention" for layer in text_config.layer_types),
            )
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [text_config.num_attention_heads, seq_len, seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine.
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
                len(self_attentions),
                sum(layer == "deepseek_sparse_attention" for layer in text_config.layer_types),
            )
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [text_config.num_attention_heads, seq_len, seq_len],
            )

    def test_hidden_states_output(self):
        """Override to account for the difference in MHC and the final state shapes"""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.output_hidden_states = True

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            text_config = model.config.get_text_config()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), text_config.num_hidden_layers + 1)

            batch_size, seq_len = inputs_dict["input_ids"].shape

            # Raw MHC shapes
            for layer_hidden_states in hidden_states[:-1]:
                self.assertEqual(
                    layer_hidden_states.shape,
                    (
                        batch_size,
                        seq_len,
                        text_config.hc_mult,
                        text_config.hidden_size,
                    ),
                )

            # Final output is standard again
            self.assertEqual(
                hidden_states[-1].shape,
                (
                    batch_size,
                    seq_len,
                    text_config.hidden_size,
                ),
            )

    def test_mismatching_num_image_tokens(self):
        """
        Overridden as flattened over patches, so slicing one row removes one patch rather than one complete image.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # Test 1: remove one image but leave the image token in text
            # Key change: Handle flattened patches properly
            image_grid_thw = curr_input_dict["image_grid_thw"][-1:, ...]
            curr_input_dict["image_grid_thw"] = image_grid_thw
            num_patches = int(image_grid_thw.prod(dim=-1).sum().item())
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-num_patches:, ...]
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = curr_input_dict["image_sizes"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 2: simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            # First, take just the first item from each tensor
            curr_input_dict = {
                key: val if key == "pixel_values" else val[:1]  # only slice pixel values
                for key, val in curr_input_dict.items()
            }

            # Double the batch size for all batch-dimension tensors except pixel_values
            # This simulates having 2 prompts (each with image tokens) but only 1 image
            batch_tensors_to_double = ["input_ids", "attention_mask", "token_type_ids"]
            for key in batch_tensors_to_double:
                if key in curr_input_dict and curr_input_dict[key] is not None:
                    curr_input_dict[key] = torch.cat([curr_input_dict[key], curr_input_dict[key]], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 3: two images and two image tokens don't raise an error
            curr_input_dict["pixel_values"] = torch.cat(
                [curr_input_dict["pixel_values"], curr_input_dict["pixel_values"]], dim=0
            )
            curr_input_dict["image_grid_thw"] = torch.cat(
                [curr_input_dict["image_grid_thw"], curr_input_dict["image_grid_thw"]], dim=0
            )
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = torch.cat(
                    [curr_input_dict["image_sizes"], curr_input_dict["image_sizes"]], dim=0
                )
            _ = model(**curr_input_dict)

    @pytest.mark.generate
    @pytest.mark.torch_compile_test
    @require_torch_greater_or_equal("2.6")  # Uses torch.compiler.set_stance
    def test_generate_compile_model_forward_fullgraph(self):
        """
        Overriden as the batch logic can not be applied to flattened patches.

        NOTE: Former GLM 4 vision models only surived this test due to lucky broadcasting
              (`adapted_pos_embed` is added --> forcing a broadcast)
        """
        for model_class in self.all_generative_model_classes:
            # 1. Test exclusion criteria
            if not model_class._can_compile_fullgraph:
                self.skipTest("This model doesn't support compilation without graph breaks")

            # 2. Prepares two sets of inputs
            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=4)
            set_config_for_less_flaky_test(config)
            model = model_class(config).to(torch_device)
            set_model_for_less_flaky_test(model)
            model.eval()  # otherwise `self.training` is `True` -- this flag is used at attn mask creation time

            # Some composite models have a custom generate and will call an inner model's generate -> that inner model
            # is the one that gets compiled.
            # (Note for the future: if BLIP starts causing problems, let's stop testing it)
            if "blip" in model.__class__.__name__.lower():
                model_to_be_compiled = model.language_model
            else:
                model_to_be_compiled = model

            # creates two sets of *different* inputs with the same shape
            main_input = inputs_dict[model.main_input_name].to(torch_device)
            half_batch_size = main_input.shape[0] // 2

            input_1 = {}
            input_2 = {}

            # Key difference: split flattened image patches using the image grids
            image_grid_thw_1 = inputs_dict["image_grid_thw"][:half_batch_size]
            image_grid_thw_2 = inputs_dict["image_grid_thw"][half_batch_size : 2 * half_batch_size]
            num_patches_1 = int(image_grid_thw_1.prod(dim=-1).sum().item())
            num_patches_2 = int(image_grid_thw_2.prod(dim=-1).sum().item())

            for key, value in inputs_dict.items():
                if not isinstance(value, torch.Tensor):
                    input_1[key] = value
                    input_2[key] = value
                elif key == "pixel_values":
                    input_1[key] = value[:num_patches_1].to(torch_device)
                    input_2[key] = value[num_patches_1 : num_patches_1 + num_patches_2].to(torch_device)
                else:
                    input_1[key] = value[:half_batch_size].to(torch_device)
                    input_2[key] = value[half_batch_size : 2 * half_batch_size].to(torch_device)

            model_input_sets = [input_1, input_2]
            self.assertTrue(
                model_input_sets[0][model.main_input_name].shape == model_input_sets[1][model.main_input_name].shape
            )

            # 3. compilation-specific setup and generation parameterization
            torch.compiler.reset()  # prevent cached compilation from being used in the test
            has_defined_cache_implementation = model.generation_config.cache_implementation is not None
            compile_config = CompileConfig(fullgraph=True, dynamic=False)  # Error out on dynamic shapes
            compile_config._compile_all_devices = True  # force compilation (e.g. fast CI, CPU)

            generation_kwargs = {
                "use_cache": True,
                "do_sample": False,
                "max_new_tokens": 5,
                "return_dict_in_generate": True,
                "output_scores": True,
                "compile_config": compile_config,
            }

            # 4. get eager + dynamic cache results for future comparison
            dynamic_outputs = []
            # Ignores all `torch.compile` usage, useful to test models that that have non-default compilable caches
            # (who would have used compilation in this section)
            with torch.compiler.set_stance("force_eager"):
                for model_inputs in model_input_sets:
                    gen_out = model.generate(**model_inputs, **generation_kwargs)
                    dynamic_outputs.append(gen_out)

                    # sanity checks for the default cache implementation
                    if not has_defined_cache_implementation:
                        decoder_cache = (
                            gen_out.past_key_values.self_attention_cache
                            if config.is_encoder_decoder
                            else gen_out.past_key_values
                        )
                        self.assertIsInstance(decoder_cache, DynamicCache)

                        # Recurrent / hybrid SSM models (mamba2, lfm2, ...) populate the default DynamicCache
                        # with statically-shaped recurrent layers, so the cache is compileable by default and
                        # auto-compile kicks in. Skip the "default cache is non-compileable" sanity check for
                        # those models — they're tested under their compileable path further down.
                        if not decoder_cache.is_compileable:
                            # our auto compile should NOT have been called
                            self.assertFalse(hasattr(model_to_be_compiled, "_compiled_call"))

            # 5. get compiled results -- relies on the automatic compilation triggered by specific compilable caches
            if not has_defined_cache_implementation:
                generation_kwargs["cache_implementation"] = "static"

            compiled_outputs = []

            # Uses a context manager to catch recompilation logs. If there is any recompilation, this test fails.
            # Try/Finally is used to ensure that the log options are reset even if an error is raised.
            try:
                torch._logging.set_logs(recompiles_verbose=True)
                logger = logging.get_logger("torch._dynamo.guards")

                with CaptureLogger(logger) as cl:
                    for model_inputs in model_input_sets:
                        # with torch.compiler.set_stance("fail_on_recompile"):
                        gen_out = model.generate(**model_inputs, **generation_kwargs)
                        compiled_outputs.append(gen_out)

                        # sanity checks
                        decoder_cache = (
                            gen_out.past_key_values.self_attention_cache
                            if config.is_encoder_decoder
                            else gen_out.past_key_values
                        )
                        self.assertNotIsInstance(decoder_cache, DynamicCache)
                        self.assertTrue(decoder_cache.is_compileable)

                        # our auto compile should have been called
                        self.assertTrue(hasattr(model_to_be_compiled, "_compiled_call"))
            finally:
                torch._logging.set_logs()

            # Compilation of sliding layers necessarily has recompiles with `dynamic=False` - however this test
            # still checks that `fullgraph=True` is supported in this case, as compilation with `dynamic=None`
            # is the default and does not actually lead to too many recompiles
            has_sliding_layers = any(decoder_cache.is_sliding)
            has_recompilation = "Recompiling" in cl.out or ("guard" in cl.out and "failure" in cl.out)
            if not has_sliding_layers and has_recompilation:
                raise RuntimeError(
                    f"`torch.compile` recompiled part of the forward pass in {model.__class__.__name__}. "
                    "See the test logs for more details."
                )

            if is_moe_model(config):
                atol = rtol = 1e-3
            else:
                atol = rtol = 1e-5

            for dynamic_result, compiled_result in zip(dynamic_outputs, compiled_outputs):
                assert_similar_generate_outputs(dynamic_result, compiled_result, atol=atol, rtol=rtol)

    @unittest.skip("Fundamentally incompatible with indexer - indexer has no boundary offset telling sequences apart")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Fundamentally incompatible with indexer - indexer has no boundary offset telling sequences apart")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("MLA creates different head dims which avoids invoking the FA backend")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@require_torch_accelerator
@slow
@unittest.skip(reason="No model weights yet, add after release")
class Glm5NextIntegrationTest(unittest.TestCase):
    pass
