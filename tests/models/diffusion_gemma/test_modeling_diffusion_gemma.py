# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch DiffusionGemma model."""

import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import CaptureStdout, cleanup, require_torch, slow, tooslow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoProcessor,
        AutoTokenizer,
        Cache,
        DiffusionGemmaConfig,
        DiffusionGemmaForBlockDiffusion,
        DiffusionGemmaGenerationConfig,
        DiffusionGemmaModel,
        DynamicCache,
        EntropyBoundSamplerConfig,
        StaticCache,
        TextDiffusionStreamer,
        set_seed,
    )


class DiffusionGemmaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        text_config={
            # default text config test values
            "vocab_size": 99,
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 32,
            "hidden_act": "gelu",
            "max_position_embeddings": 512,
            "initializer_range": 0.02,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "moe_intermediate_size": 8,
            "moe_num_shared_experts": 1,
            "num_experts": 4,
            "num_hidden_layers": 2,
            # model-specific text config values
            "layer_types": ["sliding_attention", "full_attention"],  # we want to test both types
            "num_global_key_value_heads": 2,  # key introduced by the gemma4 family
            "global_head_dim": 32 // 2,  # hidden_size // num_attention_heads
            "top_k_experts": 2,  # key introduced by the gemma4 family
            "use_bidirectional_attention": "vision",  # Test if bidirectional image mask path works
        },
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 16,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        mm_tokens_per_image=2,
        image_token_id=4,
        boi_token_id=5,
        eoi_token_id=6,
        seq_length=25,
        self_conditioning_size=16,
        canvas_length=16,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        # `image_token_id` is set to 0 to pass "resize_embeddings" test, do not modify
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_id = image_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.is_training = is_training
        self.self_conditioning_size = self_conditioning_size
        self.canvas_length = canvas_length

        self.pad_token_id = text_config["pad_token_id"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]

        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]

    def get_config(self):
        return DiffusionGemmaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            mm_tokens_per_image=self.mm_tokens_per_image,
            self_conditioning_size=self.self_conditioning_size,
            canvas_length=self.canvas_length,
        )

    def prepare_config_and_inputs_for_common(self):
        # 1. config preparation
        config = self.get_config()
        config.vision_config.pooling_kernel_size = 2

        # 2. image inputs preparation
        # (num_images, max_num_patches, patch_size * patch_size * num_channels)
        patch_size = config.vision_config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["image_size"],
                patch_size * patch_size * self.vision_config["num_channels"],
            ]
        )
        # (num_images, max_num_patches, 2) for height/width positions. Let it be all ones for testign
        pixel_position_ids = torch.ones(self.vision_config["image_size"], device=torch_device, dtype=torch.long)
        pixel_position_ids = pixel_position_ids[None, :, None].repeat(self.batch_size, 1, 2)

        # 3. text input preparation
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        decoder_input_ids = ids_tensor([self.batch_size, self.canvas_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Ensure no tokens accidentally match special token IDs
        input_ids[input_ids == config.image_token_id] = self.pad_token_id
        input_ids[:, :1] = config.image_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == config.image_token_id] = 1

        # 4. build the input dict
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_position_ids": pixel_position_ids,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return config, inputs_dict


@require_torch
class DiffusionGemmaVisionText2TextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (DiffusionGemmaModel, DiffusionGemmaForBlockDiffusion) if is_torch_available() else ()
    all_generative_model_classes = ()  # No class inherits `GenerationMixin`
    additional_model_inputs = ["mm_token_type_ids", "decoder_input_ids"]

    test_torch_exportable = False  # This model always returns cache -> export test fails in `_get_leaf_tensors`

    def setUp(self):
        self.model_tester = DiffusionGemmaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DiffusionGemmaConfig, has_text_modality=False)

    # Tests adapted from `CausalLMModelTest` (when appropriate, as this is not an AR model). We can use the test
    # name to check the original equivalent that served as inspiration.
    def test_config(self):
        """Tests basic config properties"""
        self.config_tester.run_common_tests()

    def test_model(self):
        """Tests that we can run a forward pass with the base model"""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaModel(config=config).to(torch_device).eval()
        expected_shape = (self.model_tester.batch_size, self.model_tester.canvas_length, self.model_tester.hidden_size)

        result = model(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])
        self.assertEqual(result.last_hidden_state.shape, expected_shape)

        result = model(inputs_dict["input_ids"], decoder_input_ids=inputs_dict["decoder_input_ids"])
        self.assertEqual(result.last_hidden_state.shape, expected_shape)

        result = model(inputs_dict["input_ids"])
        self.assertEqual(result.last_hidden_state.shape, expected_shape)

    # Tests overwritten from `ModelTesterMixin`.
    @unittest.skip(reason="TODO")
    def test_attention_outputs(self):
        # TODO(joaogante): Write a custom test for this model. The base test can't be used because
        # the decoder has query len = canvas len and key/value shape = kv cache + canvas len.
        # On top of this, we should also check the encoder attentions.
        pass

    @unittest.skip(reason="TODO")
    def test_hidden_states_output(self):
        # TODO(joaogante): Same as `test_attention_outputs`
        pass

    @unittest.skip(reason="TODO")
    def test_can_init_all_missing_weights(self):
        # TODO(joaogante): failing on tied weights. Explore later.
        pass

    @unittest.skip(reason="TODO")
    def test_init_weights_can_init_buffers(self):
        # TODO(joaogante): probably related to `test_can_init_all_missing_weights`
        pass

    @unittest.skip(reason="TODO")
    def test_tp_plan_matches_params(self):
        # TODO(joaogante): explore TP config after other issues are sorted
        pass

    @unittest.skip(reason="Hard to specify `self.model_split_percents` due to tied weights. Skip for now.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Hard to specify `self.model_split_percents` due to tied weights. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Hard to specify `self.model_split_percents` due to tied weights. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass

    # Tests designed specifically for `DiffusionGemma`, excluding generation tests.
    def test_tied_weights(self):
        """
        Tests that randomly initialized models have their weights properly tied.

        There are two sets of ties in DiffusionGemma:
        1. The text encoder trainable params are the same as the decoder trainable params, if we exclude the
           decoder self-conditioning weights
        2. The LM Head weights are the same as the embedding weights -- in both the text encoder and in the decoder!
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # 1. text encoder trainable params == decoder trainable params (excluding decoder self-conditioning)
        # In other words, all text encoder trainable params must be decoder trainable params.
        decoder_trainable_params = dict(model.model.decoder.named_parameters())
        for name, encoder_text_param in model.model.encoder.language_model.named_parameters():
            decoder_trainable_param = decoder_trainable_params[name]
            self.assertTrue(encoder_text_param is decoder_trainable_param)

        # 2. lm head weights == text encoder embedding weights == decoder embedding weights
        self.assertTrue(model.lm_head.weight is model.model.decoder.embed_tokens.weight)
        self.assertTrue(model.lm_head.weight is model.model.encoder.language_model.embed_tokens.weight)

    def test_use_cache_raises_exception(self):
        """
        DiffusionGemma always use cache. Therefore, the common kwarg `use_cache` isn't used -- and we raise an
        exception
        """
        config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        with self.assertRaises(ValueError):
            model(**model_inputs, use_cache=False)
        with self.assertRaises(ValueError):
            model(**model_inputs, use_cache=True)

    def test_diffusion_decoder_mask_no_cache_raises_exception(self):
        """
        DiffusionGemma has a custom function to create an attention mask for the decoder. Contrarily
        to other mask creation functions, it requires a KV cache
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        dummy_canvas = torch.ones((2, 4), dtype=torch.int32, device=torch_device)
        with self.assertRaises(ValueError):
            _ = model.model.decoder.create_diffusion_decoder_attention_mask(
                config=config.text_config,
                inputs_embeds=dummy_canvas.unsqueeze(-1),
                past_key_values=None,  # this will trigger an exception
            )

    def test_diffusion_decoder_mask_dynamic_cache(self):
        """
        This is the simplest test: the resulting mask should be full of 1s. As a shortcut, in eager execution
        + dynamic cache, it simply returns `None` attention masks
        """

        prefill_length = 8
        canvas_length = 4
        concat_kv_length = prefill_length + canvas_length
        batch_size = 2

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # Apply prefill (smaller than sliding window)
        past_key_values = DynamicCache(config=config)
        prefill_input_ids = torch.ones((batch_size, prefill_length), dtype=torch.int32, device=torch_device) * 50
        past_key_values = model.model.encoder(prefill_input_ids, past_key_values=past_key_values).past_key_values

        # Get the mask
        decoder_attention_mask = torch.ones((batch_size, concat_kv_length), dtype=torch.bool, device=torch_device)
        dummy_canvas = torch.ones((batch_size, canvas_length), dtype=torch.int32, device=torch_device)
        mask_mapping = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=config.text_config,
            inputs_embeds=dummy_canvas.unsqueeze(-1),
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
        )
        self.assertIsNone(mask_mapping["full_attention"])
        self.assertIsNone(mask_mapping["sliding_attention"])

        # We get the same result if we pass `attention_mask = None`
        mask_mapping = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=config.text_config,
            inputs_embeds=dummy_canvas.unsqueeze(-1),
            past_key_values=past_key_values,
            decoder_attention_mask=None,
        )
        self.assertIsNone(mask_mapping["full_attention"])
        self.assertIsNone(mask_mapping["sliding_attention"])

    def test_diffusion_decoder_mask_dynamic_cache_left_padding(self):
        """
        This test is similar to `test_diffusion_decoder_mask_dynamic_cache`, but with left-padding on one of the
        rows. There should be a few zeros on the final mask, corresponding to the left-padded items.
        """

        prefill_length = 8
        canvas_length = 4
        concat_kv_length = prefill_length + canvas_length
        batch_size = 2
        left_padding_length = 2  # only applied on batch item 0
        expected_non_zero = (
            ((concat_kv_length - left_padding_length) * canvas_length)  # batch item 0
            + (concat_kv_length * canvas_length)  # batch item 1
        )
        expected_attention_mask_shape = (batch_size, 1, canvas_length, concat_kv_length)

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # Apply prefill (smaller than sliding window)
        past_key_values = DynamicCache(config=config)
        prefill_input_ids = torch.ones((batch_size, prefill_length), dtype=torch.int32, device=torch_device) * 50
        past_key_values = model.model.encoder(prefill_input_ids, past_key_values=past_key_values).past_key_values

        # Get the mask
        decoder_attention_mask = torch.ones((batch_size, concat_kv_length), dtype=torch.bool, device=torch_device)
        decoder_attention_mask[0, :left_padding_length] = 0
        dummy_canvas = torch.ones((batch_size, canvas_length), dtype=torch.int32, device=torch_device)
        mask_mapping = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=config.text_config,
            inputs_embeds=dummy_canvas.unsqueeze(-1),
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
        )
        self.assertEqual(mask_mapping["full_attention"].shape, expected_attention_mask_shape)
        self.assertEqual(mask_mapping["full_attention"].sum(), expected_non_zero)
        self.assertEqual(mask_mapping["full_attention"][0, 0, :, :left_padding_length].sum(), 0)
        self.assertEqual(mask_mapping["sliding_attention"].shape, expected_attention_mask_shape)
        self.assertEqual(mask_mapping["sliding_attention"].sum(), expected_non_zero)
        self.assertEqual(mask_mapping["sliding_attention"][0, 0, :, :left_padding_length].sum(), 0)

    def test_diffusion_decoder_mask_dynamic_cache_beyond_sliding_window(self):
        """
        This tests builds upon `test_diffusion_decoder_mask_dynamic_cache_left_padding`: it tests the case with
        left-padding AND going beyond the sliding window. The prefill length is longer than the sliding window
        length. The short left-padding should become invisible in the sliding window layers, after prefill.
        """

        prefill_length = 16
        sliding_window_length = 8
        canvas_length = 4
        concat_kv_length_full = prefill_length + canvas_length
        # -1 -> the DYNAMIC sliding window kv cache has len=window-1
        concat_kv_length_sliding = sliding_window_length + canvas_length - 1
        batch_size = 2
        left_padding_length = 2  # only applied on batch item 0
        expected_non_zero_full = (
            ((concat_kv_length_full - left_padding_length) * canvas_length)  # batch item 0
            + (concat_kv_length_full * canvas_length)  # batch item 1
        )
        expected_attention_mask_shape_full = (batch_size, 1, canvas_length, concat_kv_length_full)
        expected_non_zero_sliding = concat_kv_length_sliding * batch_size * canvas_length
        expected_attention_mask_shape_sliding = (batch_size, 1, canvas_length, concat_kv_length_sliding)
        # Double-check test assumption that left-padding should be past the sliding window
        self.assertTrue(prefill_length - left_padding_length > sliding_window_length)

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.sliding_window = sliding_window_length
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # Apply prefill (LARGER than sliding window)
        past_key_values = DynamicCache(config=config)
        prefill_input_ids = torch.ones((batch_size, prefill_length), dtype=torch.int32, device=torch_device) * 50
        past_key_values = model.model.encoder(prefill_input_ids, past_key_values=past_key_values).past_key_values

        # Get the mask
        decoder_attention_mask = torch.ones((batch_size, concat_kv_length_full), dtype=torch.bool, device=torch_device)
        decoder_attention_mask[0, :left_padding_length] = 0
        dummy_canvas = torch.ones((batch_size, canvas_length), dtype=torch.int32, device=torch_device)
        mask_mapping = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=config.text_config,
            inputs_embeds=dummy_canvas.unsqueeze(-1),
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
        )
        self.assertEqual(mask_mapping["full_attention"].shape, expected_attention_mask_shape_full)
        self.assertEqual(mask_mapping["full_attention"].sum(), expected_non_zero_full)
        self.assertEqual(mask_mapping["full_attention"][0, 0, :, :left_padding_length].sum(), 0)
        self.assertEqual(mask_mapping["sliding_attention"].shape, expected_attention_mask_shape_sliding)
        self.assertEqual(mask_mapping["sliding_attention"].sum(), expected_non_zero_sliding)
        # sliding window got beyond padding, so the first tokens in the sliding window will not be masked
        self.assertEqual(
            mask_mapping["sliding_attention"][0, 0, :, :left_padding_length].sum(), left_padding_length * canvas_length
        )

    def test_diffusion_decoder_mask_static_cache(self):
        """
        Same as `test_diffusion_decoder_mask_dynamic_cache`, but with a Static Cache. Contrarily to the
        original test, the mask is materialized (a materialized mask is needed at compile time)
        """

        prefill_length = 8
        canvas_length = 4
        static_cache_length = 16
        concat_kv_length = static_cache_length + canvas_length
        batch_size = 2
        expected_non_zero = (prefill_length + canvas_length) * canvas_length * batch_size
        expected_attention_mask_shape = (batch_size, 1, canvas_length, concat_kv_length)

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # Apply prefill (smaller than sliding window)
        past_key_values = StaticCache(config=config, max_cache_len=static_cache_length)
        prefill_input_ids = torch.ones((batch_size, prefill_length), dtype=torch.int32, device=torch_device) * 50
        past_key_values = model.model.encoder(prefill_input_ids, past_key_values=past_key_values).past_key_values

        # Get the mask (correctly designed for the static cache)
        decoder_attention_mask = torch.ones((batch_size, concat_kv_length), dtype=torch.bool, device=torch_device)
        decoder_attention_mask[:, prefill_length:static_cache_length] = 0  # unfilled KV cache values
        dummy_canvas = torch.ones((batch_size, canvas_length), dtype=torch.int32, device=torch_device)
        mask_mapping = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=config.text_config,
            inputs_embeds=dummy_canvas.unsqueeze(-1),
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
        )
        self.assertEqual(mask_mapping["full_attention"].shape, expected_attention_mask_shape)
        self.assertEqual(mask_mapping["full_attention"].sum(), expected_non_zero)
        self.assertEqual(mask_mapping["sliding_attention"].shape, expected_attention_mask_shape)
        self.assertEqual(mask_mapping["sliding_attention"].sum(), expected_non_zero)

    def test_diffusion_decoder_mask_static_cache_bad_attention_mask(self):
        """
        Same as `test_diffusion_decoder_mask_static_cache`, but assuming the user forgot to set to 0
        the attention mask entries corresponding to unfilled cache positions. It will raise an exception
        """

        prefill_length = 8
        canvas_length = 4
        static_cache_length = 16
        concat_kv_length = static_cache_length + canvas_length
        batch_size = 2

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # Apply prefill (smaller than sliding window)
        past_key_values = StaticCache(config=config, max_cache_len=static_cache_length)
        prefill_input_ids = torch.ones((batch_size, prefill_length), dtype=torch.int32, device=torch_device) * 50
        past_key_values = model.model.encoder(prefill_input_ids, past_key_values=past_key_values).past_key_values

        # Get the mask (INCORRECTLY designed for the static cache)
        decoder_attention_mask = torch.ones((batch_size, concat_kv_length), dtype=torch.bool, device=torch_device)
        dummy_canvas = torch.ones((batch_size, canvas_length), dtype=torch.int32, device=torch_device)
        with self.assertRaises(ValueError):
            _ = model.model.decoder.create_diffusion_decoder_attention_mask(
                config=config.text_config,
                inputs_embeds=dummy_canvas.unsqueeze(-1),
                past_key_values=past_key_values,
                decoder_attention_mask=decoder_attention_mask,
            )

    def test_diffusion_decoder_mask_static_cache_beyond_sliding_window(self):
        """
        Same as `test_diffusion_decoder_mask_dynamic_cache_beyond_sliding_window`, but with a Static Cache.
        This is the most complex mask preparation test, including static caches, left-padding, and mask preparation
        beyond the sliding window length.
        """
        prefill_length = 16
        sliding_window_length = 8
        canvas_length = 4
        static_cache_length = 32
        concat_kv_length_full = static_cache_length + canvas_length
        # No -1 -> the STATIC sliding window kv cache has len=window
        concat_kv_length_sliding = sliding_window_length + canvas_length
        batch_size = 2
        left_padding_length = 2  # only applied on batch item 0
        expected_non_zero_full = (
            ((prefill_length + canvas_length - left_padding_length) * canvas_length)  # batch item 0
            + ((prefill_length + canvas_length) * canvas_length)  # batch item 1
        )
        expected_attention_mask_shape_full = (batch_size, 1, canvas_length, concat_kv_length_full)
        expected_non_zero_sliding = concat_kv_length_sliding * batch_size * canvas_length
        expected_attention_mask_shape_sliding = (batch_size, 1, canvas_length, concat_kv_length_sliding)
        # Double-check test assumption that left-padding should be past the sliding window
        self.assertTrue(prefill_length - left_padding_length > sliding_window_length)

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.sliding_window = sliding_window_length
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # Apply prefill (LARGER than sliding window)
        past_key_values = StaticCache(config=config, max_cache_len=static_cache_length)
        prefill_input_ids = torch.ones((batch_size, prefill_length), dtype=torch.int32, device=torch_device) * 50
        past_key_values = model.model.encoder(prefill_input_ids, past_key_values=past_key_values).past_key_values

        # Get the mask (correctly designed for the static cache)
        decoder_attention_mask = torch.ones((batch_size, concat_kv_length_full), dtype=torch.bool, device=torch_device)
        decoder_attention_mask[0, :left_padding_length] = 0
        decoder_attention_mask[:, prefill_length:static_cache_length] = 0  # unfilled KV cache values
        dummy_canvas = torch.ones((batch_size, canvas_length), dtype=torch.int32, device=torch_device)
        mask_mapping = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=config.text_config,
            inputs_embeds=dummy_canvas.unsqueeze(-1),
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
        )
        self.assertEqual(mask_mapping["full_attention"].shape, expected_attention_mask_shape_full)
        self.assertEqual(mask_mapping["full_attention"].sum(), expected_non_zero_full)
        self.assertEqual(mask_mapping["full_attention"][0, 0, :, :left_padding_length].sum(), 0)
        self.assertEqual(mask_mapping["sliding_attention"].shape, expected_attention_mask_shape_sliding)
        self.assertEqual(mask_mapping["sliding_attention"].sum(), expected_non_zero_sliding)
        # sliding window got beyond padding, so the first tokens in the sliding window will not be masked
        self.assertEqual(
            mask_mapping["sliding_attention"][0, 0, :, :left_padding_length].sum(), left_padding_length * canvas_length
        )

    # Generation tests
    @parameterized.expand(
        [
            # (sub_test_name, {generate_kwargs}),
            # Cache interface
            ("dynamic_cache", {}),  # default cache
            ("static_cache", {"cache_implementation": "static"}),  # static cache -> triggers compilation
            # Diffusion-specific common kwargs
            ("custom_sampler", {"sampler_config": EntropyBoundSamplerConfig(0.1)}),
            (
                "custom_temperature",
                {"t_min": 0.1, "t_max": 0.2},
            ),
            (
                "custom_stopping",
                {"stability_threshold": 0, "confidence_threshold": 1e-1},
            ),
        ]
    )
    def test_generate(self, name, generate_kwargs):
        """Tests `generate` calls with common flags"""
        config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()
        model.generation_config.eos_token_id = None  # force generation up to `max_new_tokens`

        generation_outputs = model.generate(
            **model_inputs, max_new_tokens=16, max_denoising_steps=2, **generate_kwargs
        )

        expected_shape = (model_inputs["input_ids"].shape[0], model_inputs["input_ids"].shape[1] + 16)
        self.assertEqual(generation_outputs.shape, expected_shape)

    def test_generate_text_only(self):
        """Same as `test_generate`, but only with text inputs (no images)"""
        config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()
        model.generation_config.eos_token_id = None  # force generation up to `max_new_tokens`

        generation_outputs = model.generate(model_inputs["input_ids"], max_new_tokens=16, max_denoising_steps=2)

        expected_shape = (model_inputs["input_ids"].shape[0], model_inputs["input_ids"].shape[1] + 16)
        self.assertEqual(generation_outputs.shape, expected_shape)

    def test_generate_from_generation_config(self):
        """
        Same as the base case in `test_generate`, but we parameterize the generation call with a
        `DiffusionGemmaGenerationConfig`
        """
        config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # force `eos_token_id = None` to not stop before reaching `max_new_tokens`
        generation_config = DiffusionGemmaGenerationConfig(eos_token_id=None, max_new_tokens=16, max_denoising_steps=2)
        generation_outputs = model.generate(**model_inputs, generation_config=generation_config)

        expected_shape = (model_inputs["input_ids"].shape[0], model_inputs["input_ids"].shape[1] + 16)
        self.assertEqual(generation_outputs.shape, expected_shape)

    def test_generate_kwarg_overrides(self):
        """
        Same as `test_generate_from_generation_config`, but we override some parameters with kwargs in the
        `generate` call. Remember: kwargs > generation config
        """
        config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # force `eos_token_id = None` to not stop before reaching `max_new_tokens`
        generation_config = DiffusionGemmaGenerationConfig(eos_token_id=None, max_new_tokens=16,
                                                           max_denoising_steps=2, return_dict_in_generate=True)
        generation_outputs = model.generate(**model_inputs, generation_config=generation_config, max_new_tokens=32)

        expected_shape = (model_inputs["input_ids"].shape[0], model_inputs["input_ids"].shape[1] + 32)
        self.assertEqual(generation_outputs.sequences.shape, expected_shape)

    def test_generate_with_past_key_values(self):
        """
        Tests that we can pass in past_key_values outputted from a previous `generate` into a subsequent call.
        This pattern is used in chat sessions.
        """
        config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()
        model.generation_config.eos_token_id = None  # force generation up to `max_new_tokens`
        model.generation_config.return_dict_in_generate = True

        # 1st generate call, without KV cache
        generation_1_outputs = model.generate(**model_inputs, max_new_tokens=16, max_denoising_steps=2)
        self.assertIsInstance(generation_1_outputs.past_key_values, Cache)

        expected_shape = (model_inputs["input_ids"].shape[0], model_inputs["input_ids"].shape[1] + 16)
        self.assertEqual(generation_1_outputs.sequences.shape, expected_shape)

        # 2nd generate call, with KV cache from previous call
        # NOTE: `input_ids` in `generate` can only contain **unprocessed** tokens
        new_input_ids = torch.cat((generation_1_outputs.sequences, model_inputs["input_ids"]), dim=-1)
        generation_2_outputs = model.generate(
            input_ids=new_input_ids,
            past_key_values=generation_1_outputs.past_key_values,
            max_new_tokens=16,
            max_diffusion_steps=2,
        )

        # in total, we feed in `model_inputs["input_ids"]` twice, and get 16 new tokens twice
        expected_shape = (model_inputs["input_ids"].shape[0], (model_inputs["input_ids"].shape[1] + 16) * 2)
        self.assertEqual(generation_2_outputs.sequences.shape, expected_shape)

    @parameterized.expand(
        [
            ("dynamic_cache", None),  # default cache
            ("static_cache", "static"),  # static cache -> triggers compilation under the hood
        ]
    )
    @slow
    def test_generate_beyond_sliding_window(self, name, cache_implementation):
        """Tests that generate can run beyond the sliding window length"""
        config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.sliding_window = 16
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()
        model.generation_config.eos_token_id = None  # force generation up to `max_new_tokens`

        generation_outputs = model.generate(
            **model_inputs, max_new_tokens=32, max_denoising_steps=2, cache_implementation=cache_implementation
        )
        self.assertTrue(generation_outputs.shape[1] > config.text_config.sliding_window)

        expected_shape = (model_inputs["input_ids"].shape[0], model_inputs["input_ids"].shape[1] + 32)
        self.assertEqual(generation_outputs.shape, expected_shape)

    @unittest.skip(reason="TODO(joaogante): red in CI, fix me")
    def test_diffusion_streaming(self):
        """Tests `TextDiffusionStreamer`"""
        # TODO(huggingface team): after a tiny diffusion model is created, this test can be moved into
        # its right place, i.e. tests/generation/test_streamers.py

        config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = model_inputs["input_ids"][:1, :]  # streaming requires bsz=1
        model = DiffusionGemmaForBlockDiffusion(config=config).to(torch_device).eval()

        # Any tokenizer will do, as long as it has the same vocab size.
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        with CaptureStdout() as cs:
            streamer = TextDiffusionStreamer(tokenizer)
            model_out = model.generate(input_ids, max_new_tokens=16, max_denoising_steps=2, streamer=streamer)
        streamer_text = cs.out
        generate_text = tokenizer.decode(model_out[0])

        # the streamer ends with a `\n`
        self.assertTrue(streamer_text.endswith("\n"))
        # Important for this test: max_new_tokens==canvas_length. The last piece of text thrown by the streamer
        # is the finalized canvas.
        self.assertEqual(streamer_text[-(len(generate_text) + 1) : -1], generate_text)
        # `\x1b7` = save cursor position; `\x1b8` = restore cursor position;
        # Used to overwrite draft tokens while streaming.
        self.assertIn("\x1b7", streamer_text)
        self.assertIn("\x1b8", streamer_text)


@require_torch
class DiffusionGemmaIntegrationTest(unittest.TestCase):
    _model_path = "google/diffusiongemma-26B-A4B-it"

    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_diffusion_gemma_chat_template(self):
        # fmt: off
        expected_input_ids = [
            [
                2,    105,   2364,    107,   6974,    496,   1440,  20494,   1003,
                23613, 236761,    106,    107,    105,   4368,    107,    100,  45518,
                107,    101
            ]
        ]
        # fmt: on
        expected_decoded_tokens = [
            "<bos><|turn>user\nWrite a long essay about Portugal.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        ]

        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            {"role": "user", "content": "Write a long essay about Portugal."},
        ]
        input_ids = processor.apply_chat_template(chat, tokenize=True, return_tensors="pt", add_generation_prompt=True)
        self.assertEqual(input_ids.tolist(), expected_input_ids)

        decoded_tokens = processor.decode(input_ids)
        self.assertIn("<|turn>", decoded_tokens[0])
        self.assertIn("<turn|>", decoded_tokens[0])
        self.assertIn("<|channel>", decoded_tokens[0])
        self.assertIn("<channel|>", decoded_tokens[0])
        self.assertNotIn("<|think|>", decoded_tokens[0])  # shouldn't have the think token
        self.assertEqual(decoded_tokens, expected_decoded_tokens)

    @slow
    def test_diffusion_gemma_chat_template_image(self):
        image_tokens = [255999, 258880, 258882]  # These tokens must be present in the `input_ids`
        # TODO(joao): this should be 280! Something is wrong with processing?
        image_token_count = 256

        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
                    },
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }
        ]
        model_inputs = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True
        )
        for token in image_tokens:
            self.assertIn(token, model_inputs["input_ids"])
        self.assertTrue((model_inputs["input_ids"] == 258880).sum() == image_token_count)

        for expected_model_input in ("attention_mask", "pixel_values", "image_position_ids", "mm_token_type_ids"):
            self.assertIn(expected_model_input, model_inputs)

    @slow
    def test_diffusion_gemma_chat_template_with_thinking(self):
        # fmt: off
        expected_input_ids = [
            [
                2,    105,   9731,    107,     98,    107,    106,    107,    105,
                2364,    107,   6974,    496,   1440,  20494,   1003,  23613, 236761,
                106,    107,    105,   4368,    107
            ]
        ]
        # fmt: on
        expected_decoded_tokens = [
            "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nWrite a long essay about Portugal.<turn|>\n<|turn>model\n"
        ]

        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            {"role": "user", "content": "Write a long essay about Portugal."},
        ]
        input_ids = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, enable_thinking=True
        )
        self.assertEqual(input_ids.tolist(), expected_input_ids)

        decoded_tokens = processor.decode(input_ids)
        self.assertIn("<|turn>", decoded_tokens[0])
        self.assertIn("<turn|>", decoded_tokens[0])
        self.assertIn("<|think|>", decoded_tokens[0])  # must have the think token
        self.assertEqual(decoded_tokens, expected_decoded_tokens)

    # Only 1 checkpoint got released: the 26B MoE. At release time, MoE quantization was not working properly,
    # and the HF CI didn't have 80GB GPUs. As such, tests against the original model are tagged with `@tooslow`.
    # Meaning: these tests won't run in the `py.test` command, and you have to uncomment the line. These tests
    # exist to keep track of the model's original numerics.
    def _load_model(self, minified=False):
        if minified:
            config = AutoConfig.from_pretrained(self._model_path)
            config.text_config.num_hidden_layers = 6  # loaded model requires ~12.7 GB
            config.text_config.layer_types = config.text_config.layer_types[:6]
            model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                self._model_path,
                config=config,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                self._model_path,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        return model

    @tooslow
    def test_diffusion_gemma_forward_text_only(self):
        """bsz=1, no image"""
        # fmt: off
        # print(model_out.logits[:, :10, :12])
        # Printed after running `torch.set_printoptions(precision=8, threshold=200, linewidth=100)`
        expected_logits = [
            [
                [ 4.12986183, 20.66659164,  6.75712347,  4.16051483, 13.11566544,  3.59206009,  3.46879172,
                3.86895919,  3.14465618,  2.47871161,  4.43598938,  4.77164030],
                [ 3.20645595, 22.52679825,  7.78647614,  2.71133161, 14.73279667,  4.55818033,  4.16051483,
                3.03644228,  3.91504526,  1.58446193,  4.52764702,  4.46655130],
                [ 4.77164030, 23.55075455,  5.80103159,  3.65364885, 15.66178513,  4.19115925,  5.01498747,
                2.92814803,  4.55818033,  2.92814803,  5.28794241,  7.87383986],
                [ 4.64972258, 24.27903175,  6.01130199,  3.66904116, 15.57065392,  3.17555952,  5.25765753,
                3.40711212,  4.95421267,  4.28303957,  5.71073294,  8.10610390],
                [ 5.16673803, 24.27903175,  6.42998266,  3.36083388, 16.89584923,  4.49710369,  5.77094412,
                4.40541792,  5.22736216,  4.12986183,  7.05306482,  8.68214703],
                [ 5.28794241, 24.36471176,  7.58207989,  3.53044105, 17.31759644,  4.49710369,  5.31821585,
                4.40541792,  4.98460531,  3.88432312,  7.46494198,  8.33732700],
                [ 5.10607004, 24.27903175,  7.17103910,  3.34540391, 16.55123901,  3.96111226,  4.09919930,
                4.25242186,  4.34424734,  4.46655130,  5.71073294,  8.27962112],
                [ 6.81642532, 24.19219208,  8.56747913,  5.37872887, 17.15010071,  5.80103159,  5.86117029,
                4.25242186,  6.34047985,  4.49710369,  7.61132526,  9.36425304],
                [ 6.99399042, 24.44923782,  8.96760273,  5.34847832, 16.81029892,  5.77094412,  4.95421267,
                4.49710369,  5.68060923,  4.68021679,  6.93485880,  8.85363007],
                [ 7.93200207, 23.83295822,  7.81561375,  6.51936626, 17.72932053,  5.98130083,  5.83110666,
                5.43919706,  6.60862780,  5.28794241,  7.66977119,  9.86893463]
            ]
        ]
        # fmt: on

        model = self._load_model()
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            {"role": "user", "content": "Write a long essay about Portugal."},
        ]
        model_inputs = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)

        # hardcodes input canvas to avoid depending on random canvas sampling
        decoder_input_ids = torch.arange(model.config.canvas_length, dtype=torch.int32, device=model.device).unsqueeze(
            0
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids

        model_out = model(**model_inputs)
        self.assertEqual(
            list(model_out.logits.shape), [1, model.config.canvas_length, model.config.text_config.vocab_size]
        )
        torch.testing.assert_close(model_out.logits[:, :10, :12].cpu(), torch.tensor(expected_logits))

    @tooslow
    def test_diffusion_gemma_forward_with_image(self):
        """bsz=1, with image"""
        # fmt: off
        # print(model_out.logits[:, :10, :12])
        # Printed after running `torch.set_printoptions(precision=8, threshold=200, linewidth=100)`
        expected_logits = [
            [
                [ 6.93485880, 20.60072327,  3.89968514,  7.49424887,  6.99399042, -0.03344725,  0.44332713,
                -0.53900456,  2.18363142,  0.35545215,  1.24927723,  3.53044105],
                [ 6.81642532, 19.92110252,  4.92381001,  5.74084425, 10.31235313,  4.83253860,  0.94890219,
                1.05425322,  0.87475199,  0.16503741,  0.06933582,  2.74232340],
                [ 6.40016174, 22.63516045,  2.88171268,  6.75712347, 12.08642769,  5.25765753,  2.60281467,
                2.60281467,  5.55999660,  2.66483307,  2.05925679,  5.98130083],
                [ 9.86893463, 21.78874207,  2.35452294,  7.78647614, 15.01560497,  7.11208105,  5.07571983,
                5.80103159,  4.68021679,  4.86297274,  3.45337439,  7.78647614],
                [ 5.01498747, 21.18004990,  1.31166339,  4.95421267, 11.13002872,  2.54077387,  0.40427244,
                2.40110350,  0.42770541,  1.23367894, -0.29100651,  1.06205606],
                [ 5.74084425, 21.42788887,  2.12145352,  4.77164030, 11.82355690,  3.12920213,  3.66904116,
                3.43795562,  2.64933062,  2.88171268,  2.71133161,  4.89339685],
                [ 6.28074551, 22.30597687,  4.16051483,  4.52764702, 12.08642769,  4.64972258,  4.31364775,
                5.89122152,  1.52212954,  3.74597287,  2.13699985,  6.31061935],
                [ 6.37032795, 23.92457008,  9.02448273,  3.28366685, 10.64161205,  3.22190189,  3.00550938,
                4.83253860,  0.88255781,  1.63899159,  1.81808197,  5.49961996],
                [ 7.46494198, 25.78687286,  9.36425304,  3.63825440, 13.06506538,  2.02815175,  3.23734570,
                5.37872887,  3.02097654,  2.27686334,  2.16808867,  6.63835430],
                [ 6.16112089, 26.16215897,  5.71073294,  3.34540391, 11.92896748,  1.55329728,  4.77164030,
                4.12986183,  1.02304065,  3.49961996,  1.73244548,  6.45979071]
           ]
        ]
        # fmt: on

        model = self._load_model()
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
                    },
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }
        ]
        model_inputs = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)

        # hardcodes input canvas to avoid depending on random canvas sampling
        decoder_input_ids = torch.arange(model.config.canvas_length, dtype=torch.int32, device=model.device).unsqueeze(
            0
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids

        # sanity check: has image inputs
        self.assertIn("pixel_values", model_inputs)

        model_out = model(**model_inputs)
        self.assertEqual(
            list(model_out.logits.shape), [1, model.config.canvas_length, model.config.text_config.vocab_size]
        )
        torch.testing.assert_close(model_out.logits[:, :10, :12].cpu(), torch.tensor(expected_logits))

    @tooslow
    def test_diffusion_gemma_forward_batched(self):
        """bsz>1, no image"""
        # fmt: off
        # print(model_out.logits[:, :5, :12])
        # Printed after running `torch.set_printoptions(precision=8, threshold=200, linewidth=100)`
        expected_logits = [
            [
                [-1.56108880e+00,  1.99211025e+01,  1.60004282e+00,  1.01073846e-01,  7.58207989e+00,
                -5.98130083e+00, -4.40541792e+00, -1.73244548e+00, -1.42860627e+00, -5.97577214e-01,
                -1.81808197e+00,  8.74021053e-02],
                [-1.02304065e+00,  2.26888294e+01,  9.53315258e+00, -3.37876350e-01,  8.91065121e+00,
                1.29394531e-02,  5.35099506e-01, -2.52526045e+00, -7.91013837e-02, -2.02815175e+00,
                2.27686334e+00,  8.86460662e-01],
                [-2.33393744e-01,  2.34054356e+01,  5.04535866e+00, -4.94095922e-01,  1.10760899e+01,
                1.23367894e+00,  9.56706762e-01, -1.94259071e+00,  1.51433694e+00,  1.03864717e+00,
                2.61832142e+00,  5.65047359e+00],
                [-1.21808004e+00,  2.25267982e+01,  4.43598938e+00, -1.23367894e+00,  8.62484741e+00,
                6.79571211e-01,  7.10804522e-01, -2.02815175e+00, -4.82380331e-01,  7.57651389e-01,
                2.01259756e+00,  4.92381001e+00],
                [ 4.56995904e-01,  2.34541931e+01,  4.09919930e+00, -5.39550222e-02,  1.30143728e+01,
                7.14708507e-01,  1.21028030e+00,  5.11669159e-01,  1.81029797e+00,  1.63120234e+00,
                4.68021679e+00,  7.72815514e+00]
            ],
            [
                [ 1.15722090e-01,  2.06007233e+01,  3.12920213e+00, -2.12145352e+00,  5.37872887e+00,
                -3.46879172e+00, -5.80103159e+00, -4.12986183e+00, -2.47871161e+00, -5.52981424e+00,
                -5.01498747e+00, -8.16204786e-01],
                [ 2.88171268e+00,  2.27421646e+01,  8.62484741e+00,  5.11669159e-01,  8.51004219e+00,
                2.41662765e+00,  5.89767814e-01,  2.88171268e+00, -5.19479334e-01, -2.66483307e+00,
                -2.52526045e+00,  2.33899355e+00],
                [ 5.37872887e+00,  2.53779068e+01,  8.96760273e+00,  1.32725811e+00,  1.13449488e+01,
                1.28826988e+00, -6.79571211e-01,  3.45337439e+00,  3.66904116e+00, -2.52526045e+00,
                -1.18688023e+00,  2.61832142e+00],
                [ 5.04535866e+00,  2.42790318e+01,  8.68214703e+00,  4.43598938e+00,  9.47692776e+00,
                3.78886133e-01, -4.61921787e+00,  6.28814161e-01,  4.99953747e-01, -3.69981956e+00,
                -6.49413019e-02,  3.63825440e+00],
                [ 6.13118267e+00,  2.50096397e+01,  8.68214703e+00,  4.52764702e+00,  1.17707224e+01,
                -9.52148531e-03, -3.34540391e+00,  2.16808867e+00,  3.08282948e+00, -3.32997227e+00,
                -6.32718682e-01,  3.26822829e+00]
            ]
        ]
        # fmt: on

        model = self._load_model()
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            [
                {"role": "user", "content": "Write a long essay about Portugal."},
            ],
            [
                {"role": "user", "content": "Why is 6 affraid of 7?"},
            ],
        ]
        model_inputs = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)

        # hardcodes input canvas to avoid depending on random canvas sampling
        decoder_input_ids = torch.arange(model.config.canvas_length, dtype=torch.int32, device=model.device)
        decoder_input_ids = (
            torch.ones((2, model.config.canvas_length), dtype=torch.int32, device=model.device) * decoder_input_ids
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids

        # sanity check: the input has bsz=2
        self.assertEqual(model_inputs["input_ids"].shape[0], 2)
        # sanity check: has padding, and it is left-padding
        self.assertIn(model.config.text_config.pad_token_id, model_inputs["input_ids"])
        self.assertFalse((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, -1]).any())
        self.assertTrue((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, 0]).any())

        model_out = model(**model_inputs)
        self.assertEqual(
            list(model_out.logits.shape), [2, model.config.canvas_length, model.config.text_config.vocab_size]
        )
        torch.testing.assert_close(model_out.logits[:, :5, :12].cpu(), torch.tensor(expected_logits))

    @tooslow
    def test_diffusion_gemma_generate_with_image_batched(self):
        """
        bsz>1, with image, calling `generate`.

        This test has a LOT of non-determinism (sampling canvases, sampling tokens), so it is normal
        if different platforms need to set different expected outcomes.
        """
        # fmt: off
        # print(generated_tokens)
        # Printed after running `torch.set_printoptions(threshold=10000, linewidth=100)`
        expected_sequences = [
            [
                1018,    818,  21065,  41025, 236787,    562,  61774,  77985,    529,  23613,   1018,
                108,   5141,   3059,    580,    506,  16425,   3374,   7377,    529,  49566,   3879,
                236764,   1298,    506, 211050,   2601,   6861,  15741,    531,    506,  12529,  21065,
                18414, 236764,  23613,    563,    496,   7097,   5221,    684,   1061,   4191,    607,
                506,   5442, 236761,   1701,  24744, 236764,    625,    815,   7779,    618,    496,
                44701,   1534,    506,  10687,   4109,    532,    506,   1799, 236764,    496,  11825,
                1534,  20117,    532,  70548, 236761,   2282,   3050,  23613,    563,    531,   3050,
                496,   3996, 105362,    529,  49094,  27877, 236764,  44040,  11858,  24637, 236764,
                532,    496,   4709,   9226,  78729,   3224,    618,    808,   4834,    689,   1007,
                236829, 237028, 236746,   5268, 236764,  85539,   7833,  86508,    573,   2613,   5745,
                653,   8229,   2752,   1765, 236761,    108,  10354,    562,  34641,  84181, 236787,
                4934, 117838,    576,   4848,    531,  41254,    108,    818,   4083,    529,  23613,
                563,    886,    529,  35691,    532,  10534, 236761,  42251,    618,    496,  21880,
                528,    506, 236743, 236770, 236778,    594,   7691,   1913,    506, 117838,    576,
                4848,    699,  78763,   1044,   6157, 236764,  23613,   5452,    886,    529,    506,
                24625,   7097, 236772,  26582,    528,   3879, 236761,   5978,   1061,  23789,    964,
                3187, 204045,    528,   6145,  77486,  42469, 236764,  23613,   6976,  44630, 236761,
                236743,    108,    818, 236743, 236770, 236810,    594,    532, 236743, 236770, 236825,
                594,  24744,  11373,   1061,  16522,  17884, 236764,    506,  17884,    529,  41254,
                236761,   8382,    506,  91295,    529,   9732,   1133,  18934,  12297,    506,  60137,
                236764,  41082,   2556,  41516,  19482,   6998,    506,    623,  46811,    529, 106819,
                2098,  15468,    506,   9992,   9383,    532,  17154,   4673, 236761, 125620,   1776,
                150225,  11788,    506,   5442,   9116,    531,   4673,    528, 236743, 236770, 236812,
                236819, 236828, 236764
            ],
            [
                2094,   2471,   3831,    506,  16522,  26010,  17936,    528,   5054,  14322, 236764,
                7151, 236761,    669,   4429,    563,   3523,    699,    506,  15891,   2678, 236764,
                18482,   9975,  10654,    528,    506,  44529,    532,    506,  69480,  10901,   6615,
                528,    506,   1695, 236761,    106,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0
            ]
        ]
        # fmt: on

        model = self._load_model()
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            [
                {"role": "user", "content": "Write a long essay about Portugal."},
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
                        },
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                }
            ],
        ]
        model_inputs = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)

        # sanity check: the input has bsz=2
        self.assertEqual(model_inputs["input_ids"].shape[0], 2)
        # sanity check: the input has an image component
        self.assertIn("pixel_values", model_inputs)
        # sanity check: has padding, and it is left-padding
        self.assertIn(model.config.text_config.pad_token_id, model_inputs["input_ids"])
        self.assertFalse((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, -1]).any())
        self.assertTrue((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, 0]).any())

        set_seed(42)
        gen_out = model.generate(**model_inputs, max_new_tokens=256)
        generated_tokens = gen_out[:, -256:]
        self.assertEqual(generated_tokens.tolist(), expected_sequences)

    @tooslow
    def test_diffusion_gemma_generate_with_image_batched_long(self):
        """
        Ultimate test: bsz>1, with image, with thinking on, calling `generate` with
        max_new_tokens > sliding attention length.

        This test has a LOT of non-determinism (sampling canvases, sampling tokens), so it is normal
        if different platforms need to set different expected outcomes.
        """
        # fmt: off
        # print(generated_tokens)
        # Printed after running `torch.set_printoptions(threshold=10000, linewidth=100)`
        expected_sequences = [
            [
                21042,   2846, 236764,  23613,    691,    506,   3988,    529,    506,   1902, 236764,
                614,  38613,    600,  39349,    699,  14600,    531, 126382, 236764,    532,  73420,
                531,  72449, 236761,   1174,   6933,   6111,  36628,  12821,    532,   9226,   8770,
                236764, 175676,    528,    506,  94079,  37704,    688,  31035,   3117, 236764,  17202,
                684,   1061,  82647,  59491,   1133,  70186, 236764,  35809, 236764,    532,  83798,
                47397, 236761,  16961, 236764,    506,   3825,    529,  38613,  10734,   5378,    531,
                16670, 236764,   6641,    684,   1440,  13443,    529,   6658,  23049,    532,    506,
                56845,    808,  66856, 107130, 236829, 102708,   1208, 187920,    569,  81237, 118882,
                236761,   1030,    691,    711,   3097,    506,  33275,    567,  24817,    529, 236743,
                236770, 236819, 236832, 236812, 237028, 236746,  73887, 236764,    569, 236772,  18377,
                236748,   4806,   1933,   7820,  28622,   1298,  11838,   7006,  37067,    847,    528,
                506,    520, 204532,    529,  18187, 236789,  79689, 237028,   7705,  23613, 130661,
                1131,    506,  28239, 236764,  29810,   6693,   8927, 236772,  14531,   7097,    625,
                563,   3124, 236761,    108,   8551,  69565, 236764,  23613,    563,  33887,  12801,
                9785,   1061,  30997,   2425, 236761,    669,   4695,    563,  52803,    532,   3826,
                236764,   2033,    531,    506, 176165,  15706,    530,  11849, 236764,    506,   1902,
                236858, 236751,  24625, 115368,    774,  10135,   4128, 236764,   1298,   6944,   3776,
                81026, 103534,    531,  25465,  26607,   1949, 236761,    669,   3988,    563,  27222,
                684,    506,   1429,   3194,   5400, 236764,    496,  12529, 236764,  13935, 152990,
                529,  26012, 154863,    532,  64192,  32049,  24599,   1298,   1972,  12489,    657,
                496,  31878,  17723, 236761,   2282,    506,   8710, 236764,    506, 201172,   5469,
                20997,  62260, 236764,  11497,   5442,    505,  21294, 236764,    532,    496,   6962,
                32920,  10022, 236761,  43725,    506,   2891, 236764,    506,  21065,  18414,   7474,
                506,   4512,  61102
            ],
            [
                563,  88076,    580,   1903,    529,    672,   5441, 236761,    108,    902,    506,
                1695, 236764,   3418,    506,   1813, 236764,    506,  19519, 236764, 127150, 236772,
                30497,  26607,    529,    506,  69480,  10901,   6615,    659,  11325,   1208,    506,
                6425,   1933,   7217, 236761,    669,  15408,    563,   7804,    532,   1982, 236764,
                22169,    496,  17635,   1719, 236764,  30439,   3538,  37676,    580,    506,  31035,
                4065,    529,    506,  11825,    532,    506,   7845, 236761,    106,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                0,      0,      0
            ]
        ]
        # fmt: on

        model = self._load_model()
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            [
                {"role": "user", "content": "Write a long essay about Portugal."},
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
                        },
                        {"type": "text", "text": "Describe the image in detail."},
                    ],
                }
            ],
        ]
        model_inputs = processor.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=True,
        ).to(model.device)

        # sanity check: the input has bsz=2
        self.assertEqual(model_inputs["input_ids"].shape[0], 2)
        # sanity check: the input has an image component
        self.assertIn("pixel_values", model_inputs)
        # sanity check: has padding, and it is left-padding
        self.assertIn(model.config.text_config.pad_token_id, model_inputs["input_ids"])
        self.assertFalse((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, -1]).any())
        self.assertTrue((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, 0]).any())

        set_seed(42)
        gen_out = model.generate(**model_inputs, max_new_tokens=1280)

        # sanity check: we went beyond the sliding window length
        self.assertTrue(gen_out.shape[1] > model.config.text_config.sliding_window)
        generated_tokens = gen_out[:, -256:]
        self.assertEqual(generated_tokens.tolist(), expected_sequences)

    # The tests below use a botched model, using the first six layers, to respect memory constraints of HF's CI.
    # Six layers = minimum number of layers to have both sliding window attention and full attention.
    @slow
    def test_minified_diffusion_gemma_forward_text_only(self):
        """[Minified (6 first layers)] bsz=1, no image"""
        # fmt: off
        # print(model_out.logits[:, :10, :12])
        # Printed after running `torch.set_printoptions(precision=8, threshold=200, linewidth=100)`
        expected_logits = [
            [
                [ 21.36647987, -29.92865181,  -3.86895919,  29.14220428,  29.96334648,  28.18346024,
                27.68740082,   0.78888065, -27.24345016, -29.00568390,  -1.03864717, -18.21017647],
                [ 29.49378586,  17.56583405,  26.10184860,  29.87437057,  29.99975395, -17.89120674,
                -29.92623520, -19.99079323, -14.92172623, -28.82887459, -29.78616714, -29.99817467],
                [ -2.33899355, -29.99688721, -29.98895645,  26.50611115,  13.36728001, -24.69602203,
                -4.52764702, -18.28892136, -29.44104195, -29.94533730, -25.58675003, -29.89002419],
                [ -1.58446193, -29.44104195,  16.28858566,  26.71912193,  29.87012291,  14.35028839,
                21.96432686, -10.20196629, -29.24793625, -28.15386963,   2.15254474, -23.83295822],
                [ 28.06227112, -28.93834877,  26.96818924,  29.90688133,  29.90992928,  18.28892136,
                27.83115768,   4.12986183, -24.69602203, -28.92082787,  24.53262520, -22.30597687],
                [  4.16051483, -29.89710426,   5.43919706,  28.68712234,  29.92115784, -14.39843941,
                21.30470467, -18.67665863, -29.31869316, -29.09888268,   5.71073294, -27.49481583],
                [ 22.07961273, -25.51811981, -23.30696678,  27.15444756,  27.53452492,  -9.42062664,
                    4.03784895, -16.98100090, -29.70199203, -25.37790680,  24.69602203,   1.42081141],
                [ 20.79719925, -29.45920181,  17.89120674,  29.67082214,  29.83608246,  -9.25128651,
                26.82098389, -13.16617393, -29.42227936, -29.45920181,   4.74117613, -22.19349098],
                [  7.99010134, -29.54159164,  14.82745647,  29.19678497,  29.38286209, -16.63799286,
                28.66555023, -12.55404091, -28.37808609, -29.71171379,  -6.81642532, -18.28892136],
                [ 18.13103485, -29.98779488,  29.29586983,  29.34078217,  29.71171379, -25.00963974,
                    7.75732422, -28.03077507, -29.52617455, -29.92865181, -25.51811981, -26.77043152]
            ]
        ]
        # fmt: on

        model = self._load_model(minified=True)
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            {"role": "user", "content": "Write a long essay about Portugal."},
        ]
        model_inputs = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)

        # hardcodes input canvas to avoid depending on random canvas sampling
        decoder_input_ids = torch.arange(model.config.canvas_length, dtype=torch.int32, device=model.device).unsqueeze(
            0
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids

        model_out = model(**model_inputs)
        self.assertEqual(
            list(model_out.logits.shape), [1, model.config.canvas_length, model.config.text_config.vocab_size]
        )
        torch.testing.assert_close(model_out.logits[:, :10, :12].cpu(), torch.tensor(expected_logits))

    @slow
    def test_minified_diffusion_gemma_forward_with_image(self):
        """[Minified (6 first layers)] bsz=1, with image"""
        # fmt: off
        # print(model_out.logits[:, :10, :12])
        # Printed after running `torch.set_printoptions(precision=8, threshold=200, linewidth=100)`
        expected_logits = [
            [
                [ 18.28892136, -29.96683311, -26.33790970,  27.61213493,  29.29586983,  24.27903175,
                5.71073294,   7.81561375, -19.12884521, -22.41707611,  -0.74984390, -25.00963974],
                [ 28.40418053,  26.91985321,  28.92082787,  29.64826202,  29.99992561,  -9.75730991,
                -29.90047264, -25.23368835,  -0.45113790, -29.24793625, -29.90992928, -29.99850464],
                [ -2.30793095, -29.59842873, -29.96082115,  24.77605438,   5.46941423, -25.37790680,
                -18.36726570, -27.32979774, -28.93834877, -29.95215416, -27.86572838, -29.98458862],
                [ -6.51936626, -26.45086479, -25.65441132,  26.39480209,  29.91287804,  -2.77330947,
                4.31364775,  -4.00716066, -27.99878311,  -9.25128651, -10.96796322, -25.58675003],
                [ 28.72925377, -29.79991150,  28.88492966,  29.91287804,  29.98179245,   9.47692776,
                27.61213493,  -9.36425304, -13.41732597, -26.71912193,  28.55237961, -22.30597687],
                [ -2.05925679, -29.81889534,  -6.60862780,  25.85169601,  29.65972710, -19.27641296,
                4.19115925, -21.66990089, -29.54159164, -28.12381363,  -4.34424734, -28.48003769],
                [-20.33346367, -27.68740082, -27.79604530,  14.92172623,  -7.37692833, -26.39480209,
                -18.21017647, -25.00963974, -29.68155670,  -4.03784895,  25.91559219,  -8.96760273],
                [  3.17555952, -29.27229309,  11.92896748,  28.57573891,  26.91985321, -23.92457008,
                3.17555952, -23.74012756, -29.74754906, -29.63641167,  -5.80103159, -26.71912193],
                [-13.81431580, -29.06881332,   8.91065121,  26.96818924,  26.16215897, -22.95215607,
                14.35028839, -15.75251865, -28.77006149, -29.31869316,  -9.98025513, -27.37199974],
                [  3.08282948, -29.98509216,  28.68712234,  28.80958176,  26.71912193, -28.45515060,
                -16.11148643, -28.50453377, -29.19678497, -29.77148056, -26.87078667, -28.45515060]
          ]
        ]
        # fmt: on

        model = self._load_model(minified=True)
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
                    },
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }
        ]
        model_inputs = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)

        # hardcodes input canvas to avoid depending on random canvas sampling
        decoder_input_ids = torch.arange(model.config.canvas_length, dtype=torch.int32, device=model.device).unsqueeze(
            0
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids

        # sanity check: has image inputs
        self.assertIn("pixel_values", model_inputs)

        model_out = model(**model_inputs)
        self.assertEqual(
            list(model_out.logits.shape), [1, model.config.canvas_length, model.config.text_config.vocab_size]
        )
        torch.testing.assert_close(model_out.logits[:, :10, :12].cpu(), torch.tensor(expected_logits))

    @slow
    def test_minified_diffusion_gemma_forward_batched(self):
        """[Minified (6 first layers)] bsz>1, no image"""
        # fmt: off
        # print(model_out.logits[:, :5, :12])
        # Printed after running `torch.set_printoptions(precision=8, threshold=200, linewidth=100)`
        expected_logits = [
            [
                [ 20.26569748, -29.95372391,  -4.92381001,  28.84785843,  29.96454811,  27.89977074,
                28.26949692,   4.58870411, -27.49481583, -29.11355782,   0.93719494, -22.47210884],
                [ 29.55650711,  -9.42062664,  24.77605438,  29.89710426,  29.99937248, -20.99029922,
                -29.93754959, -22.41707611, -17.89120674, -28.80958176, -29.81277657, -29.99840164],
                [ -0.89426625, -29.99745178, -29.98966789,  27.10892868,  14.49445248, -24.61488152,
                -7.14156723, -18.59990692, -29.57094383, -29.93098831, -26.22159958, -29.91572952],
                [ -0.71080452, -29.27229309,  16.98100090,  26.82098389,  29.84144211,  13.56690025,
                21.72950172, -11.45190430, -29.42227936, -27.65005684,   2.43215036, -24.36471176],
                [ 27.96628952, -28.93834877,  27.28695297,  29.90373039,  29.91287804,  18.21017647,
                27.89977074,   3.56125450, -25.00963974, -28.92082787,  24.36471176, -22.41707611]
            ],
            [
                [ 20.66659164, -29.81889534, -14.15672493,  29.40289116,  29.85164261,  27.53452492,
                22.95215607,   9.86893463, -28.29728889, -28.09328270,  -1.10106778, -25.16004562],
                [ 29.15618324,  20.40085030,  20.33346367,  29.89710426,  29.99983406, -17.89120674,
                -29.87012291, -14.82745647, -18.13103485, -28.48003769, -29.71171379, -29.99777031],
                [ 10.96796322, -29.97776413, -29.97542572,  28.88492966,  18.67665863, -19.99079323,
                    4.34424734, -12.08642769, -29.54159164, -29.87848091, -22.90015602, -29.73022270],
                [-10.91377544, -28.50453377,  15.93279934,  27.01581001,  29.71171379,  16.02234268,
                16.55123901, -15.20219231, -29.49378586, -27.86572838,  -3.29910398, -23.92457008],
                [ 25.44850922, -29.51024628,  27.53452492,  29.89710426,  29.63641167,  13.26691246,
                24.10417366,   5.28794241, -27.10892868, -28.45515060,  19.63847351, -24.27903175]
           ]
        ]
        # fmt: on

        model = self._load_model(minified=True)
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            [
                {"role": "user", "content": "Write a long essay about Portugal."},
            ],
            [
                {"role": "user", "content": "Why is 6 affraid of 7?"},
            ],
        ]
        model_inputs = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)

        # hardcodes input canvas to avoid depending on random canvas sampling
        decoder_input_ids = torch.arange(model.config.canvas_length, dtype=torch.int32, device=model.device)
        decoder_input_ids = (
            torch.ones((2, model.config.canvas_length), dtype=torch.int32, device=model.device) * decoder_input_ids
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids

        # sanity check: the input has bsz=2
        self.assertEqual(model_inputs["input_ids"].shape[0], 2)
        # sanity check: has padding, and it is left-padding
        self.assertIn(model.config.text_config.pad_token_id, model_inputs["input_ids"])
        self.assertFalse((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, -1]).any())
        self.assertTrue((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, 0]).any())

        model_out = model(**model_inputs)
        self.assertEqual(
            list(model_out.logits.shape), [2, model.config.canvas_length, model.config.text_config.vocab_size]
        )
        torch.testing.assert_close(model_out.logits[:, :5, :12].cpu(), torch.tensor(expected_logits))

    @slow
    def test_minified_diffusion_gemma_generate_with_image_batched(self):
        """
        [Minified (6 first layers)] bsz>1, with image, calling `generate`.

        This test has a LOT of non-determinism (sampling canvases, sampling tokens), so it is normal
        if different platforms need to set different expected outcomes.
        """
        # fmt: off
        # print(generated_tokens)
        # Printed after running `torch.set_printoptions(threshold=10000, linewidth=100)`
        expected_sequences = [
            [
                48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,
                371, 371,  48,  48, 371, 371, 371,  48, 371, 379, 379,  48, 595, 371, 371, 516,  48, 371,
                48, 371, 516, 595,  48, 957,  48, 371, 371, 595, 516, 379, 371,  48, 371, 371,  48, 516,
                595, 371, 371,  48, 371, 549,  48, 173,  48, 371, 595, 371, 658, 371, 983,  48, 516, 595,
                48, 516, 516, 516, 595, 595, 371, 371, 595, 832, 379, 371, 728, 371, 516, 371, 516, 516,
                371,  48, 371,  48, 371, 371, 498, 371, 371,  48,  48,  48, 371,  48,  48,  48, 595,  48,
                48,  48, 516,  48, 371,  48, 379,  48, 516,  48,  48,  48, 371,  48,  48,  48,  48,  48,
                48, 595,  48, 841, 173,  48,  48,  48,  48,  48,  48,  48, 516,  48,  48,  48, 405,  48,
                48, 371,  48,  48, 371,  48,  48,  48,  48,  48,  48,  48, 173,  48, 371,  48,  48,  48,
                48,  48, 516, 371,  48, 516,  48,  48, 516,  48, 841, 371,  48,  48,  48, 516, 595, 371,
                48,  48, 516, 371,  48,  48, 371,  48, 516, 371, 371,  48,  48,  48,  48, 371, 595,  48,
                48,  48,  48,  48,  48,  48,  48,  48, 371, 516,  48,  48,  48,  48,  48, 379, 371,  48,
                371,  48,  48,  48,  48, 371,  48,  48,  48, 371, 371,  48,  48,  48,  48, 516, 371,  48,
                48, 172, 516, 371, 516,  48, 371,  48,  48,  48, 173,  48, 173, 371,  48,  48,  48,  48,
                379,  48,  48,  48
            ],
            [
                5, 530, 595, 371,  48, 371,  48, 371,  48, 371,  48,  48, 595, 371, 859,  48, 371, 595,
                379, 371, 371, 371,  48, 371, 595, 595,  48, 595, 841, 379,  48, 379, 634,  48,  48, 371,
                48, 371, 841,  48,  48,  48,  48, 379,  48, 379,  48, 379, 371, 371, 379,  48,  48,  48,
                48,  48,  48,  48, 832, 371, 371, 371,  48,  48, 371,  48, 371,  48, 371, 371,  48, 595,
                595, 654, 516, 516,  48, 371,  48,  48,  48, 595, 371, 371, 517,  48,  48, 832, 371, 983,
                48,  48,  48, 841,  48,  48,  48,  48,  48,  48,  48, 841,  48,  48,  48,  48,  48,  48,
                48,  48,  48,  48, 371,  48,  48,  48,  48,  48,  48,  48,  48, 841,  48,  48,  48,  48,
                48,  48,  48,  48,  48,  48,  48,  48,  48,  48, 500, 379,  48,  48,  48,  48,  48,  48,
                48,  48, 595,  48,  48,  48,  48, 371,  48, 379, 371,  48,  48, 371,  48,  48, 832, 371,
                48, 371, 371,  48,  48,  48,  48,  48,  48, 371,  48,  48, 174,  48,  48, 841,  48,  48,
                48,  48,  48,  48, 371,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,
                48,  48, 371,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48, 371,  48, 841,
                371,  48, 371,  48,  48,  48,  48,  48,  48,  48,  48, 841,  48,  48,  48,  48,  48,  48,
                516, 371,  48,  48,  48,  48, 832, 841,  48,  48,  48,  48, 841,  48,  48,  48,  48,  48,
                48,  48,  48, 371
            ]
        ]
        # fmt: on

        model = self._load_model(minified=True)
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            [
                {"role": "user", "content": "Write a long essay about Portugal."},
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
                        },
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                }
            ],
        ]
        model_inputs = processor.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)

        # sanity check: the input has bsz=2
        self.assertEqual(model_inputs["input_ids"].shape[0], 2)
        # sanity check: the input has an image component
        self.assertIn("pixel_values", model_inputs)
        # sanity check: has padding, and it is left-padding
        self.assertIn(model.config.text_config.pad_token_id, model_inputs["input_ids"])
        self.assertFalse((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, -1]).any())
        self.assertTrue((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, 0]).any())

        set_seed(42)
        gen_out = model.generate(**model_inputs, max_new_tokens=256)
        generated_tokens = gen_out[:, -256:]
        self.assertEqual(generated_tokens.tolist(), expected_sequences)

    @slow
    def test_minified_diffusion_gemma_generate_with_image_batched_long(self):
        """
        [Minified (6 first layers)] Ultimate test: bsz>1, with image, with thinking on, calling `generate` with
        max_new_tokens > sliding attention length.

        This test has a LOT of non-determinism (sampling canvases, sampling tokens), so it is normal
        if different platforms need to set different expected outcomes.
        """
        # fmt: off
        # print(generated_tokens)
        # Printed after running `torch.set_printoptions(threshold=10000, linewidth=100)`
        expected_sequences = [
            [
                48,  48,  48,  48,  48,  48, 379, 379, 379,  48, 379, 379, 405, 379,  48, 379, 379, 379,
                379, 379, 595, 379,  48, 379,  48, 379,  48, 379, 379, 379,  48,  48,  48, 516, 379,  48,
                48,  48, 379, 379, 379,  48, 379, 379, 379,  48,  48, 379,  48,  48,  48,  48,  48,  48,
                48,  48, 379, 379,  48,  48, 379, 379,  48, 379,  48,  48,  48, 379,  48,  48,  48,  48,
                379,  48,  48,  48,  48, 516,  48,  48, 379, 379,  48, 379, 379,  48, 379,  48, 379,  48,
                371,  48, 379,  48,  48,  48,  48,  48,  48, 418,  48, 379, 379,  48,  48, 379, 516,  48,
                379,  48,  48,  48,  48, 379, 379, 379, 379,  48,  48,  48, 379, 379, 379, 516, 379,  48,
                379,  48, 379,  48,  48, 405,  48,  48,  48,  48, 379, 379,  48,  48,  48,  48, 379,  48,
                379,  48, 379,  48,  48, 379,  48,  48, 516, 379,  48,  48,  48,  48,  48, 379,  48, 379,
                379,  48,  48,  48, 418, 379, 379, 379,  48, 516,  48,  48,  48, 379,  48,  48, 379, 516,
                48,  48,  48, 379, 379, 379, 516,  48,  48,  48,  48, 379, 379, 516, 379,  48,  48, 379,
                48,  48,  48,  48, 379,  48,  48,  48,  48,  48,  48,  48, 841, 379,  48, 379, 499,  48,
                48,  48,  48,  48, 379,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48, 405, 379,  48,
                48,  48,  48,  48,  48,  48,  48, 379,  48, 379,  48, 379,  48, 379,  48,  48,  48,  48,
                48,  48,  48,  48
            ],
            [
                48,  48,  48,  48,  48,  48, 379, 379, 379, 379, 379, 516, 379,  48,  48, 379,  48, 379,
                48, 379, 379, 379, 977, 379, 379, 379,  48,  48, 379, 379, 379, 379,  48,  48, 379, 379,
                379, 379,  48,  48, 379,  48, 379,  48,  48,  48,  48,  48, 379, 516, 379,  48,  48, 379,
                517,  48,  48, 379, 379,  48, 379,  48,  48, 516, 516, 379,  48,  48, 379, 379,  48, 379,
                379,  48, 379, 379, 516, 379,  48,  48, 379,  48, 379,  48, 379,  48,  48,  48,  48, 379,
                379, 418,  48, 379, 379, 379,  48, 379,  48, 379, 379, 379, 379, 379, 379, 379,  48, 379,
                371,  48,  48,  48, 379,  48,  48, 379,  48,  48, 379,  48, 379,  48,  48, 379, 379, 379,
                48,  48, 499,  48,  48, 379,  48,  48,  48,  48,  48,  48,  48,  48,  48, 379,  48,  48,
                379, 371,  48, 379,  48, 379, 405,  48,  48, 516,  48, 379, 379,  48,  48, 379, 379, 379,
                379, 379,  48, 379, 379, 379, 379,  48, 379, 516,  48, 379, 379,  48,  48, 379, 499, 516,
                379,  48,  48, 379, 379,  48,  48,  48,  48,  48, 379, 379,  48,  48,  48, 379, 379,  48,
                516,  48,  48,  48,  48, 379,  48, 379,  48,  48, 379, 379,  48,  48,  48,  48, 379,  48,
                517,  48, 371,  48,  48,  48, 379,  48,  48,  48, 379, 379,  48,  48, 379, 379,  48, 379,
                48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48,  48, 379,  48,  48,  48,  48,  48,
                48, 379,  48,  48
            ]
        ]
        # fmt: on

        model = self._load_model(minified=True)
        processor = AutoProcessor.from_pretrained(self._model_path)
        chat = [
            [
                {"role": "user", "content": "Write a long essay about Portugal."},
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
                        },
                        {"type": "text", "text": "Describe the image in detail."},
                    ],
                }
            ],
        ]
        model_inputs = processor.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=True,
        ).to(model.device)

        # sanity check: the input has bsz=2
        self.assertEqual(model_inputs["input_ids"].shape[0], 2)
        # sanity check: the input has an image component
        self.assertIn("pixel_values", model_inputs)
        # sanity check: has padding, and it is left-padding
        self.assertIn(model.config.text_config.pad_token_id, model_inputs["input_ids"])
        self.assertFalse((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, -1]).any())
        self.assertTrue((model.config.text_config.pad_token_id == model_inputs["input_ids"][:, 0]).any())

        set_seed(42)
        gen_out = model.generate(**model_inputs, max_new_tokens=1280)

        # sanity check: we went beyond the sliding window length
        self.assertTrue(gen_out.shape[1] > model.config.text_config.sliding_window)
        generated_tokens = gen_out[:, -256:]
        self.assertEqual(generated_tokens.tolist(), expected_sequences)
