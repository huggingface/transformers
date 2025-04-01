# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch cosmos model."""

import unittest

import pytest
import requests

from transformers import CosmosConfig, CosmosProcessor, CosmosTextConfig, is_torch_available, is_vision_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    require_torch_large_gpu,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ..t5.test_modeling_t5 import T5ModelTester


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

    from transformers import (
        CosmosForConditionalGeneration,
        CosmosModel,
    )
    from transformers.cache_utils import (
        DynamicCache,
        StaticCache,
    )
    from transformers.generation import (
        GenerateDecoderOnlyOutput,
        GreedySearchDecoderOnlyOutput,
        SampleDecoderOnlyOutput,
    )


class CosmosModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        is_training=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=3,
        image_size=30,
        codebook_size=20,
        temporal_downsample_factor=1,
        base_channels=32,
        vq_channel_multiplier=[1, 1],
        vq_img_token_start_id=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.temporal_downsample_factor = temporal_downsample_factor
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_img_token_start_id = vq_img_token_start_id
        self.base_channels = base_channels
        self.seq_length = 42
        self.vision_seq_length = 42

    def prepare_config_and_inputs(self):
        config = self.get_config()

        pixel_values_videos = floats_tensor(
            [
                self.batch_size,
                9,
                3,
                self.image_size,
                self.image_size,
            ]
        )
        return config, pixel_values_videos

    def get_config(self):
        text_config = CosmosTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            cross_attn_hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            rope_latent_shape=[3, 2, 3],
        )

        vq_config = {
            "codebook_size": self.codebook_size,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "base_channels": self.base_channels,
            "channel_multiplier": self.vq_channel_multiplier,
            "hidden_size": self.base_channels,
            "levels": [2, 2, 2, 2, 2, 2],
        }

        config = CosmosConfig(
            text_config=text_config,
            vq_config=vq_config,
            image_token_id=self.image_token_id,
        )
        return config

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values_videos,
        ) = config_and_inputs
        inputs_dict = {
            "pixel_values_videos": pixel_values_videos,
        }
        return config, inputs_dict


@require_torch
class CosmosModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            CosmosModel,
            CosmosForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = ()  # Cosmos generates only video as output, so we use custom tests
    _custom_generative_model_classes = (CosmosForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {}
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = CosmosModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=CosmosConfig, has_text_modality=False, common_properties=["image_token_id"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @pytest.mark.generate
    def test_greedy_generate(self):
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)
            self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length)

    @pytest.mark.generate
    def test_greedy_generate_dict_outputs(self):
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                inputs_dict=inputs_dict,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            self.assertTrue(
                output_generate.sequences.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length
            )
            self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)
            self.assertIsInstance(output_generate, GreedySearchDecoderOnlyOutput)
            self._check_generate_outputs(output_generate, model.config)

    @pytest.mark.generate
    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            output_generate = self._greedy_generate(
                model=model,
                inputs_dict=inputs_dict,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=True,  # Enable cache
            )

            self.assertTrue(
                output_generate.sequences.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length
            )
            self._check_generate_outputs(output_generate, model.config, use_cache=True)

    @pytest.mark.generate
    def test_sample_generate(self):
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).to(torch_device).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)
            self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length)

    @pytest.mark.generate
    def test_sample_generate_dict_output(self):
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            model = model_class(config).to(torch_device).eval()
            output_generate = self._sample_generate(
                model=model,
                inputs_dict=inputs_dict,
                num_return_sequences=2,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            self.assertTrue(
                output_generate.sequences.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length
            )
            self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)
            self.assertIsInstance(output_generate, SampleDecoderOnlyOutput)

            self._check_generate_outputs(output_generate, model.config, num_return_sequences=2)

    @pytest.mark.generate
    def test_beam_search_generate(self):
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).to(torch_device).eval()
            beam_kwargs = self._get_beam_kwargs()
            output_generate = self._beam_search_generate(model=model, inputs_dict=inputs_dict, beam_kwargs=beam_kwargs)
            self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length)

    @pytest.mark.generate
    def test_generate_with_static_cache(self):
        """
        Tests that generating with static cache give almost same results as with dynamic cache, and the output cache
        has the expected shapes
        """
        set_model_tester_for_less_flaky_test(self)
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            set_config_for_less_flaky_test(config)

            config.is_decoder = True
            batch_size = self.model_tester.batch_size
            seq_length = self.model_tester.vision_seq_length
            max_new_tokens = 20

            for dtype in (torch.float32, torch.float16):
                model = model_class(config).to(torch_device).to(dtype).eval()
                inputs_dict = {
                    k: v.to(dtype) if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v
                    for k, v in inputs_dict.items()
                }
                set_model_for_less_flaky_test(model)

                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "return_dict_in_generate": True,  # Required to return `past_key_values`
                    "output_scores": True,
                    "use_cache": True,
                }

                static_cache_generation = model.generate(
                    **generation_kwargs, **inputs_dict, cache_implementation="static"
                )

                # Check 1: The cache shapes must match the expected shapes
                max_cache_len = seq_length + max_new_tokens - 1  # cache len = gen len - 1, the last token has no cache
                text_config = config.text_config if hasattr(config, "text_config") else config
                head_dim = (
                    text_config.head_dim
                    if hasattr(text_config, "head_dim")
                    else text_config.hidden_size // text_config.num_attention_heads
                )
                num_key_value_heads = (
                    text_config.num_attention_heads
                    if getattr(text_config, "num_key_value_heads", None) is None
                    else text_config.num_key_value_heads
                )
                num_hidden_layers = text_config.num_hidden_layers
                cache_shape = (batch_size, num_key_value_heads, max_cache_len, head_dim)
                self.assertTrue(isinstance(static_cache_generation.past_key_values, StaticCache))
                self.assertTrue(len(static_cache_generation.past_key_values.key_cache) == num_hidden_layers)
                self.assertTrue(static_cache_generation.past_key_values.key_cache[0].shape == cache_shape)

                # Check 2: The outputs must be similar to the case with dynamic cache
                dynamic_cache_generation = model.generate(**generation_kwargs, **inputs_dict)
                self._check_similar_generate_outputs(dynamic_cache_generation, static_cache_generation)

    @pytest.mark.generate
    def test_generate_compile_model_forward(self):
        """
        Tests that `.generate` is compatible with torch.compile without graph breaks, keeping the same results.
        ⚠️ Runs two sequential generations to ensure the cache doesn't get stuck after the first compiled run! ⚠️
        """
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=4)

            model = model_class(config).to(torch_device)
            model.eval()  # otherwise `self.training` is `True` -- this flag is used at attn mask creation time

            half_batch_size = self.model_tester.batch_size // 2
            input_1 = {}
            input_2 = {}
            for key, value in inputs_dict.items():
                if isinstance(value, torch.Tensor):
                    input_1[key] = value[:half_batch_size, :].to(torch_device)
                    input_2[key] = value[half_batch_size : half_batch_size * 2, :].to(torch_device)
                else:
                    input_1[key] = value
                    input_2[key] = value
            model_input_sets = [input_1, input_2]
            self.assertTrue(
                model_input_sets[0]["pixel_values_videos"].shape == model_input_sets[1]["pixel_values_videos"].shape
            )

            torch.compiler.reset()  # prevent cached compilation from being used in the test
            model.generation_config.compile_config._compile_all_devices = True

            generation_kwargs = {
                "do_sample": False,
                "max_new_tokens": 5,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

            # get eager + dynamic cache results for future comparison
            dynamic_outputs = []
            for model_inputs in model_input_sets:
                gen_out = model.generate(**model_inputs, **generation_kwargs)
                dynamic_outputs.append(gen_out)
                # sanity checks for the default cache implementation
                decoder_cache = (
                    gen_out.past_key_values.self_attention_cache
                    if config.is_encoder_decoder
                    else gen_out.past_key_values
                )
                self.assertTrue(isinstance(decoder_cache, DynamicCache))
                self.assertFalse(decoder_cache.is_compileable)
                self.assertFalse(hasattr(model, "_compiled_call"))  # our auto compile should NOT have been called

            generation_kwargs["cache_implementation"] = "static"
            compiled_outputs = []
            for model_inputs in model_input_sets:
                gen_out = model.generate(**model_inputs, **generation_kwargs)
                compiled_outputs.append(gen_out)
                # sanity checks
                decoder_cache = (
                    gen_out.past_key_values.self_attention_cache
                    if config.is_encoder_decoder
                    else gen_out.past_key_values
                )
                self.assertFalse(isinstance(decoder_cache, DynamicCache))
                self.assertTrue(decoder_cache.is_compileable)

                self.assertTrue(hasattr(model, "_compiled_call"))  # our auto compile should have been called

            for dynamic_result, compiled_result in zip(dynamic_outputs, compiled_outputs):
                self._check_similar_generate_outputs(dynamic_result, compiled_result)


class CosmosVideoWorldModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        is_training=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=3,
        image_size=30,
        codebook_size=20,
        temporal_downsample_factor=1,
        base_channels=32,
        vq_channel_multiplier=[1, 1],
        vq_img_token_start_id=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.temporal_downsample_factor = temporal_downsample_factor
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_img_token_start_id = vq_img_token_start_id
        self.base_channels = base_channels
        self.seq_length = seq_length

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = input_ids.ne(1).to(torch_device)

        pixel_values_videos = floats_tensor(
            [
                self.batch_size,
                9,
                3,
                self.image_size,
                self.image_size,
            ]
        )
        return config, input_ids, attention_mask, pixel_values_videos

    def get_config(self):
        text_config = CosmosTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            rope_latent_shape=[3, 2, 3],
            is_video_to_world=True,
        )

        vq_config = {
            "codebook_size": self.codebook_size,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "base_channels": self.base_channels,
            "channel_multiplier": self.vq_channel_multiplier,
            "hidden_size": self.base_channels,
            "levels": [2, 2, 2, 2, 2, 2],
        }

        prompt_encoder_config = T5ModelTester(self).get_config()

        config = CosmosConfig(
            text_config=text_config,
            vq_config=vq_config,
            prompt_encoder_config=prompt_encoder_config,
            image_token_id=self.image_token_id,
            is_encoder_decoder=True,
        )
        return config

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values_videos,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values_videos": pixel_values_videos,
        }
        return config, inputs_dict


@require_torch
class CosmosVideoWorldModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            CosmosModel,
            CosmosForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    # all_generative_model_classes = () # Cosmos generates only video as output, so we use custom tests
    pipeline_model_mapping = {}
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = CosmosVideoWorldModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=CosmosConfig, has_text_modality=False, common_properties=["image_token_id"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()


@require_torch
class IntegrationTest(unittest.TestCase):
    @slow
    @require_bitsandbytes
    def test_model_generation(self):
        model = CosmosForConditionalGeneration.from_pretrained("BAAI/-Chat-hf", load_in_4bit=True)
        processor = CosmosProcessor.from_pretrained("BAAI/-Chat-hf")

        image = Image.open(requests.get("https://picsum.photos/id/237/200/200", stream=True).raw)
        prompt = "USER: <image>Describe what do you see here and tell me about the history behind it? ASSISTANT:"

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETION = ['USER: 64*64Describe what do you see here and tell me about the history behind it? ASSISTANT: The image captures a moment of tranquility with a black Labrador Retriever resting on a wooden floor. The dog, with its glossy black coat, is lying down with its front legs stretched out in']  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_bitsandbytes
    @require_torch_large_gpu
    def test_model_generation_batched(self):
        model = CosmosForConditionalGeneration.from_pretrained("BAAI/-Chat-hf", load_in_4bit=True)
        processor = CosmosProcessor.from_pretrained("BAAI/-Chat-hf")
        processor.tokenizer.padding_side = "left"

        image = Image.open(requests.get("https://picsum.photos/id/237/50/50", stream=True).raw)
        image_2 = Image.open(requests.get("https://picsum.photos/id/247/50/50", stream=True).raw)
        prompts = [
            "USER: <image>Describe what do you see here? ASSISTANT:",
            "USER: <image>What can you say about the image? ASSISTANT:",
        ]

        inputs = processor(images=[image, image_2], text=prompts, padding=True, return_tensors="pt").to(
            model.device, torch.float16
        )

        # greedy generation outputs
        EXPECTED_TEXT_COMPLETION = [
            "USER: 64*64Describe what do you see here? ASSISTANT: The image depicts a black panther in a crouched position. The panther's body is elongated and curved, with its head lowered and ears pointed forward, suggesting alertness or focus.",
            'USER: 64*64What can you say about the image? ASSISTANT: The image depicts a serene natural landscape. The foreground consists of a grassy area with some patches of bare earth. The middle ground shows a steep, reddish-brown cliff, which could be a'
        ]  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
