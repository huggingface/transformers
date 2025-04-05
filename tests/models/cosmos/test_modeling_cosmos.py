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

import inspect
import tempfile
import unittest

import numpy as np
import pytest
import requests
from parameterized import parameterized

from transformers import CosmosConfig, CosmosProcessor, CosmosTextConfig, is_torch_available, is_vision_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    require_torch_large_gpu,
    require_torch_sdpa,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils import is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, sdpa_kernel
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
        EncoderDecoderCache,
        StaticCache,
    )
    from transformers.generation import (
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GreedySearchDecoderOnlyOutput,
        GreedySearchEncoderDecoderOutput,
        SampleDecoderOnlyOutput,
        SampleEncoderDecoderOutput,
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
    _is_composite = True

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
    @unittest.skip("Cannot return attentions due to clashing keys, ask @gante if similar models")
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
    @unittest.skip("Cannot return attentions due to clashing keys, ask @gante if similar models")
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
    @unittest.skip("Cannot return attentions due to clashing keys, ask @gante if similar models")
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
            batch_size = inputs_dict["pixel_values_videos"].shape[0]
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
                self.assertTrue(isinstance(decoder_cache, StaticCache))
                self.assertTrue(decoder_cache.is_compileable)
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
                self.assertTrue(isinstance(decoder_cache, StaticCache))
                self.assertTrue(decoder_cache.is_compileable)

                self.assertTrue(hasattr(model, "_compiled_call"))  # our auto compile should have been called

            for dynamic_result, compiled_result in zip(dynamic_outputs, compiled_outputs):
                self._check_similar_generate_outputs(dynamic_result, compiled_result)

    def _check_generate_outputs(self, output, config, use_cache=False, num_return_sequences=1, num_beams=1):
        input_batch_size = int(output.sequences.shape[0] / num_return_sequences)
        internal_batch_size = (
            input_batch_size * num_beams if num_beams > 1 else input_batch_size * num_return_sequences
        )

        prompt_length = getattr(self.model_tester, "seq_length", None)
        config = config.text_config if hasattr(config, "text_config") else config

        generated_length = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - prompt_length
        )
        decoder_past_key_values = getattr(output, "past_key_values", None)
        if config.is_encoder_decoder and isinstance(decoder_past_key_values, EncoderDecoderCache):
            decoder_past_key_values = decoder_past_key_values.self_attention_cache

        # in some models we subsample the sequence length in inner layers
        if hasattr(self.model_tester, "get_subsampled_output_lengths"):
            prompt_length = self.model_tester.get_subsampled_output_lengths(prompt_length)

        # scores
        self._check_scores(
            batch_size=internal_batch_size, scores=output.scores, generated_length=generated_length, config=config
        )

        # unprocessed logits
        self._check_logits(batch_size=internal_batch_size, logits=output.logits, config=config)

        self._check_attentions_for_generate(
            batch_size=internal_batch_size,
            attentions=output.decoder_attentions,
            prompt_length=prompt_length,
            output_length=output.sequences.shape[-1],
            config=config,
            decoder_past_key_values=decoder_past_key_values,
        )

        self._check_hidden_states_for_generate(
            batch_size=internal_batch_size,
            hidden_states=output.decoder_hidden_states,
            prompt_length=prompt_length,
            output_length=output.sequences.shape[-1],
            config=config,
            use_cache=use_cache,
        )

        if use_cache:
            cache_length = output.sequences.shape[-1] - 1
            self._check_past_key_values_for_generate(
                batch_size=internal_batch_size,
                decoder_past_key_values=decoder_past_key_values,
                cache_length=cache_length,
                config=config,
            )
        elif use_cache is False:
            self.assertTrue(decoder_past_key_values is None)

    @unittest.skip("Cosmos Video has no input ids")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("Cosmos Video has no input ids")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("Cosmos Video has no input ids")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(
        "Needs to check prompt encoder config, skip instead of overriding because we already check attn in generate tests"
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip(
        "Needs to check prompt encoder config, skip instead of overriding because we already check hiddens in generate tests"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip("We write custom generation tests")
    def test_generation_tester_mixin_inheritance(self):
        pass

    @unittest.skip("Cosmos does not support flex in vq backbone")
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skip("VQ-VAE module doesn't initialize weights properly")
    def test_initialization(self):
        pass

    @unittest.skip("Cosmos doesn't accept text tokens, thus cannot resize embeddings")
    def test_resize_embeddings_untied(self):
        pass

    @unittest.skip("Cosmos does not support torchscript")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip("Cosmos does not support torchscript")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip("Cosmos does not support torchscript")
    def test_torchscript_simple(self):
        pass

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self.all_model_classes[0]._supports_sdpa:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        if torch_dtype == "float16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_dtype == "bfloat16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Not sure whether it's fine to put torch.XXX in a decorator if torch is not available so hacking it here instead.
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        def get_mean_reldiff(failcase, x, ref, atol, rtol):
            return f"{failcase}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        set_model_tester_for_less_flaky_test(self)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            set_config_for_less_flaky_test(config)
            model = model_class(config)
            # FIXME: we deactivate boolean mask for models using "use_mask_token" in their constructors.
            # These models support masking only in the case `use_mask_token=True`. Otherwise they cannot consume an input mask.
            # This means that the class needs to be instantiated much later, after `use_mask` is set, which means a significant refactor of the code.
            # However masking there is not done at any layers that matters (i.e self-attention), therefore we can safely deactivate it.
            deactivate_mask = "use_mask_token" in inspect.signature(model_class).parameters
            is_encoder_decoder = model.config.is_encoder_decoder

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                try:
                    model_sdpa = model_class.from_pretrained(
                        tmpdirname, torch_dtype=torch_dtype, attn_implementation="sdpa"
                    )
                except ValueError:
                    model_sdpa = model_class.from_pretrained(tmpdirname, torch_dtype=torch_dtype)
                model_sdpa = model_sdpa.eval().to(torch_device, dtype=torch_dtype)

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device, dtype=torch_dtype)

                set_model_for_less_flaky_test(model_eager)
                set_model_for_less_flaky_test(model_sdpa)

                # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving 16 times the model,
                # but it would be nicer to have an efficient way to use parameterized.expand
                fail_cases = []
                for padding_side in ["left", "right"]:
                    for use_mask in [False, True]:
                        for output_attentions in [True, False]:
                            can_output_attn = "output_attentions" in inspect.signature(model_sdpa.forward).parameters
                            if not (self.has_attentions and can_output_attn) and output_attentions:
                                continue
                            # TODO: if we can also check with `batch_size=1` without being flaky?
                            for batch_size in [7]:
                                dummy_input = inputs_dict["pixel_values_videos"]

                                if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                    dummy_input = dummy_input.to(torch_dtype)

                                dummy_input = dummy_input[:batch_size]
                                if dummy_input.shape[0] != batch_size:
                                    if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                        extension = torch.rand(
                                            batch_size - dummy_input.shape[0],
                                            *dummy_input.shape[1:],
                                            dtype=torch_dtype,
                                            device=torch_device,
                                        )
                                        dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)
                                    else:
                                        extension = torch.randint(
                                            high=5,
                                            size=(batch_size - dummy_input.shape[0], *dummy_input.shape[1:]),
                                            dtype=dummy_input.dtype,
                                            device=torch_device,
                                        )
                                        dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)

                                if not use_mask:
                                    dummy_attention_mask = None
                                else:
                                    dummy_attention_mask = inputs_dict.get("attention_mask", None)
                                    if dummy_attention_mask is None:
                                        if is_encoder_decoder:
                                            seqlen = inputs_dict.get("decoder_input_ids", dummy_input).shape[-1]
                                        else:
                                            seqlen = dummy_input.shape[-1]
                                        dummy_attention_mask = (
                                            torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)
                                        )

                                    dummy_attention_mask = dummy_attention_mask[:batch_size]
                                    if dummy_attention_mask.shape[0] != batch_size:
                                        extension = torch.ones(
                                            batch_size - dummy_attention_mask.shape[0],
                                            *dummy_attention_mask.shape[1:],
                                            dtype=dummy_attention_mask.dtype,
                                            device=torch_device,
                                        )
                                        dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)
                                        dummy_attention_mask = dummy_attention_mask.to(torch_device)

                                    dummy_attention_mask[:] = 1
                                    if padding_side == "left":
                                        dummy_attention_mask[-1, :2] = 0
                                        dummy_attention_mask[-1, 2:] = 1
                                    elif padding_side == "right":
                                        dummy_attention_mask[-1, -2:] = 0
                                        dummy_attention_mask[-1, :-2] = 1

                                for enable_kernels in [False, True]:
                                    failcase = f"padding_side={padding_side}, use_mask={use_mask}, enable_kernels={enable_kernels}"
                                    if is_encoder_decoder:
                                        decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[
                                            :batch_size
                                        ]
                                        if decoder_input_ids.shape[0] != batch_size:
                                            extension = torch.ones(
                                                batch_size - decoder_input_ids.shape[0],
                                                *decoder_input_ids.shape[1:],
                                                dtype=decoder_input_ids.dtype,
                                                device=torch_device,
                                            )
                                            decoder_input_ids = torch.cat((decoder_input_ids, extension), dim=0)
                                            decoder_input_ids = decoder_input_ids.to(torch_device)

                                        # TODO: never an `attention_mask` arg here?
                                        processed_inputs = {
                                            "pixel_values_videos": dummy_input,
                                            "decoder_input_ids": decoder_input_ids,
                                            "decoder_attention_mask": dummy_attention_mask,
                                            "output_hidden_states": True,
                                        }
                                    else:
                                        processed_inputs = {
                                            "pixel_values_videos": dummy_input,
                                            "output_hidden_states": True,
                                        }

                                        # Otherwise fails for e.g. WhisperEncoderModel
                                        if "attention_mask" in inspect.signature(model_eager.forward).parameters:
                                            processed_inputs["attention_mask"] = dummy_attention_mask

                                        if (
                                            self.has_attentions
                                            and "output_attentions" in inspect.signature(model_sdpa.forward).parameters
                                        ):
                                            processed_inputs["output_attentions"] = output_attentions
                                    if not deactivate_mask and (
                                        "bool_masked_pos" in inspect.signature(model_eager.forward).parameters
                                    ):
                                        dummy_mask = torch.ones((self.model_tester.num_masks,))

                                        # In case of additional token (like class) we define a custom `mask_length`
                                        if hasattr(self.model_tester, "mask_length"):
                                            mask_length = self.model_tester.mask_length - dummy_mask.size(0)
                                        else:
                                            mask_length = self.model_tester.seq_length - dummy_mask.size(0)
                                        dummy_mask = torch.cat([dummy_mask, torch.zeros(mask_length)])
                                        dummy_bool_masked_pos = dummy_mask.expand(batch_size, -1).bool()
                                        processed_inputs["bool_masked_pos"] = dummy_bool_masked_pos.to(torch_device)

                                    if "noise" in inspect.signature(model_eager.forward).parameters:
                                        np.random.seed(2)
                                        num_patches = int(
                                            (self.model_tester.image_size // self.model_tester.patch_size) ** 2
                                        )
                                        noise = np.random.uniform(size=(batch_size, num_patches))
                                        processed_inputs["noise"] = torch.from_numpy(noise)

                                    # TODO: test gradients as well (& for FA2 as well!)
                                    with torch.no_grad():
                                        with sdpa_kernel(
                                            enable_flash=enable_kernels,
                                            enable_math=True,
                                            enable_mem_efficient=enable_kernels,
                                        ):
                                            prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
                                            outputs_eager = model_eager(**prepared_inputs)
                                            outputs_sdpa = model_sdpa(**prepared_inputs)

                                    if hasattr(outputs_eager, "vision_hidden_states"):
                                        logits_eager = outputs_eager.vision_hidden_states[-1]
                                        logits_sdpa = outputs_sdpa.vision_hidden_states[-1]
                                    else:
                                        logits_eager = outputs_eager.decoder_hidden_states[-1]
                                        logits_sdpa = outputs_sdpa.decoder_hidden_states[-1]

                                    if torch_device in ["cpu", "cuda"]:
                                        atol = atols[torch_device, enable_kernels, torch_dtype]
                                        rtol = rtols[torch_device, enable_kernels, torch_dtype]
                                    elif torch_device == "xpu":
                                        # As of PyTorch 2.5 XPU backend supports only torch.nn.attention.SDPBackend.MATH
                                        # which is implemented on PyTorch level using aten operators and is
                                        # device agnostic with respect to implementation of each aten operator.
                                        atol = atols["cuda", False, torch_dtype]
                                        rtol = rtols["cuda", False, torch_dtype]
                                    else:
                                        atol = 1e-7
                                        rtol = 1e-4

                                    # Masked tokens output slightly deviates - we don't mind that.
                                    if use_mask:
                                        _logits_sdpa = torch.zeros_like(input=logits_sdpa)
                                        _logits_eager = torch.zeros_like(input=logits_eager)

                                        _logits_sdpa[:-1] = logits_sdpa[:-1]
                                        _logits_eager[:-1] = logits_eager[:-1]

                                        if padding_side == "left":
                                            _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, 2:]
                                            _logits_eager[-1:, 2:] = logits_eager[-1:, 2:]

                                        elif padding_side == "right":
                                            _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, :-2]
                                            _logits_eager[-1:, 2:] = logits_eager[-1:, :-2]

                                        logits_sdpa = _logits_sdpa
                                        logits_eager = _logits_eager

                                    results = [
                                        torch.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
                                        for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
                                    ]
                                    # If 80% batch elements have matched results, it's fine
                                    if np.mean(results) < 0.8:
                                        fail_cases.append(
                                            get_mean_reldiff(failcase, logits_sdpa, logits_eager, atol, rtol)
                                        )

                self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are calles "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implamentation" is a dict after the model
        is loaded, because we manually replicate requested attn implementation on each sub-config when loading.
        See https://github.com/huggingface/transformers/pull/32238 for more info

        The test tries to cover most general cases of composite models, VLMs with vision and text configs. Any model
        that has a different set of sub-configs has to overwrite this test.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)
                model_sdpa = getattr(model_sdpa, "model", model_sdpa)  # base model

                vision_model_names = {"vqmodel"}
                language_model_names = {"language_model"}
                vision_model_name = [name for name in vision_model_names if hasattr(model_sdpa, name)][0]
                language_model_name = [name for name in language_model_names if hasattr(model_sdpa, name)][0]

                vision_model_sdpa = getattr(model_sdpa, vision_model_name)
                language_model_sdpa = getattr(model_sdpa, language_model_name)
                text_attn = "sdpa" if language_model_sdpa._supports_sdpa else "eager"
                vision_attn = "sdpa" if vision_model_sdpa._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(language_model_sdpa.config._attn_implementation == text_attn)
                self.assertTrue(vision_model_sdpa.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                model_eager = getattr(model_eager, "model", model_eager)  # base model
                self.assertTrue(getattr(model_eager, language_model_name).config._attn_implementation == "eager")
                self.assertTrue(getattr(model_eager, vision_model_name).config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if (
                        class_name.endswith("Attention")
                        and getattr(submodule, "config", None)
                        and submodule.config._attn_implementation == "sdpa"
                    ):
                        raise ValueError("The eager model should not have SDPA attention layers")


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
        insert_cross_attn_layers=[0, 1],
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
        self.insert_cross_attn_layers = insert_cross_attn_layers
        self.base_channels = base_channels
        self.vision_seq_length = 42  # video seq length when tokenizer
        self.encoder_seq_length = seq_length
        self.seq_length = self.vision_seq_length + 1

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, 7], config.text_config.vocab_size)
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
            is_video_to_world=True,
            insert_cross_attn_layers=self.insert_cross_attn_layers,
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
    all_generative_model_classes = ()  # Cosmos generates only video as output, so we use custom tests
    _custom_generative_model_classes = (CosmosForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {}
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    _is_composite = True

    def setUp(self):
        self.model_tester = CosmosVideoWorldModelTester(self)
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
            self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length + 1)

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
                output_generate.sequences.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length + 1
            )
            self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)
            self.assertIsInstance(output_generate, GreedySearchEncoderDecoderOutput)
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
                output_generate.sequences.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length + 1
            )
            self._check_generate_outputs(output_generate, model.config, use_cache=True)

    @pytest.mark.generate
    def test_sample_generate(self):
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).to(torch_device).eval()
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)
            self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length + 1)

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
                output_generate.sequences.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length + 1
            )
            self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)
            self.assertIsInstance(output_generate, SampleEncoderDecoderOutput)

            self._check_generate_outputs(output_generate, model.config, num_return_sequences=2)

    @pytest.mark.generate
    def test_beam_search_generate(self):
        for model_class in self._custom_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            model = model_class(config).to(torch_device).eval()
            beam_kwargs = self._get_beam_kwargs()
            output_generate = self._beam_search_generate(model=model, inputs_dict=inputs_dict, beam_kwargs=beam_kwargs)
            self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + self.model_tester.vision_seq_length + 1)

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
            batch_size = inputs_dict["pixel_values_videos"].shape[0]
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
                max_cache_len = seq_length + max_new_tokens
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
                cache = static_cache_generation.past_key_values.self_attention_cache
                self.assertTrue(isinstance(cache, StaticCache))
                self.assertTrue(len(cache.key_cache) == num_hidden_layers)
                self.assertTrue(cache.key_cache[0].shape == cache_shape)

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
                self.assertTrue(isinstance(decoder_cache, StaticCache))  # Cosmos uses only static cache
                self.assertTrue(decoder_cache.is_compileable)
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
                self.assertTrue(isinstance(decoder_cache, StaticCache))
                self.assertTrue(decoder_cache.is_compileable)

                self.assertTrue(hasattr(model, "_compiled_call"))  # our auto compile should have been called

            for dynamic_result, compiled_result in zip(dynamic_outputs, compiled_outputs):
                self._check_similar_generate_outputs(dynamic_result, compiled_result)

    def _check_generate_outputs(self, output, config, use_cache=False, num_return_sequences=1, num_beams=1):
        input_batch_size = int(output.sequences.shape[0] / num_return_sequences)
        internal_batch_size = (
            input_batch_size * num_beams if num_beams > 1 else input_batch_size * num_return_sequences
        )

        prompt_length = getattr(self.model_tester, "encoder_seq_length", None)
        text_config = config.text_config if hasattr(config, "text_config") else config

        generated_length = output.sequences.shape[-1] - 1 - self.model_tester.vision_seq_length
        decoder_past_key_values = getattr(output, "past_key_values", None)
        if config.is_encoder_decoder and isinstance(decoder_past_key_values, EncoderDecoderCache):
            decoder_past_key_values = decoder_past_key_values.self_attention_cache

        # scores
        self._check_scores(
            batch_size=internal_batch_size, scores=output.scores, generated_length=generated_length, config=config
        )

        # unprocessed logits
        self._check_logits(batch_size=internal_batch_size, logits=output.logits, config=config)

        # Attentions
        self._check_encoder_attention_for_generate(
            attentions=output.encoder_attentions,
            batch_size=input_batch_size,
            config=config.prompt_encoder_config,
            prompt_length=prompt_length,
        )
        self._check_attentions_for_generate(
            batch_size=internal_batch_size,
            attentions=output.decoder_attentions,
            prompt_length=1 + self.model_tester.vision_seq_length,  # the BOS token
            output_length=output.sequences.shape[-1],
            config=text_config,
            decoder_past_key_values=decoder_past_key_values,
        )

        # Hidden States
        self._check_encoder_hidden_states_for_generate(
            hidden_states=output.encoder_hidden_states,
            batch_size=input_batch_size,
            config=config.prompt_encoder_config,
            prompt_length=prompt_length,
        )
        self._check_hidden_states_for_generate(
            batch_size=internal_batch_size,
            hidden_states=output.decoder_hidden_states,
            prompt_length=1 + self.model_tester.vision_seq_length,  # the BOS token
            output_length=output.sequences.shape[-1],
            config=text_config,
            use_cache=use_cache,
        )

        if use_cache:
            cache_length = output.sequences.shape[-1] - 1
            self._check_past_key_values_for_generate(
                batch_size=internal_batch_size,
                decoder_past_key_values=decoder_past_key_values,
                cache_length=cache_length,
                config=text_config,
            )
        elif use_cache is False:
            self.assertTrue(decoder_past_key_values is None)

    @unittest.skip("VQ-VAE module doesn't initialize weights properly")
    def test_initialization(self):
        pass

    @unittest.skip(
        "Needs to check prompt encoder config, skip instead of overriding because we already check attn in generate tests"
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip(
        "Needs to check prompt encoder config, skip instead of overriding because we already check hiddens in generate tests"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip("We write custom generation tests")
    def test_generation_tester_mixin_inheritance(self):
        pass

    @unittest.skip("Cosmos does not support flex in vq backbone")
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skip("Cosmos does not support torchscript")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip("Cosmos does not support torchscript")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip("Cosmos does not support torchscript")
    def test_torchscript_simple(self):
        pass

    @unittest.skip("Cosmos uses T5 which doesn't support sdpa")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip("Cosmos uses T5 which doesn't support sdpa")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Cosmos uses T5 which doesn't support sdpa")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip("Cosmos uses T5 which doesn't support sdpa")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are calles "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implamentation" is a dict after the model
        is loaded, because we manually replicate requested attn implementation on each sub-config when loading.
        See https://github.com/huggingface/transformers/pull/32238 for more info

        The test tries to cover most general cases of composite models, VLMs with vision and text configs. Any model
        that has a different set of sub-configs has to overwrite this test.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)
                model_sdpa = getattr(model_sdpa, "model", model_sdpa)  # base model

                vision_model_names = {"vqmodel"}
                language_model_names = {"language_model"}
                vision_model_name = [name for name in vision_model_names if hasattr(model_sdpa, name)][0]
                language_model_name = [name for name in language_model_names if hasattr(model_sdpa, name)][0]

                vision_model_sdpa = getattr(model_sdpa, vision_model_name)
                language_model_sdpa = getattr(model_sdpa, language_model_name)
                text_attn = "sdpa" if language_model_sdpa._supports_sdpa else "eager"
                vision_attn = "sdpa" if vision_model_sdpa._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(language_model_sdpa.config._attn_implementation == text_attn)
                self.assertTrue(vision_model_sdpa.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                model_eager = getattr(model_eager, "model", model_eager)  # base model
                self.assertTrue(getattr(model_eager, language_model_name).config._attn_implementation == "eager")
                self.assertTrue(getattr(model_eager, vision_model_name).config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if (
                        class_name.endswith("Attention")
                        and getattr(submodule, "config", None)
                        and submodule.config._attn_implementation == "sdpa"
                    ):
                        raise ValueError("The eager model should not have SDPA attention layers")


@require_torch
class IntegrationTest(unittest.TestCase):
    @slow
    @require_bitsandbytes
    def test_model_generation(self):
        model = CosmosForConditionalGeneration.from_pretrained("NVIDIA/Cosmos-4B", load_in_4bit=True)
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
        model = CosmosForConditionalGeneration.from_pretrained("NVIDIA/Cosmos-4B", load_in_4bit=True)
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
