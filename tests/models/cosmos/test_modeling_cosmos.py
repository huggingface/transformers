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
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    sdpa_kernel,
)
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

    @unittest.skip("Cosmos is not expected to generate from embeddings because it doesn't accept text tokens.")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(
        self, name, torch_dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        # convert shorthand name to torch.dtype
        if torch_dtype == "fp16" or torch_dtype == "bf16":
            self.skipTest("Cosmos uses 3D pooling which is not implemented in half precision.")
        elif torch_dtype == "fp32":
            torch_dtype = torch.float32

        if not is_torch_fp16_available_on_device(torch_device) and torch_dtype == torch.float16:
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if not is_torch_bf16_available_on_device(torch_device) and torch_dtype == torch.bfloat16:
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Dictionary of tolerances for eager <> sdpa tests. Key = (device, sdpa_kernels_enabled, dtype)
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
            ("cuda", True, torch.bfloat16): 3e-2,  # (different from others)
            ("cuda", True, torch.float16): 5e-3,
        }

        set_model_tester_for_less_flaky_test(self)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            set_config_for_less_flaky_test(config)
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_from_pretrained_kwargs = {
                    "pretrained_model_name_or_path": tmpdirname,
                    "torch_dtype": torch_dtype,
                }

                if (
                    hasattr(config, "use_mask_token")
                    or "use_mask_token" in inspect.signature(model.__init__).parameters
                ):
                    model_from_pretrained_kwargs["use_mask_token"] = True

                # TODO: remove this try/except, models should have a shared API
                try:
                    model_sdpa = model_class.from_pretrained(
                        **model_from_pretrained_kwargs, attn_implementation="sdpa"
                    )
                except ValueError:
                    model_sdpa = model_class.from_pretrained(**model_from_pretrained_kwargs)
                model_sdpa = model_sdpa.eval().to(torch_device, dtype=torch_dtype)

                model_eager = model_class.from_pretrained(**model_from_pretrained_kwargs, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device, dtype=torch_dtype)

            set_model_for_less_flaky_test(model_eager)
            set_model_for_less_flaky_test(model_sdpa)

            can_output_attn = "output_attentions" in inspect.signature(model_sdpa.forward).parameters
            if not (self.has_attentions and can_output_attn) and output_attentions:
                self.skipTest(reason="Model does not support output_attentions")

            # TODO: if we can also check with `batch_size=1` without being flaky?
            for batch_size in [7]:
                # musicgen decoder models; TODO: find better abstraction
                if hasattr(self.model_tester, "num_codebooks") and not hasattr(model_eager, "text_encoder"):
                    input_data_batch_size = batch_size * self.model_tester.num_codebooks
                else:
                    input_data_batch_size = batch_size

                processed_inputs = {}
                processed_inputs["pixel_values_videos"] = inputs_dict["pixel_values_videos"]

                for key, value in processed_inputs.items():
                    if torch.is_floating_point(value):
                        value = value.to(torch_dtype)

                    # extend value to have at least `input_data_batch_size` elements
                    if value.shape[0] < input_data_batch_size:
                        size = (input_data_batch_size - value.shape[0], *value.shape[1:])
                        if torch.is_floating_point(value):
                            extension = torch.rand(size=size, dtype=value.dtype, device=torch_device)
                        else:
                            extension = torch.randint(high=5, size=size, dtype=value.dtype, device=torch_device)
                        value = torch.cat((value, extension), dim=0).to(torch_device)

                    processed_inputs[key] = value[:input_data_batch_size]

                if not use_attention_mask:
                    dummy_attention_mask = None
                else:
                    dummy_attention_mask = inputs_dict.get("attention_mask", None)
                    if dummy_attention_mask is None:
                        seqlen = self.model_tester.vision_seq_length
                        dummy_attention_mask = torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)

                    # extend dummy_attention_mask to have at least `batch_size` elements
                    if dummy_attention_mask.shape[0] < batch_size:
                        size = (batch_size - dummy_attention_mask.shape[0], *dummy_attention_mask.shape[1:])
                        extension = torch.ones(size=size, dtype=dummy_attention_mask.dtype, device=torch_device)
                        dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)

                    dummy_attention_mask = dummy_attention_mask[:batch_size].to(torch_device)

                    dummy_attention_mask[:] = 1
                    if padding_side == "left":
                        dummy_attention_mask[-1, :2] = 0
                        dummy_attention_mask[-1, 2:] = 1
                    elif padding_side == "right":
                        dummy_attention_mask[-1, -2:] = 0
                        dummy_attention_mask[-1, :-2] = 1

                processed_inputs.update(
                    {
                        "output_hidden_states": True,
                    }
                )

                # Otherwise fails for e.g. WhisperEncoderModel
                if "attention_mask" in inspect.signature(model_eager.forward).parameters:
                    processed_inputs["attention_mask"] = dummy_attention_mask
                if self.has_attentions and "output_attentions" in inspect.signature(model_sdpa.forward).parameters:
                    processed_inputs["output_attentions"] = output_attentions

                # TODO: test gradients as well (& for FA2 as well!)
                with torch.no_grad():
                    with sdpa_kernel(
                        enable_flash=enable_kernels,
                        enable_math=True,
                        enable_mem_efficient=enable_kernels,
                    ):
                        prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
                        prepared_inputs = {
                            k: v.to(torch_device) if isinstance(v, torch.Tensor) else v
                            for k, v in prepared_inputs.items()
                        }
                        outputs_eager = model_eager(**prepared_inputs)
                        outputs_sdpa = model_sdpa(**prepared_inputs)

                key = "decoder_hidden_states"

                # TODO: rename logits -> hidden_states
                logits_eager = outputs_eager[key]
                logits_sdpa = outputs_sdpa[key]

                if key in ["vision_hidden_states", "decoder_hidden_states", "hidden_states"]:
                    logits_eager = logits_eager[-1]
                    logits_sdpa = logits_sdpa[-1]

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
                if use_attention_mask:
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
                    mean_relative_diff = ((logits_sdpa - logits_eager).abs() / (logits_eager.abs() + 1e-12)).mean()
                    raise ValueError(
                        f"mean relative difference for {key}: {mean_relative_diff:.3e}, torch atol = {atol}, torch rtol = "
                        f"{rtol}"
                    )

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
        "Cosmos can't be initialized on meta, it has `tensor.item()` in init which is not possible with meta tensors."
    )
    def test_can_be_initialized_on_meta(self):
        pass

    @unittest.skip(
        "Cosmos can't be initialized on meta, it has `tensor.item()` in init which is not possible with meta tensors."
    )
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip("Prompt encoder embeddings are not initalized properly")
    def test_can_init_all_missing_weights(self):
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

    @unittest.skip("Cosmos is not expected to generate from input embeds")
    def test_inputs_embeds_matches_input_ids(self):
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
