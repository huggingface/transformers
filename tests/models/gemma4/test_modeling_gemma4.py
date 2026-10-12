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
"""Testing suite for the PyTorch Gemma4 model."""

import tempfile
import unittest
from contextlib import contextmanager

import pytest
from parameterized import parameterized

from transformers import (
    AutoTokenizer,
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    is_torch_available,
    logging,
    set_seed,
)
from transformers.testing_utils import (
    CaptureLogger,
    Expectations,
    cleanup,
    require_deterministic_for_accelerator,
    require_deterministic_for_xpu,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import ModelOutput

from ...alm_tester import ALMModelTest, ALMModelTester
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        Gemma4ForCausalLM,
        Gemma4ForConditionalGeneration,
        Gemma4Model,
        Gemma4Processor,
        Gemma4TextModel,
    )
    from transformers.cache_utils import StaticCache
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_vision_model


GEMMA4_RANDOM_MOE_FA2_SKIP_REASON = (
    "Randomly initialized Gemma4 MoE routers are too sensitive to tiny eager/FA2 input differences"
)


class Gemma4TextModelTester(CausalLMModelTester):
    forced_config_args = ["pad_token_id", "per_layer_config"]

    if is_torch_available():
        config_class = Gemma4TextConfig
        base_model_class = Gemma4TextModel
        causal_lm_class = Gemma4ForCausalLM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hidden_layers = 4  # override to correctly test sharing cache pattern
        self.num_kv_shared_layers = 2  # important to override
        self.layer_types = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]  # similarly we want to test sharing on both types
        self.per_layer_config = {
            layer_idx: {"head_dim": 2 * self.head_dim}
            for layer_idx, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        }  # gemma4 use a different head_dim for full and sliding layers

        # To make model small
        self.vocab_size_per_layer_input = 99
        self.hidden_size_per_layer_input = 16

        # To activate moe blocks
        self.enable_moe_block = True
        self.moe_intermediate_size = 16
        self.top_k_experts = 2

        # Test if bidirectional image mask path works
        self.use_bidirectional_attention = "vision"


@require_torch
class Gemma4TextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Gemma4TextModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Gemma4ForCausalLM if is_torch_available() else None

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    def test_bidirectional_sliding_window_survives_save_and_reload(self):
        config = Gemma4TextConfig(sliding_window=512, use_bidirectional_attention="all")
        self.assertEqual(config.sliding_window, 257)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            reloaded = Gemma4TextConfig.from_pretrained(tmpdirname)

        self.assertEqual(reloaded.sliding_window, config.sliding_window)

    @unittest.skip(
        "Gemma4 cannot use random inputs_embeds, as it needs to reverse them when input_ids is not provided"
    )
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip(
        "Flaky on CI, but not locally on Mac. If model is set to fp32 instead of bf16, not flaky anymore."
        "TODO Cyril: investigate where the loss of precision between bf16 and fp32 comes from."
    )
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "Fails after fully removing the unused weights, even if `forward` is exactly the same. Investigate why."
    )
    def test_tp_generation_quantized(self):
        pass

    @unittest.skip(GEMMA4_RANDOM_MOE_FA2_SKIP_REASON)
    def test_flash_attn_2_equivalence(self):
        pass

    @unittest.skip(GEMMA4_RANDOM_MOE_FA2_SKIP_REASON)
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip(GEMMA4_RANDOM_MOE_FA2_SKIP_REASON)
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    def test_all_bidirectional_attention_uses_bidirectional_mask(self):
        self.model_tester.use_bidirectional_attention = "all"
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config._attn_implementation = "eager"

        model = Gemma4TextModel(config).to(torch_device)
        model.eval()

        input_ids = inputs_dict["input_ids"][:1]
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True)

        for attention in out.attentions:
            self.assertTrue((attention[..., :4, :4] != 0).all().item())

    def test_model_training(self):
        pass

    @unittest.skip(
        "Under non-bf16 dtypes, MoE grouped_mm falls back to "
        "_grouped_mm_fallback_backward which is incompatible with torch.compile under 'reduce-overhead' mode"
    )
    def test_flash_attn_2_can_compile_with_attention_mask_None_without_graph_break(self):
        pass

    @unittest.skip(
        "Under non-bf16 dtypes, MoE grouped_mm falls back to "
        "_grouped_mm_fallback_backward which is incompatible with torch.compile under 'reduce-overhead' mode"
    )
    def test_torch_compile_for_training(self):
        pass


class Gemma4Audio2TextModelTester(ALMModelTester):
    base_model_class = Gemma4Model
    conditional_generation_class = Gemma4ForConditionalGeneration
    config_class = Gemma4Config
    text_config_class = Gemma4TextConfig
    audio_config_class = Gemma4AudioConfig
    audio_mask_key = "input_features_mask"

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("boi_token_id", 5)
        kwargs.setdefault("eoi_token_id", 6)
        kwargs.setdefault("audio_token_id", 7)
        kwargs.setdefault("boa_token_id", 8)
        kwargs.setdefault("eoa_token_index", 9)
        kwargs.setdefault("video_token_id", 10)
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("seq_length", 50)
        kwargs.setdefault("feat_seq_length", 96)
        kwargs.setdefault("num_mel_bins", 16)
        kwargs.setdefault("num_hidden_layers", 4)
        kwargs.setdefault("num_kv_shared_layers", 2)
        kwargs.setdefault(
            "layer_types", ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
        )
        kwargs.setdefault("vocab_size_per_layer_input", 99)
        kwargs.setdefault("hidden_size_per_layer_input", 16)
        kwargs.setdefault("enable_moe_block", True)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("top_k_experts", 2)
        kwargs.setdefault("subsampling_conv_channels", [16, 8])
        kwargs.setdefault("conv_kernel_size", 3)
        kwargs.setdefault("attention_chunk_size", 4)
        kwargs.setdefault("attention_context_left", 5)
        kwargs.setdefault("attention_context_right", 0)
        kwargs.setdefault("output_proj_dims", 32)
        # Clipped linears register inf/-inf buffers which cause NaN in test_torch_save_load's
        # comparison logic (inf - inf = NaN). Disable for testing.
        kwargs.setdefault("use_clipped_linears", False)
        super().__init__(parent, **kwargs)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.per_layer_config = {
            layer_idx: {"head_dim": 2 * self.head_dim}
            for layer_idx, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        }

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.image_token_id, self.video_token_id}

    @property
    def text_config_args(self):
        return super().text_config_args + ["per_layer_config"]

    def create_attention_mask(self, input_ids):
        return input_ids.ne(self.pad_token_id).to(torch_device)

    def create_audio_features(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        # (num_audios, num_frames, num_mel_bins)
        return floats_tensor([batch_size, self.feat_seq_length, self.num_mel_bins])

    def create_audio_mask(self, batch_size: int | None = None):
        return super().create_audio_mask(batch_size).bool()

    def get_audio_embeds_mask(self, audio_mask):
        # Each of the two stride-2 subsampling convs keeps every other mask position
        return audio_mask[:, ::4]


@require_torch
class Gemma4Audio2TextModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = Gemma4Audio2TextModelTester

    @unittest.skip("The tester has no image in input dict and mm-encoder-output don't yet support audio")
    def test_generate_from_multimodal_encoder_outputs_and_raw_data(self):
        pass

    @unittest.skip("The tester has no image in input dict and mm-encoder-output don't yet support audio")
    def test_generate_from_multimodal_encoder_outputs(self):
        pass

    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Gemma4 needs correct embeddings for per-layer-input computation, random won't work!")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip(GEMMA4_RANDOM_MOE_FA2_SKIP_REASON)
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip(GEMMA4_RANDOM_MOE_FA2_SKIP_REASON)
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    def test_audio_rel_pos_encoding_uses_context_size_from_config(self):
        """Regression test for #45468; attention context size is properly read from config"""
        from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioRelPositionalEncoding

        config = Gemma4AudioConfig(
            hidden_size=32,
            attention_chunk_size=6,
            attention_context_left=5,
            attention_context_right=1,
            use_clipped_linears=False,
        )

        module = Gemma4AudioRelPositionalEncoding(config)
        hidden_states = torch.zeros(1, 3, config.hidden_size)

        pos = module(hidden_states)

        context_size = config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        expected_len = context_size // 2 + 1

        self.assertEqual(pos.shape, (1, expected_len, config.hidden_size))

        position_ids = torch.arange(context_size // 2, -1, -1, device=hidden_states.device)[..., None]
        scaled_time = position_ids * module.inv_timescales.to(device=hidden_states.device)
        expected = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1).to(hidden_states.dtype)

        torch.testing.assert_close(pos, expected)


class Gemma4Vision2TextModelTester(VLMModelTester):
    base_model_class = Gemma4Model
    conditional_generation_class = Gemma4ForConditionalGeneration
    config_class = Gemma4Config
    text_config_class = Gemma4TextConfig
    vision_config_class = Gemma4VisionConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("boi_token_id", 5)
        kwargs.setdefault("eoi_token_id", 6)
        kwargs.setdefault("video_token_id", 7)
        kwargs.setdefault("audio_token_id", 8)
        kwargs.setdefault("patch_size", 5)
        kwargs.setdefault("pooling_kernel_size", 2)
        kwargs.setdefault("num_image_tokens", 5)
        kwargs.setdefault("seq_length", 25)
        kwargs.setdefault("num_hidden_layers", 4)
        kwargs.setdefault("num_kv_shared_layers", 2)
        kwargs.setdefault(
            "layer_types", ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
        )
        kwargs.setdefault("vocab_size_per_layer_input", 99)
        kwargs.setdefault("hidden_size_per_layer_input", 16)
        kwargs.setdefault("enable_moe_block", True)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("top_k_experts", 2)
        kwargs.setdefault("use_bidirectional_attention", "vision")
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(parent, **kwargs)
        self.per_layer_config = {
            layer_idx: {"head_dim": 2 * self.head_dim}
            for layer_idx, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        }

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.video_token_id, self.audio_token_id}

    @property
    def text_config_args(self):
        return super().text_config_args + ["per_layer_config"]

    def create_attention_mask(self, input_ids):
        return input_ids.ne(self.pad_token_id).to(torch_device)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        # (num_images, max_num_patches, patch_size * patch_size * num_channels)
        num_patches = self.num_image_tokens * self.pooling_kernel_size**2
        return floats_tensor([batch_size, num_patches, self.patch_size**2 * self.num_channels])

    def create_image_position_ids(self, num_images):
        # (num_images, max_num_patches, 2) grid of (x, y) coords for a non-square image
        num_patches = self.num_image_tokens * self.pooling_kernel_size**2
        h = int(num_patches**0.5)
        w = num_patches // h
        xs = torch.arange(w).repeat(h)
        ys = torch.arange(h).repeat_interleave(w)
        position_ids = torch.stack([xs, ys], dim=-1).to(device=torch_device)
        return position_ids.unsqueeze(0).repeat(num_images, 1, 1)

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == config.image_token_id] = 1
        return {
            "image_position_ids": self.create_image_position_ids(batch_size),
            "mm_token_type_ids": mm_token_type_ids,
        }


@require_torch
class Gemma4Vision2TextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Gemma4Vision2TextModelTester
    additional_model_inputs = ["mm_token_type_ids", "image_position_ids"]
    model_split_percents = [0.85, 0.9]

    def setUp(self):
        super().setUp()
        self.skip_flash_attn_inference_equivalence_tests()

    def skip_flash_attn_inference_equivalence_tests(self):
        skippable_tests = [
            "test_flash_attn_2_inference_equivalence",
            "test_flash_attn_3_inference_equivalence",
            "test_flash_attn_4_inference_equivalence",
        ]
        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(
                    reason="The base test does not pass image_position_ids and mm_token_type_ids required by Gemma4"
                )

    def test_training(self):
        # Overwrite to test training with text-only samples, should not raise errors
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        model = Gemma4ForConditionalGeneration(config)
        model.to(torch_device)
        model.train()
        inputs = self._prepare_for_class(inputs_dict, Gemma4ForConditionalGeneration, return_labels=True)
        loss = model(**inputs).loss
        loss.backward()

        # pop out image-related inputs and try to run forward
        inputs.pop("mm_token_type_ids", None)
        inputs.pop("pixel_values", None)
        loss = model(**inputs).loss
        loss.backward()

    def test_vision_axial_rope(self):
        # override -> model shipped weirdly to from the start, pos IDs have actual batch dim

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        rope_class = None
        base_model = Gemma4Model(config)
        for name, module in base_model.named_modules():
            if hasattr(module, "compute_axial_rope_parameters"):
                rope_class = type(module)
                vision_config = module.config
                break

        if rope_class is None:
            self.skipTest("Couldn't infer RoPE layer for this model class.")

        # First make sure that validation on default config raises no rope-related warnings
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("", cl.out)
        logger.warning_once.cache_clear()

        # Axial rope type expects only `rope_theta`, otherwise raises warning
        vision_config.rope_parameters["factor"] = 0.25
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("Unrecognized keys in `rope_parameters` for 'rope_type'='axial': {'factor'}\n", cl.out)
        del vision_config.rope_parameters["factor"]
        logger.warning_once.cache_clear()

        inv_freq, attention_scale = rope_class.compute_axial_rope_parameters(config=vision_config)
        rope_module = rope_class(vision_config).to(device=torch_device)

        self.assertTrue(hasattr(rope_module, "inv_freq"))
        self.assertTrue(hasattr(rope_module, "attention_scaling"))
        self.assertEqual(attention_scale, 1.0)  # attention scale is always 1
        torch.testing.assert_close(inv_freq, rope_module.inv_freq.cpu())

        # create 2D position IDs for a single grid of one row and 10 cols `size=(10, 2)`
        position_ids = torch.stack(
            [
                torch.arange(10, dtype=torch.long, device=torch_device),
                torch.zeros(10, dtype=torch.long, device=torch_device),
            ]
        ).transpose(0, 1)
        position_ids = position_ids[None, ...].repeat(3, 1, 1)  # batch size of `3`
        # and an empty hidden states used only to infer device/dtype
        hidden_states = torch.empty(1, dtype=torch.float32, device=torch_device)
        cos, sin = rope_module(hidden_states, position_ids)
        self.assertEqual(cos.shape[-1], inv_freq.shape[-1] * 4)  # the freq are `//4` of head dim
        self.assertEqual(cos.shape[0], 3)  # angles presserve batch

    @unittest.skip("The tester has no audios in input dict")
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no audios in input dict")
    def test_get_audio_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no audios in input dict")
    def test_get_audio_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Gemma4 needs correct embeddings for per-layer-input computation, random won't work!")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip(
        "Randomly starts failing after module order changed in the __init__ because accelertate is not robust enough"
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        "Randomly starts failing after module order changed in the __init__ because accelertate is not robust enough"
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        "Randomly starts failing after module order changed in the __init__ because accelertate is not robust enough"
    )
    def test_disk_offload_safetensors(self):
        pass

    def test_per_layer_inputs_are_correctly_forwarded(self):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        model = Gemma4ForConditionalGeneration(config).to(torch_device)
        model.eval()

        input_ids = torch.randint(20, 50, (1, 10), device=torch_device)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        per_layer_inputs = model.model.language_model.get_per_layer_inputs(input_ids, None)

        @contextmanager
        def count_get_per_layer_inputs_calls():
            original = Gemma4TextModel.get_per_layer_inputs
            counter = {"call_count": 0}

            def count_calls(*args, **kwargs):
                nonlocal counter
                counter["call_count"] += 1
                return original(*args, **kwargs)

            Gemma4TextModel.get_per_layer_inputs = count_calls
            try:
                yield counter
            finally:
                Gemma4TextModel.get_per_layer_inputs = original

        # We should never call `get_per_layer_input_embeddings` if we provide both inputs_embeds and per_layer_inputs
        with count_get_per_layer_inputs_calls() as counter:
            _ = model(inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs)
            self.assertEqual(counter["call_count"], 0)

        # We should call it once if we provide only input_ids
        with count_get_per_layer_inputs_calls() as counter:
            _ = model(input_ids)
            self.assertEqual(counter["call_count"], 1)

        # We should call it once as well if we provide only inputs_embeds
        with count_get_per_layer_inputs_calls() as counter:
            _ = model(inputs_embeds=inputs_embeds)
            self.assertEqual(counter["call_count"], 1)

    @parameterized.expand([True, False, None])
    def test_get_image_features_output(self, return_dict: bool | None):
        "Override to infer last hidden states' `batch_size` from image position ids"
        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_image_features"):
                continue

            config, inputs_dict = self._image_features_prepare_config_and_inputs()
            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval()
            model = model.to(torch_device)

            set_seed(42)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs_dict)

            if return_dict in (True, None):
                self.assertTrue(isinstance(outputs, ModelOutput), "get_image_features() must return a BaseModelOutput")
                self.assertTrue(
                    hasattr(outputs, "last_hidden_state"),
                    "get_image_features() must return a BaseModelOutput with last_hidden_state",
                )
                self.assertTrue(
                    hasattr(outputs, "pooler_output"),
                    "get_image_features() must return a BaseModelOutput with pooler_output",
                )
                self.assertTrue(
                    hasattr(outputs, "hidden_states"),
                    "get_image_features() must return a BaseModelOutput with hidden_states",
                )
                if self.has_attentions:
                    self.assertTrue(
                        hasattr(outputs, "attentions"),
                        "get_image_features() must return a BaseModelOutput with attentions",
                    )

                if getattr(self, "skip_test_image_features_output_shape", False):
                    return

                last_hidden_state_shape = outputs.last_hidden_state.shape
                batch_size = (
                    inputs_dict["pixel_values"].shape[0]
                    if "pixel_values" in inputs_dict
                    else inputs_dict["pixel_values_images"].shape[0]
                )
                output_length = inputs_dict["pixel_values"].shape[-2] // (
                    model.config.vision_config.pooling_kernel_size**2
                )
                k_squared = int((inputs_dict["image_position_ids"].shape[1] // output_length) ** 0.5) ** 2
                batch_size *= inputs_dict["image_position_ids"].shape[1] // k_squared

                self.assertEqual(
                    last_hidden_state_shape[0],
                    batch_size,
                    f"batch_size mismatch, full shape: {last_hidden_state_shape}",
                )

                vision_config = config.vision_config if hasattr(config, "vision_config") else config
                vision_config = (
                    vision_config.backbone_config if hasattr(vision_config, "backbone_config") else vision_config
                )
                vision_config = vision_config.vq_config if hasattr(vision_config, "vq_config") else vision_config
                vision_config = vision_config.model_args if hasattr(vision_config, "model_args") else vision_config
                attribute_candidates = [
                    "embed_dim_per_stage",
                    "embed_dim",
                    "embed_dims",
                    "out_hidden_size",
                    "hidden_size",
                    "hidden_dim",
                ]
                hidden_size = None
                for attr in attribute_candidates:
                    if hasattr(vision_config, attr):
                        hidden_size = getattr(vision_config, attr)
                        break
                    elif isinstance(vision_config, dict) and attr in vision_config:
                        hidden_size = vision_config[attr]
                        break
                else:
                    raise ValueError("Cannot find the hidden size attribute in vision_config")
                if isinstance(hidden_size, (list, tuple)):
                    hidden_size = hidden_size[-1]
                self.assertEqual(
                    last_hidden_state_shape[-1],
                    hidden_size,
                    f"hidden_size mismatch, full shape: {last_hidden_state_shape}",
                )

                self.assertEqual(
                    len(outputs.pooler_output),
                    self.model_tester.batch_size,
                    f"batch_size mismatch for `pooler_output`: {len(outputs.pooler_output)} != {self.model_tester.batch_size}",
                )
                self.assertEqual(
                    outputs.pooler_output[0].ndim,
                    2,
                    f"each sample in `pooler_output` should be a 2D array but got {outputs.pooler_output[0].ndim}",
                )
            else:
                self.assertIsInstance(outputs, tuple, "get_image_features() must return a tuple if return_dict=False")

    def test_attention_mask_composition(self):
        config = self.model_tester.get_config()
        config.text_config._attn_implementation = "eager"

        # Override sliding window to a known small value to test truncation
        sliding_window = 4
        config.text_config.sliding_window = sliding_window

        # Create a sequence of 13 tokens: 0..4 text, 5..11 image (7 tokens), 12 text
        # block_sequence_ids maps image tokens to group 0, and text tokens to -1
        block_sequence_ids = torch.tensor([[-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1]], dtype=torch.long)
        attention_mask = torch.ones((1, 13), dtype=torch.bool)
        position_ids = torch.arange(13).unsqueeze(0)
        inputs_embeds = torch.randn(1, 13, config.text_config.hidden_size)

        mask_dict = create_masks_for_vision_model(
            config=config.text_config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
            block_sequence_ids=block_sequence_ids,
        )

        full_mask = mask_dict["full_attention"]
        sliding_mask = mask_dict["sliding_attention"]

        # In full_attention (global layers), Gemma 4 uses causal-only masking —
        # no bidirectional attention on vision tokens. This matches the internal
        # Gemax/Gemini3 transformer which sets bidirectional_segment_ids=None
        # for GLOBAL layers.
        # Token 5 looking ahead at token 11 -> MASKED (causal prevents look-ahead)
        self.assertLess(full_mask[0, 0, 5, 11].item(), -1000)
        # Token 11 looking back at token 5 -> VISIBLE (causal allows look-back)
        self.assertEqual(full_mask[0, 0, 11, 5].item(), 0.0)

        # In sliding_attention (local layers), bidirectional IS applied within the window.
        # Token 8 looking back at 5 (dist 3 < 4) -> VISIBLE
        self.assertEqual(sliding_mask[0, 0, 8, 5].item(), 0.0)
        # Token 5 looking ahead at 8 (dist 3 < 4, same image block) -> VISIBLE (bidirectional)
        self.assertEqual(sliding_mask[0, 0, 5, 8].item(), 0.0)

        # In sliding_attention, look-back outside the sliding window is strictly masked
        # Token 11 looking back at 5 (dist 6 > 4) -> MASKED
        self.assertLess(sliding_mask[0, 0, 11, 5].item(), -1000)

        # Verify that causal masking still applies correctly to text
        # Token 11 (image) looking ahead at Token 12 (text) -> MASKED
        self.assertLess(full_mask[0, 0, 11, 12].item(), -1000)

    def test_vision_mask_with_cache_beyond_sliding_window(self):
        """Regression test, see the Gemma 3 test of the same name.

        Once the cache is longer than the sliding window, sliding and full attention layers report
        different `kv_length`s. The vision mask has to be built for a sliding layer, otherwise the
        sliding mask ends up sized against a full attention layer and the forward pass crashes.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config._attn_implementation = "eager"
        config.text_config.sliding_window = 4

        model = Gemma4ForConditionalGeneration(config).to(torch_device).eval()
        batch_size, prompt_length = inputs_dict["input_ids"].shape
        past_key_values = StaticCache(
            config=config.get_text_config(),
            max_batch_size=batch_size,
            max_cache_len=prompt_length + 8,  # longer than the sliding window
            device=torch_device,
            dtype=model.dtype,
        )

        with torch.no_grad():
            model(**inputs_dict, past_key_values=past_key_values, use_cache=True)


@slow
@require_torch_accelerator
class Gemma4IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "google/gemma-4-E2B-it"
        self.processor = Gemma4Processor.from_pretrained(self.model_name)

        self.url1 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        self.url2 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/australia.jpg"
        )
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url1},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_deterministic_for_xpu
    def test_model_with_image(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean** in the background under a **clear'],
                ("xpu", 5): ['This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean** in the background under a **clear'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_with_image_batch(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": self.url1,
                    },
                    {"type": "image", "url": self.url2},
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [self.messages, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): [
                    "This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background",
                    "No, these images are **not identical**.\n\nHere's a breakdown of the differences:\n\n1.  **Image 1 (Cow on",
                ],
                ("xpu", 5): [
                    "This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean** in the background under a **clear",
                    "No, these images are **not identical**.\n\nHere's a breakdown of the differences:\n\n1.  **Image 1 (Cow on",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_multiimage(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url2},
                    {"type": "text", "text": "What do you see here?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['Based on the image, here is a description of what I see:\n\n**Foreground & Street Scene:**\n* **Roadway:** There is an'],
                ("cuda", (9, 0)): ['Based on the image, here is a description of what I see:\n\n**Foreground & Street Scene:**\n* **Roadway:** There is an'],
                ("xpu", 5): ['Based on the image, here is a description of what I see:\n\n**Foreground & Street Scene:**\n* **Roadway:** There is an'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_multi_gpu
    def test_model_text_only_multigpu(self):
        """Accelerate destroys the input dict `shared_kv_states` if it's not passed as kwarg and part of
        `_skip_keys_device_placement`, so test this to avoid regresions.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA loom of logic, spun from endless thread,\nWhere data streams in, and the patterns spread.\nNo'],
                ("cuda", (9, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_text_only(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA loom of logic, spun from endless thread,\nWhere data streams in, and the patterns spread.\nNo'],
                ("cuda", (9, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("xpu", 5): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_states_sharing_with_and_without_cache(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Who are you? What can you do?"}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)
        input_size = inputs.input_ids.shape[-1]

        # With and without cache generatiom should share kv states the same way
        output_with_cache = model.generate(**inputs, max_new_tokens=30, do_sample=False, use_cache=True)
        output_without_cache = model.generate(**inputs, max_new_tokens=30, do_sample=False, use_cache=False)

        output_text_with_cache = tokenizer.batch_decode(output_with_cache[:, input_size:], skip_special_tokens=True)
        output_text_without_cache = tokenizer.batch_decode(
            output_without_cache[:, input_size:], skip_special_tokens=True
        )

        self.assertEqual(output_text_with_cache, output_text_without_cache)

    # Note: we do not test FA2 as the head dim is 512 on some layers, which is not compatible with the kernels
    @parameterized.expand([("sdpa",), ("eager",)])
    @require_deterministic_for_accelerator(devices=["cuda"])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. Outputs for every attention functions
        should be coherent and identical.
        """

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding="left")
        input_text = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in input_text
        ]
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = Gemma4ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=torch_device,
            attn_implementation=attn_implementation,
        )

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.get_text_config().sliding_window)

        out = model.generate(**inputs, max_new_tokens=16, do_sample=False, cache_implementation="static")
        output_text = tokenizer.batch_decode(out[:, input_size:])

        EXPECTED_COMPLETIONS = Expectations(
            {
                ("cuda", 8): [
                    "That sounds lovely! It seems like you're really enjoying the place you'"
                    if attn_implementation == "sdpa"
                    else "That sounds like a very pleasant place! It seems like you're really enjoying",
                    "Here are a few ways you could use or expand upon that list, depending on",
                ],
                ("xpu", 5): [
                    "That sounds lovely! It seems like you're really enjoying the place you'",
                    "Here are a few ways you could use or expand upon that list, depending on",
                ],
            }
        )
        self.assertEqual(output_text, EXPECTED_COMPLETIONS.get_expectation())

    @pytest.mark.torch_export_test
    def test_export_text_only(self):
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        # Run on CPU: the full E2B model (~4 GiB bfloat16) + torch.export tracing overhead
        # (~4 GiB) exceeds the 22.3 GiB GPU memory available in CI. CPU avoids the OOM.
        # max_cache_len=19 covers the prompt (~16 tokens) + 3 new tokens with a small buffer.
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        exportable_module = TorchExportableModuleForDecoderOnlyLM(model, batch_size=1, max_cache_len=19, device="cpu")
        exported_program = exportable_module.export(
            input_ids=torch.tensor([[1]], device="cpu", dtype=torch.long),
        )

        # Test generation with the exported model
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        max_new_tokens_to_generate = 3
        # Generate text with the exported model
        export_generated_text = TorchExportableModuleForDecoderOnlyLM.generate(
            exported_program, tokenizer, prompt, max_new_tokens=max_new_tokens_to_generate, device="cpu"
        )

        input_text = tokenizer(prompt, return_tensors="pt").to("cpu")
        eager_outputs = model.generate(
            **input_text,
            max_new_tokens=max_new_tokens_to_generate,
            do_sample=False,  # Use greedy decoding to match the exported model
        )

        eager_generated_text = tokenizer.decode(eager_outputs[0], skip_special_tokens=True)
        self.assertEqual(export_generated_text, eager_generated_text)
