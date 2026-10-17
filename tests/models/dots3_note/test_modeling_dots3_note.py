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
"""Tests for the PyTorch Dots 3 Note Preview model."""

import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pytest
from parameterized import parameterized
from safetensors import safe_open

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    Dots3NoteAudioConfig,
    Dots3NoteConfig,
    Dots3NoteForCausalLM,
    Dots3NoteForConditionalGeneration,
    Dots3NoteModel,
    Dots3NoteTextConfig,
    Dots3NoteTextModel,
    Dots3NoteVisionConfig,
    is_torch_available,
)
from transformers.cache_utils import (
    StaticCache,
    StaticIndexedLayer,
    StaticSlidingWindowLayer,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.testing_utils import (
    require_accelerate,
    require_torch,
    require_torch_multi_gpu,
    require_torchvision,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    sdpa_kernel,
)
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers.models.dots3_note.modeling_dots3_note import (
        Dots3NoteAudioModel,
        Dots3NoteTextAttention,
        Dots3NoteVisionModel,
        eager_attention_forward,
    )
    from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids


def get_tiny_config(use_dsa=False):
    vision_config = {
        "embed_dim": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "pyramid_num_routed": [-1, 2],
        "capacity_factor": 2,
        "adapter_in_dim": 32,
        "adapter_out_dim": 32,
        "adapter_merge_size": 2,
    }
    audio_config = {
        "whisper_config": {
            "d_model": 32,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64,
            "encoder_layers": 2,
            "num_mel_bins": 8,
            "max_source_positions": 32,
            "activation_function": "swiglu",
        },
        "feature_size": 8,
        "hop_length": 4,
        "downsample_hidden_size": 4,
        "adapter_input_size": 32,
        "adapter_output_size": 32,
    }
    config = Dots3NoteConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        head_dim=16,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window_size=4,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=4,
        swa_q_lora_rank=16,
        swa_kv_lora_rank=16,
        swa_head_dim=16,
        swa_qk_nope_head_dim=8,
        swa_qk_rope_head_dim=8,
        swa_v_head_dim=8,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=4,
        use_dsa=use_dsa,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_experts_intermediate_size=16,
        first_k_dense_replace=1,
        image_token_id=120,
        video_token_id=122,
        audio_token_id=121,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        vision_config=vision_config,
        audio_config=audio_config,
    )
    config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    return config


class Dots3NoteTextModelTester(CausalLMModelTester):
    # The common tester's constructor-signature filter omits the inherited per_layer_config argument.
    forced_config_args = ["pad_token_id", "per_layer_config"]
    base_model_class = Dots3NoteTextModel
    causal_lm_class = Dots3NoteForCausalLM

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("intermediate_size", 256)
        super().__init__(parent=parent, **kwargs)
        self.q_lora_rank = 16
        self.kv_lora_rank = 16
        self.qk_rope_head_dim = self.head_dim // 2
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.v_head_dim = self.head_dim
        self.layer_types = ["indexed_attention", "sliding_attention"] * (self.num_hidden_layers // 2)
        self.sliding_window = self.max_position_embeddings
        self.per_layer_config = {}
        self.index_n_heads = 2
        self.index_head_dim = 128
        self.index_topk = 4
        self.n_routed_experts = 4
        self.n_shared_experts = 1
        self.num_experts_per_tok = 2
        self.moe_intermediate_size = 64
        self.shared_experts_intermediate_size = 16


@require_torch
class Dots3NoteTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Dots3NoteTextModelTester
    test_all_params_have_gradient = False  # The shared DSA indexer has no auxiliary training loss.

    def setUp(self):
        super().setUp()
        # DSA and SWA have different head counts, so no global num_attention_heads exists.
        self.config_tester.common_properties = ["hidden_size", "num_hidden_layers"]

    @unittest.skip("DSA hard top-k selection is sensitive to sequence packing, as in GLM-MoE-DSA.")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("DSA hard top-k selection is sensitive to sequence packing, as in GLM-MoE-DSA.")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass


@require_torch
class Dots3NoteVision2TextModelTester(VLMModelTester):
    base_model_class = Dots3NoteModel
    config_class = Dots3NoteConfig
    text_config_class = Dots3NoteTextConfig
    vision_config_class = Dots3NoteVisionConfig
    conditional_generation_class = Dots3NoteForConditionalGeneration
    modality = "vision"
    pipeline_model_mapping = {}

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("audio_token_id", 5)
        kwargs.setdefault("image_size", 4)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("embed_dim", 64)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("moe_intermediate_size", 64)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault("mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("num_local_experts", 2)
        kwargs.setdefault("num_experts_per_tok", 2)
        kwargs.setdefault("adapter_in_dim", 64)
        kwargs.setdefault("adapter_out_dim", 64)
        kwargs.setdefault("adapter_merge_size", 2)
        super().__init__(parent, **kwargs)
        self.encoder_seq_length = self.seq_length

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.image_token_id, self.video_token_id, self.audio_token_id}

    def get_text_config(self):
        return Dots3NoteTextModelTester(
            self.parent, num_attention_heads=self.num_attention_heads, num_key_value_heads=self.num_attention_heads
        ).get_config()

    def create_pixel_values(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return torch.randn(batch_size * 4, self.num_channels * self.patch_size**2, device=torch_device)

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        input_ids[:, 1] = config.video_token_id
        grid = torch.tensor([[1, 2, 2]], device=torch_device).repeat(batch_size, 1)
        return {
            "image_grid_thw": grid,
            "pixel_values_videos": modality_inputs.clone(),
            "video_grid_thw": grid.clone(),
        }


class Dots3NoteAudio2TextModelTester(ALMModelTester):
    base_model_class = Dots3NoteModel
    config_class = Dots3NoteConfig
    text_config_class = Dots3NoteTextConfig
    audio_config_class = Dots3NoteAudioConfig
    audio_mask_key = "chunk_sample_lengths"
    conditional_generation_class = Dots3NoteForConditionalGeneration
    modality = "audio"
    pipeline_model_mapping = {}

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("audio_token_id", 5)
        kwargs.setdefault("feat_seq_length", 16)
        kwargs.setdefault("num_mel_bins", 8)
        kwargs.setdefault("feature_size", 8)
        kwargs.setdefault("hop_length", 4)
        kwargs.setdefault("downsample_hidden_size", 4)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("adapter_input_size", 64)
        kwargs.setdefault("adapter_output_size", 64)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("intermediate_size", 64)
        super().__init__(parent, **kwargs)
        self.encoder_seq_length = self.seq_length

    def get_text_config(self):
        return Dots3NoteTextModelTester(
            self.parent, num_attention_heads=self.num_attention_heads, num_key_value_heads=self.num_attention_heads
        ).get_config()

    def create_audio_mask(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return torch.ones(batch_size, self.feat_seq_length, device=torch_device, dtype=torch.long)

    def get_audio_embeds_mask(self, audio_mask):
        return audio_mask[:, ::8]

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return {
            "chunk_sample_lengths": torch.full(
                (batch_size,), self.feat_seq_length * self.hop_length, device=torch_device
            )
        }


class Dots3NoteMultimodalTestMixin:
    test_all_params_have_gradient = False
    additional_model_inputs = ["image_grid_thw", "pixel_values_videos", "video_grid_thw", "chunk_sample_lengths"]

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        # Keep packed patches and their grids together instead of slicing them as batch-first tensors.
        dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
        for model_class in self.all_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
            set_config_for_less_flaky_test(config)
            inputs = {key: value.to(dtype) if value.is_floating_point() else value for key, value in inputs.items()}
            input_ids = inputs["input_ids"]
            padding = torch.full_like(input_ids[:, :2], config.text_config.pad_token_id)
            left = padding_side == "left"
            inputs["input_ids"] = torch.cat((padding, input_ids) if left else (input_ids, padding), dim=1)
            mask = torch.cat(
                (torch.zeros_like(padding), torch.ones_like(input_ids))
                if left
                else (torch.ones_like(input_ids), torch.zeros_like(padding)),
                dim=1,
            )
            inputs["attention_mask"] = mask if use_attention_mask else None
            inputs["output_attentions"] = output_attentions
            model = model_class(config).to(device=torch_device, dtype=dtype).eval()
            set_model_for_less_flaky_test(model)
            with torch.no_grad(), sdpa_kernel(enable_kernels, True, enable_kernels):
                model.set_attn_implementation("eager")
                for layer in model.base_model.language_model.layers:
                    self.assertEqual(layer.self_attn.config._attn_implementation, "eager")
                eager = model(**inputs)
                model.set_attn_implementation("sdpa")
                for layer in model.base_model.language_model.layers:
                    self.assertEqual(layer.self_attn.config._attn_implementation, "sdpa")
                sdpa = model(**inputs)
            key = "logits" if "logits" in eager else "last_hidden_state"
            tolerance = 1e-5 if dtype == torch.float32 else 2e-3 if dtype == torch.float16 else 2e-2
            torch.testing.assert_close(eager[key][mask.bool()], sdpa[key][mask.bool()], atol=tolerance, rtol=tolerance)

    def setUp(self):
        super().setUp()
        if self._testMethodName in ("test_reverse_loading_mapping", "test_sdpa_can_dispatch_composite_models"):
            configs = {
                **Dots3NoteVision2TextModelTester(self)._build_modality_sub_configs(),
                **Dots3NoteAudio2TextModelTester(self)._build_modality_sub_configs(),
            }
            self.enterContext(
                patch.object(
                    self.model_tester,
                    "_build_modality_sub_configs",
                    return_value=configs,
                )
            )
        modality = self.model_tester.modality
        if "_audio_features_" in self._testMethodName and modality != "audio":
            self.skipTest("This tester has no audio inputs")
        if (
            "_image_features_" in self._testMethodName or "_video_features_" in self._testMethodName
        ) and modality != "vision":
            self.skipTest("This tester has no image/video inputs")


@require_torch
class Dots3NoteVision2TextModelTest(Dots3NoteMultimodalTestMixin, VLMModelTest, unittest.TestCase):
    model_tester_class = Dots3NoteVision2TextModelTester


@require_torch
class Dots3NoteAudio2TextModelTest(Dots3NoteMultimodalTestMixin, ALMModelTest, unittest.TestCase):
    model_tester_class = Dots3NoteAudio2TextModelTester

    def _audio_features_prepare_config_and_inputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        return config, {key: inputs[key] for key in ("input_features", "chunk_sample_lengths")}


@require_torch
class Dots3NoteIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    @parameterized.expand(["nested", "flat", "legacy"])
    def test_text_rope_parameters_roundtrip(self, parameter_format):
        values = get_tiny_config(use_dsa=True).text_config.to_dict()
        rope_parameters = {"rope_type": "linear", "rope_theta": 12345.0, "factor": 2.0}
        if parameter_format == "nested":
            values["rope_parameters"]["indexed_attention"] = rope_parameters.copy()
        elif parameter_format == "flat":
            values["rope_parameters"] = rope_parameters.copy()
        else:
            values.pop("rope_parameters")
            values["rope_scaling"] = rope_parameters.copy()
        config = Dots3NoteTextConfig(**values)

        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = Dots3NoteTextConfig.from_pretrained(directory)
        for candidate in (config, restored):
            for key, value in rope_parameters.items():
                self.assertEqual(candidate.rope_parameters["indexed_attention"][key], value)
            model = Dots3NoteTextModel(candidate)
            dim = candidate.per_layer_config["indexed_attention"].qk_rope_head_dim
            expected = 1.0 / (12345.0 ** (torch.arange(0, dim, 2).float() / dim)) / 2.0
            torch.testing.assert_close(model.rotary_emb.indexed_attention_inv_freq, expected)

    @parameterized.expand([False, True])
    def test_audio_chunk_groups_are_preserved_for_beams(self, use_inputs_embeds):
        torch.manual_seed(19)
        config = get_tiny_config()
        model = Dots3NoteForConditionalGeneration(config).eval()
        input_ids = torch.tensor([[1, 121, 121, 121, 2], [0, 0, 0, 1, 2], [0, 1, 121, 121, 2]])
        inputs = {
            "attention_mask": input_ids.ne(0),
            "input_features": torch.randn(3, config.audio_config.feature_size, 16),
            "chunk_sample_lengths": torch.tensor([32, 64, 64]),
        }
        if use_inputs_embeds:
            inputs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
        else:
            inputs["input_ids"] = input_ids
        with torch.no_grad():
            greedy = model.generate(
                **inputs, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_logits=True
            )
            beams = model.generate(
                **inputs,
                num_beams=2,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )
        torch.testing.assert_close(beams.logits[0], greedy.logits[0].repeat_interleave(2, dim=0))

    def test_vision_head_override_roundtrip(self):
        config = get_tiny_config().vision_config
        config_class = type(config)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            config = config_class.from_pretrained(directory, num_attention_heads=8)
            self.assertEqual(Dots3NoteVisionModel(config).blocks[0].attn.num_heads, 8)
            self.assertNotIn("num_heads", config.to_dict())
            config.save_pretrained(directory)
            self.assertEqual(config_class.from_pretrained(directory).num_attention_heads, 8)
        legacy = config.to_dict()
        legacy.pop("num_attention_heads")
        legacy["num_heads"] = 4
        converted = config_class(**legacy)
        self.assertEqual(converted.num_attention_heads, 4)
        self.assertNotIn("num_heads", converted.to_dict())
        config = config_class(capacity_factor=2, router_scale=0.5)
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            config = config_class.from_pretrained(directory, num_experts_per_tok=1, router_scaling_factor=2.0)
            config.save_pretrained(directory)
            restored = config_class.from_pretrained(directory)
        self.assertEqual(restored.num_experts_per_tok, 1)
        self.assertEqual(restored.router_scaling_factor, 2.0)
        self.assertEqual(restored.capacity_factor, 2)
        self.assertEqual(restored.router_scale, 0.5)

    def test_heterogeneous_attention_dimensions(self):
        config = get_tiny_config(use_dsa=True).text_config
        values = config.to_dict()
        values["per_layer_config"]["1"] = {
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "q_lora_rank": 8,
            "kv_lora_rank": 8,
            "head_dim": 32,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 24,
        }
        config = Dots3NoteTextConfig(**values)
        self.assertEqual(config.rope_parameters["sliding_attention"]["partial_rotary_factor"], 0.25)
        model = Dots3NoteForCausalLM(config).eval()
        attention = model.model.layers[1].self_attn
        self.assertEqual(attention.num_heads, 2)
        self.assertEqual(attention.q_a_proj.out_features, 8)
        self.assertEqual(attention.kv_lora_rank, 8)
        with torch.no_grad():
            output = model(torch.tensor([[1, 7, 2]]), use_cache=True)
        self.assertEqual(output.logits.shape, (1, 3, config.vocab_size))

    def test_legacy_text_config_loading(self):
        config = get_tiny_config()
        legacy = config.to_dict()
        legacy.update(legacy.pop("text_config"))
        legacy["model_type"] = "dots3_note"
        legacy.pop("sliding_window", None)
        legacy["sliding_window_size"] = 513
        quantization_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        quantized = Dots3NoteConfig(**(legacy | {"quantization_config": quantization_config}))
        self.assertEqual(quantized.quantization_config, quantization_config)
        self.assertEqual(quantized.text_config.model_type, "dots3_note_text")
        self.assertEqual(quantized.text_config.sliding_window, 513)
        model = AutoModelForCausalLM.from_config(Dots3NoteConfig(**legacy)).eval()
        input_ids = torch.tensor([[1, 7, 2]])
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            Dots3NoteConfig(**legacy).save_pretrained(directory)
            loaded = AutoModelForCausalLM.from_pretrained(directory).eval()
            self.assertEqual(loaded.config.sliding_window, 513)
            with torch.no_grad():
                torch.testing.assert_close(model(input_ids).logits, loaded(input_ids).logits)

    all_model_classes = (
        (
            Dots3NoteAudioModel,
            Dots3NoteForCausalLM,
            Dots3NoteForConditionalGeneration,
            Dots3NoteModel,
            Dots3NoteVisionModel,
        )
        if is_torch_available()
        else ()
    )

    def test_dsa_shared_mask_reaches_indexer_and_attention(self):
        config = get_tiny_config(use_dsa=True).text_config
        config._attn_implementation = "eager"
        attention = Dots3NoteTextAttention(config, layer_idx=0).eval()
        allowed = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()
        allowed[..., :2] = False
        mask = torch.zeros_like(allowed, dtype=torch.float).masked_fill(~allowed, -10000)
        indices = torch.tensor([[[0, 3, 4]] * 5], dtype=torch.int32)
        received = []

        def attention_spy(module, query, key, value, attention_mask, **kwargs):
            received.append((attention_mask, kwargs["position_ids"]))
            return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)

        hidden = torch.randn(1, 5, config.hidden_size)
        cos = torch.ones(1, 5, config.qk_rope_head_dim)
        position_ids = torch.tensor([[0, 0, 0, 1, 2]])
        with (
            patch.object(attention.indexer, "forward", return_value=indices) as indexer,
            patch.object(ALL_ATTENTION_FUNCTIONS, "get_interface", return_value=attention_spy),
            torch.no_grad(),
        ):
            attention(hidden, (cos, torch.zeros_like(cos)), mask, position_ids=position_ids)
        torch.testing.assert_close(indexer.call_args.args[3], mask[:, 0])
        self.assertIs(received[0][1], position_ids)
        torch.testing.assert_close(received[0][0][0, 0, -1] == 0, torch.tensor([False, False, False, True, True]))

    def test_dsa_chunked_prefill_matches_one_shot(self):
        config = get_tiny_config(use_dsa=True).text_config
        # Keep every key in this cache-parity test. Randomly initialized tiny indexers can produce exact
        # score ties, for which `topk` is allowed to pick different tied keys as the cached key width grows.
        # Sparse top-k math and sparse cached generation are covered independently below.
        config.index_topk = config.max_position_embeddings
        model = Dots3NoteForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 7, 11, 9, 6, 13, 12, 5, 8, 10, 2]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            expected = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
            past_key_values = None
            chunks = []
            for start in range(0, input_ids.shape[1], 3):
                stop = min(start + 3, input_ids.shape[1])
                outputs = model(
                    input_ids[:, start:stop],
                    attention_mask=attention_mask[:, :stop],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                chunks.append(outputs.logits)

        actual = torch.cat(chunks, dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_dsa_left_padded_batch_cache_matches_unpadded_decode(self):
        torch.manual_seed(0)
        config = get_tiny_config(use_dsa=True).text_config
        config.layer_types = ["indexed_attention"] * config.num_hidden_layers
        # Select every key so this test targets physical cache/padding coordinates rather than
        # the numerical discontinuity of hard top-k selection on a randomly initialized indexer.
        config.index_topk = config.max_position_embeddings
        model = Dots3NoteForCausalLM(config).eval()
        model.set_experts_implementation("eager")

        sequences = [
            torch.tensor([[1, 7, 11, 9, 2]]),
            torch.tensor([[1, 8, 6, 13, 12, 5, 10, 2]]),
        ]
        next_tokens = torch.tensor([[14], [15]])
        max_length = max(sequence.shape[1] for sequence in sequences)
        input_ids = torch.cat(
            [torch.nn.functional.pad(sequence, (max_length - sequence.shape[1], 0)) for sequence in sequences]
        )
        attention_mask = input_ids.ne(config.pad_token_id).long()
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)

        with torch.no_grad():
            prefill = model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            batched = model(
                next_tokens,
                attention_mask=torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=-1),
                position_ids=attention_mask.sum(-1, keepdim=True),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            ).logits[:, -1]

            unpadded = []
            for sequence, next_token in zip(sequences, next_tokens):
                sequence_mask = torch.ones_like(sequence)
                sequence_prefill = model(
                    sequence,
                    attention_mask=sequence_mask,
                    position_ids=torch.arange(sequence.shape[1]).unsqueeze(0),
                    use_cache=True,
                )
                unpadded.append(
                    model(
                        next_token.view(1, 1),
                        attention_mask=torch.ones(1, sequence.shape[1] + 1, dtype=torch.long),
                        position_ids=torch.tensor([[sequence.shape[1]]]),
                        past_key_values=sequence_prefill.past_key_values,
                        use_cache=True,
                    ).logits[:, -1]
                )

        torch.testing.assert_close(batched, torch.cat(unpadded), rtol=1e-4, atol=1e-4)

    def test_dsa_static_cache_matches_dynamic_cache(self):
        torch.manual_seed(0)
        config = get_tiny_config(use_dsa=True).text_config
        model = Dots3NoteForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 7, 11, 9]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            dynamic = model(input_ids, attention_mask=attention_mask, use_cache=True)
            next_token = dynamic.logits[:, -1:].argmax(dim=-1)
            decode_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=-1)
            dynamic_decode = model(
                next_token,
                attention_mask=decode_mask,
                past_key_values=dynamic.past_key_values,
                use_cache=True,
            )

            static_cache = StaticCache(config=config, max_cache_len=16)
            static = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=static_cache,
                use_cache=True,
            )
            static_decode = model(
                next_token,
                attention_mask=decode_mask,
                past_key_values=static.past_key_values,
                use_cache=True,
            )

        self.assertIsInstance(static_cache.layers[0], StaticIndexedLayer)
        self.assertIsInstance(static_cache.layers[1], StaticSlidingWindowLayer)
        torch.testing.assert_close(static_decode.logits, dynamic_decode.logits, rtol=1e-4, atol=1e-4)

    def test_dsa_precomputed_mask_uses_dispatched_layer_type(self):
        model = Dots3NoteForCausalLM(get_tiny_config(use_dsa=True).text_config).eval()
        full_mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        sliding_mask = torch.eye(2, dtype=torch.bool).view(1, 1, 2, 2)

        received = []
        handles = [
            layer.register_forward_pre_hook(
                lambda module, args, kwargs: received.append(kwargs["attention_mask"]), with_kwargs=True
            )
            for layer in model.model.layers
        ]
        try:
            with torch.no_grad():
                model(
                    torch.tensor([[1, 2]]),
                    attention_mask={"indexed_attention": full_mask, "sliding_attention": sliding_mask},
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        self.assertIs(received[0], full_mask)
        self.assertIs(received[1], sliding_mask)

    @parameterized.expand([("eager", False), ("eager", True), ("sdpa", False)])
    def test_vision_forward(self, backend, config_attentions):
        config = get_tiny_config().vision_config
        config = type(config)(**(config.to_dict() | {"is_causal": False}))
        config._attn_implementation = backend
        config.output_attentions = config_attentions
        model = Dots3NoteVisionModel(config).eval()
        pixel_values = torch.randn(20, 3 * config.patch_size**2)
        grid_thw = torch.tensor([[1, 2, 2], [1, 4, 4]])
        with torch.no_grad():
            reference = model(pixel_values, grid_thw, output_attentions=False)
            outputs = model(
                pixel_values,
                grid_thw,
                output_hidden_states=True,
                **({} if config_attentions else {"output_attentions": True}),
            )
        self.assertEqual(outputs.last_hidden_state.shape, (20, config.embed_dim))
        self.assertEqual(outputs.pooler_output.shape, (5, config.adapter_out_dim))
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
        torch.testing.assert_close(outputs.last_hidden_state, reference.last_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(outputs.pooler_output, reference.pooler_output, rtol=0, atol=0)
        if backend == "eager":
            self.assertEqual(len(outputs.attentions), config.num_hidden_layers)
            for layer_attentions in outputs.attentions:
                self.assertEqual(len(layer_attentions), 2)
                for attention, length in zip(layer_attentions, (4, 16)):
                    self.assertEqual(attention.shape, (1, config.num_attention_heads, length, length))
                    self.assertTrue((attention >= 0).all())
                    torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))
        else:
            self.assertEqual(outputs.attentions, ())

    @parameterized.expand(["eager", "sdpa"])
    @pytest.mark.torch_compile_test
    def test_vision_compile_fullgraph(self, backend):
        config = get_tiny_config().vision_config
        config._attn_implementation = backend
        model = Dots3NoteVisionModel(config).to(torch_device).eval()
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)

        for grid_values in ([[1, 2, 2], [1, 4, 4]], [[2, 2, 2], [1, 4, 4]], [[1, 4, 4], [1, 2, 4]]):
            with self.subTest(grid=grid_values):
                grid_thw = torch.tensor(grid_values, device=torch_device)
                num_patches = sum(t * h * w for t, h, w in grid_values)
                pixel_values = torch.randn(num_patches, 3 * config.patch_size**2, device=torch_device)
                metadata = {
                    "position_ids": get_vision_position_ids(grid_thw, config.spatial_merge_size),
                    "cu_seqlens": get_vision_cu_seqlens(grid_thw),
                }
                with torch.no_grad():
                    expected = model(pixel_values, grid_thw, **metadata)
                    actual = compiled(pixel_values, grid_thw, **metadata)
                torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(actual.pooler_output, expected.pooler_output, atol=1e-5, rtol=1e-5)

    def test_audio_forward(self):
        config = get_tiny_config().audio_config
        config._attn_implementation = "eager"
        model = Dots3NoteAudioModel(config).eval()
        inputs = {
            "input_features": torch.randn(3, config.feature_size, 32),
            "chunk_sample_lengths": torch.tensor([33, 128, 65]),
        }
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        self.assertEqual(outputs.last_hidden_state.shape, (3, 4, config.hidden_size))
        self.assertEqual(outputs.pooler_output.shape, (9, config.hidden_size))
        token_lengths = torch.tensor([2, 4, 3])
        valid_tokens = outputs.last_hidden_state[torch.arange(4)[None, :] < token_lengths[:, None]]
        torch.testing.assert_close(outputs.pooler_output, valid_tokens, rtol=0, atol=0)
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
        self.assertEqual(len(outputs.attentions), config.num_hidden_layers)

    @parameterized.expand([0, 1])
    @require_accelerate
    @require_torch_multi_gpu
    def test_audio_forward_split_devices(self, main_device):
        from accelerate import dispatch_model

        config = get_tiny_config().audio_config
        config._attn_implementation = "eager"
        model = Dots3NoteAudioModel(config).eval().to(f"{torch_device}:0")
        inputs = {
            "input_features": torch.randn(3, config.feature_size, 32, device=f"{torch_device}:0"),
            "chunk_sample_lengths": torch.tensor([33, 128, 65], device=f"{torch_device}:0"),
        }
        with torch.no_grad():
            expected = model(**inputs).pooler_output
            dispatch_model(
                model,
                main_device=f"{torch_device}:{main_device}",
                device_map={
                    "conv_stem": 1,
                    "rotary_embedding": 1,
                    "layers.0": 1,
                    "layers.1": 0,
                    "layer_norm": 0,
                },
            )
            actual = model(**inputs).pooler_output
        torch.testing.assert_close(actual.to(expected.device), expected)

    @parameterized.expand(["base", "conditional_generation"])
    def test_multimodal_forward(self, variant):
        config = get_tiny_config()
        model_class = {
            "base": Dots3NoteModel,
            "conditional_generation": Dots3NoteForConditionalGeneration,
        }[variant]
        model = model_class(config).eval()
        input_ids = torch.tensor([[1, 120, 5, 121, 121, 122, 2]])
        pixel_values = torch.randn(4, config.vision_config.num_channels * config.vision_config.patch_size**2)
        media_inputs = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "pixel_values_videos": pixel_values,
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "input_features": torch.randn(1, config.audio_config.feature_size, 16),
            "chunk_sample_lengths": torch.tensor([64]),
        }

        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False, output_router_logits=True, **media_inputs)
            self.assertEqual(len(outputs.router_logits), config.text_config.mlp_layer_types.count("sparse"))
            for logits in outputs.router_logits:
                self.assertEqual(logits.shape, (input_ids.numel(), config.text_config.n_routed_experts))
                self.assertTrue(torch.isfinite(logits).all())
            embedded_outputs = model(
                inputs_embeds=model.get_input_embeddings()(input_ids), use_cache=False, **media_inputs
            )
            torch.testing.assert_close(embedded_outputs[0], outputs[0])
            backbone = model if variant == "base" else model.model
            encoder_outputs = {
                "image": backbone.get_image_features(pixel_values, media_inputs["image_grid_thw"]),
                "video": backbone.get_video_features(pixel_values, media_inputs["video_grid_thw"]),
            }
            with patch.object(backbone.vision_tower, "forward", wraps=backbone.vision_tower.forward) as vision:
                cached_outputs = model(
                    input_ids=input_ids,
                    use_cache=False,
                    mm_encoder_outputs=encoder_outputs,
                    input_features=media_inputs["input_features"],
                    chunk_sample_lengths=media_inputs["chunk_sample_lengths"],
                )
            vision.assert_not_called()
            torch.testing.assert_close(cached_outputs[0], outputs[0], rtol=0, atol=0)
            precomputed = {}
            for modality in ("image", "video"):
                grid = media_inputs[f"{modality}_grid_thw"]
                precomputed[f"{modality}_position_ids"] = get_vision_position_ids(
                    grid, config.vision_config.spatial_merge_size
                )
                precomputed[f"{modality}_cu_seqlens"] = get_vision_cu_seqlens(grid)
                precomputed[f"{modality}_max_seqlen"] = 4
            with patch.object(backbone.vision_tower, "forward", wraps=backbone.vision_tower.forward) as vision:
                actual = model(input_ids=input_ids, use_cache=False, **media_inputs, **precomputed)
            torch.testing.assert_close(actual[0], outputs[0], rtol=0, atol=0)
            self.assertEqual(vision.call_count, 2)
            for modality, call in zip(("image", "video"), vision.call_args_list):
                for name in ("position_ids", "cu_seqlens", "max_seqlen"):
                    self.assertIs(call.kwargs[name], precomputed[f"{modality}_{name}"])
                self.assertFalse(any(key.startswith(("image_", "video_")) for key in call.kwargs))
            for modality, args in (
                ("image", (pixel_values, media_inputs["image_grid_thw"])),
                ("video", (pixel_values, media_inputs["video_grid_thw"])),
                ("audio", (media_inputs["input_features"], media_inputs["chunk_sample_lengths"])),
            ):
                with self.subTest(modality=modality):
                    get_features = getattr(backbone, f"get_{modality}_features")
                    expected = get_features(*args, return_dict=True).to_tuple()
                    actual = get_features(*args, return_dict=False)
                    self.assertIsInstance(actual, tuple)
                    torch.testing.assert_close(actual, expected)
                    features = get_features(*args, return_dict=True).pooler_output
                    if modality != "audio":
                        features = torch.cat(features, dim=0)
                    token_id = getattr(config, f"{modality}_token_id")
                    inputs_embeds = model.get_input_embeddings()(input_ids)
                    for ids in (input_ids, None):
                        mask = backbone.get_placeholder_mask(ids, inputs_embeds, features, token_id)
                        torch.testing.assert_close(
                            mask, (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
                        )
                        for count in (features.shape[0] - 1, features.shape[0] + 1):
                            with self.assertRaisesRegex(ValueError, "features and placeholder tokens do not match"):
                                backbone.get_placeholder_mask(
                                    ids, inputs_embeds, features.new_zeros(count, features.shape[-1]), token_id
                                )
            config.return_dict = False
            tuple_outputs = model(input_ids=input_ids, use_cache=False, return_dict=False, **media_inputs)
            torch.testing.assert_close(tuple_outputs[0], outputs[0])
            config.return_dict = True
            if variant == "conditional_generation":
                generated = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=2,
                    min_new_tokens=2,
                    do_sample=False,
                    **media_inputs,
                )
                self.assertEqual(generated.shape, (1, input_ids.shape[1] + 2))
        output = outputs.last_hidden_state if variant == "base" else outputs.logits
        width = config.text_config.hidden_size if variant == "base" else config.text_config.vocab_size
        self.assertEqual(output.shape, (1, input_ids.shape[1], width))
        self.assertTrue(torch.isfinite(output).all())

    def test_multimodal_chunked_prefill_processes_cached_media(self):
        torch.manual_seed(0)
        config = get_tiny_config(use_dsa=True)
        config.text_config.index_topk = config.text_config.max_position_embeddings
        model = Dots3NoteForConditionalGeneration(config).eval()
        prefix_ids = torch.tensor([[1, 5]])
        patch_width = config.vision_config.num_channels * config.vision_config.patch_size**2
        cases = {
            "image": (
                torch.tensor([[config.image_token_id, 7, 2]]),
                {
                    "pixel_values": torch.randn(4, patch_width),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
            ),
            "video": (
                torch.tensor([[config.video_token_id, 7, 2]]),
                {
                    "pixel_values_videos": torch.randn(4, patch_width),
                    "video_grid_thw": torch.tensor([[1, 2, 2]]),
                },
            ),
            "audio": (
                torch.tensor([[config.audio_token_id, config.audio_token_id, 7, 2]]),
                {
                    "input_features": torch.randn(1, config.audio_config.feature_size, 16),
                    "chunk_sample_lengths": torch.tensor([64]),
                },
            ),
        }

        for modality, (suffix_ids, media_inputs) in cases.items():
            with self.subTest(modality=modality):
                input_ids = torch.cat((prefix_ids, suffix_ids), dim=-1)
                attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    expected = model(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        **media_inputs,
                    ).logits[:, -suffix_ids.shape[1] :]
                    prefix = model(prefix_ids, attention_mask=torch.ones_like(prefix_ids), use_cache=True)
                    actual = model(
                        suffix_ids,
                        attention_mask=attention_mask,
                        past_key_values=prefix.past_key_values,
                        use_cache=True,
                        **media_inputs,
                    ).logits

                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_expert_checkpoint_conversion_roundtrip(self):
        torch.manual_seed(13)
        model = Dots3NoteForConditionalGeneration(get_tiny_config()).eval()
        input_ids = torch.tensor([[1, 5, 6, 2]])
        with torch.no_grad():
            expected = model(input_ids, use_cache=False).logits

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Released checkpoints also contain an MTP prediction layer outside the text decoder.
            state_dict = model.state_dict() | {"model.language_model.layers.46.eh_proj.weight": torch.zeros(32, 64)}
            model.save_pretrained(tmpdirname, state_dict=state_dict)
            with safe_open(f"{tmpdirname}/model.safetensors", framework="pt") as checkpoint:
                checkpoint_keys = set(checkpoint.keys())
            self.assertIn("model.layers.1.mlp.experts.0.gate_proj.weight", checkpoint_keys)
            self.assertIn("model.layers.1.mlp.experts.0.up_proj.weight", checkpoint_keys)
            self.assertIn("model.layers.1.mlp.experts.0.down_proj.weight", checkpoint_keys)
            self.assertNotIn("model.layers.1.mlp.experts.gate_up_proj", checkpoint_keys)
            self.assertIn("model.vision_encoder.blocks.1.mlp.experts.0.fc1.weight", checkpoint_keys)
            self.assertIn("model.vision_encoder.blocks.1.mlp.gate_weight", checkpoint_keys)

            model.config.architectures = ["Dots3NoteForCausalLM"]
            legacy_config = model.config.to_dict() | {"rope_scaling": None}
            text_config = legacy_config.pop("text_config")
            legacy_config.update(text_config)
            legacy_config["model_type"] = "dots3_note"
            legacy_config["architectures"] = ["Dots3NoteForCausalLM"]
            with open(f"{tmpdirname}/config.json", "w", encoding="utf-8") as config_file:
                json.dump(legacy_config, config_file)
            reloaded, info = AutoModelForMultimodalLM.from_pretrained(tmpdirname, output_loading_info=True)
            self.assertIsInstance(reloaded, Dots3NoteForConditionalGeneration)
            self.assertFalse(info["missing_keys"])
            self.assertFalse(info["unexpected_keys"])
            torch.testing.assert_close(reloaded.state_dict(), model.state_dict(), rtol=0, atol=0)

        with torch.no_grad():
            actual = reloaded(input_ids, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@slow
@require_torch
@require_torchvision
class Dots3NoteHubIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "miraclezqc/TestForDots3"
    revision = "14258a1269bdeba3b9d1ef69ea72673470464300"

    @parameterized.expand(
        [
            ("text", [247, 103, 217, 187, 59, 295, 13, 246]),
            ("image", [247, 103, 217, 187, 59, 295, 13, 246]),
            ("audio", [247, 103, 217, 187, 59, 92, 306, 158]),
            ("video", [247, 103, 217, 187, 59, 295, 13, 246]),
        ]
    )
    def test_generation(self, modality, expected):
        processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        model, loading_info = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            experts_implementation="eager",
            output_loading_info=True,
        )
        # Fix the device because short-audio BF16 generation differs between CPU and CUDA.
        model = model.to("cpu").eval()
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            self.assertFalse(loading_info.get(key), loading_info)
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()))

        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        video = np.stack([image, np.flip(image, axis=0).copy(), image, np.flip(image, axis=1).copy()])
        audio = (0.1 * np.sin(2 * np.pi * 440 * np.arange(2400) / 16000)).astype(np.float32)
        content = [] if modality == "text" else [{"type": modality}]
        content.append({"type": "text", "text": "Describe."})
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        media = {
            "text": {},
            "image": {"images": [image]},
            "audio": {"audio": [audio], "sampling_rate": 16000},
            "video": {
                "videos": [video],
                "do_sample_frames": False,
                "input_data_format": "channels_last",
                "cap_pixels_per_frame": False,
            },
        }[modality]
        inputs = processor(text=prompt, return_tensors="pt", **media).to("cpu", dtype=model.dtype)
        logits = model(**inputs, use_cache=False).logits
        self.assertTrue(torch.isfinite(logits).all())
        if modality != "text":
            media_key = {"image": "pixel_values", "audio": "input_features", "video": "pixel_values_videos"}[modality]
            # Random weights may generate the same tokens for different media, so also check the logits depend on it.
            ablated_inputs = {**inputs, media_key: torch.zeros_like(inputs[media_key])}
            ablated_logits = model(**ablated_inputs, use_cache=False).logits
            self.assertGreater((logits.float() - ablated_logits.float()).abs().max().item(), 0)
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=8)
        self.assertEqual(generated[0, inputs.input_ids.shape[-1] :].tolist(), expected)


if __name__ == "__main__":
    unittest.main()
