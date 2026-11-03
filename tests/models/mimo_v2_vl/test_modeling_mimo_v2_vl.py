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

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parameterized import parameterized

from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    MiMoV2VLAudioConfig,
    MiMoV2VLAudioTokenizerConfig,
    MiMoV2VLConfig,
    MiMoV2VLForConditionalGeneration,
    MiMoV2VLModel,
    MiMoV2VLTextConfig,
    MiMoV2VLTextModel,
    MiMoV2VLVisionConfig,
)
from transformers.testing_utils import require_accelerate, require_torch, require_torch_gpu, torch_device
from transformers.utils import is_torch_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_expert_parallel_mixin import ExpertParallelTesterMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_tensor_parallel_mixin import TensorParallelTesterMixin


if is_torch_available():
    import torch

    from transformers.models.mimo_v2_vl import convert_mimo_v2_vl_weights_to_hf as convert_script
    from transformers.models.mimo_v2_vl.modeling_mimo_v2_vl import (
        MiMoV2VLAudioTokenizerRotaryEmbedding,
        apply_rotary_pos_emb_fp32,
        eager_attention_forward,
    )


def get_config():
    return MiMoV2VLConfig(
        image_token_id=4,
        video_token_id=5,
        audio_token_id=6,
        text_config=MiMoV2VLTextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            v_head_dim=4,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
            swa_head_dim=8,
            swa_v_head_dim=4,
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 1.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
            },
            sliding_window=4,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            layer_types=["full_attention", "sliding_attention"],
            mlp_layer_types=["dense", "sparse"],
            pad_token_id=0,
            eos_token_id=3,
        ),
        vision_config=MiMoV2VLVisionConfig(
            hidden_size=16,
            intermediate_size=24,
            depth=3,
            num_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            layer_types=["full_attention", "sliding_attention", "sliding_attention"],
            patch_orders=["row_major", "row_major", "column_major"],
            sliding_window=2,
        ),
        audio_config=MiMoV2VLAudioConfig(
            num_codebooks=4,
            group_size=2,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            head_dim=8,
            intermediate_size=24,
            out_hidden_size=32,
            codebook_vocab_sizes=[16] * 4,
            codebook_pad_token_ids=[15] * 4,
        ),
        audio_tokenizer_config=MiMoV2VLAudioTokenizerConfig(
            hidden_size=16,
            num_mel_bins=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=24,
            skip_layer_idx=0,
            codebook_sizes=[16] * 4,
            segment_length=8,
        ),
    )


class MiMoV2VLModelTester:
    if is_torch_available():
        conditional_generation_class = MiMoV2VLForConditionalGeneration
        base_model_class = MiMoV2VLModel

    def __init__(self, parent, text_only=False):
        self.parent = parent
        self.text_only = text_only
        self.batch_size = 2
        self.seq_length = 7
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.vocab_size = 64
        self.is_training = True

    def get_config(self):
        config = get_config()
        config.text_config.max_position_embeddings = 64
        config.text_config.bos_token_id = 2
        return config.text_config if self.text_only else config

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 7) + 7
        attention_mask = torch.ones_like(input_ids, device=torch_device)
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


class MiMoV2VLCommonTestMixin:
    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("MiMo V2 VL uses per-layer-type RoPE parameters, which the common RoPE tests do not cover.")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    @unittest.skip("MiMo V2 VL uses per-layer-type RoPE parameters, which the common RoPE tests do not cover.")
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip("Partial rotation is configured per layer type and has dedicated tests.")
    def test_model_rope_with_partial_rotation(self):
        pass

    @unittest.skip("MiMo V2 VL does not return router logits, it has no auxiliary load-balancing loss.")
    def test_moe_models_record_router_logits(self):
        pass

    @unittest.skip("SDPA has no API for the learned attention sinks, so MiMo V2 VL does not support SDPA.")
    def test_can_set_attention_dynamically_composite_model(self):
        pass


@require_torch
class MiMoV2VLCommonModelTest(
    MiMoV2VLCommonTestMixin,
    ModelTesterMixin,
    GenerationTesterMixin,
    TensorParallelTesterMixin,
    ExpertParallelTesterMixin,
    unittest.TestCase,
):
    all_model_classes = (MiMoV2VLModel, MiMoV2VLForConditionalGeneration) if is_torch_available() else ()
    _is_composite = True
    has_attentions = False
    test_torch_exportable = False
    # Common API tests use text inputs. Modality-specific gradients and frozen audio encoding are tested below.
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = MiMoV2VLModelTester(self)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        text_config = config.get_text_config()
        self.assertEqual(len(past_key_values.layers), text_config.num_hidden_layers)
        for index, layer in enumerate(past_key_values.layers):
            sliding = text_config.layer_types[index] == "sliding_attention"
            num_heads = text_config.swa_num_key_value_heads if sliding else text_config.num_key_value_heads
            key_dim = text_config.swa_head_dim if sliding else text_config.head_dim
            value_dim = text_config.swa_v_head_dim if sliding else text_config.v_head_dim
            length = min(seq_length, text_config.sliding_window - 1) if layer.is_sliding else seq_length
            self.assertEqual(layer.keys.shape, (batch_size, num_heads, length, key_dim))
            self.assertEqual(layer.values.shape, (batch_size, num_heads, length, value_dim))

    # Audio features are grouped codec frames, not one sequence per input.
    skip_test_audio_features_output_shape = True

    def _feature_config(self):
        config = self.model_tester.get_config()
        # The shape checks compare the vision tower width with `out_hidden_size`.
        config.vision_config.hidden_size = config.vision_config.out_hidden_size
        return config

    def _image_features_prepare_config_and_inputs(self):
        grid_thw = torch.tensor([[1, 2, 2], [1, 2, 4]], device=torch_device)
        pixel_values = floats_tensor([int(grid_thw.prod(-1).sum()), 24]).to(torch_device)
        return self._feature_config(), {"pixel_values": pixel_values, "image_grid_thw": grid_thw}

    def _video_features_prepare_config_and_inputs(self):
        grid_thw = torch.tensor([[2, 2, 2], [1, 2, 2]], device=torch_device)
        pixel_values = floats_tensor([int(grid_thw.prod(-1).sum()), 24]).to(torch_device)
        return self._feature_config(), {"pixel_values_videos": pixel_values, "video_grid_thw": grid_thw}

    def _audio_features_prepare_config_and_inputs(self):
        codes = ids_tensor([self.model_tester.batch_size, 3, 2, 4], 15).to(torch_device)
        return self._feature_config(), {"input_audio_codes": codes}

    def test_generate_from_multimodal_encoder_outputs_and_raw_data(self):
        # The common inputs are text-only; add one image per sample so that `mm_encoder_outputs` are exercised.
        prepare_text_inputs = self.prepare_config_and_inputs_for_generate

        def prepare_image_inputs(batch_size=2):
            config, inputs = prepare_text_inputs(batch_size)
            inputs["input_ids"][:, 1] = config.image_token_id
            inputs["pixel_values"] = floats_tensor([batch_size * 4, 24]).to(torch_device)
            inputs["image_grid_thw"] = torch.tensor([[1, 2, 2]] * batch_size, device=torch_device)
            return config, inputs

        with patch.object(self, "prepare_config_and_inputs_for_generate", prepare_image_inputs):
            super().test_generate_from_multimodal_encoder_outputs_and_raw_data()

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        return self._feature_config().vision_config.depth

    def _video_features_get_expected_num_attentions(self, model_tester=None):
        return self._feature_config().vision_config.depth

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        return self._feature_config().audio_config.num_hidden_layers


@require_torch
class MiMoV2VLTextCommonModelTest(MiMoV2VLCommonTestMixin, ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MiMoV2VLTextModel,) if is_torch_available() else ()
    has_attentions = False
    test_torch_exportable = False
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = MiMoV2VLModelTester(self, text_only=True)


@require_torch
class MiMoV2VLModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MiMoV2VLForConditionalGeneration(get_config()).eval()

    def image_inputs(self):
        return {
            "input_ids": torch.tensor([[2, 4, 7, 8]]),
            "pixel_values": torch.randn(4, 24),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }

    @require_accelerate
    @require_torch_gpu
    def test_multimodal_encoders_with_cpu_offload(self):
        from accelerate import dispatch_model

        model = dispatch_model(
            self.model,
            device_map={
                "model.language_model": 0,
                "model.visual": "cpu",
                "model.audio_encoder": "cpu",
                "model.audio_tokenizer": "cpu",
                "lm_head": 0,
            },
        )
        inputs = {key: value.cuda() for key, value in self.image_inputs().items()}
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(**inputs).logits).all())
            audio_inputs = {
                "input_ids": torch.tensor([[2, 6, 6, 8]], device="cuda"),
                "input_features": torch.randn(1, 9, 4, device="cuda"),
                "feature_attention_mask": torch.ones(1, 9, device="cuda", dtype=torch.long),
                "audio_feature_lens": torch.tensor([2], device="cuda"),
            }
            self.assertTrue(torch.isfinite(model(**audio_inputs).logits).all())

    @parameterized.expand([("image",), ("video",), ("audio",), ("waveform",)])
    def test_modal_embeddings_are_fused(self, modality):
        if modality == "image":
            inputs = self.image_inputs()
        elif modality == "video":
            inputs = {
                "input_ids": torch.tensor([[2, 5, 5, 8]]),
                "pixel_values_videos": torch.randn(8, 24),
                "video_grid_thw": torch.tensor([[2, 2, 2]]),
            }
        elif modality == "audio":
            inputs = {
                "input_ids": torch.tensor([[2, 6, 6, 8]]),
                "input_audio_codes": torch.randint(0, 15, (1, 2, 2, 4)),
                "audio_feature_lens": torch.tensor([2]),
            }
        else:
            inputs = {
                "input_ids": torch.tensor([[2, 6, 6, 8]]),
                "input_features": torch.randn(1, 9, 4),
                "feature_attention_mask": torch.ones(1, 9, dtype=torch.long),
                "audio_feature_lens": torch.tensor([2]),
            }
        with torch.no_grad():
            fused = self.model.model.get_multimodal_embeddings(**inputs)
            text_only = self.model.get_input_embeddings()(inputs["input_ids"])
            self.assertFalse(torch.equal(fused, text_only))
            actual = self.model(**inputs, use_cache=False).logits
            expected = self.model(inputs_embeds=fused, use_cache=False).logits
        torch.testing.assert_close(actual, expected)

    def test_image_content_changes_logits(self):
        inputs = self.image_inputs()
        with torch.no_grad():
            first = self.model(**inputs).logits
            inputs["pixel_values"] = torch.randn_like(inputs["pixel_values"])
            second = self.model(**inputs).logits
        self.assertFalse(torch.allclose(first, second))

    @parameterized.expand([(False, False), (False, True), (True, False)])
    def test_return_dict_config_and_call_overrides(self, outer_return_dict, text_return_dict):
        inputs = self.image_inputs()
        with torch.no_grad():
            expected = self.model(**inputs, use_cache=False).logits
            expected_ids = self.model.generate(**inputs, max_new_tokens=2, do_sample=False)
            self.model.config.return_dict = outer_return_dict
            self.model.config.text_config.return_dict = text_return_dict
            output = self.model(**inputs, use_cache=False)
            if outer_return_dict:
                torch.testing.assert_close(output.logits, expected, rtol=0, atol=0)
            else:
                self.assertIsInstance(output, tuple)
                torch.testing.assert_close(output[0], expected, rtol=0, atol=0)
            explicit_dict = self.model(**inputs, use_cache=False, return_dict=True)
            explicit_tuple = self.model(**inputs, use_cache=False, return_dict=False)
            torch.testing.assert_close(explicit_dict.logits, expected, rtol=0, atol=0)
            torch.testing.assert_close(explicit_tuple[0], expected, rtol=0, atol=0)
            generated_ids = self.model.generate(**inputs, max_new_tokens=2, do_sample=False)
            torch.testing.assert_close(generated_ids, expected_ids, rtol=0, atol=0)

    def test_save_load_after_updating_native_config(self):
        config = get_config()
        config.text_config.sliding_window = 9
        config.text_config.layer_types = ["sliding_attention", "full_attention"]
        config.text_config.rope_parameters["full_attention"]["rope_theta"] = 20000.0
        model = MiMoV2VLForConditionalGeneration(config).eval()
        inputs = {"input_ids": torch.tensor([[2, 7, 8, 9]])}
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            expected = model(**inputs, use_cache=False).logits
            model.save_pretrained(directory)
            restored = AutoModelForImageTextToText.from_pretrained(directory, local_files_only=True).eval()
            self.assertEqual(restored.config.text_config.sliding_window, 9)
            self.assertEqual(restored.config.text_config.layer_types, config.text_config.layer_types)
            self.assertEqual(restored.config.text_config.rope_parameters, config.text_config.rope_parameters)
            torch.testing.assert_close(restored(**inputs, use_cache=False).logits, expected, rtol=0, atol=0)

    def test_float_and_boolean_audio_masks_match_integer_masks(self):
        tokenizer = self.model.model.audio_tokenizer
        for codebook in tokenizer.quantizer.codebooks:
            codebook.embed.copy_(torch.randn_like(codebook.embed))
        features = torch.randn(2, 12, 4)
        mask = torch.tensor([[1] * 9 + [0] * 3, [0] * 3 + [1] * 9])
        expected = tokenizer(features, mask).audio_codes
        self.assertEqual(len(expected), 2)
        for dtype in (torch.float32, torch.bool):
            actual = tokenizer(features, mask.to(dtype)).audio_codes
            self.assertEqual(len(actual), 2)
            for codes, reference in zip(actual, expected):
                torch.testing.assert_close(codes, reference, rtol=0, atol=0)

    def test_placeholder_mismatch_raises(self):
        inputs = self.image_inputs()
        inputs["input_ids"] = torch.tensor([[2, 4, 4, 8]])
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.model(**inputs)

    def test_cache_matches_full_sequence_across_sliding_window(self):
        tokens = torch.tensor([[2, 7, 8, 9, 10, 11, 12]])
        with torch.no_grad():
            expected = self.model(input_ids=tokens, use_cache=False).logits
            cached = self.model(input_ids=tokens[:, :3], use_cache=True)
            for index in range(3, tokens.shape[1]):
                cached = self.model(
                    input_ids=tokens[:, index : index + 1], past_key_values=cached.past_key_values, use_cache=True
                )
                torch.testing.assert_close(cached.logits[:, -1], expected[:, index], rtol=1e-4, atol=1e-5)

    def test_padding_does_not_change_text_logits(self):
        with torch.no_grad():
            expected = self.model(input_ids=torch.tensor([[2, 7, 8]]), use_cache=False).logits
            actual = self.model(
                input_ids=torch.tensor([[0, 0, 2, 7, 8]]),
                attention_mask=torch.tensor([[0, 0, 1, 1, 1]]),
                position_ids=torch.tensor([[0, 0, 0, 1, 2]]),
                use_cache=False,
            ).logits
        torch.testing.assert_close(actual[:, -3:], expected, rtol=1e-4, atol=1e-5)

    @parameterized.expand([(1,), (2,)])
    def test_generate_encodes_image_once(self, num_beams):
        inputs = self.image_inputs()
        with patch.object(self.model.model.visual, "forward", wraps=self.model.model.visual.forward) as encode:
            generated = self.model.generate(
                **inputs, max_new_tokens=3, min_new_tokens=3, num_beams=num_beams, do_sample=False
            )
        self.assertEqual(generated.shape, (1, 7))
        self.assertEqual(encode.call_count, 1)

    def test_generate_video_and_waveform(self):
        inputs = {
            "input_ids": torch.tensor([[2, 5, 5, 6, 6, 8]]),
            "pixel_values_videos": torch.randn(8, 24),
            "video_grid_thw": torch.tensor([[2, 2, 2]]),
            "input_features": torch.randn(1, 9, 4),
            "feature_attention_mask": torch.ones(1, 9, dtype=torch.long),
            "audio_feature_lens": torch.tensor([2]),
            "audio_segment_offsets": torch.tensor([[0, 0, 1], [0, 1, 1]]),
        }
        with patch.object(
            self.model.model.audio_tokenizer, "forward", wraps=self.model.model.audio_tokenizer.forward
        ) as encode:
            generated = self.model.generate(**inputs, max_new_tokens=2, min_new_tokens=2, do_sample=False)
        self.assertEqual(generated.shape, (1, 8))
        self.assertEqual(encode.call_count, 1)

    def test_loss_backpropagates_through_vision(self):
        self.model.train()
        inputs = self.image_inputs()
        loss = self.model(**inputs, labels=inputs["input_ids"], use_cache=False).loss
        loss.backward()
        gradient = self.model.model.visual.patch_embed.proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum(), 0)

    @parameterized.expand([("text",), ("image",), ("audio",), ("video",), ("mixed",)])
    def test_save_load_native_auto_model(self, modality):
        inputs = {"input_ids": torch.tensor([[2, 7, 8]])}
        if modality in ("image", "mixed"):
            inputs.update(self.image_inputs())
        if modality in ("video", "mixed"):
            inputs.update(
                pixel_values_videos=torch.randn(8, 24),
                video_grid_thw=torch.tensor([[2, 2, 2]]),
                input_ids=torch.tensor([[2, 5, 5, 8]]),
            )
        if modality in ("audio", "mixed"):
            inputs.update(
                input_features=torch.randn(1, 9, 4),
                feature_attention_mask=torch.ones(1, 9, dtype=torch.long),
                audio_feature_lens=torch.tensor([2]),
                input_ids=torch.tensor([[2, 6, 6, 8]]),
            )
        if modality == "mixed":
            inputs["input_ids"] = torch.tensor([[2, 4, 5, 5, 6, 6, 8]])
            inputs["audio_segment_offsets"] = torch.tensor([[0, 0, 2]])

        generation_kwargs = {"max_new_tokens": 2, "min_new_tokens": 2, "do_sample": False}
        with torch.no_grad():
            expected_logits = self.model(**inputs).logits
            expected_ids = self.model.generate(**inputs, **generation_kwargs)
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            config = AutoConfig.from_pretrained(directory, local_files_only=True)
            self.assertIsInstance(config, MiMoV2VLConfig)
            restored = AutoModelForImageTextToText.from_pretrained(directory, local_files_only=True).eval()
            self.assertIsInstance(restored, MiMoV2VLForConditionalGeneration)
            with torch.no_grad():
                torch.testing.assert_close(expected_logits, restored(**inputs).logits, rtol=0, atol=0)
                torch.testing.assert_close(
                    expected_ids, restored.generate(**inputs, **generation_kwargs), rtol=0, atol=0
                )

    def test_sink_is_extra_softmax_mass(self):
        from types import SimpleNamespace

        query = torch.zeros(1, 2, 1, 4)
        key = torch.zeros(1, 1, 2, 4)
        value = torch.ones(1, 1, 2, 3)
        module = SimpleNamespace(
            sinks=torch.tensor([0.0, torch.log(torch.tensor(2.0))]), num_key_value_groups=2, training=False
        )
        actual, _ = eager_attention_forward(module, query, key, value, attention_mask=None, scaling=0.5)
        # Two keys with equal logits, plus a sink of logit 0 (head 0) or log(2) (head 1).
        expected = torch.tensor([2 / 3, 1 / 2]).reshape(1, 1, 2, 1).expand_as(actual)
        torch.testing.assert_close(actual, expected)

    def test_vision_sliding_window_is_local(self):
        # Block 1 of the test config is a sliding-window block of window 2, block 0 a full-attention block.
        visual = self.model.model.visual
        hidden_states = torch.randn(8, visual.config.hidden_size)
        cu_seqlens = torch.tensor([0, 8])
        head_dim = visual.config.head_dim
        position_embeddings = (torch.ones(8, head_dim), torch.zeros(8, head_dim))
        perturbed = hidden_states.clone()
        perturbed[-1] += 1.0
        with torch.no_grad():
            for block, is_local in ((visual.blocks[1], True), (visual.blocks[0], False)):
                expected = block.attn(hidden_states, cu_seqlens, position_embeddings)
                actual = block.attn(perturbed, cu_seqlens, position_embeddings)
                # The first patch only sees the last one with full attention.
                self.assertEqual(torch.allclose(actual[0], expected[0]), is_local)

    def test_segmented_audio_token_count_and_left_padding(self):
        from transformers import MiMoV2VLFeatureExtractor

        model = self.model.model
        model.config.audio_tokenizer_config.segment_length = 9
        extractor = MiMoV2VLFeatureExtractor(segment_length=9, group_size=2)
        length = extractor._get_audio_token_length(27)
        self.assertEqual(length, 5)
        features = torch.randn(1, 27, 4)
        mask = torch.ones(1, 27, dtype=torch.long)
        with torch.no_grad():
            expected = model.get_audio_features(
                input_features=features, feature_attention_mask=mask, audio_feature_lens=[length]
            ).pooler_output
            padded = torch.nn.functional.pad(features, (0, 0, 5, 0))
            padded_mask = torch.nn.functional.pad(mask, (5, 0))
            actual = model.get_audio_features(
                input_features=padded, feature_attention_mask=padded_mask, audio_feature_lens=[length]
            ).pooler_output
        torch.testing.assert_close(actual, expected)

    def test_bfloat16_save_load_preserves_codebook_precision(self):
        model = self.model
        for name, parameter in model.named_parameters():
            if not name.endswith("e_score_correction_bias"):
                parameter.data = parameter.data.to(torch.bfloat16)
        for codebook in model.model.audio_tokenizer.quantizer.codebooks:
            codebook.embed.copy_(torch.randn_like(codebook.embed))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = AutoModelForImageTextToText.from_pretrained(directory, dtype="auto")
            for original, loaded in zip(
                model.model.audio_tokenizer.quantizer.codebooks, restored.model.audio_tokenizer.quantizer.codebooks
            ):
                self.assertEqual(loaded.embed.dtype, torch.float32)
                torch.testing.assert_close(original.embed, loaded.embed, atol=0, rtol=0)
            buffers = dict(restored.named_buffers())
            for name, buffer in model.named_buffers():
                if name.endswith("inv_freq"):
                    torch.testing.assert_close(buffer, buffers[name], atol=0, rtol=0)

    def test_prepare_generation_without_cache_preserves_media(self):
        inputs = self.image_inputs()
        prepared = self.model.prepare_inputs_for_generation(**inputs, use_cache=True)
        self.assertIn("pixel_values", prepared)
        with torch.no_grad():
            cache = self.model(**inputs, use_cache=True).past_key_values
        prepared = self.model.prepare_inputs_for_generation(**inputs, past_key_values=cache, use_cache=True)
        self.assertNotIn("pixel_values", prepared)

    @parameterized.expand([(dtype, dim) for dtype in ("float16", "bfloat16", "float32", "float64") for dim in (4, 8)])
    def test_rotary_dtype_and_partial_rotation(self, dtype, rotary_dim):
        dtype = getattr(torch, dtype)
        query = torch.randn(1, 2, 3, 8, dtype=dtype, device=torch_device, requires_grad=True)
        key = torch.randn(1, 1, 3, 8, dtype=dtype, device=torch_device, requires_grad=True)
        positions = torch.tensor([[0, 49000, 49001]], device=torch_device)
        config = MiMoV2VLAudioTokenizerConfig(
            hidden_size=rotary_dim, num_attention_heads=1, rope_parameters={"rope_type": "default", "rope_theta": 1e4}
        )
        cos, sin = MiMoV2VLAudioTokenizerRotaryEmbedding(config).to(torch_device)(query, positions)
        self.assertEqual(cos.dtype, torch.float32)
        self.assertEqual(sin.dtype, torch.float32)
        actual = apply_rotary_pos_emb_fp32(query, key, cos, sin)
        for value, original in zip(actual, (query, key)):
            self.assertEqual(value.dtype, dtype)
            self.assertEqual(value.shape, original.shape)
            torch.testing.assert_close(
                value[:, :, 0, :rotary_dim], original[:, :, 0, :rotary_dim].float().to(dtype), rtol=0, atol=0
            )
            torch.testing.assert_close(value[..., rotary_dim:], original[..., rotary_dim:], rtol=0, atol=0)
        # Q/K and V must retain matching dtypes, including asymmetric value head dimensions.
        value = torch.randn(1, 2, 3, 4, dtype=dtype, device=torch_device)
        output = torch.nn.functional.scaled_dot_product_attention(actual[0], actual[1].repeat_interleave(2, 1), value)
        self.assertEqual(output.dtype, dtype)
        output.float().sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(key.grad).all())

    def test_rotary_bfloat16_rounds_after_rotation_and_gradient_accumulation(self):
        query = torch.ones(1, 1, 1, 2, dtype=torch.bfloat16, device=torch_device, requires_grad=True)
        cos = torch.full((1, 1, 2), 0.6, dtype=torch.float32, device=torch_device)
        sin = torch.full((1, 1, 2), 0.8, dtype=torch.float32, device=torch_device)
        rotated, _ = apply_rotary_pos_emb_fp32(query, query.detach(), cos, sin)
        # Rotating (1, 1) gives (-0.2, 1.4); rounding either the phases or intermediate products changes this.
        expected = torch.tensor([[[[-0.2, 1.4]]]], dtype=torch.bfloat16, device=torch_device)
        torch.testing.assert_close(rotated, expected, rtol=0, atol=0)
        rotated.sum().backward()
        expected_gradient = torch.tensor([[[[1.4, -0.2]]]], dtype=torch.bfloat16, device=torch_device)
        torch.testing.assert_close(query.grad, expected_gradient, rtol=0, atol=0)

    @require_torch_gpu
    def test_rotary_reinitialization_matches_cpu_construction(self):
        for theta in (10000.0, 10000000.0, 640000.0):
            config = MiMoV2VLAudioTokenizerConfig(rope_parameters={"rope_type": "default", "rope_theta": theta})
            rotary = MiMoV2VLAudioTokenizerRotaryEmbedding(config)
            expected = rotary.inv_freq.clone()
            rotary.cuda()
            self.model._init_weights(rotary)
            torch.testing.assert_close(rotary.inv_freq.cpu(), expected, rtol=0, atol=0)

    @require_torch_gpu
    def test_bfloat16_gpu_round_trip_at_long_positions(self):
        config = get_config()
        config.text_config.head_dim = config.text_config.swa_head_dim = 192
        for parameters in config.text_config.rope_parameters.values():
            parameters["partial_rotary_factor"] = 0.334
        model = MiMoV2VLForConditionalGeneration(config).eval()
        for name, parameter in model.named_parameters():
            if not name.endswith("e_score_correction_bias"):
                parameter.data = parameter.data.to(torch.bfloat16)
        model.cuda()
        inputs = {
            "input_ids": torch.tensor([[2, 7, 8, 9]], device="cuda"),
            "position_ids": torch.arange(49000, 49004, device="cuda")[None],
        }
        with torch.no_grad(), tempfile.TemporaryDirectory() as directory:
            expected = model(**inputs, use_cache=False).logits
            model.save_pretrained(directory)
            restored = AutoModelForImageTextToText.from_pretrained(directory, dtype="auto", device_map="cuda")
            actual = restored(**inputs, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


ORIGINAL_SPECIAL_TOKENS = [
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|mimo_audio_start|>",
    "<|mimo_audio_end|>",
    "<|mimo_video_start|>",
    "<|mimo_video_end|>",
]


def write_original_checkpoint(model, directory, multimodal=True, tp_size=2, fp8_layer=None):
    """Saves `model` in the layout of the original MiMo V2 checkpoints (flat config, fused TP-interleaved QKV)."""
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    from transformers import PreTrainedTokenizerFast

    root = Path(directory)
    native = root / "native"
    model.save_pretrained(native)
    native_weights = {}
    for shard in native.glob("*.safetensors"):
        native_weights.update(load_file(shard))

    # Inverse of the converter's key mappings: native names -> original names.
    native_to_original = {
        r"^model\.language_model\.": "model.",
        r"^model\.visual\.merger\.norm\.": "visual.merger.ln_q.",
        r"^model\.visual\.merger\.linear_fc1\.": "visual.merger.mlp.0.",
        r"^model\.visual\.merger\.linear_fc2\.": "visual.merger.mlp.2.",
        r"^model\.visual\.": "visual.",
        r"^model\.audio_encoder\.embed_tokens\.": "speech_embeddings.",
        r"^model\.audio_encoder\.projection\.linear_fc1\.": "audio_encoder.projection.mlp.0.",
        r"^model\.audio_encoder\.projection\.linear_fc2\.": "audio_encoder.projection.mlp.2.",
        r"^model\.audio_encoder\.(layers|norm)\.": r"audio_encoder.input_local_transformer.\1.",
        r"^model\.audio_tokenizer\.downsample_conv\.": "encoder.down_sample_layer.0.",
        r"^model\.audio_tokenizer\.downsample_norm\.": "encoder.down_sample_norm.",
        r"^model\.audio_tokenizer\.quantizer\.codebooks\.(\d+)\.embed$": r"encoder.quantizer.vq.layers.\1._codebook.embed",
        r"^model\.audio_tokenizer\.": "encoder.",
    }
    weights, audio_weights = {}, {}
    for name, tensor in native_weights.items():
        if not multimodal and not name.startswith(("model.language_model.", "lm_head.")):
            continue
        name = convert_script.convert_key(name, native_to_original)
        if name.startswith("encoder."):
            audio_weights[name] = tensor
            if name.endswith("._codebook.embed"):
                # Training statistics of the codebooks, unused by the model.
                prefix = name.removesuffix("embed")
                audio_weights[prefix + "inited"] = torch.ones(1)
                audio_weights[prefix + "cluster_size"] = torch.ones(tensor.shape[0])
                audio_weights[prefix + "embed_avg"] = torch.randn_like(tensor)
        else:
            weights[name] = tensor

    text_config = model.config.text_config
    for layer_idx in range(text_config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.self_attn."
        projections = [weights.pop(f"{prefix}{name}_proj.weight").chunk(tp_size) for name in "qkv"]
        weights[prefix + "qkv_proj.weight"] = torch.cat(
            [projection[rank] for rank in range(tp_size) for projection in projections]
        )
    scales = {}
    if fp8_layer is not None:
        # Store a dense weight in FP8 with a separate (block-wise inverse) scale file.
        name = f"model.layers.{fp8_layer}.mlp.down_proj.weight"
        weights[name] = (weights[name] * 2).to(torch.float8_e4m3fn)
        scales[name.replace(".weight", ".weight_scale_inv")] = torch.tensor([[0.5]])

    original_config = {
        "model_type": "mimo_v2",
        "architectures": ["MiMoV2ForCausalLM"],
        "attention_projection_layout": "fused_qkv",
        "vocab_size": text_config.vocab_size,
        "hidden_size": text_config.hidden_size,
        "intermediate_size": text_config.intermediate_size,
        "moe_intermediate_size": text_config.moe_intermediate_size,
        "num_hidden_layers": text_config.num_hidden_layers,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": text_config.num_key_value_heads,
        "head_dim": text_config.head_dim,
        "v_head_dim": text_config.v_head_dim,
        "swa_num_attention_heads": text_config.swa_num_attention_heads,
        "swa_num_key_value_heads": text_config.swa_num_key_value_heads,
        "swa_head_dim": text_config.swa_head_dim,
        "swa_v_head_dim": text_config.swa_v_head_dim,
        "n_routed_experts": text_config.n_routed_experts,
        "num_experts_per_tok": text_config.num_experts_per_tok,
        "hybrid_layer_pattern": [int(kind == "sliding_attention") for kind in text_config.layer_types],
        "moe_layer_freq": [int(kind == "sparse") for kind in text_config.mlp_layer_types],
        "rope_theta": text_config.rope_parameters["full_attention"]["rope_theta"],
        "swa_rope_theta": text_config.rope_parameters["sliding_attention"]["rope_theta"],
        "partial_rotary_factor": text_config.rope_parameters["full_attention"]["partial_rotary_factor"],
        "sliding_window_size": text_config.sliding_window - 1,
        "layernorm_epsilon": text_config.rms_norm_eps,
        "routed_scaling_factor": None,
        "pad_token_id": text_config.pad_token_id,
        "eos_token_id": text_config.eos_token_id,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }
    if multimodal:
        vision = model.config.vision_config
        vision_config = {
            "depth": vision.depth,
            "hidden_size": vision.hidden_size,
            "intermediate_size": vision.intermediate_size,
            "num_heads": vision.num_heads,
            "num_key_value_heads": vision.num_key_value_heads,
            "qk_channels": vision.head_dim,
            "in_chans": vision.in_channels,
            "patch_size": vision.patch_size,
            "temporal_patch_size": vision.temporal_patch_size,
            "spatial_merge_size": vision.spatial_merge_size,
            "out_hidden_size": vision.out_hidden_size,
            "visual_token_window_size": vision.sliding_window,
            "use_sink": vision.use_attention_sinks,
            "fullatt_block_indexes": [i for i, kind in enumerate(vision.layer_types) if kind == "full_attention"],
            "vit_window_attn_types": [
                -1 if kind == "full_attention" else int(order == "column_major")
                for kind, order in zip(vision.layer_types, vision.patch_orders)
            ],
        }
        audio = model.config.audio_config
        audio_config = {
            "audio_channels": audio.num_codebooks,
            "group_size": audio.group_size,
            "input_local_dim": audio.hidden_size,
            "input_local_layers": audio.num_hidden_layers,
            "input_local_attn_heads": audio.num_attention_heads,
            "input_local_head_dim": audio.head_dim,
            "input_local_intermediate_size": audio.intermediate_size,
            "input_full_attention": not audio.is_causal,
            "add_post_norm": audio.use_final_norm,
            "out_hidden_size": audio.out_hidden_size,
            "projection_layers": audio.projection_layers,
            "rope_theta": int(audio.rope_parameters["rope_theta"]),
            "partial_rotary_factor": 1.0,
            "speech_vocab_size": "-".join(map(str, audio.codebook_vocab_sizes)),
            "speech_zeroemb_idx": "-".join(map(str, audio.codebook_pad_token_ids)),
            "audio_segment_size": model.config.audio_tokenizer_config.segment_length,
        }
        token_ids = dict(
            zip(["image", "video", "audio", "vision_start", "vision_end", "audio_start", "audio_end"], range(4, 11))
        )
        original_config.update(
            vision_config=vision_config,
            audio_config=audio_config,
            **{f"{name}_token_id": token_id for name, token_id in token_ids.items()},
            processor_config={
                "patch_size": 2,
                "merge_size": 2,
                "image_min_pixels": 16,
                "image_max_pixels": 256,
                "video_start_token_id": 11,
                "video_end_token_id": 12,
                "audio_sampling_rate": 16000,
                "audio_n_mels": 4,
                "fps": 1.0,
            },
        )
        (root / "audio_tokenizer").mkdir()
        tokenizer = model.config.audio_tokenizer_config
        audio_tokenizer_config = {
            "d_model": tokenizer.hidden_size,
            "n_mels": tokenizer.num_mel_bins,
            "kernel_size": tokenizer.conv_kernel_size,
            "stride_size": tokenizer.conv_stride,
            "encoder_layers": tokenizer.num_hidden_layers,
            "encoder_attention_heads": tokenizer.num_attention_heads,
            "encoder_ffn_dim": tokenizer.intermediate_size,
            "activation_function": tokenizer.hidden_act,
            "encoder_causal": tokenizer.is_causal,
            "encoder_attn_window_size": [tokenizer.sliding_window, 0],
            "hybrid_attention": True,
            "swa_per_block": 2,
            "encoder_skip_layer_id": tokenizer.skip_layer_idx + 1,
            "avg_pooler": tokenizer.downsample_factor,
            "num_quantizers": len(tokenizer.codebook_sizes),
            "codebook_size": tokenizer.codebook_sizes,
            "rope_theta": int(tokenizer.rope_parameters["rope_theta"]),
            "decoder_layers": 2,
        }
        (root / "audio_tokenizer" / "config.json").write_text(json.dumps(audio_tokenizer_config), encoding="utf-8")
        save_file(audio_weights, root / "audio_tokenizer" / "model.safetensors")

    (root / "config.json").write_text(json.dumps(original_config), encoding="utf-8")
    (root / "generation_config.json").write_text(
        json.dumps({"do_sample": False, "top_p": 0.95, "eos_token_id": [3, 7], "max_new_tokens": 13}),
        encoding="utf-8",
    )
    weight_map = {**dict.fromkeys(weights, "model.safetensors"), **dict.fromkeys(scales, "scales.safetensors")}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"tp_size": tp_size}, "weight_map": weight_map}), encoding="utf-8"
    )
    save_file(weights, root / "model.safetensors")
    if scales:
        save_file(scales, root / "scales.safetensors")

    vocab = {
        token: i for i, token in enumerate(["<unk>", "<pad>", "<bos>", "<eos>"] + ORIGINAL_SPECIAL_TOKENS + ["a"])
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        additional_special_tokens=ORIGINAL_SPECIAL_TOKENS,
    ).save_pretrained(root)
    return root


@require_torch
class MiMoV2VLConversionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = MiMoV2VLForConditionalGeneration(get_config()).eval()

    def test_mxfp4_unpacking_and_scaling(self):
        packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
        scales = torch.tensor([[127, 128]], dtype=torch.uint8)
        actual = convert_script.dequantize_mxfp4(packed, scales, block_size=8)
        expected = torch.tensor([[0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -1, -2, -3, -4, -6, -8, -12]])
        torch.testing.assert_close(actual, expected)

    def test_qkv_deinterleaving(self):
        config = self.model.config.text_config
        query = torch.arange(32 * 3).reshape(32, 3)
        key = torch.arange(16 * 3).reshape(16, 3) + 1000
        value = torch.arange(8 * 3).reshape(8, 3) + 2000
        interleaved = torch.cat(
            [part for rank in range(2) for part in (query.chunk(2)[rank], key.chunk(2)[rank], value.chunk(2)[rank])]
        )
        actual = convert_script.split_interleaved_qkv(interleaved, config, layer_idx=0, tp_size=2)
        for result, expected in zip(actual, (query, key, value)):
            torch.testing.assert_close(result, expected)

    def test_tp_size_candidates(self):
        config = self.model.config.text_config
        # With KV replication, the number of rows identifies the TP size.
        self.assertEqual(convert_script.get_tp_size_candidates(config, layer_idx=0, num_rows=80), {4})
        # Otherwise it is ambiguous...
        self.assertEqual(convert_script.get_tp_size_candidates(config, layer_idx=0, num_rows=56), {1, 2})
        # ...unless the FP8 scales, padded on every rank, disambiguate it.
        self.assertEqual(
            convert_script.get_tp_size_candidates(config, layer_idx=0, num_rows=56, num_scale_rows=8, block_rows=8),
            {2},
        )

    def test_fp8_qkv_scales_are_applied_per_rank(self):
        weight = torch.ones(56, 4).to(torch.float8_e4m3fn)
        scales = torch.arange(1, 9, dtype=torch.float32)[:, None]
        query, key, value = convert_script.split_interleaved_qkv(
            weight, self.model.config.text_config, layer_idx=0, tp_size=2, scales=scales, block_size=(8, 4)
        )
        expected_query = torch.tensor([1] * 8 + [2] * 8 + [5] * 8 + [6] * 8)[:, None].expand(32, 4).float()
        expected_key = torch.tensor([3] * 8 + [7] * 8)[:, None].expand(16, 4).float()
        expected_value = torch.tensor([4] * 4 + [8] * 4)[:, None].expand(8, 4).float()
        torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
        torch.testing.assert_close(value, expected_value, rtol=0, atol=0)

    def test_convert_multimodal_checkpoint(self):
        from transformers import AutoProcessor, GenerationConfig

        with tempfile.TemporaryDirectory() as directory:
            original = write_original_checkpoint(self.model, Path(directory) / "original", fp8_layer=0)
            with torch.no_grad():
                down_proj = self.model.model.language_model.layers[0].mlp.down_proj
                down_proj.weight.copy_((down_proj.weight * 2).to(torch.float8_e4m3fn).float() * 0.5)
            output = Path(directory) / "converted"
            convert_script.convert_checkpoint(original, output, dtype=torch.float32)

            restored = AutoModelForImageTextToText.from_pretrained(output, local_files_only=True).eval()
            self.assertIsInstance(restored, MiMoV2VLForConditionalGeneration)
            state_dict = restored.state_dict()
            for name, expected in self.model.state_dict().items():
                torch.testing.assert_close(state_dict[name], expected, rtol=0, atol=0, msg=name)

            generation_config = GenerationConfig.from_pretrained(output)
            self.assertEqual(generation_config.eos_token_id, [3, 7])
            self.assertIsNone(generation_config.top_p)

            processor = AutoProcessor.from_pretrained(output)
            self.assertEqual(processor.image_token, "<|image_pad|>")
            self.assertEqual(processor.image_token_id, restored.config.image_token_id)
            self.assertEqual(processor.video_start_token, "<|mimo_video_start|>")
            self.assertEqual(processor.image_processor.size["longest_edge"], 256)
            self.assertEqual(processor.feature_extractor.num_codebooks, restored.config.audio_config.num_codebooks)
            self.assertEqual(processor.feature_extractor.sampling_rate, 16000)

    def test_convert_text_checkpoint(self):
        from transformers import AutoModelForCausalLM, MiMoV2FlashForCausalLM

        with tempfile.TemporaryDirectory() as directory:
            original = write_original_checkpoint(self.model, Path(directory) / "original", multimodal=False)
            output = Path(directory) / "converted"
            convert_script.convert_checkpoint(original, output, dtype=torch.float32)
            restored = AutoModelForCausalLM.from_pretrained(output, local_files_only=True).eval()
            self.assertIsInstance(restored, MiMoV2FlashForCausalLM)
            inputs = {"input_ids": torch.tensor([[2, 7, 8, 9, 10, 11]])}
            with torch.no_grad():
                torch.testing.assert_close(
                    restored(**inputs).logits, self.model(**inputs).logits, rtol=1e-4, atol=1e-5
                )

    def test_dry_run_only_reads_headers(self):
        import struct

        with tempfile.TemporaryDirectory() as directory:
            original = write_original_checkpoint(self.model, Path(directory) / "original", fp8_layer=0)
            # Keep only the safetensors headers, as when validating a Hub repository without its weights.
            for path in original.rglob("*.safetensors"):
                data = path.read_bytes()
                (header_size,) = struct.unpack("<Q", data[:8])
                path.write_bytes(data[: 8 + header_size])
            output = Path(directory) / "converted"
            config, processor = convert_script.convert_checkpoint(original, output, dry_run=True)
            self.assertFalse(output.exists())
            self.assertEqual(config.architectures, ["MiMoV2VLForConditionalGeneration"])
            self.assertEqual(config.model_type, "mimo_v2_vl")
            self.assertEqual(processor.image_token_id, config.image_token_id)

    def test_dry_run_detects_missing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            original = write_original_checkpoint(self.model, Path(directory) / "original")
            index_path = original / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            del index["weight_map"]["visual.merger.ln_q.weight"]
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model.visual.merger.norm.weight"):
                convert_script.convert_checkpoint(original, dry_run=True)

    def test_ambiguous_tp_size_requires_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            original = write_original_checkpoint(self.model, Path(directory) / "original", tp_size=2)
            # Without the TP size in the index metadata, BF16 fused QKV weights with 2 and 4 KV heads fit TP=1 or 2.
            index_path = original / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index.pop("metadata")
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"candidates: \[1, 2\]"):
                convert_script.convert_checkpoint(original, dry_run=True)
            convert_script.convert_checkpoint(original, tp_size=2, dry_run=True)
