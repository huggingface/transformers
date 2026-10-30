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
    AutoModel,
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
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import modeling_mimo_v2_utils
    from transformers.modeling_mimo_v2_utils import dequantize_mxfp4, split_interleaved_qkv
    from transformers.models.mimo_v2_vl.modeling_mimo_v2_vl import attention_forward


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
            moe_router_dtype="float32",
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
            qk_channels=8,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            fullatt_block_indexes=[0],
            vit_window_attn_types=[-1, 0, 1],
            visual_token_window_size=2,
        ),
        audio_config=MiMoV2VLAudioConfig(
            audio_channels=4,
            group_size=2,
            input_local_dim=16,
            input_local_layers=2,
            input_local_attn_heads=2,
            input_local_head_dim=8,
            input_local_intermediate_size=24,
            out_hidden_size=32,
            speech_vocab_size=16,
            speech_zeroemb_idx=15,
            audio_segment_size=8,
        ),
        audio_tokenizer_config=MiMoV2VLAudioTokenizerConfig(
            d_model=16,
            n_mels=4,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=24,
            encoder_skip_layer_id=1,
            num_quantizers=4,
            codebook_size=[16] * 4,
        ),
    )


class MiMoV2VLModelTester:
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
    @unittest.skip("MiMo currently supports default RoPE only; unsupported scaling is tested explicitly.")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    @unittest.skip("MiMo currently supports default RoPE only; unsupported scaling is tested explicitly.")
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip("MiMo's rotary module takes dimensions rather than a config; partial RoPE has dedicated tests.")
    def test_model_rope_with_partial_rotation(self):
        pass


@require_torch
class MiMoV2VLCommonModelTest(MiMoV2VLCommonTestMixin, ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
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

    def _image_features_prepare_config_and_inputs(self):
        self.skipTest("MiMo feature helpers return fused tensors, covered by test_modal_embeddings_are_fused.")

    def _audio_features_prepare_config_and_inputs(self):
        self.skipTest("MiMo feature helpers return fused tensors, covered by test_modal_embeddings_are_fused.")

    def _video_features_prepare_config_and_inputs(self):
        self.skipTest("MiMo feature helpers return fused tensors, covered by test_modal_embeddings_are_fused.")


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
                "model.speech_embeddings": "cpu",
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

    def test_noncontiguous_image_patches(self):
        inputs = self.image_inputs()
        pixels = inputs["pixel_values"]
        noncontiguous = pixels.repeat_interleave(2, dim=-1)[:, ::2]
        self.assertFalse(noncontiguous.is_contiguous())
        with torch.no_grad():
            expected = self.model(**inputs).logits
            inputs["pixel_values"] = noncontiguous
            actual = self.model(**inputs).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

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

    @parameterized.expand(
        [
            ("audio_feature_lens", [1.5]),
            ("audio_feature_lens", [2.0]),
            ("audio_feature_lens", [True]),
            ("audio_feature_lens", [[1]]),
            ("audio_feature_lens", 1),
            ("audio_segment_offsets", [[0, 0.5, 1]]),
            ("audio_segment_offsets", [[0, 0, 1.0]]),
            ("audio_segment_offsets", [[True, False, True]]),
            ("audio_segment_offsets", [0, 0, 1]),
            ("audio_segment_offsets", [[0, 1]]),
        ]
    )
    def test_audio_metadata_requires_integers(self, name, values):
        codes = torch.zeros(1, 2, 2, 4, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, name):
            self.model.model.get_audio_features(input_audio_codes=codes, **{name: values})

    @parameterized.expand([("bool",), ("complex64",), ("float32",)])
    def test_audio_codes_require_integer_dtype(self, dtype):
        codes = torch.zeros(1, 2, 2, 4, dtype=getattr(torch, dtype))
        with self.assertRaisesRegex(ValueError, "codec IDs must be integers"):
            self.model.model.get_audio_features(input_audio_codes=codes)

    @parameterized.expand([((1, 9),), ((2, 8),), ((9,),)])
    def test_audio_feature_mask_shape_must_match(self, mask_shape):
        features = torch.randn(2, 9, 4)
        with self.assertRaisesRegex(ValueError, "feature_attention_mask must match"):
            self.model.model.audio_tokenizer(features, torch.ones(mask_shape, dtype=torch.long), 8)

    @parameterized.expand([(2.0,), (-1.0,), (float("nan"),)])
    def test_audio_feature_mask_must_be_binary(self, value):
        with self.assertRaisesRegex(ValueError, "zero and one"):
            self.model.model.audio_tokenizer(torch.randn(1, 9, 4), torch.full((1, 9), value), 8)

    @parameterized.expand([((0, 9, 4),), ((1, 0, 4),), ((9, 4),), ((1, 9, 5),)])
    def test_audio_feature_dimensions_are_validated(self, shape):
        with self.assertRaisesRegex(ValueError, "input_features"):
            self.model.model.audio_tokenizer(torch.zeros(shape), None, 8)

    def test_float_and_boolean_audio_masks_match_integer_masks(self):
        tokenizer = self.model.model.audio_tokenizer
        for layer in tokenizer.encoder.quantizer.layers:
            layer._codebook.embed.copy_(torch.randn_like(layer._codebook.embed))
        features = torch.randn(2, 12, 4)
        mask = torch.tensor([[1] * 9 + [0] * 3, [0] * 3 + [1] * 9])
        expected = tokenizer(features, mask, 8)
        self.assertEqual(len(expected), 2)
        for dtype in (torch.float32, torch.bool):
            actual = tokenizer(features, mask.to(dtype), 8)
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
        query = torch.zeros(1, 2, 1, 4)
        key = torch.zeros(1, 1, 2, 4)
        value = torch.ones(1, 1, 2, 3)
        sinks = torch.tensor([0.0, torch.log(torch.tensor(2.0))])
        actual = attention_forward(query, key, value, sinks=sinks)
        expected = torch.tensor([2 / 3, 1 / 2]).reshape(1, 2, 1, 1).expand_as(actual)
        torch.testing.assert_close(actual, expected)

    @parameterized.expand([("eager", None), ("sdpa", None), ("eager", (3, 0)), ("sdpa", (2, 2))])
    def test_chunked_attention_matches_dense(self, implementation, window):
        query = torch.randn(2, 4, 11, 8, requires_grad=True)
        key = torch.randn(2, 2, 14, 8, requires_grad=True)
        value = torch.randn(2, 2, 14, 4, requires_grad=True)
        sinks = torch.randn(4, requires_grad=True)
        mask = torch.zeros(2, 1, 11, 14)
        mask[0, :, :, :2] = -torch.inf
        expected_mask = mask.clone()
        if window is not None:
            queries = torch.arange(11) + 3
            keys = torch.arange(14)
            allowed = (keys[None] >= queries[:, None] - window[0]) & (keys[None] <= queries[:, None] + window[1])
            expected_mask = expected_mask.masked_fill(~allowed, -torch.inf)
        expected = attention_forward(
            query, key, value, expected_mask, sinks, implementation="eager", query_chunk_size=32
        )
        actual = attention_forward(
            query, key, value, mask, sinks, implementation=implementation, window_size=window, query_chunk_size=3
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual_grads = torch.autograd.grad(actual.sum(), (query, key, value, sinks), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.sum(), (query, key, value, sinks))
        for actual_gradient, expected_gradient in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=1e-5)

    def test_segmented_audio_token_count_and_left_padding(self):
        from transformers import MiMoV2FeatureExtractor

        model = self.model.model
        model.config.audio_config.audio_segment_size = 9
        extractor = MiMoV2FeatureExtractor(audio_segment_size=9, audio_group_size=2)
        length = extractor._get_audio_token_length(27)
        self.assertEqual(length, 5)
        features = torch.randn(1, 27, 4)
        mask = torch.ones(1, 27, dtype=torch.long)
        with torch.no_grad():
            expected = model.get_audio_features(
                input_features=features, feature_attention_mask=mask, audio_feature_lens=[length]
            )
            padded = torch.nn.functional.pad(features, (0, 0, 5, 0))
            padded_mask = torch.nn.functional.pad(mask, (5, 0))
            actual = model.get_audio_features(
                input_features=padded, feature_attention_mask=padded_mask, audio_feature_lens=[length]
            )
        torch.testing.assert_close(actual, expected)

    def test_bfloat16_save_load_preserves_codebook_precision(self):
        model = self.model
        for name, parameter in model.named_parameters():
            if not name.endswith("e_score_correction_bias"):
                parameter.data = parameter.data.to(torch.bfloat16)
        for layer in model.model.audio_tokenizer.encoder.quantizer.layers:
            layer._codebook.embed.copy_(torch.randn_like(layer._codebook.embed))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = AutoModelForImageTextToText.from_pretrained(directory, dtype="auto")
            for original, loaded in zip(
                model.model.audio_tokenizer.encoder.quantizer.layers,
                restored.model.audio_tokenizer.encoder.quantizer.layers,
            ):
                self.assertEqual(loaded._codebook.embed.dtype, torch.float32)
                torch.testing.assert_close(original._codebook.embed, loaded._codebook.embed, atol=0, rtol=0)
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

    def test_legacy_text_config_fields(self):
        config = MiMoV2VLTextConfig(
            num_hidden_layers=2,
            hybrid_layer_pattern=[1, 0],
            moe_layer_freq=[0, 1],
            sliding_window_size=16,
            rope_theta=20000,
            swa_rope_theta=30000,
            partial_rotary_factor=0.5,
        )
        self.assertEqual(config.layer_types, ["sliding_attention", "full_attention"])
        self.assertEqual(config.mlp_layer_types, ["dense", "sparse"])
        self.assertEqual(config.sliding_window, 17)
        self.assertEqual(config.rope_parameters["full_attention"]["rope_theta"], 20000)
        self.assertEqual(config.rope_parameters["sliding_attention"]["rope_theta"], 30000)
        self.assertEqual(config.rope_parameters["sliding_attention"]["partial_rotary_factor"], 0.5)
        serialized = config.to_dict()
        self.assertFalse(
            {"hybrid_layer_pattern", "moe_layer_freq", "sliding_window_size", "rope_theta"} & set(serialized)
        )
        self.assertEqual(MiMoV2VLTextConfig.from_dict(serialized).to_dict(), serialized)
        with self.assertRaisesRegex(ValueError, "default RoPE"):
            MiMoV2VLTextConfig(rope_parameters={"rope_type": "linear", "factor": 2})

    @require_torch_gpu
    def test_router_fallback_without_mm_output_dtype(self):
        router = self.model.model.language_model.layers[1].mlp.gate.cuda()
        router.config.moe_router_dtype = "bfloat16"
        hidden_states = torch.randn(4, router.config.hidden_size, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            expected = router(hidden_states)
            with patch(
                "transformers.models.mimo_v2_vl.modeling_mimo_v2_vl.is_torch_greater_or_equal", return_value=False
            ):
                actual = router(hidden_states)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference)

    @parameterized.expand([(dtype, dim) for dtype in ("float16", "bfloat16", "float32", "float64") for dim in (4, 8)])
    def test_rotary_dtype_and_partial_rotation(self, dtype, rotary_dim):
        from transformers.models.mimo_v2_vl.modeling_mimo_v2_vl import (
            MiMoV2VLRotaryEmbedding,
            apply_rotary_pos_emb_fp32,
        )

        dtype = getattr(torch, dtype)
        query = torch.randn(1, 2, 3, 8, dtype=dtype, device=torch_device, requires_grad=True)
        key = torch.randn(1, 1, 3, 8, dtype=dtype, device=torch_device, requires_grad=True)
        positions = torch.tensor([[0, 49000, 49001]], device=torch_device)
        cos, sin = MiMoV2VLRotaryEmbedding(rotary_dim, 10000.0).to(torch_device)(query, positions)
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
        from transformers.models.mimo_v2_vl.modeling_mimo_v2_vl import apply_rotary_pos_emb_fp32

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
        from transformers.models.mimo_v2_vl.modeling_mimo_v2_vl import MiMoV2VLRotaryEmbedding

        for theta in (10000.0, 10000000.0, 640000.0):
            rotary = MiMoV2VLRotaryEmbedding(64, theta)
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

    def test_legacy_generation_defaults_can_be_saved(self):
        from transformers import GenerationConfig
        from transformers.modeling_mimo_v2_utils import _load_generation_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "do_sample": False,
                "top_p": 0.95,
                "temperature": 0.7,
                "eos_token_id": [3, 7],
                "max_new_tokens": 2048,
            }
            (root / "generation_config.json").write_text(json.dumps(source), encoding="utf-8")
            config = _load_generation_config(root)
            self.assertFalse(config.do_sample)
            self.assertIsNone(config.top_p)
            self.assertIsNone(config.temperature)
            self.assertEqual(config.eos_token_id, [3, 7])
            self.assertEqual(config.max_new_tokens, 2048)
            config.save_pretrained(root / "native")
            restored = GenerationConfig.from_pretrained(root / "native")
            self.assertEqual(restored.eos_token_id, [3, 7])
            self.assertEqual(json.loads((root / "generation_config.json").read_text(encoding="utf-8")), source)

    @parameterized.expand(
        [
            ({}, "bfloat16"),
            ({"dtype": None}, "bfloat16"),
            ({"dtype": None, "torch_dtype": "float32"}, "float32"),
            ({"dtype": "float16", "torch_dtype": "float32"}, "float16"),
        ]
    )
    def test_original_checkpoint_auto_dtype(self, dtype_config, expected_dtype):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps({"model_type": "mimo_v2", "vision_config": {"depth": 1}, **dtype_config}), encoding="utf-8"
            )
            with patch.object(modeling_mimo_v2_utils, "load_original_checkpoint", return_value=self.model) as load:
                MiMoV2VLForConditionalGeneration.from_pretrained(directory, dtype="auto", local_files_only=True)
            self.assertEqual(load.call_args.kwargs["dtype"], getattr(torch, expected_dtype))

    def test_mxfp4_unpacking_and_scaling(self):
        packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
        scales = torch.tensor([[127, 128]], dtype=torch.uint8)
        actual = dequantize_mxfp4(packed, scales, block_size=8)
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
        actual = split_interleaved_qkv(interleaved, config, layer_idx=0, tp_size=2)
        for result, expected in zip(actual, (query, key, value)):
            torch.testing.assert_close(result, expected)

    def test_tp_inference_requires_unambiguous_kv_replication(self):
        from transformers.modeling_mimo_v2_utils import _infer_checkpoint_tp_size

        config = self.model.config.text_config
        self.assertEqual(_infer_checkpoint_tp_size(config, layer_idx=0, rows=80), 4)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _infer_checkpoint_tp_size(config, layer_idx=0, rows=56)

    def test_fp8_qkv_scales_preserve_per_rank_padding(self):
        weight = torch.ones(56, 4).to(torch.float8_e4m3fn)
        scales = torch.arange(1, 9, dtype=torch.float32)[:, None]
        query, key, value = split_interleaved_qkv(
            weight, self.model.config.text_config, layer_idx=0, tp_size=2, scales=scales, block_size=(8, 4)
        )
        expected_query = torch.tensor([1] * 8 + [2] * 8 + [5] * 8 + [6] * 8)[:, None].expand(32, 4).float()
        expected_key = torch.tensor([3] * 8 + [7] * 8)[:, None].expand(16, 4).float()
        expected_value = torch.tensor([4] * 4 + [8] * 4)[:, None].expand(8, 4).float()
        torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
        torch.testing.assert_close(value, expected_value, rtol=0, atol=0)

    @parameterized.expand([(False, False), (True, False), (True, True)])
    def test_original_checkpoint_loading_round_trip(self, multimodal, split_scales):
        from safetensors.torch import save_file

        from transformers import AutoModelForCausalLM, GenerationConfig, MiMoV2FlashForCausalLM, MiMoV2FlashModel

        model = self.model
        weights = {}
        audio_weights = {}
        for name, tensor in model.state_dict().items():
            if name.startswith("model.audio_tokenizer."):
                name = name.removeprefix("model.audio_tokenizer.")
                name = name.replace(".quantizer.", ".quantizer.vq.").replace(
                    ".down_sample_layer.", ".down_sample_layer.0."
                )
                audio_weights[name] = tensor.clone()
            elif name.startswith("model.language_model."):
                weights["model." + name.removeprefix("model.language_model.")] = tensor.clone()
            else:
                name = name.removeprefix("model.")
                for prefix in ("visual.merger.", "audio_encoder.projection."):
                    name = name.replace(prefix + "linear_fc1.", prefix + "mlp.0.").replace(
                        prefix + "linear_fc2.", prefix + "mlp.2."
                    )
                weights[name] = tensor.clone()
        for layer_idx in range(model.config.text_config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.self_attn."
            projections = [
                weights.pop(prefix + projection + ".weight").chunk(2) for projection in ("q_proj", "k_proj", "v_proj")
            ]
            weights[prefix + "qkv_proj.weight"] = torch.cat(
                [projection[rank] for rank in range(2) for projection in projections]
            )
        config_dict = model.config.text_config.to_dict()
        config_dict.update(
            model_type="mimo_v2",
            vision_config=model.config.vision_config.to_dict(),
            audio_config=model.config.audio_config.to_dict(),
            attention_projection_layout="fused_qkv",
            image_token_id=4,
            video_token_id=5,
            audio_token_id=6,
        )
        config_dict["architectures"] = ["MiMoV2ForCausalLM"]
        separate_scales = {}
        if split_scales:
            name = "model.layers.0.mlp.down_proj.weight"
            weights[name] = (weights[name] * 2).to(torch.float8_e4m3fn)
            separate_scales[name.replace(".weight", ".weight_scale_inv")] = torch.tensor([[0.5]])
            with torch.no_grad():
                model.model.language_model.layers[0].mlp.down_proj.weight.copy_(weights[name].float() * 0.5)
        if not multimodal:
            config_dict.pop("vision_config")
            config_dict.pop("audio_config")
            weights = {name: value for name, value in weights.items() if name.startswith(("model.", "lm_head."))}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio_tokenizer").mkdir()
            (root / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
            (root / "generation_config.json").write_text(
                json.dumps({"eos_token_id": [3, 7], "max_new_tokens": 13, "do_sample": True, "temperature": 0.7}),
                encoding="utf-8",
            )
            (root / "audio_tokenizer/config.json").write_text(
                model.config.audio_tokenizer_config.to_json_string(), encoding="utf-8"
            )
            weight_map = {
                **dict.fromkeys(weights, "model.safetensors"),
                **dict.fromkeys(separate_scales, "scales.safetensors"),
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"tp_size": 2}, "weight_map": weight_map}),
                encoding="utf-8",
            )
            save_file(weights, root / "model.safetensors")
            if separate_scales:
                save_file(separate_scales, root / "scales.safetensors")
            save_file(audio_weights, root / "audio_tokenizer/model.safetensors")
            restored = AutoModelForCausalLM.from_pretrained(
                root, device_map="cpu", dtype=torch.float32, trust_remote_code=False, local_files_only=True
            )
            self.assertIsInstance(restored, MiMoV2VLForConditionalGeneration if multimodal else MiMoV2FlashForCausalLM)
            self.assertEqual(restored.name_or_path, str(root))
            self.assertEqual(restored.generation_config.eos_token_id, [3, 7])
            self.assertEqual(restored.generation_config.max_new_tokens, 13)
            self.assertEqual(restored.generation_config.temperature, 0.7)
            if multimodal and not split_scales:
                overridden = MiMoV2VLForConditionalGeneration.from_pretrained(
                    root,
                    device_map="cpu",
                    dtype=torch.float32,
                    trust_remote_code=False,
                    local_files_only=True,
                    generation_config=GenerationConfig(do_sample=False, max_new_tokens=7, eos_token_id=2),
                )
                self.assertEqual(overridden.generation_config.max_new_tokens, 7)
                self.assertEqual(overridden.generation_config.eos_token_id, 2)
                self.assertEqual(overridden.name_or_path, str(root))
            if multimodal:
                for name, expected in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
                inputs = self.image_inputs()
            else:
                inputs = {"input_ids": torch.tensor([[2, 7, 8]])}
            with torch.no_grad():
                torch.testing.assert_close(restored(**inputs).logits, model(**inputs).logits, rtol=1e-4, atol=1e-5)
            if not split_scales:
                device_map = {"model": "cpu", "lm_head": "cpu"}
                expected_map = {"": "cpu"}
                if multimodal:
                    expected_map = dict.fromkeys(
                        ["language_model", "visual", "speech_embeddings", "audio_encoder", "audio_tokenizer"], "cpu"
                    )
                    device_map = {
                        **{f"model.{name}": device for name, device in expected_map.items()},
                        "lm_head": "cpu",
                    }
                backbone = AutoModel.from_pretrained(
                    root, device_map=device_map, dtype=torch.float32, trust_remote_code=False, local_files_only=True
                )
                self.assertIsInstance(backbone, MiMoV2VLModel if multimodal else MiMoV2FlashModel)
                self.assertEqual(backbone.name_or_path, str(root))
                self.assertEqual(backbone.hf_device_map, expected_map)
                with torch.no_grad():
                    torch.testing.assert_close(
                        backbone(**inputs).last_hidden_state,
                        restored.model(**inputs).last_hidden_state,
                        rtol=0,
                        atol=0,
                    )
