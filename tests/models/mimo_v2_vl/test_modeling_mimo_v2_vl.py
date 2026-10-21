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
    MiMoV2VLTextConfig,
    MiMoV2VLVisionConfig,
)
from transformers.testing_utils import require_accelerate, require_torch, require_torch_gpu
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

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
            partial_rotary_factor=1.0,
            sliding_window_size=3,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            moe_router_dtype="float32",
            hybrid_layer_pattern=[0, 1],
            moe_layer_freq=[0, 1],
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

    def test_save_load_native_auto_model(self):
        inputs = self.image_inputs()
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            config = AutoConfig.from_pretrained(directory)
            self.assertIsInstance(config, MiMoV2VLConfig)
            restored = AutoModelForImageTextToText.from_pretrained(directory).eval()
            with torch.no_grad():
                torch.testing.assert_close(self.model(**inputs).logits, restored(**inputs).logits)

    def test_sink_is_extra_softmax_mass(self):
        query = torch.zeros(1, 2, 1, 4)
        key = torch.zeros(1, 1, 2, 4)
        value = torch.ones(1, 1, 2, 3)
        sinks = torch.tensor([0.0, torch.log(torch.tensor(2.0))])
        actual = attention_forward(query, key, value, sinks=sinks)
        expected = torch.tensor([2 / 3, 1 / 2]).reshape(1, 2, 1, 1).expand_as(actual)
        torch.testing.assert_close(actual, expected)

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

        from transformers import AutoModelForCausalLM, MiMoV2FlashForCausalLM

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
            if multimodal:
                for name, expected in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
                inputs = self.image_inputs()
            else:
                inputs = {"input_ids": torch.tensor([[2, 7, 8]])}
            with torch.no_grad():
                torch.testing.assert_close(restored(**inputs).logits, model(**inputs).logits, rtol=1e-4, atol=1e-5)
