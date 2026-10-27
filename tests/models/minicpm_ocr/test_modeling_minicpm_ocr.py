# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MiniCPM-OCR 1.0 model."""

import unittest

from transformers import (
    AutoProcessor,
    MiniCPMOcrConfig,
    is_torch_available,
)
from transformers.models.minicpm_ocr.configuration_minicpm_ocr import MiniCPMOcrVisionConfig
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import MiniCPMOcrForConditionalGeneration, MiniCPMOcrModel
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class MiniCPMOcrVisionText2TextModelTester(VLMModelTester):
    base_model_class = MiniCPMOcrModel
    config_class = MiniCPMOcrConfig
    text_config_class = Qwen3_5TextConfig
    vision_config_class = MiniCPMOcrVisionConfig
    conditional_generation_class = MiniCPMOcrForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("image_token_id", 100)
        # `downsample_mode="4x"` skips the ViT insert merger, so only the downsample MLP merges:
        # patch_size=8, image_size=32 → 4×4 grid → merger [2×2] = 4 tokens
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 8)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("vocab_size", 256)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 32)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 512)
        kwargs.setdefault("rope_parameters", {"type": "default", "rope_theta": 10_000, "mrope_section": [2, 1, 1]})
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 1)
        kwargs.setdefault("pad_token_id", 2)
        # Qwen3.5 hybrid attention
        kwargs.setdefault("layer_types", ["full_attention", "linear_attention"])
        kwargs.setdefault("linear_conv_kernel_dim", 2)
        kwargs.setdefault("linear_key_head_dim", 16)
        kwargs.setdefault("linear_value_head_dim", 16)
        kwargs.setdefault("linear_num_key_heads", 4)
        kwargs.setdefault("linear_num_value_heads", 8)
        # Vision config overrides
        kwargs.setdefault("vision_hidden_act", "gelu_pytorch_tanh")
        kwargs.setdefault("vision_intermediate_size", 128)
        # MiniCPM-OCR specific
        kwargs.setdefault("insert_layer_id", 0)
        super().__init__(parent, **kwargs)

    def create_pixel_values(self, batch_size: int | None = None):
        # Override to 5D for patch-based models
        batch_size = batch_size if batch_size is not None else self.batch_size
        C = self.num_channels
        P = self.patch_size
        h_patches = self.image_size // self.patch_size
        w_patches = self.image_size // self.patch_size
        total_L = batch_size * h_patches * w_patches * P
        return floats_tensor([1, C, P, total_L])

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        h_patches = self.image_size // self.patch_size
        w_patches = self.image_size // self.patch_size
        target_sizes = torch.tensor([[h_patches, w_patches]] * batch_size, dtype=torch.int32, device=torch_device)
        return {"target_sizes": target_sizes}

    def get_config(self):
        text_config = {
            "model_type": "qwen3_5_text",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "hidden_act": "silu",
            "max_position_embeddings": self.max_position_embeddings,
            "rope_parameters": self.rope_parameters,
            "tie_word_embeddings": self.tie_word_embeddings,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "layer_types": self.layer_types,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
        }
        vision_config = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.vision_intermediate_size,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_channels": self.num_channels,
            "hidden_act": self.vision_hidden_act,
            "insert_layer_id": self.insert_layer_id,
        }
        return MiniCPMOcrConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=self.image_token_id,
            insert_layer_id=self.insert_layer_id,
        )


@require_torch
class MiniCPMOcrModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMOcrVisionText2TextModelTester
    # The default `downsample_mode="4x"` bypasses the ViT insert merger, so `vision_tower.vit_merger`
    # gets no gradient. Its weights are still loaded and used when running with `downsample_mode="16x"`.
    test_all_params_have_gradient = False

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            key: value
            for key, value in inputs_dict.items()
            if ("pixel" in key or "image" in key or key == "target_sizes") and "video" not in key
        }
        return config, inputs_dict

    def _video_features_prepare_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        return config, {
            "pixel_values_videos": inputs_dict["pixel_values"],
            "target_sizes_videos": inputs_dict["target_sizes"],
        }

    @unittest.skip(reason="Conversion only for CausalLM loading from saved ConditionalLM")
    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        pass

    @unittest.skip(
        reason="NaViT packs all images into a single tensor (batch dim=1); "
        "generic batch-splitting logic cannot separate individual samples"
    )
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="Packed attention doesn't return attn weights yet")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip(reason="Packed attention doesn't return attn weights yet")
    def test_get_video_features_attentions(self):
        pass

    @unittest.skip(reason="FA works on generate test, inference needs override to pass target sizes")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip(reason="FA works on generate, inference needs override to pass target sizes")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    def _get_conv_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        intermediate_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

        return (batch_size, intermediate_size, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim

        return (batch_size, num_v_heads, head_k_dim, head_v_dim)

    def test_attention_outputs(self):
        """Overwritten: Qwen3.5 alternates between full attention and gated deltanet layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(
                len(attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )

            del inputs_dict["output_attentions"]
            config.text_config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(
                len(attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )
            self.assertListEqual(
                list(attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )
            out_len = len(outputs)

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
                len(self_attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )

    def test_4x_downsample_keeps_four_times_more_visual_tokens(self):
        """`"4x"` skips the ViT insert merger, so the patch grid is only merged once."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = self.model_tester.base_model_class(config).to(torch_device).eval()

        pixel_values = inputs_dict["pixel_values"].to(torch_device)
        target_sizes = inputs_dict["target_sizes"]

        with torch.no_grad():
            features_4x = model.get_image_features(pixel_values, target_sizes, downsample_mode="4x")
            features_16x = model.get_image_features(pixel_values, target_sizes, downsample_mode="16x")

        num_tokens_4x = sum(feature.shape[0] for feature in features_4x.pooler_output)
        num_tokens_16x = sum(feature.shape[0] for feature in features_16x.pooler_output)
        self.assertEqual(num_tokens_4x, 4 * num_tokens_16x)


@slow
@require_torch_accelerator
class MiniCPMOcrIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "openbmb/MiniCPM-OCR-1.0"

    @slow
    def test_small_model_logits(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMOcrForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)

        # The template renders an already-closed thinking block when `enable_thinking=False` is
        # passed, so the prompt ends with the closed thinking tags and the model answers directly.
        expected_input_ids = Expectations(
            {
                ("cuda", (8, 0)): [248045, 846, 198, 12675, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271],
            }
        ).get_expectation()
        self.assertEqual(inputs["input_ids"][0].tolist(), expected_input_ids)

        with torch.no_grad():
            logits = model(**inputs).logits.float()

        self.assertEqual(logits.shape, torch.Size([1, 13, 248100]))
        self.assertTrue(torch.isfinite(logits).all().item())

        expected_slice = Expectations(
            {
                ("cuda", (8, 0)): [2.765625, 6.0, 0.61328125, 4.8125, 2.0, -2.796875, 3.59375, 3.71875],
            }
        ).get_expectation()
        torch.testing.assert_close(
            logits[0, -1, :8], torch.tensor(expected_slice, device=torch_device), rtol=1e-3, atol=1e-3
        )
        expected_argmax = Expectations({("cuda", (8, 0)): 12675}).get_expectation()
        self.assertEqual(logits[0, -1].argmax().item(), expected_argmax)

    @slow
    def test_small_model_vision_generation(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMOcrForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
                    },
                    {"type": "text", "text": "What kind of animal is this?"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device, dtype=torch.bfloat16)

        # 960x686 at `downsample_mode="4x"`: one overview patch plus a 2x2 slice grid, each
        # aligned to `patch_size * 2 = 28`.
        self.assertEqual(inputs["target_sizes"].tolist(), [[28, 38]] * 5)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        expected_text = Expectations(
            {
                ("cuda", (8, 0)): (
                    "This is a **Tibet Wildcat** (*Lynx rufus dovondo*).\n\n### Key Features:\n- **Morph"
                ),
            }
        ).get_expectation()
        self.assertEqual(decoded_text, expected_text)

    @slow
    def test_small_model_vision_generation_batch(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMOcrForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
                    },
                    {"type": "text", "text": "What kind of animal is this?"},
                ],
            }
        ]
        batch_messages = [messages, messages]

        inputs = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            enable_thinking=False,
        ).to(model.device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_texts = processor.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        expected_text = Expectations(
            {
                ("cuda", (8, 0)): (
                    "This is a **Tibet Wildcat** (*Lynx rufus dovondo*).\n\n### Key Features:\n- **Morph"
                ),
            }
        ).get_expectation()
        self.assertEqual(len(decoded_texts), 2)
        self.assertEqual(decoded_texts[0], decoded_texts[1])
        # Batching the same sample twice must not change it w.r.t. the single-sample test above.
        self.assertEqual(decoded_texts[0], expected_text)
