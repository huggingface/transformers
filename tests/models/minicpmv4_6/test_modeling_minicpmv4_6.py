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
"""Testing suite for the PyTorch MiniCPM-V 4.6 model."""

import unittest

import pytest

from transformers import (
    AutoProcessor,
    MiniCPMV4_6Config,
    is_torch_available,
)
from transformers.models.minicpmv4_6.configuration_minicpmv4_6 import MiniCPMV4_6VisionConfig
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import MiniCPMV4_6ForConditionalGeneration, MiniCPMV4_6Model
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class MiniCPMV4_6VisionText2TextModelTester(VLMModelTester):
    base_model_class = MiniCPMV4_6Model if is_torch_available() else None
    config_class = MiniCPMV4_6Config
    text_config_class = Qwen3_5TextConfig if is_torch_available() else None
    vision_config_class = MiniCPMV4_6VisionConfig
    conditional_generation_class = MiniCPMV4_6ForConditionalGeneration if is_torch_available() else None

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("image_token_id", 100)
        # patch_size=8, image_size=32 → 4×4 grid → vit_merger [2×2] → merger [1×1] = 1 token
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 8)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("vocab_size", 256)
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 512)
        kwargs.setdefault("rope_parameters", {"rope_type": "default"})
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
        # MiniCPM-V 4.6 specific
        kwargs.setdefault("insert_layer_id", 0)
        super().__init__(parent, **kwargs)

    def _navit_pixel_values(self, batch_size):
        """Build NaViT-packed pixel_values: (1, C, patch_size, total_L)."""
        C = self.num_channels
        P = self.patch_size
        h_patches = self.image_size // self.patch_size
        w_patches = self.image_size // self.patch_size
        total_L = batch_size * h_patches * w_patches * P
        return floats_tensor([1, C, P, total_L])

    def _target_sizes(self, batch_size):
        h_patches = self.image_size // self.patch_size
        w_patches = self.image_size // self.patch_size
        return torch.tensor([[h_patches, w_patches]] * batch_size, dtype=torch.int32)

    def create_pixel_values(self):
        return self._navit_pixel_values(self.batch_size)

    def get_additional_inputs(self, config, input_ids, pixel_values):
        return {"target_sizes": self._target_sizes(self.batch_size)}

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
            "rope_theta": 10000,
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
        }
        return MiniCPMV4_6Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=self.image_token_id,
            image_size=self.image_size,
            drop_vision_last_layer=False,
            insert_layer_id=self.insert_layer_id,
        )


@require_torch
class MiniCPMV4_6ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMV4_6VisionText2TextModelTester

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)
        inputs_dict["pixel_values"] = self.model_tester._navit_pixel_values(batch_size)
        inputs_dict["target_sizes"] = self.model_tester._target_sizes(batch_size)
        return config, inputs_dict

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

    @unittest.skip(
        "NaViT packing puts all images in a single tensor with dim-0 = 1; "
        "the default test cannot correctly simulate image count mismatches"
    )
    def test_mismatching_num_image_tokens(self):
        pass

    @unittest.skip(reason="MiniCPM-V uses custom pixel_values format (list-of-list), skipping common input tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MiniCPM-V uses custom pixel_values format (list-of-list), skipping common input tests")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="Compile not yet supported for MiniCPM-V models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip("FlashAttention only supports fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip("The Qwen3.5 hybrid cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="MiniCPM-V 4.6 uses Qwen3.5 hybrid cache layers that are incompatible with QuantizedCache.")
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip(reason="Conversion only for CausalLM loading from saved ConditionalLM")
    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        pass

    @unittest.skip(
        reason="NaViT packs all images into a single tensor (batch dim=1); "
        "generic batch-splitting logic cannot separate individual samples"
    )
    def test_batching_equivalence(self):
        pass

    @unittest.skip(
        reason="NaViT packs all images into a single tensor (batch dim=1); "
        "generic batch-splitting logic cannot separate individual samples"
    )
    def test_model_forward_default_config_values(self):
        pass

    @unittest.skip(
        reason="get_image_features uses a custom pipeline (vision_tower -> vit_merger -> merger) "
        "that does not accept output_attentions/output_hidden_states kwargs"
    )
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip(
        reason="get_image_features uses a custom pipeline (vision_tower -> vit_merger -> merger) "
        "that does not accept output_attentions/output_hidden_states kwargs"
    )
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip(
        reason="get_video_features uses a custom pipeline that does not accept "
        "output_attentions/output_hidden_states kwargs"
    )
    def test_get_video_features_attentions(self):
        pass

    @unittest.skip(
        reason="get_video_features uses a custom pipeline that does not accept "
        "output_attentions/output_hidden_states kwargs"
    )
    def test_get_video_features_hidden_states(self):
        pass

    @unittest.skip(
        "MiniCPM-V generate creates vision-aware embeddings via _build_vlm_inputs; "
        "text-only get_input_embeddings bypass produces different outputs"
    )
    def test_generate_from_inputs_embeds(self):
        pass

    @unittest.skip(reason="Same as test_generate_from_inputs_embeds: vision-aware vs text-only embeddings mismatch")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(
        "Manual left-padding in test does not adjust image_bound offsets, "
        "causing vision features to be placed at wrong positions"
    )
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip(reason="Batch splitting in compile test incompatible with list-of-list pixel_values")
    @pytest.mark.torch_compile_test
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip(reason="Batch splitting in compile test incompatible with list-of-list pixel_values")
    @pytest.mark.torch_compile_test
    def test_generate_compilation_all_outputs(self):
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


@slow
@require_torch_accelerator
@unittest.skip(reason="waiting for release")
class MiniCPMV4_6IntegrationTest(unittest.TestCase):
    model_id = "openbmb/MiniCPM-V-4_6"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_logits(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits.float().cpu()

        self.assertEqual(logits.shape[0], 1)
        self.assertTrue(torch.isfinite(logits).all().item())

    @slow
    def test_small_model_vision_generation(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "What kind of animal is this?"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_text = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertIn("cat", decoded_text.lower())

    @slow
    def test_small_model_vision_generation_batch(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
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
        ).to(model.device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_texts = processor.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        self.assertEqual(len(decoded_texts), 2)
        for text in decoded_texts:
            self.assertIn("cat", text.lower())

    @slow
    def test_small_model_vision_generation_batch_mixed(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto", dtype=torch.bfloat16
        )

        image_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "What kind of animal is this?"},
                ],
            }
        ]
        text_only_message = [{"role": "user", "content": [{"type": "text", "text": "Who are you?"}]}]
        batch_messages = [image_message, text_only_message]

        inputs = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        decoded_texts = processor.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        self.assertEqual(len(decoded_texts), 2)
        self.assertTrue(len(decoded_texts[0]) > 0)
        self.assertTrue(len(decoded_texts[1]) > 0)
