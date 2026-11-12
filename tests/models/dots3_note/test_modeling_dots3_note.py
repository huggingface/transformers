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

import unittest

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Dots3NoteConfig,
    Dots3NoteForCausalLM,
    Dots3NoteForConditionalGeneration,
    Dots3NoteModel,
    Dots3NoteTextModel,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torchvision,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


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
        self.index_head_dim = 32
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

    @unittest.skip("DSA/SWA require attention masks, which the Flash-only SDPA backend does not support.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

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
    conditional_generation_class = Dots3NoteForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 7)
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("audio_token_id", 5)
        kwargs.setdefault("image_size", 8)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("seq_length", 16)
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
        kwargs.setdefault("audio_feature_size", 8)
        kwargs.setdefault("audio_hop_length", 4)
        kwargs.setdefault("audio_downsample_hidden_size", 4)
        kwargs.setdefault("audio_adapter_input_size", 64)
        kwargs.setdefault("text_intermediate_size", 256)
        kwargs.setdefault("text_num_key_value_heads", kwargs["num_attention_heads"])
        kwargs.setdefault("text_q_lora_rank", 16)
        kwargs.setdefault("text_kv_lora_rank", 16)
        kwargs.setdefault("text_head_dim", 16)
        kwargs.setdefault("text_qk_rope_head_dim", 8)
        kwargs.setdefault("text_qk_nope_head_dim", 8)
        kwargs.setdefault("text_v_head_dim", 16)
        kwargs.setdefault("text_layer_types", ["indexed_attention", "sliding_attention"])
        kwargs.setdefault("text_mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("text_per_layer_config", {})
        kwargs.setdefault("text_index_head_dim", 32)
        kwargs.setdefault("text_index_topk", 4)
        kwargs.setdefault("text_n_routed_experts", 4)
        kwargs.setdefault("text_shared_experts_intermediate_size", 16)
        super().__init__(parent, **kwargs)
        self.encoder_seq_length = self.seq_length

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.image_token_id, self.video_token_id, self.audio_token_id}

    def create_pixel_values(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return torch.randn(batch_size, 16, self.num_channels * self.patch_size**2, device=torch_device)

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        input_ids[:, self.num_image_tokens : 2 * self.num_image_tokens] = config.video_token_id
        grid = torch.tensor([[1, 4, 4]], device=torch_device).repeat(batch_size, 1)
        return {
            "image_grid_thw": grid,
            "pixel_values_videos": modality_inputs.clone(),
            "video_grid_thw": grid.clone(),
        }


class Dots3NoteAudio2TextModelTester(ALMModelTester):
    base_model_class = Dots3NoteModel
    config_class = Dots3NoteConfig
    audio_mask_key = "chunk_sample_lengths"
    conditional_generation_class = Dots3NoteForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("audio_token_id", 5)
        kwargs.setdefault("feat_seq_length", 16)
        kwargs.setdefault("num_mel_bins", 8)
        kwargs.setdefault("feature_size", 8)
        kwargs.setdefault("hop_length", 4)
        kwargs.setdefault("downsample_hidden_size", 4)
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("adapter_input_size", 64)
        kwargs.setdefault("adapter_out_dim", 64)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("vision_patch_size", 2)
        kwargs.setdefault("vision_spatial_merge_size", 2)
        kwargs.setdefault("vision_adapter_in_dim", 64)
        kwargs.setdefault("vision_adapter_merge_size", 2)
        kwargs.setdefault("vision_mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("vision_num_local_experts", 2)
        kwargs.setdefault("vision_num_experts_per_tok", 2)
        kwargs.setdefault("text_intermediate_size", 256)
        kwargs.setdefault("text_moe_intermediate_size", 64)
        kwargs.setdefault("text_num_key_value_heads", kwargs["num_attention_heads"])
        kwargs.setdefault("text_q_lora_rank", 16)
        kwargs.setdefault("text_kv_lora_rank", 16)
        kwargs.setdefault("text_qk_rope_head_dim", 8)
        kwargs.setdefault("text_qk_nope_head_dim", 8)
        kwargs.setdefault("text_v_head_dim", 16)
        kwargs.setdefault("text_layer_types", ["indexed_attention", "sliding_attention"])
        kwargs.setdefault("text_mlp_layer_types", ["dense", "sparse"])
        kwargs.setdefault("text_per_layer_config", {})
        kwargs.setdefault("text_index_head_dim", 32)
        kwargs.setdefault("text_index_topk", 4)
        kwargs.setdefault("text_n_routed_experts", 4)
        kwargs.setdefault("text_num_experts_per_tok", 2)
        kwargs.setdefault("text_shared_experts_intermediate_size", 16)
        kwargs.setdefault("text_initializer_range", 0.02)
        kwargs.setdefault("text_tie_word_embeddings", False)
        super().__init__(parent, **kwargs)
        self.encoder_seq_length = self.seq_length

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


@require_torch
class Dots3NoteVision2TextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Dots3NoteVision2TextModelTester
    test_all_params_have_gradient = False
    additional_model_inputs = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]
    # TODO: Cover packed/variable-resolution feature shapes instead of batch-first assertions.
    skip_test_image_features_output_shape = True
    skip_test_video_features_output_shape = True

    @unittest.skip("DSA/SWA require attention masks, which the Flash-only SDPA backend does not support.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def _audio_features_prepare_config_and_inputs(self):
        self.skipTest("Audio features are covered by Dots3NoteAudio2TextModelTest")

    def test_get_image_features_attentions(self, modality="image"):
        """The reused GLM-OCR vision attention does not return weights, even when capture is requested."""
        config, inputs = getattr(self, f"_{modality}_features_prepare_config_and_inputs")()
        config.vision_config.output_attentions = True
        for model_class in self.all_model_classes:
            model = model_class(config).base_model.to(torch_device).eval()
            model.set_attn_implementation("eager")
            with torch.no_grad():
                for kwargs in ({}, {"output_attentions": True}):
                    outputs = getattr(model, f"get_{modality}_features")(**inputs, **kwargs)
                    self.assertIsNone(outputs.attentions)

    def test_get_video_features_attentions(self):
        self.test_get_image_features_attentions("video")


@require_torch
class Dots3NoteAudio2TextModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = Dots3NoteAudio2TextModelTester
    test_all_params_have_gradient = False
    additional_model_inputs = ["chunk_sample_lengths"]

    @unittest.skip("DSA/SWA require attention masks, which the Flash-only SDPA backend does not support.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def _image_features_prepare_config_and_inputs(self):
        self.skipTest("Image features are covered by Dots3NoteVision2TextModelTest")

    def _video_features_prepare_config_and_inputs(self):
        self.skipTest("Video features are covered by Dots3NoteVision2TextModelTest")

    def _audio_features_prepare_config_and_inputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        return config, {key: inputs[key] for key in ("input_features", "chunk_sample_lengths")}


@slow
@require_torch
@require_torchvision
class Dots3NoteHubIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "hf-internal-testing/dummy-dots3"

    def test_text_generation(self):
        expected = Expectations(
            {
                ("cpu", None): (
                    "\ufffd3\u0229\u0016\ufffd\ufffd\ufffd<no_think>\ufffd<dots_function_response>H\u001f"
                    "\ufffd1\ufffd\u0016\ufffd\ufffd\ufffd<no_think>\ufffd<dots_function_response>H\u001f"
                    "\ufffdG\ufffd"
                ),
                ("cuda", None): (
                    "\ufffd3\u0229\u0016\ufffd\ufffd\ufffd<no_think>\ufffd<dots_function_response>H\u001f"
                    "\ufffd1\ufffd(\u007fj\ufffd<dots_function_response>H\u001f\ufffdG\ufffd\u007fj\ufffd"
                ),
            }
        )
        self._check_generation("text", None, "Briefly introduce yourself.", expected)

    def test_image_generation(self):
        expected = Expectations(
            {
                ("cpu", None): (
                    "\ufffd3\u0229(\u0006\ufffd<dots_function_response>H\u001f\ufffdG\ufffd\u007fj\ufffd<do"
                    "ts_function_response>H\u001f\ufffd1\ufffd\u0016H\u001f\ufffd1\ufffd\u0016"
                ),
                ("cuda", None): (
                    "\ufffd3\u0229\u0016\ufffd\ufffd\ufffd<no_think>\ufffd<dots_function_response>H\u001f"
                    "\ufffdG\ufffd\u007fj\ufffd<dots_function_response>H\u001f\ufffdG\ufffd\u007fj\ufffd<do"
                    "ts_function_response>H"
                ),
            }
        )
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        self._check_generation("image", url, "What is shown in this image?", expected)

    def test_audio_generation(self):
        expected = Expectations(
            {
                (None, None): (
                    "\ufffd3\u0229\u0016H\u001f\ufffd1\ufffd\u0016H\u001f\ufffd1\ufffd\u0016H\u001f\ufffd1"
                    "\ufffd\u0016H\u001f\ufffdH"
                ),
            }
        )
        url = "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/audio/mr_quiller.flac"
        self._check_generation("audio", url, "Transcribe this recording.", expected)

    def test_video_generation(self):
        expected = Expectations(
            {
                ("cpu", None): (
                    "\ufffd3\u0229\u0016\ufffd\ufffd\ufffd<no_think>\ufffd<dots_function_response>H\u001fW"
                    "\ufffd\ufffdd!\u0014\ufffdo\u001f\ufffd1\ufffd\u0016H\u001f\ufffd"
                ),
                ("cuda", None): (
                    "\ufffd3\u0229(\u007fj\ufffd<dots_function_response>H\u001f\ufffd1\ufffd\ufffd\ufffd"
                    "\ufffd<no_think>\ufffd<dots_function_response>H\u001f\ufffdG\ufffd\u007fj\ufffd<dots_f"
                    "unction_response>H"
                ),
            }
        )
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tiny_video.mp4"
        self._check_generation("video", url, "Describe this video.", expected)

    def _check_generation(self, modality, url, question, expected):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model, loading_info = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            experts_implementation="eager",
            output_loading_info=True,
        )
        model = model.to(torch_device).eval()
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            self.assertFalse(loading_info.get(key), loading_info)
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in model.parameters()))

        content = [] if url is None else [{"type": modality, "url": url_to_local_path(url)}]
        content.append({"type": "text", "text": question})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(torch_device, dtype=model.dtype)
        logits = model(**inputs, use_cache=False).logits
        self.assertTrue(torch.isfinite(logits).all())
        if modality != "text":
            media_key = {"image": "pixel_values", "audio": "input_features", "video": "pixel_values_videos"}[modality]
            # Random weights may generate the same tokens for different media, so also check the logits depend on it.
            ablated_inputs = {**inputs, media_key: torch.zeros_like(inputs[media_key])}
            ablated_logits = model(**ablated_inputs, use_cache=False).logits
            self.assertGreater((logits.float() - ablated_logits.float()).abs().max().item(), 0)

        generated = model.generate(**inputs, do_sample=False, max_new_tokens=32)
        decoded = processor.batch_decode(generated[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        self.assertEqual(decoded[0], expected.get_expectation())
