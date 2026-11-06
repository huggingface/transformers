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

from parameterized import parameterized

from transformers import (
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

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 7)
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

    def _build_modality_sub_configs(self):
        return {
            "vision_config": self.get_vision_config(),
            "audio_config": Dots3NoteAudio2TextModelTester(self.parent).get_audio_config(),
        }

    def create_pixel_values(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return torch.randn(batch_size, 4, self.num_channels * self.patch_size**2, device=torch_device)

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
        super().__init__(parent, **kwargs)
        self.encoder_seq_length = self.seq_length

    def get_text_config(self):
        return Dots3NoteTextModelTester(
            self.parent, num_attention_heads=self.num_attention_heads, num_key_value_heads=self.num_attention_heads
        ).get_config()

    def create_audio_mask(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        return torch.ones(batch_size, self.feat_seq_length, device=torch_device, dtype=torch.long)

    def _build_modality_sub_configs(self):
        return {
            "audio_config": self.get_audio_config(),
            "vision_config": Dots3NoteVision2TextModelTester(self.parent).get_vision_config(),
        }

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

    def _audio_features_prepare_config_and_inputs(self):
        self.skipTest("Audio features are covered by Dots3NoteAudio2TextModelTest")

    def test_get_image_features_attentions(self, modality="image"):
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
    model_id = "miraclezqc/TestForDots3"

    @parameterized.expand(
        [
            ("text", None, "Briefly introduce yourself.", [247, 103, 217, 187, 59, 295, 13, 246]),
            (
                "image",
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                "What is shown in this image?",
                [247, 103, 217, 187, 59, 92, 279, 219],
                [247, 103, 217, 187, 59, 295, 13, 246],
            ),
            (
                "audio",
                "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/audio/mr_quiller.flac",
                "Transcribe this recording.",
                [247, 103, 217, 187, 59, 295, 124, 304],
            ),
            (
                "video",
                "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tiny_video.mp4",
                "Describe this video.",
                [247, 103, 217, 187, 59, 295, 13, 246],
                [247, 103, 217, 187, 59, 92, 306, 158],
            ),
        ]
    )
    def test_generation(self, modality, url, question, expected, cuda_expected=None):
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
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=8)
        expected = Expectations({("cpu", None): expected, ("cuda", None): cuda_expected or expected}).get_expectation()
        self.assertEqual(generated[0, inputs.input_ids.shape[-1] :].tolist(), expected)
