# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Lfm3AudioForConditionalGeneration,
    Lfm3OmniConfig,
    Lfm3OmniForConditionalGeneration,
    Lfm3OmniModel,
    is_torch_available,
)
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor
from ..lfm3_audio import test_modeling_lfm3_audio as lfm3_audio
from ..lfm3_audio.test_modeling_lfm3_audio import Lfm3AudioModelTester


if is_torch_available():
    import torch


# the LFM3-Audio behaviour tests build the LFM3-Audio classes, which the LFM3-Audio test runs
AUDIO_CLASS_TESTS = (
    "test_model_without_audio_encoder_or_depth_decoder",
    "test_audio_features_ignore_padding",
    "test_streaming_encoder_matches_offline",
    "test_windowed_repetition_penalty",
    "test_generate_speech_with_depth_decoder_settings",
    "test_generate_interleaved_mode_alternates_blocks",
    "test_cached_generation_matches_full_prompt_in_the_speech_modes",
)


class Lfm3OmniModelTester(Lfm3AudioModelTester):
    config_class = Lfm3OmniConfig
    base_model_class = Lfm3OmniModel
    conditional_generation_class = Lfm3OmniForConditionalGeneration
    # the common inputs hold one image per row; off for the test that doubles the prompt and the clips only
    with_images = True

    small_config_overrides = {
        **Lfm3AudioModelTester.small_config_overrides,
        # inside the tester's vocabulary, next to the speech tokens
        "image_token_id": 9,
        "video_token_id": 3,
        # a 2x2 patch grid per image, one language token after the 2x2 merge; head_dim 16: the FA4 kernels need 16
        "vision_num_hidden_layers": 2,
        "vision_num_attention_heads": 2,
        "vision_num_key_value_heads": 1,
        "vision_layer_types": ["conv", "full_attention"],
        "vision_patch_size": 2,
        "vision_temporal_patch_size": 2,
        "vision_spatial_merge_size": 2,
        "projector_hidden_size": 32,
    }

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.image_token_id, self.video_token_id}

    def create_patches(self, grid_thw):
        patch_dim = 3 * self.vision_temporal_patch_size * self.vision_patch_size**2
        return floats_tensor([int(grid_thw.prod(-1).sum()), patch_dim])

    def _prepare_modality_inputs(self, input_ids, config, batch_size: int | None = None):
        input_ids, modality_inputs = super()._prepare_modality_inputs(input_ids, config, batch_size)
        if self.with_images:
            batch_size = batch_size if batch_size is not None else self.batch_size
            input_ids = input_ids.clone()
            input_ids[:, 0] = self.image_token_id
            image_grid_thw = torch.tensor([[1, 2, 2]] * batch_size, device=torch_device)
            modality_inputs.update(pixel_values=self.create_patches(image_grid_thw), image_grid_thw=image_grid_thw)
        return input_ids, modality_inputs

    def prepare_config_and_inputs_for_media(self):
        """The speech prompt behind an image (one token) and before a video of two temporal patches (two tokens)."""
        config, inputs_dict = self.prepare_config_and_inputs_for_speech()
        input_ids = inputs_dict["input_ids"]
        image = input_ids.new_full((1, 1), self.image_token_id)
        video = input_ids.new_full((1, 2), self.video_token_id)
        inputs_dict["input_ids"] = torch.cat([input_ids[:, :1], image, input_ids[:, 1:], video], dim=1)
        image_grid_thw = torch.tensor([[1, 2, 2]], device=torch_device)
        video_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)
        inputs_dict.update(
            pixel_values=self.create_patches(image_grid_thw),
            image_grid_thw=image_grid_thw,
            pixel_values_videos=self.create_patches(video_grid_thw),
            video_grid_thw=video_grid_thw,
        )
        return config, inputs_dict


@require_torch
class Lfm3OmniModelTest(lfm3_audio.Lfm3AudioModelTest):
    model_tester_class = Lfm3OmniModelTester

    def setUp(self):
        super().setUp()
        if self._testMethodName in AUDIO_CLASS_TESTS:
            self.skipTest("builds the LFM3-Audio classes: run by the LFM3-Audio test")

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # the parent builds the speech targets for its own head only
        if model_class is Lfm3OmniForConditionalGeneration:
            model_class = Lfm3AudioForConditionalGeneration
        return super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

    def test_mismatching_num_audio_tokens(self):
        # its last case doubles the prompt and the clips, which the images can not follow
        with patch.object(self.model_tester, "with_images", False):
            super().test_mismatching_num_audio_tokens()

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.vision_layer_types.count("full_attention")

    def _image_features_get_expected_num_hidden_states(self, model_tester=None):
        return self.model_tester.vision_num_hidden_layers + 1

    def _video_features_get_expected_num_attentions(self, model_tester=None):
        return self._image_features_get_expected_num_attentions()

    def _video_features_get_expected_num_hidden_states(self, model_tester=None):
        return self._image_features_get_expected_num_hidden_states()

    def test_media_match_explicit_embedding_assembly(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_media()
        model = Lfm3OmniForConditionalGeneration(config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]

        with torch.no_grad():
            image = model.get_image_features(inputs_dict["pixel_values"], inputs_dict["image_grid_thw"])
            video = model.get_video_features(inputs_dict["pixel_values_videos"], inputs_dict["video_grid_thw"])
            audio = model.get_audio_features(inputs_dict["input_features"], inputs_dict["input_features_mask"])
            features = {
                config.image_token_id: torch.cat(image.pooler_output),
                config.video_token_id: torch.cat(video.pooler_output),
                config.audio_token_id: audio.pooler_output,
                config.audio_out_token_id: model.model.get_audio_input_embeds(inputs_dict["audio_input_ids"]),
            }
            embeds = model.get_input_embeddings()(input_ids)
            for token_id, tokens in features.items():
                embeds[input_ids == token_id] = tokens
            expected = model.lm_head(model.model.language_model(inputs_embeds=embeds).last_hidden_state)
            actual = model(**inputs_dict).logits

        torch.testing.assert_close(actual, expected)

    def test_generate_encodes_the_media_once_and_matches_cache_free(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_media()
        model = Lfm3OmniForConditionalGeneration(config).to(torch_device).eval()
        model.generation_config.eos_token_id = None
        # text only: no mode switch, and the placeholders are never written as text
        suppress = [
            config.audio_start_token_id,
            config.mixed_start_token_id,
            config.text_end_token_id,
            config.audio_token_id,
            config.audio_out_token_id,
            config.image_token_id,
            config.video_token_id,
        ]

        vision_tower = model.model.vision_tower
        with torch.no_grad(), patch.object(vision_tower, "forward", wraps=vision_tower.forward) as encode:
            outputs = model.generate(**inputs_dict, max_new_tokens=4, suppress_tokens=suppress)
            # the image and the video, before the prefill; the cache replays them
            self.assertEqual(encode.call_count, 2)

            input_ids = inputs_dict["input_ids"]
            for _ in range(4):
                logits = model(**{**inputs_dict, "input_ids": input_ids}, use_cache=False).logits[:, -1]
                logits[:, suppress] = -float("inf")
                input_ids = torch.cat((input_ids, logits.argmax(-1, keepdim=True)), dim=-1)

        torch.testing.assert_close(outputs, input_ids)

    def test_training_reaches_every_component(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_media()
        model = Lfm3OmniForConditionalGeneration(config).to(torch_device).train()
        input_ids = inputs_dict["input_ids"]
        labels = input_ids.clone()
        for token_id in (
            config.image_token_id,
            config.video_token_id,
            config.audio_token_id,
            config.audio_out_token_id,
        ):
            labels[labels == token_id] = -100

        outputs = model(**inputs_dict, labels=labels, audio_labels=inputs_dict["audio_input_ids"])
        self.assertTrue(torch.isfinite(outputs.text_loss) and torch.isfinite(outputs.audio_loss))
        outputs.loss.backward()

        for name in (
            "model.language_model.layers.0.conv.in_proj.weight",
            "model.audio_model.subsampling.conv_in.weight",
            "model.multi_modal_projector.linear_1.weight",
            "model.vision_tower.patch_embed.weight",
            "model.audio_projector.linear_1.weight",
            "model.embed_audio_tokens.embed_audio_tokens.weight",
            "depth_decoder.model.backbone_projector.weight",
        ):
            gradient = model.get_parameter(name).grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

    def test_speech_modes_with_media(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_media()
        config.interleaved_n_text, config.interleaved_n_audio = 2, 3
        model = Lfm3OmniForConditionalGeneration(config).to(torch_device).eval()
        prompt = inputs_dict.pop("input_ids")

        # speech: frames only; interleaved: 2 text tokens then 3 frames, and again
        for marker, is_frame in (
            (config.audio_start_token_id, [True] * 8),
            (config.mixed_start_token_id, [False, False, True, True, True, False, False, True]),
        ):
            with self.subTest(marker=marker):
                input_ids = torch.cat([prompt, prompt.new_full((1, 1), marker)], dim=1)
                outputs = model.generate(
                    input_ids,
                    **inputs_dict,
                    max_new_tokens=8,
                    logits_processor=lfm3_audio.force_tokens(*([10] * 8)),
                    return_dict_in_generate=True,
                    **lfm3_audio.only_code(config, 1),
                )
                new_tokens = outputs.sequences[0, input_ids.shape[1] :]
                self.assertEqual((new_tokens == config.audio_out_token_id).tolist(), is_frame)
                self.assertEqual(outputs.audio_sequences.tolist(), [[[1] * sum(is_frame)] * config.num_codebooks])
                self.assertTrue(outputs.audio_sequences_mask.all())


@unittest.skip(reason="Update after release, the Hub checkpoint still points at remote code")
@require_torch
class Lfm3OmniIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """Goldens: token ids and speech codes of the released checkpoint, speech greedy (`depth_decoder_do_sample=False`)
    with a seeded MaskGIT reveal order, in float32."""

    checkpoint = "LiquidAI/LFM3-Omni-2B"
    image = "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/coco_sample.png"
    samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent.parent.parent / "fixtures/lfm3_omni/expected_results.json"
        with open(path, encoding="utf-8") as handle:
            cls.expected = json.load(handle)

    def load(self, dtype=None):
        model = AutoModelForMultimodalLM.from_pretrained(self.checkpoint, dtype=dtype, device_map=torch_device)
        processor = AutoProcessor.from_pretrained(self.checkpoint, device_map=torch_device)
        return model, processor

    @slow
    def test_image_description(self):
        model, processor = self.load()
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=32)
        generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1] :]

        self.assertEqual(generated_ids.tolist(), self.expected["image_description"]["generated_ids"])
        self.assertEqual(
            processor.decode(generated_ids, skip_special_tokens=True), self.expected["image_description"]["text"]
        )

    @slow
    def test_image_and_audio_question(self):
        model, processor = self.load()
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image},
                    {"type": "audio", "url": self.samples + "librispeech_mr_quilter.wav"},
                    {"type": "text", "text": "What is in the image, and what is said in the audio? Answer briefly."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, max_new_tokens=32)
        generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1] :]

        self.assertEqual(generated_ids.tolist(), self.expected["image_and_audio_question"]["generated_ids"])

    @slow
    def test_speech_about_an_image(self):
        model, processor = self.load(dtype=torch.float32)
        conversation = [
            {"role": "system", "content": "Answer with speech, in American English using a female voice."},
            {
                "role": "user",
                "content": [{"type": "image", "url": self.image}, {"type": "text", "text": "What do you see?"}],
            },
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            generation_mode="audio",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        torch.manual_seed(0)
        outputs = model.generate(**inputs, max_new_tokens=256, depth_decoder_do_sample=False)
        audio = processor.decode(outputs.audio_sequences, outputs.audio_sequences_mask)

        expected_codes = torch.tensor(self.expected["speech_about_an_image"]["audio_codes"])
        num_frames = expected_codes.shape[-1]
        self.assertEqual(outputs.audio_sequences.shape[-1], num_frames)
        torch.testing.assert_close(outputs.audio_sequences[0].cpu(), expected_codes)
        # end-of-audio frame not decoded; 1920 samples per frame
        self.assertEqual(outputs.audio_sequences_mask[0].sum().item(), num_frames - 1)
        self.assertEqual([waveform.shape[-1] for waveform in audio], [(num_frames - 1) * 1920])
