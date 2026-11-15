# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch AudioVisualFlamingo model."""

import unittest

from transformers import (
    AudioVisualFlamingoConfig,
    AudioVisualFlamingoForConditionalGeneration,
    AudioVisualFlamingoModel,
    AutoProcessor,
    is_torch_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...alm_tester import ALMModelTest, ALMModelTester
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers.models.audiovisualflamingo.modeling_audiovisualflamingo import temporal_pool


# Tiny sub-configs shared by the vision and audio testers. The vision tower sees 8x8 tiles of 2x2 patches, merged into
# 4 tokens, and the audio encoder windows hold 16 mel frames, encoded into 4 tokens.
TINY_CONFIG_KWARGS = {
    "image_token_id": 3,
    "video_token_id": 4,
    "audio_token_id": 5,
    "image_size": 8,
    "patch_size": 2,
    "s2_scales": [8, 16, 24],
    "num_mel_bins": 8,
    "max_source_positions": 8,
    "video_temporal_pool_size": 2,
    "time_embedding_dim": 8,
}


class AudioVisualFlamingoVisionTextModelTester(VLMModelTester):
    base_model_class = AudioVisualFlamingoModel
    config_class = AudioVisualFlamingoConfig
    conditional_generation_class = AudioVisualFlamingoForConditionalGeneration

    def __init__(self, parent, **kwargs):
        for key, value in TINY_CONFIG_KWARGS.items():
            kwargs.setdefault(key, value)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("num_frames", 2)
        kwargs.setdefault("seq_length", 16)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.video_token_id, self.audio_token_id}

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size=None):
        # Each sample also holds a video, whose frames are pooled into a single group of tokens
        batch_size = self.batch_size if batch_size is None else batch_size
        input_ids[:, self.num_image_tokens : 2 * self.num_image_tokens] = config.video_token_id
        pixel_values_videos = modality_inputs[:, None].repeat(1, self.num_frames, 1, 1, 1)
        return {
            "image_grid_hw": torch.ones(batch_size, 2, dtype=torch.long, device=torch_device),
            "pixel_values_videos": pixel_values_videos,
            "video_timestamps": torch.arange(self.num_frames, dtype=torch.float64, device=torch_device).repeat(
                batch_size, 1
            ),
        }


class AudioVisualFlamingoAudioTextModelTester(ALMModelTester):
    base_model_class = AudioVisualFlamingoModel
    config_class = AudioVisualFlamingoConfig
    conditional_generation_class = AudioVisualFlamingoForConditionalGeneration
    audio_mask_key = "input_features_mask"

    def __init__(self, parent, **kwargs):
        for key, value in TINY_CONFIG_KWARGS.items():
            kwargs.setdefault(key, value)
        kwargs.setdefault("feat_seq_length", 32)
        super().__init__(parent, **kwargs)

    def get_audio_embeds_mask(self, audio_mask):
        # One audio token every 4 mel frames
        return audio_mask[:, ::4]


@require_torch
class AudioVisualFlamingoVisionTextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = AudioVisualFlamingoVisionTextModelTester
    test_all_params_have_gradient = False  # the audio tower is not used by this tester
    # Video frames are encoded independently by the vision tower
    skip_test_video_features_output_shape = True

    def _audio_features_prepare_config_and_inputs(self):
        self.skipTest("Audio features are covered by AudioVisualFlamingoAudioTextModelTest")

    def test_get_image_features_s2_tiles(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = AudioVisualFlamingoModel(config).to(torch_device).eval()
        # A tiled 3x4 image (1 + 2x2 + 3x4 tiles) and a single-tile image
        pixel_values = torch.randn(1 + 4 + 12 + 1, 3, 8, 8, device=torch_device)
        image_grid_hw = torch.tensor([[3, 4], [1, 1]], device=torch_device)
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values, image_grid_hw).pooler_output

        # 4 tokens per tile of the last scale
        self.assertEqual([features.shape[0] for features in image_features], [48, 4])

    def test_interleaved_video_and_audio(self):
        config = self.model_tester.get_config()
        model = AudioVisualFlamingoModel(config).to(torch_device).eval()
        # A video of 4 frames, pooled into 2 groups of 4 tokens, and two 16 frame audio windows of 4 tokens each
        video = {
            "pixel_values_videos": torch.randn(1, 4, 3, 8, 8, device=torch_device),
            "video_timestamps": [[0.0, 1.0, 2.0, 3.0]],
        }
        audio = {
            "input_features": torch.randn(1, 8, 32, device=torch_device),
            "input_features_mask": torch.ones(1, 32, dtype=torch.bool, device=torch_device),
        }
        # Each segment holds the tokens of a group of frames, followed by the tokens of its audio
        text, video_id, audio_id = self.model_tester._safe_token_id(), config.video_token_id, config.audio_token_id
        segment = [video_id] * 4 + [audio_id] * 4
        input_ids = torch.tensor([[text] + segment + segment + [text]], device=torch_device)

        inputs_embeds = {}
        model.language_model.register_forward_pre_hook(
            lambda module, args, kwargs: inputs_embeds.update(kwargs), with_kwargs=True
        )
        with torch.no_grad():
            model(input_ids=input_ids, **video, **audio)
            video_features = model.get_video_features(**video).pooler_output[0]
            shifted_video_features = model.get_video_features(video["pixel_values_videos"], [[10.0, 11.0, 12.0, 13.0]])
            audio_features = model.get_audio_features(**audio).pooler_output[0]

        embeds = inputs_embeds["inputs_embeds"][0]
        torch.testing.assert_close(embeds[input_ids[0] == video_id], video_features)
        torch.testing.assert_close(embeds[input_ids[0] == audio_id], audio_features)
        # Video features depend on the frame times, through their time embedding
        self.assertFalse(torch.allclose(shifted_video_features.pooler_output[0], video_features))

        # Raw inputs and pre-computed features of the same modality can't be both passed
        with self.assertRaisesRegex(ValueError, "both the raw video inputs and their mm_encoder_outputs"):
            model(input_ids=input_ids, **video, **audio, mm_encoder_outputs={"video": shifted_video_features})

    def test_temporal_pool_incomplete_group(self):
        frames = torch.arange(12, dtype=torch.float).reshape(1, 3, 2, 2)
        pooled = temporal_pool(frames, pool_size=2)

        # The last group is padded with the mean value of its frames
        expected = torch.stack([frames[0, :2].mean(0), (frames[0, 2] + frames[0, 2].mean()) / 2])[None]
        torch.testing.assert_close(pooled, expected)


@require_torch
class AudioVisualFlamingoAudioTextModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = AudioVisualFlamingoAudioTextModelTester
    test_all_params_have_gradient = False  # the vision tower is not used by this tester
    # Audio is encoded in 30 second windows
    skip_test_audio_features_output_shape = True

    @unittest.skip("The tester has no image or video in its inputs")
    def test_generate_from_multimodal_encoder_outputs_and_raw_data(self):
        pass

    @unittest.skip("The first checkpointed layers belong to the vision tower, covered by the vision tester")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip("The first checkpointed layers belong to the vision tower, covered by the vision tester")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip("The first checkpointed layers belong to the vision tower, covered by the vision tester")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    def _image_features_prepare_config_and_inputs(self):
        self.skipTest("Image features are covered by AudioVisualFlamingoVisionTextModelTest")

    def test_generate_repeats_all_audio_of_a_prompt(self):
        config = self.model_tester.get_config()
        model = AudioVisualFlamingoForConditionalGeneration(config).to(torch_device).eval()
        # A single prompt holding two audio of different lengths, of 1 and 2 windows of 4 tokens
        input_features_mask = torch.ones(2, 32, dtype=torch.bool, device=torch_device)
        input_features_mask[0, 16:] = False
        text, audio_id = self.model_tester._safe_token_id(), config.audio_token_id
        inputs = {
            "input_ids": torch.tensor([[text] + [audio_id] * 4 + [text] + [audio_id] * 8], device=torch_device),
            "input_features": torch.randn(2, 8, 32, device=torch_device),
            "input_features_mask": input_features_mask,
        }

        generation_kwargs = {"max_new_tokens": 3, "return_dict_in_generate": True, "output_logits": True}
        greedy = model.generate(**inputs, do_sample=False, **generation_kwargs)
        sampled = model.generate(**inputs, do_sample=True, top_k=1, num_return_sequences=2, **generation_kwargs)

        # Each returned sequence gets all the audio of the prompt, in order
        for greedy_logits, sampled_logits in zip(greedy.logits, sampled.logits):
            torch.testing.assert_close(sampled_logits, greedy_logits.repeat(2, 1))

    def test_get_audio_features_return_tuple(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = AudioVisualFlamingoModel(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model.get_audio_features(inputs["input_features"], inputs["input_features_mask"])
            outputs_tuple = model.get_audio_features(
                inputs["input_features"], inputs["input_features_mask"], return_dict=False
            )

        torch.testing.assert_close(torch.cat(outputs_tuple[1]), torch.cat(outputs.pooler_output))

    def _video_features_prepare_config_and_inputs(self):
        self.skipTest("Video features are covered by AudioVisualFlamingoVisionTextModelTest")


@slow
@require_torch
class AudioVisualFlamingoIntegrationTest(unittest.TestCase):
    checkpoint = "nvidia/nemotron-labs-audio-visual-flamingo-hf"

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)
        cls.model = AudioVisualFlamingoForConditionalGeneration.from_pretrained(
            cls.checkpoint, device_map=torch_device, dtype=torch.bfloat16
        )

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def generate(self, conversation, max_new_tokens=32, **kwargs):
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            load_audio_from_video=True,
            **kwargs,
        ).to(torch_device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def test_image(self):
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image", "url": url}, {"type": "text", "text": "Describe this image in detail."}],
            }
        ]
        EXPECTED_OUTPUT = ["Two cats are sleeping on a pink blanket, with a remote control nearby."]
        self.assertEqual(self.generate(conversation), EXPECTED_OUTPUT)

    def test_audio(self):
        url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dogs_barking_in_sync_with_the_music.wav"
        conversation = [
            {
                "role": "user",
                "content": [{"type": "audio", "url": url}, {"type": "text", "text": "Describe the audio in detail."}],
            }
        ]
        EXPECTED_OUTPUT = ["The music is described as having a fast tempo and a strong beat."]
        self.assertEqual(self.generate(conversation), EXPECTED_OUTPUT)

    def test_video_with_audio(self):
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": url},
                    {"type": "text", "text": "Describe the video and its audio."},
                ],
            }
        ]
        EXPECTED_OUTPUT = [
            "The video takes place in an indoor tennis court with a high ceiling and bright lighting. A player dressed "
            "in a white shirt, black shorts, and white shoes is"
        ]
        self.assertEqual(self.generate(conversation), EXPECTED_OUTPUT)

    def test_batched_videos_with_audio(self):
        urls = [
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
            "https://huggingface.co/datasets/hf-internal-testing/test-videos/resolve/main/sample_demo_1_320x240.mp4",
        ]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "url": url},
                        {"type": "text", "text": "Describe the video and its audio."},
                    ],
                }
            ]
            for url in urls
        ]
        # Same outputs as each video alone
        EXPECTED_OUTPUT = [
            "The video opens with a wide shot of an indoor tennis court, featuring a player in a white shirt and black "
            "shorts positioned on the left side of the court,",
            "The video features a young child with blonde hair and glasses, dressed in a light green sleeveless shirt "
            "and pink pants, sitting on a bed and reading a book",
        ]
        self.assertEqual(self.generate(conversations, processor_kwargs={"num_frames": 32}), EXPECTED_OUTPUT)

    def test_batched_image_and_text(self):
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": url},
                        {"type": "text", "text": "Describe this image in detail."},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the capital of France? Answer in one sentence."}],
                }
            ],
        ]
        EXPECTED_OUTPUT = [
            "Two cats are sleeping on a pink blanket, with a remote control nearby.",
            "The capital of France is Paris.",
        ]
        self.assertEqual(self.generate(conversations), EXPECTED_OUTPUT)
