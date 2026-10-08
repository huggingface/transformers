# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import shutil
import unittest

import numpy as np

from transformers import EmbeddingGemma2Processor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision
from transformers.video_utils import VideoMetadata

from ...test_processing_common import ProcessorTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_vision
class EmbeddingGemma2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = EmbeddingGemma2Processor
    videos_unstructured_max_length = 570
    videos_text_kwargs_max_length = 570
    videos_text_kwargs_override_max_length = 570

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token
        cls.audio_token = processor.audio_token

    @classmethod
    def _setup_video_processor(cls):
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        video_processor_kwargs = {
            "patch_size": 28,
            "max_soft_tokens": 70,
            "pooling_kernel_size": 3,
            "fps": 1,
        }
        return video_processor_class(**video_processor_kwargs)

    @classmethod
    def _setup_feature_extractor(cls):
        feature_extractor_class = cls._get_component_class_from_processor("feature_extractor")
        return feature_extractor_class()

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor_kwargs = {
            "patch_size": 28,
            "max_soft_tokens": 70,
            "pooling_kernel_size": 3,
        }
        return image_processor_class(**image_processor_kwargs)

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        extra_special_tokens = {
            "image_token": "<|image|>",
            "video_token": "<|video|>",
            "boi_token": "<start_of_image>",
            "eoi_token": "<end_of_image>",
            "audio_token": "<|audio|>",
            "boa_token": "<start_of_audio>",
            "eoa_token": "<end_of_audio>",
        }
        tokenizer = tokenizer_class.from_pretrained(
            SAMPLE_VOCAB, keep_accents=True, extra_special_tokens=extra_special_tokens
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    # TODO: remove `_CHAT_TEMPLATE` and set `model_id` once the checkpoint is available on the Hub.
    _CHAT_TEMPLATE = (
        "{%- for msg in messages if msg.get('role') == 'system' -%}"
        "{%- if msg.get('content') is string -%}"
        "{{ msg['content'] }}"
        "{%- else -%}"
        "{%- for item in msg['content'] if item.get('type') == 'text' -%}"
        "{{ item['text'] }}"
        "{%- endfor -%}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- for msg in messages if msg.get('role') != 'system' -%}"
        "{%- if msg.get('content') is string -%}"
        "{{ msg['content'] }}"
        "{%- else -%}"
        "{%- set existing_text = msg['content'] | selectattr('type', 'equalto', 'text') | map(attribute='text') | join -%}"
        "{%- set has_manual_placeholders = ('<|image|>' in existing_text) or ('<|video|>' in existing_text) or ('<|audio|>' in existing_text) -%}"
        "{%- for item in msg['content'] -%}"
        "{%- if item.get('type') == 'text' -%}"
        "{{ item['text'] }}"
        "{%- elif not has_manual_placeholders and item.get('type') == 'image' -%}"
        "<|image|>"
        "{%- elif not has_manual_placeholders and item.get('type') == 'video' -%}"
        "<|video|>"
        "{%- elif not has_manual_placeholders and item.get('type') == 'audio' -%}"
        "<|audio|>"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{%- endif -%}"
        "{%- endfor -%}"
    )

    @staticmethod
    def prepare_processor_dict():
        return {"image_seq_length": 3}

    # Override as EmbeddingGemma 2 needs images to be an explicitly nested batch
    def prepare_images_inputs(self, batch_size: int | None = None):
        """This function prepares a list of PIL images for testing"""
        images = super().prepare_images_inputs(batch_size)
        if isinstance(images, (list, tuple)):
            images = [[image] for image in images]
        return images

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        "Tests that the helper used internally in vLLM works correctly"

        processor = self.get_processor()
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        if not hasattr(processor, "_get_num_multimodal_tokens"):
            self.skipTest("Processor doesn't support `_get_num_multimodal_tokens` yet")

        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]

        # Overwritten because EmbeddingGemma 2 (like Gemma 3/4) needs nested image inputs
        image_inputs = []
        for h, w in image_sizes:
            image_inputs.append([np.random.randint(255, size=(h, w, 3), dtype=np.uint8)])

        text = [f"This is an image {getattr(self, 'image_token', '')}"] * len(image_inputs)
        inputs = processor(text=text, images=image_inputs, padding=True, return_tensors="pt")

        # EmbeddingGemma 2 does not return `mm_token_type_ids`, so count the image placeholders
        # directly in `input_ids`.
        num_image_tokens_from_call = (inputs.input_ids == processor.image_token_id).sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])

    def test_get_num_multimodal_tokens_matches_processor_call_audio(self):
        """Tests the audio branch of the helper used internally in vLLM.

        `_compute_audio_num_tokens` derives the count analytically from the mel-frame
        and SSCP subsampling arithmetic, while the processor derives it from the audio
        tower's mask. Those are two independent implementations of the same quantity,
        and `_compute_audio_num_tokens` hardcodes the subsampling stack (2 layers,
        kernel 3 / stride 2 / padding 1), so this guards them against drifting apart.
        """

        processor = self.get_processor()
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        if not hasattr(processor, "_get_num_multimodal_tokens"):
            self.skipTest("Processor doesn't support `_get_num_multimodal_tokens` yet")

        sampling_rate = processor.feature_extractor.sampling_rate
        # Sub-second through multi-second, so the mel/subsampling arithmetic is exercised
        # at several lengths rather than a single round number.
        audio_lengths = [sampling_rate // 4, sampling_rate // 2, sampling_rate, 2 * sampling_rate]
        audio_inputs = [np.zeros(length, dtype=np.float32) for length in audio_lengths]

        text = [f"This is audio {processor.audio_token}"] * len(audio_inputs)
        inputs = processor(text=text, audio=audio_inputs, padding=True, return_tensors="pt")

        # EmbeddingGemma 2 does not return `mm_token_type_ids`, so count the audio
        # placeholders directly in `input_ids`.
        num_audio_tokens_from_call = (inputs.input_ids == processor.audio_token_id).sum(-1).tolist()
        num_audio_tokens_from_helper = processor._get_num_multimodal_tokens(audio_lengths=audio_lengths)
        self.assertListEqual(num_audio_tokens_from_call, num_audio_tokens_from_helper["num_audio_tokens"])

    @require_torch
    def test_video_timestamps(self):
        """`add_timestamps=False` omits the timestamps and concatenates one block per frame."""
        processor = self.get_processor()
        video_inputs = [np.random.randint(0, 256, size=(2, 56, 56, 3), dtype=np.uint8)]
        text = f"{processor.video_token} What is this video?"

        out = processor(text=text, videos=[video_inputs], do_sample_frames=False, return_tensors="pt")
        decoded = processor.decode(out["input_ids"][0])
        self.assertNotIn("00:00", decoded)

        num_soft_tokens = processor.video_processor(
            video_inputs, overflow_strategy="uniform", max_frames=2, fps=None, return_tensors="pt"
        )["num_soft_tokens_per_video"][0]
        expected_frame = f"{processor.boi_token}{processor.video_token * num_soft_tokens}{processor.eoi_token}"
        expected_video_str = expected_frame * 2
        self.assertIn(expected_video_str, decoded)

        # Opting back in restores the `mm:ss` timestamps. The frame rate of an already-decoded array is
        # unknowable, so timestamps require real metadata: a 2 s clip at 1 FPS -> 00:00, 00:01.
        video_metadata = [VideoMetadata(fps=1.0, total_num_frames=2, duration=2.0)]
        out_with_ts = processor(
            text=text,
            videos=[video_inputs],
            do_sample_frames=True,
            video_metadata=video_metadata,
            videos_kwargs={"add_timestamps": True, "overflow_strategy": "uniform", "max_frames": 2, "fps": None},
            return_tensors="pt",
        )
        decoded_with_ts = processor.decode(out_with_ts["input_ids"][0])
        self.assertIn("00:00", decoded_with_ts)

    @require_torch
    def test_video_timestamps_require_metadata(self):
        """Timestamps are prompt content, so a missing `fps` must fail loudly rather than be guessed."""
        processor = self.get_processor()
        video_inputs = [np.random.randint(0, 256, size=(2, 56, 56, 3), dtype=np.uint8)]
        text = f"{processor.video_token} What is this video?"

        with self.assertRaises(ValueError):
            processor(
                text=text,
                videos=[video_inputs],
                videos_kwargs={"add_timestamps": True, "fps": None},
                return_tensors="pt",
            )

    def test_single_modality_inputs_need_no_text(self):
        """Unlike Gemma 4, any single modality on its own is a valid embedding input."""
        processor = self.get_processor()

        image_input = self.prepare_images_inputs(batch_size=2)
        out = processor(images=image_input, return_tensors="np")
        self.assertIn("input_ids", out)
        self.assertIn(self.images_input_name, out)

        with self.assertRaises(ValueError):
            processor()

    @require_torch
    def test_multimodal_and_nested_inputs_without_text(self):
        """Nested per-sample lists and combined modalities with `text=None` synthesize matching placeholders per row."""
        processor = self.get_processor()
        img1 = np.random.randint(0, 256, size=(56, 56, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, size=(56, 56, 3), dtype=np.uint8)
        img3 = np.random.randint(0, 256, size=(56, 56, 3), dtype=np.uint8)
        vid1 = np.random.randint(0, 256, size=(2, 56, 56, 3), dtype=np.uint8)
        vid2 = np.random.randint(0, 256, size=(3, 56, 56, 3), dtype=np.uint8)
        vid3 = np.random.randint(0, 256, size=(2, 56, 56, 3), dtype=np.uint8)
        aud1 = np.zeros(1600, dtype=np.float32)
        aud2 = np.zeros(3200, dtype=np.float32)
        aud3 = np.zeros(1600, dtype=np.float32)

        # 1. Nested audio list: batch size 2, sample 0 has 2 audios, sample 1 has 1 audio
        out_aud = processor(audio=[[aud1, aud2], [aud3]], return_tensors="pt")
        self.assertEqual(out_aud["input_ids"].shape[0], 2)
        self.assertGreater(
            (out_aud["input_ids"][0] == processor.audio_token_id).sum().item(),
            (out_aud["input_ids"][1] == processor.audio_token_id).sum().item(),
        )

        # 2. Nested video list: batch size 2, sample 0 has 2 videos, sample 1 has 1 video
        out_vid = processor(videos=[[vid1, vid2], [vid3]], do_sample_frames=False, return_tensors="pt")
        self.assertEqual(out_vid["input_ids"].shape[0], 2)
        self.assertGreater(
            (out_vid["input_ids"][0] == processor.video_token_id).sum().item(),
            (out_vid["input_ids"][1] == processor.video_token_id).sum().item(),
        )

        # 3. Combined image + audio batch with text=None
        out_mixed = processor(
            images=[[img1, img2], [img3]],
            audio=[[aud1], [aud2, aud3]],
            return_tensors="pt",
        )
        self.assertEqual(out_mixed["input_ids"].shape[0], 2)
        self.assertGreater((out_mixed["input_ids"][0] == processor.image_token_id).sum().item(), 0)
        self.assertGreater((out_mixed["input_ids"][0] == processor.audio_token_id).sum().item(), 0)
        self.assertGreater((out_mixed["input_ids"][1] == processor.image_token_id).sum().item(), 0)
        self.assertGreater((out_mixed["input_ids"][1] == processor.audio_token_id).sum().item(), 0)

        # 4. Mismatched outer batch sizes across modalities with text=None raises ValueError
        with self.assertRaisesRegex(ValueError, "inconsistently sized modality batches"):
            processor(images=[[img1], [img2]], audio=[aud1], return_tensors="pt")

        # 5. Nested audio and video lists also work when explicit text placeholders are provided
        out_aud_with_text = processor(
            text=["<|audio|> <|audio|>", "<|audio|>"], audio=[[aud1, aud2], [aud3]], return_tensors="pt"
        )
        self.assertTrue((out_aud["input_ids"] == out_aud_with_text["input_ids"]).all())

        out_vid_with_text = processor(
            text=["<|video|> <|video|>", "<|video|>"],
            videos=[[vid1, vid2], [vid3]],
            do_sample_frames=False,
            return_tensors="pt",
        )
        self.assertTrue((out_vid["input_ids"] == out_vid_with_text["input_ids"]).all())

    @require_torch
    def test_video_token_count_matches_frames(self):
        """Ragged batch: every row expands to its own frame count.

        `pixel_values_videos` is a flat frame sequence, so indexing it by video index would silently
        yield the wrong number of placeholders instead of raising.
        """
        processor = self.get_processor()
        videos = [
            [np.random.randint(0, 256, size=(2, 56, 56, 3), dtype=np.uint8)],
            [np.random.randint(0, 256, size=(5, 56, 56, 3), dtype=np.uint8)],
        ]
        text = [f"{processor.video_token} What is this video?"] * 2

        out = processor(text=text, videos=videos, do_sample_frames=False, padding=True, return_tensors="pt")

        video_inputs = processor.video_processor(
            [video[0] for video in videos], do_sample_frames=False, return_tensors="pt"
        )
        self.assertEqual([int(n) for n in video_inputs["num_frames_per_video"]], [2, 5])

        video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        counts = [
            int(num_frames) * int(num_soft_tokens)
            for num_frames, num_soft_tokens in zip(
                video_inputs["num_frames_per_video"], video_inputs["num_soft_tokens_per_video"]
            )
        ]
        self.assertNotEqual(counts[0], counts[1], "the two rows must differ for this test to be meaningful")
        for row, expected in enumerate(counts):
            self.assertEqual((out["input_ids"][row] == video_token_id).sum().item(), expected)

    @require_torch
    def test_validate_inputs_multimodal_placeholder_counts(self):
        """Mismatched or orphan `<|image|>`, `<|video|>`, and `<|audio|>` placeholders raise `ValueError`."""
        processor = self.get_processor()
        img = np.random.randint(0, 256, size=(56, 56, 3), dtype=np.uint8)
        vid = [np.random.randint(0, 256, size=(2, 56, 56, 3), dtype=np.uint8)]
        aud = np.zeros(1600, dtype=np.float32)

        # Too few or too many placeholders per modality
        with self.assertRaisesRegex(ValueError, "image"):
            processor(text=["one <|image|>"], images=[[img, img]], return_tensors="pt")
        with self.assertRaisesRegex(ValueError, "image"):
            processor(text=["two <|image|> <|image|>"], images=[[img]], return_tensors="pt")

        with self.assertRaisesRegex(ValueError, "video"):
            processor(text=["one <|video|>"], videos=[vid, vid], do_sample_frames=False, return_tensors="pt")
        with self.assertRaisesRegex(ValueError, "video"):
            processor(text=["two <|video|> <|video|>"], videos=[vid], do_sample_frames=False, return_tensors="pt")

        with self.assertRaisesRegex(ValueError, "audio"):
            processor(text=["one <|audio|>"], audio=[aud, aud], return_tensors="pt")
        with self.assertRaisesRegex(ValueError, "audio"):
            processor(text=["two <|audio|> <|audio|>"], audio=[aud], return_tensors="pt")

        # Orphan placeholders when the corresponding modality input is None
        for token, match in (("<|image|>", "image"), ("<|video|>", "video"), ("<|audio|>", "audio")):
            with self.assertRaisesRegex(ValueError, match):
                processor(text=[f"orphan {token}"], return_tensors="pt")

    @require_torch
    def test_chat_template_ordering_and_manual_placeholders(self):
        """System prompts precede media, content order is preserved, and manual markers disable auto-insertion."""
        processor = self.get_processor()
        processor.chat_template = self._CHAT_TEMPLATE
        img = np.random.randint(0, 256, size=(56, 56, 3), dtype=np.uint8)
        aud = np.zeros(1600, dtype=np.float32)

        # 1. System prompt goes first, then caller-supplied content order is preserved
        msg_img_text = [
            {"role": "system", "content": "title: none | text: "},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "a cat"}]},
        ]
        msg_text_img = [
            {"role": "system", "content": "title: none | text: "},
            {"role": "user", "content": [{"type": "text", "text": "a cat"}, {"type": "image"}]},
        ]
        self.assertEqual(
            processor.apply_chat_template(msg_img_text, tokenize=False), "title: none | text: <|image|>a cat"
        )
        self.assertEqual(
            processor.apply_chat_template(msg_text_img, tokenize=False), "title: none | text: a cat<|image|>"
        )

        # 2. Mixed and multiple same-type modalities follow content order
        msg_multi = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare "},
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "audio"},
                ],
            }
        ]
        self.assertEqual(
            processor.apply_chat_template(msg_multi, tokenize=False), "compare <|image|><|image|><|audio|>"
        )

        # 3. Manual placeholders in text suppress automatic placeholder insertion (all-or-nothing)
        msg_manual = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "interleaved <|image|> and <|audio|> markers"},
                    {"type": "image"},
                    {"type": "audio"},
                ],
            }
        ]
        rendered_manual = processor.apply_chat_template(msg_manual, tokenize=False)
        self.assertEqual(rendered_manual, "interleaved <|image|> and <|audio|> markers")
        out_manual = processor(text=[rendered_manual], images=[[img]], audio=[aud], return_tensors="pt")
        self.assertIn("input_ids", out_manual)

        # Partial manual placeholders (manual <|image|> present, <|audio|> omitted) suppress auto <|audio|>
        # and fail count validation in the processor.
        msg_partial = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "manual <|image|> but forgot audio placeholder"},
                    {"type": "image"},
                    {"type": "audio"},
                ],
            }
        ]
        rendered_partial = processor.apply_chat_template(msg_partial, tokenize=False)
        self.assertNotIn("<|audio|>", rendered_partial)
        with self.assertRaisesRegex(ValueError, "audio"):
            processor(text=[rendered_partial], images=[[img]], audio=[aud], return_tensors="pt")
