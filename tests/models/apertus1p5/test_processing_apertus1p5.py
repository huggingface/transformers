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
"""Testing suite for the Apertus 1.5 processor."""

import os
import re
import tempfile
import unittest
import wave

import numpy as np

from transformers import Apertus1p5Config, Apertus1p5Processor, AutoProcessor, is_torch_available
from transformers.testing_utils import require_librosa, require_torch, require_torchvision, slow

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


APERTUS1P5_EXTRA_SPECIAL_TOKENS = {
    "image_token": "<|image|>",
    "audio_token": "<|audio|>",
    "boi_token": "<|img_start|>",
    "eoi_token": "<|img_end|>",
    "image_wrapper_token": "<|img_token_start|>",
    "eol_token": "<|img_end_of_row|>",
    "boa_token": "<|audio_start|>",
    "eoa_token": "<|audio_end|>",
}


@require_torchvision
class Apertus1p5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Apertus1p5Processor

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        # small pixel budget so test images stay tiny; grids stay >= 2x2
        return image_processor_class(min_pixels=32 * 32, max_pixels=64 * 64)

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        tokenizer = tokenizer_class.from_pretrained(
            "openai-community/gpt2", extra_special_tokens=APERTUS1P5_EXTRA_SPECIAL_TOKENS
        )
        tokenizer.pad_token_id = 0
        tokenizer.sep_token_id = 1
        return tokenizer

    @staticmethod
    def prepare_processor_dict():
        # a simple list-of-content-blocks chat template; the real checkpoint ships its own
        return {
            "chat_template": "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['role'].upper() + ': '}}{% endif %}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<|image|>' }}{% endfor %}{% for content in message['content'] | selectattr('type', 'equalto', 'audio') %}{{ '<|audio|>' }}{% endfor %}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] + ' '}}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ content['text'] + ' '}}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}",
        }  # fmt: skip

    @staticmethod
    def _image(height, width):
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    @staticmethod
    def _clip(num_samples):
        return np.random.randn(num_samples).astype(np.float32)

    def test_image_expansion_matches_reference_layout(self):
        """Image layouts use height-first headers and separators between rows, preserving surrounding text."""
        processor = self.get_processor()
        cases = [
            (
                (32, 32),
                "<|img_start|>2*2<|img_token_start|>"
                "<|image|><|image|><|img_end_of_row|><|image|><|image|>"
                "<|img_end|> describe",
            ),
            (
                (32, 64),
                "<|img_start|>2*4<|img_token_start|>"
                "<|image|><|image|><|image|><|image|><|img_end_of_row|>"
                "<|image|><|image|><|image|><|image|><|img_end|> describe",
            ),
        ]
        for image_size, expected in cases:
            with self.subTest(image_size=image_size):
                out = processor(text="<|image|> describe", images=[self._image(*image_size)], return_tensors="pt")
                self.assertEqual(processor.tokenizer.decode(out["input_ids"][0]), expected)

    def test_audio_expansion_matches_reference_layout(self):
        """ceil(samples / hop) placeholder tokens wrapped in audio start/end; no header."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        clips = [self._clip(hop), self._clip(hop + 1), self._clip(2 * hop)]
        out = processor(text="<|audio|>a<|audio|>b<|audio|>", audio=clips, return_tensors="pt")
        decoded = processor.tokenizer.decode(out["input_ids"][0])
        expected = (
            "<|audio_start|><|audio|><|audio_end|>a"
            "<|audio_start|><|audio|><|audio|><|audio_end|>b"
            "<|audio_start|><|audio|><|audio|><|audio_end|>"
        )
        self.assertEqual(decoded, expected)
        self.assertEqual(out["input_features"].shape, (3, 1, 2 * hop))
        self.assertEqual(out["feature_attention_mask"].sum(-1).tolist(), [hop, hop + 1, 2 * hop])
        self.assertNotIn("num_audio_codes", out)

    def test_audio_peak_normalized_to_minus_3_dbfs(self):
        processor = self.get_processor()
        out = processor(text="<|audio|>", audio=[self._clip(1200) * 20.0], return_tensors="pt")
        peak = out["input_features"].abs().amax().item()
        self.assertAlmostEqual(peak, 10 ** (-3 / 20), places=4)
        # an all-zero clip must not produce NaNs
        out = processor(text="<|audio|>", audio=[np.zeros(1200, dtype=np.float32)], return_tensors="pt")
        self.assertFalse(bool(np.isnan(out["input_features"]).any()))

    def test_flat_and_nested_batches_preserve_media_order(self):
        """Flat and nested media preserve per-sample ownership, including empty groups."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        texts = [
            "<|audio|> sound only",
            "<|image|> image only",
            "<|image|><|image|><|image|> both <|audio|><|audio|>",
        ]
        images = [[], [self._image(32, 32)], [self._image(32, 32), self._image(48, 32), self._image(32, 48)]]
        audio = [[self._clip(hop)], [], [self._clip(hop + 1), self._clip(3 * hop)]]
        nested = processor(text=texts, images=images, audio=audio, padding=True, return_tensors="pt")
        flat = processor(
            text=texts,
            images=[image for group in images for image in group],
            audio=[clip for group in audio for clip in group],
            padding=True,
            return_tensors="pt",
        )
        for key in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_sizes",
            "input_features",
            "feature_attention_mask",
        ):
            with self.subTest(key=key):
                torch.testing.assert_close(flat[key], nested[key])
        self.assertEqual(nested["pixel_values"].shape[0], 4)
        self.assertEqual(nested["image_sizes"].tolist(), [[32, 32], [32, 32], [48, 32], [32, 48]])
        self.assertEqual(nested["feature_attention_mask"].sum(-1).tolist(), [hop, hop + 1, 3 * hop])

        expected_image_headers = [[], ["2*2"], ["2*2", "3*2", "2*3"]]
        expected_audio_counts = [[1], [], [2, 3]]
        for index, ids in enumerate(nested["input_ids"]):
            with self.subTest(sample=index):
                decoded = processor.tokenizer.decode(ids)
                headers = re.findall(r"<\|img_start\|>(\d+\*\d+)<\|img_token_start\|>", decoded)
                audio_runs = re.findall(r"<\|audio_start\|>(.*?)<\|audio_end\|>", decoded)
                self.assertEqual(headers, expected_image_headers[index])
                self.assertEqual([run.count("<|audio|>") for run in audio_runs], expected_audio_counts[index])

    def test_mismatched_counts_raise(self):
        """Strict validation in both directions, for both modalities, flat and nested."""
        processor = self.get_processor()
        image, clip = self._image(32, 32), self._clip(1200)

        cases = [
            # (kwargs, expected error snippet)
            ({"text": "<|image|>", "images": [image, image]}, "placeholders in total"),
            ({"text": "<|image|><|image|>", "images": [image]}, "placeholders in total"),
            ({"text": "<|image|>"}, "no image inputs were passed"),
            ({"text": "<|audio|>", "audio": [clip, clip]}, "placeholders in total"),
            ({"text": "<|audio|><|audio|>", "audio": [clip]}, "placeholders in total"),
            ({"text": "<|audio|>"}, "no audio inputs were passed"),
            ({"text": ["a", "<|image|>b"], "images": [[image], []]}, "placeholder counts"),
            ({"text": ["<|audio|>"], "audio": [[clip], [clip]]}, "sub-lists for"),
        ]
        for kwargs, snippet in cases:
            with self.subTest(snippet=snippet, kwargs=list(kwargs)):
                with self.assertRaises(ValueError) as ctx:
                    processor(**kwargs)
                self.assertIn(snippet, str(ctx.exception))

    @require_librosa
    def test_nested_audio_files_match_flat_loading(self):
        """Nested audio files are resampled and assigned to the same samples as flat inputs."""
        processor = self.get_processor()
        source_rate = 8000
        samples = (16000 * np.sin(2 * np.pi * 440 * np.arange(800) / source_rate)).astype("<i2")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")
            with wave.open(audio_path, "wb") as audio_file:
                audio_file.setparams((1, 2, source_rate, 0, "NONE", "not compressed"))
                audio_file.writeframes(samples.tobytes())
            kwargs = {"text": ["no audio", "<|audio|>"], "padding": True, "return_tensors": "np"}
            flat = processor(audio=[audio_path], **kwargs)
            nested = processor(audio=[[], [audio_path]], **kwargs)

        np.testing.assert_allclose(nested["input_features"], flat["input_features"])
        np.testing.assert_array_equal(nested["feature_attention_mask"], flat["feature_attention_mask"])
        np.testing.assert_array_equal(nested["input_ids"], flat["input_ids"])
        expected_samples = len(samples) * processor.feature_extractor.sampling_rate // source_rate
        self.assertEqual(int(nested["feature_attention_mask"].sum()), expected_samples)
        expected_tokens = -(-expected_samples // processor.feature_extractor.hop_length)
        self.assertEqual((nested["input_ids"] == processor.audio_token_id).sum(axis=1).tolist(), [0, expected_tokens])

    def test_audio_token_count_helper_matches_processor(self):
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        audio_lengths = [1, hop, hop + 1]
        inputs = processor(
            text=["<|audio|>"] * len(audio_lengths),
            audio=[self._clip(length) for length in audio_lengths],
            padding=True,
        )
        actual_counts = [ids.count(processor.audio_token_id) for ids in inputs["input_ids"]]
        predicted_counts = processor._get_num_multimodal_tokens(audio_lengths=audio_lengths)["num_audio_tokens"]
        self.assertEqual(predicted_counts, actual_counts)

    def test_all_empty_media_treated_as_no_media(self):
        """Uniform collators may emit empty media collections for text-only batches; these must be accepted."""
        processor = self.get_processor()
        for kwargs in (
            {"images": [[], []]},
            {"audio": [[], []]},
            {"images": [], "audio": []},
            {"images": [[], []], "audio": [[], []]},
        ):
            with self.subTest(kwargs=list(kwargs)):
                out = processor(text=["plain text", "more text"], **kwargs)
                self.assertIn("input_ids", out)
                self.assertNotIn("pixel_values", out)
                self.assertNotIn("input_features", out)

    @require_torch
    def test_audio_masks_and_placeholders_match_clip_lengths(self):
        """Masks and placeholders track valid samples across padding modes and tensor formats."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        cases = [
            ([hop + 3, hop + 3], False, None, list),
            ([hop + 3, hop + 3], False, "np", np.ndarray),
            ([hop + 3, hop + 3], False, "pt", torch.Tensor),
            ([1, hop + 3], False, None, list),
            ([1, hop + 3], True, None, np.ndarray),
        ]
        for lengths, padding, tensor_type, expected_type in cases:
            with self.subTest(lengths=lengths, padding=padding, tensor_type=tensor_type):
                out = processor(
                    text=["<|audio|>"] * len(lengths),
                    audio=[self._clip(length) for length in lengths],
                    padding=padding,
                    audio_kwargs={"return_tensors": tensor_type},
                )
                self.assertIsInstance(out["feature_attention_mask"], expected_type)
                self.assertEqual(len(out["feature_attention_mask"]), len(lengths))
                self.assertEqual(len(out["input_features"]), len(lengths))
                for index, length in enumerate(lengths):
                    output_length = max(lengths) if padding else length
                    expected_mask = np.arange(output_length) < length
                    np.testing.assert_array_equal(out["feature_attention_mask"][index], expected_mask)
                    self.assertEqual(out["input_features"][index].shape, (1, output_length))
                    num_placeholders = out["input_ids"][index].count(processor.audio_token_id)
                    self.assertEqual(num_placeholders, -(-length // hop))

    def test_audio_truncation_keeps_placeholders_consistent(self):
        """Truncation must never desync the placeholder count from the returned features."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        for padding in (False, True):
            with self.subTest(padding=padding):
                out = processor(
                    text="<|audio|>",
                    audio=[self._clip(3 * hop)],
                    audio_kwargs={"padding": padding, "truncation": True, "max_length": hop},
                    return_tensors="pt",
                )
                num_placeholders = processor.tokenizer.decode(out["input_ids"][0]).count("<|audio|>")
                valid_samples = int(out["feature_attention_mask"].sum())
                self.assertEqual(valid_samples, hop)
                self.assertEqual(num_placeholders, -(-valid_samples // hop))
                self.assertEqual(out["input_features"].shape[-1], hop)

    @require_torch
    def test_processor_to_tiny_model_forward(self):
        """End-to-end: processor outputs feed a tiny Apertus1p5 model whose tokenizer sub-configs are aligned
        with the processor components (VQ factor == spatial_factor, codec hop == feature-extractor hop)."""
        from transformers import Apertus1p5ForConditionalGeneration, WavTokenizerFeatureExtractor

        from .test_modeling_apertus1p5 import Apertus1p5ModelTester

        base_processor = self.get_processor()
        tester = Apertus1p5ModelTester(
            self,
            image_size=32,
            num_hidden_layers=1,
            vq_num_res_blocks=1,
            image_token_id=base_processor.image_token_id,
            audio_token_id=base_processor.audio_token_id,
            pad_token_id=base_processor.tokenizer.pad_token_id,
        )
        tester.image_token_offset = len(base_processor.tokenizer)
        tester.audio_token_offset = tester.image_token_offset + tester.codebook_size
        tester.vocab_size = tester.audio_token_offset + tester.audio_codebook_size
        config = tester.get_config()

        processor = self.processor_class(
            image_processor=base_processor.image_processor,
            feature_extractor=WavTokenizerFeatureExtractor(hop_length=config.audio_config.hop_length),
            tokenizer=base_processor.tokenizer,
        )
        model = Apertus1p5ForConditionalGeneration(config).eval()

        inputs = processor(
            text="<|image|>hello<|audio|>",
            images=[self._image(32, 32)],
            audio=[self._clip(10)],
            images_kwargs={"spatial_factor": config.vision_config.spatial_scale_factor},
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[-1], model.lm_head.out_features)
        self.assertTrue(bool(torch.isfinite(logits).all()))


@slow
@require_torchvision
class Apertus1p5ProcessorIntegrationTest(unittest.TestCase):
    """Checkpoint processor checks without model weights; supports `APERTUS1P5_CHECKPOINT` for local assets."""

    @classmethod
    def setUpClass(cls):
        checkpoint = os.environ.get("APERTUS1P5_CHECKPOINT", "swiss-ai/Apertus-v1.5-8B")
        cls.config = Apertus1p5Config.from_pretrained(checkpoint)
        cls.processor = AutoProcessor.from_pretrained(checkpoint)
        cls.tokenizer = cls.processor.tokenizer

    @classmethod
    def tearDownClass(cls):
        del cls.processor, cls.tokenizer, cls.config

    def test_processor_placeholder_sequences(self):
        """Image and audio placeholders match the expected structure and checkpoint vocabulary IDs."""
        processor = self.processor
        config = self.config
        bos = self.tokenizer.bos_token_id

        image = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        inputs = processor(
            text="<|image|>", images=[image], images_kwargs={"min_pixels": 32 * 32}, return_tensors="pt"
        )
        digit_ids = self.tokenizer("2*2", add_special_tokens=False)["input_ids"]
        # the structure tokens have no config ids; the golden ids come from the real vocabulary
        boi, eoi, wrapper, eol = self.tokenizer.convert_tokens_to_ids(
            [processor.boi_token, processor.eoi_token, processor.image_wrapper_token, processor.eol_token]
        )
        image_id = config.image_token_id
        expected = [bos, boi, *digit_ids, wrapper, image_id, image_id, eol, image_id, image_id, eoi]
        self.assertEqual(inputs["input_ids"][0].tolist(), expected)

        clip = np.sin(2 * np.pi * 440.0 * np.arange(1200) / 24000.0).astype(np.float32)
        inputs = processor(text="<|audio|>", audio=[clip], return_tensors="pt")
        boa, eoa = self.tokenizer.convert_tokens_to_ids([processor.boa_token, processor.eoa_token])
        expected = [
            bos,
            boa,
            config.audio_token_id,
            config.audio_token_id,
            eoa,
        ]
        self.assertEqual(inputs["input_ids"][0].tolist(), expected)

    def test_chat_template_content_forms_equivalent(self):
        """String content, the upstream {'parts': [...]} mapping, and the standard list-of-blocks content must
        all render to the same prompt with the patched composite template."""
        processor = self.processor
        as_string = [{"role": "user", "content": "<|image|>Describe, then answer: <|audio|>"}]
        as_parts = [
            {
                "role": "user",
                "content": {
                    "parts": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe, then answer: "},
                        {"type": "audio"},
                    ]
                },
            }
        ]
        as_blocks = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe, then answer: "},
                    {"type": "audio"},
                ],
            }
        ]
        rendered = [
            processor.apply_chat_template(messages, add_generation_prompt=True)
            for messages in (as_string, as_parts, as_blocks)
        ]
        self.assertEqual(rendered[0], rendered[1])
        self.assertEqual(rendered[1], rendered[2])
