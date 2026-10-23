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

import math
import os
import re
import tempfile
import unittest
import wave

import numpy as np

from transformers import Apertus1p5Processor, AutoProcessor, is_torch_available
from transformers.testing_utils import require_librosa, require_torchvision, slow

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_torch_available():
    import torch


@require_torchvision
class Apertus1p5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Apertus1p5Processor

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        # small pixel budget so test images stay tiny; grids stay >= 2x2
        return image_processor_class(size={"min_pixels": 32 * 32, "max_pixels": 64 * 64})

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        extra_special_tokens = {
            "image_token": "<|image|>",
            "audio_token": "<|audio|>",
            "boi_token": "<|img_start|>",
            "eoi_token": "<|img_end|>",
            "image_wrapper_token": "<|img_token_start|>",
            "eol_token": "<|img_end_of_row|>",
            "boa_token": "<|audio_start|>",
            "eoa_token": "<|audio_end|>",
        }
        tokenizer = tokenizer_class.from_pretrained("openai-community/gpt2", extra_special_tokens=extra_special_tokens)
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
        """Images expand to a height-first `H*W` header and their grid rows joined by separators; `do_resize` and
        `spatial_factor` set the grid, and the vLLM helper predicts the same placeholder count."""
        processor = self.get_processor()
        out = processor(text="<|image|> describe", images=[self._image(32, 64)], return_tensors="pt")
        expected = (
            "<|img_start|>2*4<|img_token_start|>"
            + "<|img_end_of_row|>".join(["<|image|>" * 4] * 2)
            + "<|img_end|> describe"
        )
        self.assertEqual(processor.tokenizer.decode(out["input_ids"][0]), expected)

        image = self._image(81, 113)
        for do_resize, spatial_factor, grid in [
            (True, 16, (3, 4)),
            (False, 16, (5, 7)),
            (True, 8, (6, 9)),
            (False, 8, (10, 14)),
        ]:
            with self.subTest(do_resize=do_resize, spatial_factor=spatial_factor):
                images_kwargs = {"do_resize": do_resize, "spatial_factor": spatial_factor}
                inputs = processor(text="<|image|>", images=[image], images_kwargs=images_kwargs, return_tensors="pt")
                decoded = processor.tokenizer.decode(inputs["input_ids"][0])
                self.assertIn(f"<|img_start|>{grid[0]}*{grid[1]}<|img_token_start|>", decoded)
                self.assertEqual(decoded.count(processor.image_token), grid[0] * grid[1])
                self.assertEqual(decoded.count(processor.eol_token), grid[0] - 1)
                num_from_helper = processor._get_num_multimodal_tokens(image_sizes=[image.shape[:2]], **images_kwargs)
                self.assertEqual(num_from_helper["num_image_tokens"], [grid[0] * grid[1]])

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
        self.assertNotIn("num_audio_codes", out)

    def test_audio_peak_normalized_to_minus_3_dbfs(self):
        processor = self.get_processor()
        out = processor(text="<|audio|>", audio=[self._clip(1200) * 20.0], return_tensors="pt")
        peak = out["input_features"].abs().amax().item()
        self.assertAlmostEqual(peak, 10 ** (-3 / 20), places=4)
        # an all-zero clip must not produce NaNs
        out = processor(text="<|audio|>", audio=[np.zeros(1200, dtype=np.float32)], return_tensors="pt")
        self.assertFalse(torch.isnan(out["input_features"]).any())

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
        for index, (kwargs, snippet) in enumerate(cases):
            with self.subTest(case=index, snippet=snippet):
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
        expected_tokens = math.ceil(expected_samples / processor.feature_extractor.hop_length)
        self.assertEqual((nested["input_ids"] == processor.audio_token_id).sum(axis=1).tolist(), [0, expected_tokens])

    def test_get_num_audio_tokens(self):
        """Tests the audio path of the helper used internally in vLLM."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        audio_lengths = [1, hop, hop + 1]
        inputs = processor(
            text=["<|audio|>"] * len(audio_lengths),
            audio=[self._clip(length) for length in audio_lengths],
            padding=True,
        )
        num_from_call = [ids.count(processor.audio_token_id) for ids in inputs["input_ids"]]
        num_from_helper = processor._get_num_multimodal_tokens(audio_lengths=audio_lengths)["num_audio_tokens"]
        self.assertEqual(num_from_helper, [1, 1, 2])
        self.assertEqual(num_from_call, num_from_helper)

    def test_audio_mask_tracks_valid_samples(self):
        """Masks, features and placeholder counts follow the valid samples of unpadded, padded and truncated clips."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        cases = [
            # (clip lengths, audio kwargs, valid samples, output lengths)
            ([1, hop + 3], {"padding": False}, [1, hop + 3], [1, hop + 3]),
            ([1, hop + 3], {"padding": True}, [1, hop + 3], [hop + 3, hop + 3]),
            ([3 * hop], {"padding": False, "truncation": True, "max_length": hop}, [hop], [hop]),
            ([3 * hop], {"padding": True, "truncation": True, "max_length": hop}, [hop], [hop]),
        ]
        for lengths, audio_kwargs, valid_lengths, output_lengths in cases:
            with self.subTest(lengths=lengths, **audio_kwargs):
                out = processor(
                    text=["<|audio|>"] * len(lengths),
                    audio=[self._clip(length) for length in lengths],
                    audio_kwargs=audio_kwargs,
                )
                for index, (valid, output_length) in enumerate(zip(valid_lengths, output_lengths)):
                    np.testing.assert_array_equal(
                        out["feature_attention_mask"][index], np.arange(output_length) < valid
                    )
                    self.assertEqual(np.shape(out["input_features"][index]), (1, output_length))
                    self.assertEqual(out["input_ids"][index].count(processor.audio_token_id), math.ceil(valid / hop))

        # without padding the feature extractor returns no mask; the processor builds it in the requested format
        for return_tensors, expected_type in [(None, list), ("np", np.ndarray), ("pt", torch.Tensor)]:
            with self.subTest(return_tensors=return_tensors):
                out = processor(
                    text=["<|audio|>"] * 2,
                    audio=[self._clip(hop)] * 2,
                    audio_kwargs={"padding": False, "return_tensors": return_tensors},
                )
                self.assertIsInstance(out["feature_attention_mask"], expected_type)


@slow
@require_torchvision
class Apertus1p5ProcessorIntegrationTest(unittest.TestCase):
    """
    The expected expansions follow the token layout of the original Apertus vLLM implementation
    (`build_apertus_image_prompt` and `encode_audios`): each placeholder of the rendered prompt is replaced by its
    expansion, and the result is tokenized with the checkpoint tokenizer.
    """

    checkpoint = "swiss-ai/Apertus-v1.5-8B"
    revision = "refs/pr/5"  # TODO: drop once https://huggingface.co/swiss-ai/Apertus-v1.5-8B/discussions/5 is merged
    IMAGE_URL = url_to_local_path(
        "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
    )
    AUDIO_URL = url_to_local_path(
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
    )
    # the 640x480 image fits the default pixel budget: a 30x40 grid of 16-pixel patches, one code each
    image_expansion = (
        "<|img_start|>30*40<|img_token_start|>" + "<|img_end_of_row|>".join(["<|image|>" * 40] * 30) + "<|img_end|>"
    )
    # 93,680 samples at 16 kHz are resampled to 140,520 at 24 kHz, one code per 600 samples: 235 codes
    audio_expansion = "<|audio_start|>" + "<|audio|>" * 235 + "<|audio_end|>"

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint, revision=cls.revision)

    def _assert_expanded(self, content, expansions):
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        for placeholder, expansion in expansions:
            prompt = prompt.replace(placeholder, expansion, 1)
        expected_ids = self.processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True
        )
        self.assertEqual(inputs["input_ids"][0], expected_ids)

    def test_apply_chat_template_image(self):
        content = [{"type": "image", "url": self.IMAGE_URL}, {"type": "text", "text": "Describe this image."}]
        self._assert_expanded(content, [("<|image|>", self.image_expansion)])

    @require_librosa
    def test_apply_chat_template_audio(self):
        content = [{"type": "audio", "url": self.AUDIO_URL}, {"type": "text", "text": "Transcribe this audio."}]
        self._assert_expanded(content, [("<|audio|>", self.audio_expansion)])

    @require_librosa
    def test_apply_chat_template_image_audio(self):
        content = [
            {"type": "image", "url": self.IMAGE_URL},
            {"type": "audio", "url": self.AUDIO_URL},
            {"type": "text", "text": "What do you see and what do you hear?"},
        ]
        self._assert_expanded(content, [("<|image|>", self.image_expansion), ("<|audio|>", self.audio_expansion)])
