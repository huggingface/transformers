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

import unittest

import numpy as np

from transformers import Apertus1p5Processor, is_torch_available
from transformers.testing_utils import require_torch, require_torchvision

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
        """The 2x2 golden string from the reference vLLM implementation (vllm_swissai `apertus_integration`
        commit 40a5516b7): no eof, exactly H-1 row separators."""
        processor = self.get_processor()
        out = processor(text="<|image|> describe", images=[self._image(32, 32)], return_tensors="pt")
        decoded = processor.tokenizer.decode(out["input_ids"][0])
        expected = (
            "<|img_start|>2*2<|img_token_start|>"
            "<|image|><|image|><|img_end_of_row|><|image|><|image|>"
            "<|img_end|> describe"
        )
        self.assertEqual(decoded, expected)
        self.assertEqual(out["pixel_values"].shape, (1, 3, 32, 32))
        self.assertEqual(out["image_sizes"].tolist(), [[32, 32]])

    def test_image_expansion_counts_and_header(self):
        """A 4x4 grid: 16 placeholders, 3 row separators, one boi/wrapper/eoi; header is height-first."""
        processor = self.get_processor()
        out = processor(text="<|image|>", images=[self._image(64, 64)], return_tensors="pt")
        decoded = processor.tokenizer.decode(out["input_ids"][0])
        self.assertEqual(decoded.count("<|image|>"), 16)
        self.assertEqual(decoded.count("<|img_end_of_row|>"), 3)
        self.assertIn("<|img_start|>4*4<|img_token_start|>", decoded)
        self.assertEqual(decoded.count("<|img_end|>"), 1)

        # height-first header for a non-square image
        out = processor(text="<|image|>", images=[self._image(32, 64)], return_tensors="pt")
        decoded = processor.tokenizer.decode(out["input_ids"][0])
        self.assertIn("<|img_start|>2*4<|img_token_start|>", decoded)

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

    def test_nested_uneven_batches(self):
        """Arbitrary per-sample media counts via nested lists, including empty sub-lists."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        texts = [
            "<|audio|> sound only",
            "<|image|> image only",
            "<|image|><|image|><|image|> both <|audio|><|audio|>",
        ]
        images = [[], [self._image(32, 32)], [self._image(32, 32), self._image(48, 32), self._image(32, 48)]]
        audio = [[self._clip(hop)], [], [self._clip(hop + 1), self._clip(3 * hop)]]
        out = processor(text=texts, images=images, audio=audio, padding=True, return_tensors="pt")

        self.assertEqual(out["pixel_values"].shape[0], 4)  # total images, flattened
        self.assertEqual(out["image_sizes"].tolist(), [[32, 32], [32, 32], [48, 32], [32, 48]])
        self.assertEqual(out["input_features"].shape[0], 3)  # total clips, flattened
        # per-sample expansions are independent: sample 0 has no image structure tokens
        decoded_first = processor.tokenizer.decode(out["input_ids"][0], skip_special_tokens=False)
        self.assertNotIn("<|img_start|>", decoded_first)
        self.assertIn("<|audio_start|>", decoded_first)

    def test_flat_media_distributed_by_placeholder_order(self):
        """Flat media lists are consumed left-to-right across the batch, sample by sample."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        texts = ["<|image|><|image|>", "<|image|>"]
        out = processor(
            text=texts,
            images=[self._image(32, 32), self._image(48, 32), self._image(32, 48)],
            padding=True,
            return_tensors="pt",
        )
        # first two images belong to sample 0 (in order), the third to sample 1
        self.assertEqual(out["image_sizes"].tolist(), [[32, 32], [48, 32], [32, 48]])
        decoded_second = processor.tokenizer.decode(out["input_ids"][1], skip_special_tokens=False)
        self.assertIn("<|img_start|>2*3<|img_token_start|>", decoded_second)

        out = processor(text=["<|audio|>", "<|audio|>"], audio=[self._clip(hop), self._clip(2 * hop)], padding=True)
        decoded_second = processor.tokenizer.decode(out["input_ids"][1], skip_special_tokens=False)
        self.assertEqual(decoded_second.count("<|audio|>"), 2)

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

    def test_media_from_urls(self):
        """Image and audio entries may be URL (or path) strings; the generic layout hooks fetch them and
        audio files are resampled to the feature extractor's 24 kHz, flat and nested alike."""
        processor = self.get_processor()
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
        audio_url = (
            "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"
        )

        out = processor(text="<|image|> and <|audio|>", images=[image_url], audio=[audio_url], return_tensors="pt")
        num_samples = int(out["feature_attention_mask"].sum())
        num_placeholders = processor.tokenizer.decode(out["input_ids"][0]).count("<|audio|>")
        self.assertEqual(num_placeholders, -(-num_samples // processor.feature_extractor.hop_length))
        self.assertEqual(out["pixel_values"].ndim, 4)

        out_nested = processor(
            text=["<|image|>", "<|audio|>"],
            images=[[image_url], []],
            audio=[[], [audio_url]],
            padding=True,
            return_tensors="pt",
        )
        self.assertEqual(out_nested["pixel_values"].shape, out["pixel_values"].shape)
        self.assertEqual(int(out_nested["feature_attention_mask"].sum()), num_samples)

    def test_get_num_multimodal_tokens_math(self):
        """The pure-math helper mirrors the image processor's resize math and the audio hop arithmetic."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length

        # with the test budget (min 32*32, max 64*64, factor 16): 32x32 stays -> 2x2 grid; 64x64 stays -> 4x4;
        # 100x30 (area 3000 in budget) -> int(sqrt(3000/0.3))=100 -> 96, int(100*0.3)=30 -> 32 -> 6x2 grid
        image_sizes = [(32, 32), (64, 64), (100, 30)]
        audio_lengths = [1, hop, hop + 1, 40 * hop]

        output = processor._get_num_multimodal_tokens(image_sizes=image_sizes, audio_lengths=audio_lengths)
        self.assertEqual(output["num_image_tokens"], [4, 16, 12])
        self.assertEqual(output["num_image_patches"], [1, 1, 1])
        self.assertEqual(output["num_audio_tokens"], [1, 1, 2, 40])

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

    def test_audio_truncation_keeps_placeholders_consistent(self):
        """Truncation must never desync the placeholder count from the returned features."""
        processor = self.get_processor()
        hop = processor.feature_extractor.hop_length
        out = processor(
            text="<|audio|>",
            audio=[self._clip(3 * hop)],
            audio_kwargs={"truncation": True, "max_length": hop},
            return_tensors="pt",
        )
        num_placeholders = processor.tokenizer.decode(out["input_ids"][0]).count("<|audio|>")
        valid_samples = int(out["feature_attention_mask"].sum())
        self.assertEqual(num_placeholders, -(-valid_samples // hop))
        self.assertEqual(out["input_features"].shape[-1], hop)

    @require_torch
    def test_processor_to_tiny_model_forward(self):
        """End-to-end: processor outputs feed a tiny Apertus1p5 model whose tokenizer sub-configs are aligned
        with the processor components (VQ factor == spatial_factor, codec hop == feature-extractor hop)."""
        from transformers import Apertus1p5Config, Apertus1p5ForConditionalGeneration, WavTokenizerFeatureExtractor

        base_processor = self.get_processor()
        # dedicated feature extractor matching the tiny codec geometry (hop 4)
        processor = self.processor_class(
            image_processor=base_processor.image_processor,
            feature_extractor=WavTokenizerFeatureExtractor(hop_length=4),
            tokenizer=base_processor.tokenizer,
        )

        vocab_size = len(processor.tokenizer)
        image_token_offset = vocab_size
        audio_token_offset = image_token_offset + 20  # tiny VQ codebook
        text_config = {
            "model_type": "apertus",
            "hidden_act": "gelu",
            "vocab_size": audio_token_offset + 12,  # + tiny audio codebook
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 37,
            "pad_token_id": 0,
        }
        vision_config = {
            "codebook_size": 20,
            "base_channels": 32,
            "channel_multiplier": [1, 2, 1],  # spatial factor 4
            "num_res_blocks": 1,
            "embed_dim": 16,
            "latent_channels": 16,
            "attn_resolutions": [],
            "resolution": 32,
        }
        audio_config = {
            "model_type": "wavtokenizer",
            "num_filters": 8,
            "upsampling_ratios": [2, 2],  # hop_length 4
            "hidden_size": 32,
            "codebook_dim": 32,
            "codebook_size": 12,
            "decoder_hidden_size": 32,
            "decoder_intermediate_size": 64,
            "decoder_num_layers": 2,
        }
        config = Apertus1p5Config(
            text_config=text_config,
            vision_config=vision_config,
            audio_config=audio_config,
            image_token_id=processor.image_token_id,
            audio_token_id=processor.audio_token_id,
            image_token_offset=image_token_offset,
            audio_token_offset=audio_token_offset,
        )
        model = Apertus1p5ForConditionalGeneration(config).eval()

        inputs = processor(
            text="<|image|>hello<|audio|>",
            images=[self._image(32, 32)],
            audio=[self._clip(10)],
            images_kwargs={"spatial_factor": 4},
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[-1], config.text_config.vocab_size)
        self.assertTrue(bool(torch.isfinite(logits).all()))
