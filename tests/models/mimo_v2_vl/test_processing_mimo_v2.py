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

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from huggingface_hub import hf_hub_download, hf_hub_url
from parameterized import parameterized
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoVideoProcessor,
    MiMoV2FeatureExtractor,
    MiMoV2ImageProcessor,
    MiMoV2Processor,
    MiMoV2VideoProcessor,
    PreTrainedTokenizerFast,
)
from transformers.testing_utils import (
    require_torch,
    require_torchaudio,
    require_torchcodec,
    require_torchvision,
    require_vision,
    slow,
)
from transformers.utils import is_torch_available
from transformers.video_utils import VideoMetadata


if is_torch_available():
    import torch


SPECIAL_TOKENS = [
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|mimo_audio_start|>",
    "<|mimo_audio_end|>",
    "<|mimo_video_start|>",
    "<|mimo_video_end|>",
]


MEDIA_FIXTURES = {
    "image": ("huggingface/documentation-images", "pipeline-cat-chonk.jpeg"),
    "audio": ("hf-internal-testing/dummy-audio-samples", "f2641_0_throatclearing.wav"),
    "video": ("hf-internal-testing/test-videos", "tiny_video_320x240.mp4"),
}


MULTIMODAL_CHAT_TEMPLATE = (
    "{% for message in messages %}{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif item['type'] == 'audio' %}<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"
    "{% elif item['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
    "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %}"
)


def make_processor(**kwargs):
    vocab = {
        token: i
        for i, token in enumerate(
            ["<unk>", "<pad>", "<bos>"] + SPECIAL_TOKENS + ["before", "after", "text", "00", "01", "02", ":"]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        additional_special_tokens=SPECIAL_TOKENS,
    )
    size = {"shortest_edge": 16, "longest_edge": 256}
    return MiMoV2Processor(
        tokenizer=tokenizer,
        image_processor=MiMoV2ImageProcessor(patch_size=2, merge_size=2, size=size),
        video_processor=MiMoV2VideoProcessor(
            patch_size=2, merge_size=2, size=size, total_max_pixels=1024, min_frames=2, max_frames=16
        ),
        feature_extractor=MiMoV2FeatureExtractor(),
        **kwargs,
    )


@require_torch
@require_torchvision
@require_vision
class MiMoV2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = make_processor()
        self.image = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
        self.video = np.stack([self.image] * 5)
        self.codes = np.arange(64 * 8).reshape(64, 8)

    def video_kwargs(self):
        return {
            "do_sample_frames": False,
            "video_metadata": VideoMetadata(total_num_frames=5, fps=2, frames_indices=list(range(5))),
        }

    def test_text_only_and_bos_once(self):
        processor = self.processor
        expected = processor.tokenizer(["before text after"], return_tensors="pt")
        actual = processor(text="before text after", return_tensors="pt")
        torch.testing.assert_close(actual.input_ids, expected.input_ids)
        actual = processor(text="before <|image_pad|> after", images=self.image, return_tensors="pt")
        self.assertEqual(actual.input_ids[0].tolist().count(processor.tokenizer.bos_token_id), 1)

    @parameterized.expand([(None,), ("pt",), ("np",)])
    def test_image_batch_padding_and_full_placeholders(self, tensor_type):
        processor = self.processor
        prompts = ["<|image_pad|>", "before <|vision_start|><|image_pad|><|image_pad|><|vision_end|> after"]
        out = processor(text=prompts, images=[self.image, self.image], padding=True, return_tensors=tensor_type)
        ids = np.asarray(out.input_ids)
        self.assertEqual(ids.shape[0], 2)
        self.assertEqual(np.asarray(out.image_grid_thw).tolist(), [[1, 2, 4], [1, 2, 4]])
        for sample in ids:
            self.assertEqual(list(sample).count(processor.image_token_id), 2)
            self.assertEqual(list(sample).count(processor.vision_start_token_id), 1)
        self.assertLess(np.asarray(out.attention_mask)[0].sum(), np.asarray(out.attention_mask)[1].sum())
        self.assertEqual(prompts[0], "<|image_pad|>")

    def test_image_resize_and_pixel_normalization(self):
        image = np.full((1, 5, 3), 127, dtype=np.uint8)
        out = self.processor.image_processor(image, input_data_format="channels_last", return_tensors="pt")
        self.assertEqual(out.image_grid_thw.tolist(), [[1, 2, 10]])
        expected = (torch.tensor([127.0] * 3) - torch.tensor(self.processor.image_processor.image_mean) * 255) / (
            torch.tensor(self.processor.image_processor.image_std) * 255
        )
        torch.testing.assert_close(out.pixel_values[0].reshape(3, 2, 2, 2)[:, 0, 0, 0], expected, rtol=0, atol=0)

    def test_video_sampling_matches_mimo_intervals(self):
        metadata = VideoMetadata(total_num_frames=12, fps=6)
        np.testing.assert_array_equal(self.processor.video_processor.sample_frames(metadata, fps=2), [0, 3, 6, 9])
        np.testing.assert_array_equal(
            self.processor.video_processor.sample_frames(metadata, num_frames=6), [0, 2, 4, 6, 8, 10]
        )
        np.testing.assert_array_equal(
            self.processor.video_processor.sample_frames(VideoMetadata(total_num_frames=1, fps=6), fps=2), [0, 0]
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.processor.video_processor.sample_frames(metadata, fps=2, num_frames=6)

    def test_sampling_call_overrides_saved_defaults(self):
        video = np.stack([self.image] * 12)
        out = self.processor.video_processor(
            video,
            num_frames=6,
            video_metadata=VideoMetadata(total_num_frames=12, fps=6),
            return_metadata=True,
            return_tensors="pt",
        )
        self.assertEqual(out.video_grid_thw[0, 0].item(), 3)
        self.assertEqual(list(out.video_metadata[0].frames_indices), [0, 2, 4, 6, 8, 10])
        self.processor.video_processor.num_frames = 6
        out = self.processor.video_processor(
            video,
            fps=2,
            video_metadata=VideoMetadata(total_num_frames=12, fps=6),
            return_metadata=True,
            return_tensors="pt",
        )
        self.assertEqual(out.video_grid_thw[0, 0].item(), 2)

    def test_video_padding_timestamp_layout(self):
        p = self.processor
        out = p(
            text="<|vision_start|><|video_pad|><|vision_end|>",
            videos=self.video,
            return_tensors="pt",
            **self.video_kwargs(),
        )
        expected = (
            p.video_start_token
            + "".join(f"00:0{i}" + p.vision_start_token + p.video_token * 2 + p.vision_end_token for i in range(3))
            + p.video_end_token
        )
        self.assertEqual(out.input_ids[0].tolist(), p.tokenizer.encode(expected))
        self.assertEqual(out.video_grid_thw.tolist(), [[3, 2, 4]])
        self.assertNotIn("video_metadata", out)

    @parameterized.expand([(False,), (True,)])
    def test_video_metadata_batch_size_must_match(self, do_sample_frames):
        with self.assertRaisesRegex(ValueError, "one entry per video"):
            self.processor(
                text=[self.processor.video_token] * 2,
                videos=[self.video, self.video],
                video_metadata=[self.video_kwargs()["video_metadata"]],
                do_sample_frames=do_sample_frames,
                input_data_format="channels_last",
            )

    @parameterized.expand([(-1, False), (0, False), (1, False), (3, False), (3, True)])
    def test_video_audio_interleave(self, duration, per_grid):
        p = make_processor(video_audio_interleave_length=duration, use_per_grid_t_timestamps=per_grid)
        out = p(
            text=p.video_token,
            videos=self.video,
            audio=self.codes,
            is_tokenized=True,
            use_audio_in_video=True,
            video_audio_interleave_length=duration,
            return_tensors="pt",
            **self.video_kwargs(),
        )
        torch.testing.assert_close(out.audio_segment_offsets, torch.tensor([[0, 0, 6], [0, 6, 6], [0, 12, 3]]))
        ids = out.input_ids[0].tolist()
        self.assertEqual(ids.count(p.video_token_id), 6)
        self.assertEqual(ids.count(p.audio_token_id), 15)
        self.assertEqual(ids.count(p.audio_start_token_id), 1 if duration in (-1, 3) else 3)
        if duration == 0:
            expected = (
                p.video_start_token
                + "".join(
                    f"00:0{i}"
                    + p.vision_start_token
                    + p.video_token * 2
                    + p.vision_end_token
                    + p.audio_start_token
                    + p.audio_token * n
                    + p.audio_end_token
                    for i, n in enumerate([6, 6, 3])
                )
                + p.video_end_token
            )
            self.assertEqual(ids, p.tokenizer.encode(expected))
        if duration == 3:
            other = p(
                text=p.video_token,
                videos=self.video,
                audio=self.codes,
                is_tokenized=True,
                use_audio_in_video=True,
                use_per_grid_t_timestamps=not per_grid,
                return_tensors="pt",
                **self.video_kwargs(),
            )
            self.assertNotEqual(ids, other.input_ids[0].tolist())

    def test_mixed_standalone_audio_and_silent_video(self):
        p = self.processor
        metadata = [self.video_kwargs()["video_metadata"] for _ in range(2)]
        out = p(
            text=p.audio_token + p.video_token + p.video_token,
            videos=[self.video, self.video],
            audio=[self.codes[:8], self.codes],
            is_tokenized=True,
            use_audio_in_video=[False, True],
            do_sample_frames=False,
            video_metadata=metadata,
            return_tensors="pt",
        )
        self.assertEqual(out.audio_segment_offsets.tolist(), [[0, 0, 2], [1, 0, 6], [1, 6, 6], [1, 12, 3]])
        self.assertEqual(out.audio_feature_lens.tolist(), [2, 16])

    def test_codec_group_padding(self):
        out = self.processor.feature_extractor(
            [self.codes[:5], self.codes[:9]], is_tokenized=True, return_tensors="pt"
        )
        self.assertEqual(list(out.input_audio_codes.shape), [2, 3, 4, 8])
        self.assertEqual(out.audio_feature_lens.tolist(), [2, 3])
        torch.testing.assert_close(out.input_audio_codes[0, 1], torch.tensor(np.repeat(self.codes[4:5], 4, axis=0)))
        self.assertEqual(out.attention_mask.tolist(), [[1, 1, 0], [1, 1, 1]])

    @require_torchaudio
    def test_log_mel_matches_reference_and_padding_is_masked(self):
        import torchaudio

        extractor = self.processor.feature_extractor
        generator = torch.Generator().manual_seed(0)
        waveform = torch.randn(24000, generator=generator) * 0.1
        reference = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=24000, n_fft=960, win_length=960, hop_length=240, n_mels=128, power=1.0
            )(waveform)
            .clamp(min=1e-7)
            .log()
            .T
        )
        out = extractor([waveform.numpy(), waveform[:12000].numpy()], sampling_rate=24000, return_tensors="pt")
        torch.testing.assert_close(out.input_features[0], reference, rtol=1e-4, atol=2e-4)
        self.assertEqual(out.attention_mask.sum(-1).tolist(), [101, 51])
        self.assertEqual(out.audio_feature_lens.tolist(), [7, 4])
        self.assertTrue(torch.equal(out.input_features[1, 51:], torch.zeros_like(out.input_features[1, 51:])))

    def test_waveform_processor_and_sampling_rate(self):
        p = self.processor
        out = p(text=p.audio_token, audio=np.zeros(24000, dtype=np.float32), sampling_rate=24000, return_tensors="pt")
        self.assertEqual(out.input_ids[0].tolist().count(p.audio_token_id), 7)
        self.assertEqual(out.feature_attention_mask.sum().item(), 101)
        with self.assertRaisesRegex(ValueError, "sampling_rate"):
            p(text=p.audio_token, audio=np.zeros(24000), sampling_rate=16000)

    def test_numpy_feature_extraction_fallback(self):
        extractor = MiMoV2FeatureExtractor(win_length=480)
        waveform = np.random.default_rng(0).standard_normal(24000).astype(np.float32) * 0.1
        expected = extractor(waveform, return_tensors="np")
        with patch(
            "transformers.models.mimo_v2_vl.feature_extraction_mimo_v2.is_torch_available",
            return_value=False,
        ):
            actual = extractor(waveform, return_tensors="np")
        np.testing.assert_allclose(actual.input_features, expected.input_features, rtol=1e-4, atol=2e-4)
        np.testing.assert_array_equal(actual.audio_feature_lens, expected.audio_feature_lens)

    def test_call_time_sizes_and_compression(self):
        p = self.processor
        out = p(
            text=p.image_token,
            images=self.image,
            images_kwargs={"size": {"shortest_edge": 64, "longest_edge": 64}},
            return_tensors="pt",
        )
        self.assertNotEqual(out.image_grid_thw.tolist(), [[1, 2, 4]])
        out = p(
            text=p.video_token,
            videos=self.video,
            temporal_compression_ratio=2,
            return_tensors="pt",
            **self.video_kwargs(),
        )
        self.assertEqual(out.video_grid_thw.tolist(), [[4, 2, 4]])
        self.assertEqual(out.input_ids[0].tolist().count(p.video_token_id), 4)

    def test_missing_media_and_bad_video_metadata(self):
        p = self.processor
        for text, images in [(p.image_token * 2, self.image), ("text", self.image), (p.image_token, None)]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                p(text=text, images=images)
        with self.assertRaisesRegex(ValueError, "fps"):
            p(text=p.video_token, videos=self.video, do_sample_frames=False)
        with self.assertRaisesRegex(ValueError, "Missing audio"):
            p(text=p.video_token, videos=self.video, use_audio_in_video=True, **self.video_kwargs())
        with self.assertRaises(ValueError):
            p(text=p.image_token, images=self.image, truncation=True, max_length=2)

    def test_save_load_auto_components_and_template(self):
        template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        p = make_processor(chat_template=template, video_audio_interleave_length=3, use_per_grid_t_timestamps=True)
        with tempfile.TemporaryDirectory() as directory:
            p.save_pretrained(directory)
            restored = AutoProcessor.from_pretrained(directory, local_files_only=True)
            self.assertIsInstance(restored, MiMoV2Processor)
            self.assertEqual(restored.video_audio_interleave_length, 3)
            self.assertTrue(restored.use_per_grid_t_timestamps)
            self.assertEqual(restored.chat_template, template)
            for auto_class, expected_class in [
                (AutoImageProcessor, MiMoV2ImageProcessor),
                (AutoVideoProcessor, MiMoV2VideoProcessor),
                (AutoFeatureExtractor, MiMoV2FeatureExtractor),
            ]:
                self.assertIsInstance(auto_class.from_pretrained(directory, local_files_only=True), expected_class)
            expected = p(text=p.image_token, images=self.image, return_tensors="pt")
            actual = restored(text=p.image_token, images=self.image, return_tensors="pt")
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            self.assertEqual(
                restored.apply_chat_template([{"role": "user", "content": "text"}], tokenize=False), "text"
            )

    @parameterized.expand([("flash", 4096, 4, 8), ("pro", 6144, 8, 8)])
    def test_load_checkpoint_processor_config_over_legacy_qwen_config(
        self, variant, hidden_size, kv_heads, swa_kv_heads
    ):
        with tempfile.TemporaryDirectory(prefix=variant) as directory:
            self.processor.tokenizer.save_pretrained(directory)
            config = {
                "model_type": "mimo_v2",
                "architectures": ["MiMoV2ForCausalLM"],
                "hidden_size": hidden_size,
                "num_key_value_heads": kv_heads,
                "swa_num_key_value_heads": swa_kv_heads,
                "vision_config": {
                    "patch_size": 16,
                    "temporal_patch_size": 2,
                    "spatial_merge_size": 2,
                    "out_hidden_size": hidden_size,
                },
                "audio_config": {"audio_channels": 20, "group_size": 4, "out_hidden_size": hidden_size},
                "processor_config": {
                    "image_min_pixels": 8192,
                    "image_max_pixels": 8388608,
                    "video_total_max_pixels": 268435456,
                    "audio_sampling_rate": 24000,
                    "audio_n_mels": 128,
                    "fps": 1.0,
                    "min_frames": None,
                    "max_frames": 3600,
                    "video_start_token_id": self.processor.video_start_token_id,
                    "video_end_token_id": self.processor.video_end_token_id,
                },
            }
            Path(directory, "config.json").write_text(json.dumps(config), encoding="utf-8")
            Path(directory, "preprocessor_config.json").write_text(
                json.dumps(
                    {"processor_class": "Qwen2_5_VLProcessor", "image_processor_type": "Qwen2VLImageProcessor"}
                ),
                encoding="utf-8",
            )
            p = MiMoV2Processor.from_pretrained(directory, local_files_only=True)
            self.assertEqual(p.image_processor.patch_size, 16)
            self.assertEqual(p.image_processor.size, {"shortest_edge": 8192, "longest_edge": 8388608})
            self.assertEqual(p.video_processor.total_max_pixels, 268435456)
            self.assertEqual(p.video_processor.fps, 1.0)
            self.assertEqual(p.video_processor.min_frames, 8)
            self.assertEqual(p.feature_extractor.audio_channels, 20)
            self.assertEqual(p.video_start_token, "<|mimo_video_start|>")
            output = p(text=p.image_token, images=np.zeros((64, 128, 3), dtype=np.uint8), return_tensors="pt")
            self.assertEqual(output.pixel_values.shape, (32, 1536))
            self.assertEqual(output.input_ids[0].tolist().count(p.image_token_id), 8)
            legacy_config = Path(directory, "preprocessor_config.json").read_bytes()
            tokenizer_config = Path(directory, "tokenizer_config.json").read_bytes()
            p.to_json_file(Path(directory, "processor_config.json"))
            auto = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            self.assertIsInstance(auto, MiMoV2Processor)
            self.assertEqual(auto.image_processor.patch_size, 16)
            self.assertEqual(auto.feature_extractor.audio_channels, 20)
            self.assertEqual(Path(directory, "preprocessor_config.json").read_bytes(), legacy_config)
            self.assertEqual(Path(directory, "tokenizer_config.json").read_bytes(), tokenizer_config)
            p.save_pretrained(directory)
            restored = AutoProcessor.from_pretrained(directory, local_files_only=True)
            self.assertEqual(restored.feature_extractor.audio_channels, 20)
            self.assertEqual(restored.image_processor.patch_size, 16)

    def test_legacy_tokenizer_class_preserves_serialized_backend(self):
        tokenizer = self.processor.tokenizer
        tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            config_path = Path(directory, "tokenizer_config.json")
            tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
            tokenizer_config["tokenizer_class"] = "Qwen2Tokenizer"
            config_path.write_text(json.dumps(tokenizer_config), encoding="utf-8")
            p = MiMoV2Processor.from_pretrained(directory, local_files_only=True)
            self.assertEqual(p(text="BEFORE TEXT").input_ids, tokenizer(["BEFORE TEXT"]).input_ids)
            serialized = json.loads(p.tokenizer.backend_tokenizer.to_str())
            self.assertEqual(serialized["model"]["type"], "WordLevel")
            self.assertEqual(serialized["normalizer"], {"type": "Lowercase"})
            self.assertEqual(
                serialized["post_processor"], json.loads(tokenizer.backend_tokenizer.to_str())["post_processor"]
            )

    @parameterized.expand([("image_processor",), ("video_processor",), ("feature_extractor",)])
    def test_partial_processor_config_preserves_saved_components(self, missing_component):
        processor = self.processor
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            processor_config_path = Path(directory, "processor_config.json")
            processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
            processor_config.pop(missing_component)
            processor_config_path.write_text(json.dumps(processor_config), encoding="utf-8")
            checkpoint_config = {
                "model_type": "mimo_v2",
                "vision_config": {"patch_size": 16},
                "audio_config": {"audio_channels": 20},
                "processor_config": {"image_token_id": len(processor.tokenizer) + 1},
            }
            Path(directory, "config.json").write_text(json.dumps(checkpoint_config), encoding="utf-8")

            restored = MiMoV2Processor.from_pretrained(directory, local_files_only=True)
            self.assertEqual(restored.image_processor.patch_size, 16 if missing_component == "image_processor" else 2)
            self.assertEqual(restored.video_processor.patch_size, 16 if missing_component == "video_processor" else 2)
            self.assertEqual(
                restored.feature_extractor.audio_channels, 20 if missing_component == "feature_extractor" else 8
            )
            self.assertEqual(restored.image_token, processor.image_token)

    def test_legacy_text_checkpoint_without_multimodal_tokens(self):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, "text": 1}, unk_token="<unk>")),
            unk_token="<unk>",
            chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
        )
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            Path(directory, "config.json").write_text(json.dumps({"model_type": "mimo_v2_flash"}), encoding="utf-8")
            p = MiMoV2Processor.from_pretrained(directory, local_files_only=True)
            self.assertEqual(p(text="text").input_ids, tokenizer(["text"]).input_ids)
            self.assertEqual(p.apply_chat_template([{"role": "user", "content": "text"}], tokenize=True), [[1]])
            with self.assertRaisesRegex(ValueError, "missing.*audio"):
                p(text=p.audio_token, audio=self.codes, is_tokenized=True)
            p.save_pretrained(directory)
            self.assertEqual(AutoProcessor.from_pretrained(directory)(text="text").input_ids, [[1]])

    @staticmethod
    def prompt_ordered_codes(out):
        """Rebuild the prompt-ordered codec frames selected by `audio_segment_offsets`."""
        codes = np.asarray(out["input_audio_codes"])
        segments = [
            codes[index][start : start + length] for index, start, length in np.asarray(out["audio_segment_offsets"])
        ]
        return np.concatenate(segments).reshape(-1, codes.shape[-1])

    @parameterized.expand([("pt",), ("np",)])
    def test_audio_segment_offsets_unpad_and_align_audio(self, tensor_type):
        p = self.processor
        out = p(
            text=[p.audio_token, p.video_token],
            videos=self.video,
            audio=[self.codes[:5], self.codes],
            is_tokenized=True,
            use_audio_in_video=True,
            padding=True,
            return_tensors=tensor_type,
            **self.video_kwargs(),
        )
        expected = np.concatenate([self.codes[:5], np.repeat(self.codes[4:5], 3, axis=0), self.codes[:60]])
        np.testing.assert_array_equal(self.prompt_ordered_codes(out), expected)
        self.assertEqual(np.asarray(out.input_ids).tolist()[1].count(p.audio_token_id), 15)

    def test_single_frame_video_and_empty_final_audio_segment(self):
        p = self.processor
        out = p(
            text=p.video_token,
            videos=self.video[:1],
            video_metadata=VideoMetadata(total_num_frames=1, fps=2),
            return_tensors="pt",
            input_data_format="channels_last",
        )
        self.assertEqual(out.video_grid_thw.tolist(), [[1, 2, 4]])
        # A video grid exactly at the end of its audio may have no audio tokens.
        out = p(
            text=p.video_token,
            videos=self.video,
            audio=self.codes[:48],
            is_tokenized=True,
            use_audio_in_video=True,
            return_tensors="pt",
            **self.video_kwargs(),
        )
        self.assertEqual(out.audio_segment_offsets[-1].tolist(), [0, 12, 0])
        self.assertEqual(self.prompt_ordered_codes(out).shape, (48, 8))

    @parameterized.expand([("list",), ("tuple",), ("numpy",), ("torch",)])
    def test_batched_codec_input_containers(self, container):
        arrays = np.stack([self.codes[:9], self.codes[1:10]])
        inputs = {
            "list": arrays.tolist(),
            "tuple": tuple(arrays),
            "numpy": arrays,
            "torch": torch.from_numpy(arrays),
        }[container]
        p = self.processor
        actual = p(text=[p.audio_token] * 2, audio=inputs, is_tokenized=True, padding=True, return_tensors="pt")
        expected = p(
            text=[p.audio_token] * 2, audio=list(arrays), is_tokenized=True, padding=True, return_tensors="pt"
        )
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    @parameterized.expand([("list",), ("tuple",)])
    def test_single_codec_python_sequence(self, container):
        codes = self.codes[:5].tolist()
        if container == "tuple":
            codes = tuple(tuple(row) for row in codes)
        actual = self.processor.feature_extractor(codes, is_tokenized=True, return_tensors="np")
        expected = self.processor.feature_extractor(self.codes[:5], is_tokenized=True, return_tensors="np")
        np.testing.assert_array_equal(actual.input_audio_codes, expected.input_audio_codes)

    @parameterized.expand(
        [
            ("empty_batch", []),
            ("empty_array_batch", np.empty((0, 4, 8), dtype=np.int64)),
            ("empty_sequence", np.empty((0, 8), dtype=np.int64)),
            ("float_codes", np.zeros((4, 8), dtype=np.float32)),
            ("missing_channels", np.zeros((4, 7), dtype=np.int64)),
        ]
    )
    def test_invalid_codec_inputs(self, _, codes):
        with self.assertRaisesRegex(ValueError, "Audio"):
            self.processor.feature_extractor(codes, is_tokenized=True)

    @parameterized.expand([(float("nan"),), (float("inf"),), (-float("inf"),)])
    def test_nonfinite_waveforms_and_video_rates(self, value):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.processor.feature_extractor(np.full(1000, value, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "fps"):
            self.processor.video_processor.sample_frames(VideoMetadata(total_num_frames=8, fps=value), fps=2)
        with self.assertRaisesRegex(ValueError, "fps"):
            self.processor.video_processor.sample_frames(VideoMetadata(total_num_frames=8, fps=2), fps=value)
        with self.assertRaisesRegex(ValueError, "num_frames"):
            self.processor.video_processor.sample_frames(VideoMetadata(total_num_frames=8, fps=2), num_frames=value)
        with self.assertRaisesRegex(ValueError, "fps"):
            self.processor(
                text=self.processor.video_token,
                videos=self.video,
                do_sample_frames=False,
                input_data_format="channels_last",
                video_metadata=VideoMetadata(total_num_frames=5, fps=value),
            )

    def test_empty_waveform_batch(self):
        with self.assertRaisesRegex(ValueError, "at least one waveform"):
            self.processor.feature_extractor([])

    def test_spectral_cache_configuration_and_serialization(self):
        waveform = np.random.default_rng(0).standard_normal(24000).astype(np.float32)
        extractor = self.processor.feature_extractor
        original = extractor(waveform, return_tensors="np").input_features
        extractor.f_max = 8000
        changed = extractor(waveform, return_tensors="np").input_features
        expected = MiMoV2FeatureExtractor(f_max=8000)(waveform, return_tensors="np").input_features
        np.testing.assert_array_equal(changed, expected)
        self.assertFalse(np.array_equal(original, changed))
        with tempfile.TemporaryDirectory() as directory:
            self.processor.save_pretrained(directory)
            restored = AutoProcessor.from_pretrained(directory, local_files_only=True)
            np.testing.assert_array_equal(
                restored.feature_extractor(waveform, return_tensors="np").input_features, changed
            )

    def test_reversed_waveform(self):
        waveform = np.random.default_rng(0).standard_normal(2400).astype(np.float32)[::-1]
        extractor = self.processor.feature_extractor
        actual = extractor(waveform, return_tensors="np")
        expected = extractor(waveform.copy(), return_tensors="np")
        np.testing.assert_array_equal(actual.input_features, expected.input_features)

    @parameterized.expand([("token",), ("tokenizer",)])
    def test_token_validation_respects_changes(self, changed):
        p = self.processor
        p(text=p.image_token, images=self.image)
        if changed == "token":
            p.image_token = "text"
        else:
            p.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, "text": 1}, unk_token="<unk>")),
                unk_token="<unk>",
            )
        with self.assertRaisesRegex(ValueError, "special token"):
            p(images=self.image)

    @parameterized.expand([("image", 0), ("image", 1), ("video", 0), ("video", 2)])
    def test_flipped_numpy_media(self, modality, axis):
        processor = self.processor
        shape = self.image.shape if modality == "image" else self.video.shape
        value = np.flip(np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8), axis=axis)
        kwargs = self.video_kwargs() if modality == "video" else {}
        kwargs.update(input_data_format="channels_last", return_tensors="pt")
        media_key = "images" if modality == "image" else "videos"
        token = getattr(processor, f"{modality}_token")
        actual = processor(text=token, **{media_key: value}, **kwargs)
        expected = processor(text=token, **{media_key: value.copy()}, **kwargs)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    def test_url_messages_load_all_modalities_without_mutation(self, batched, nested_url):
        """Verify normalization and loader arguments without making network requests."""
        processor = make_processor(chat_template=MULTIMODAL_CHAT_TEMPLATE)
        image_url = hf_hub_url(*MEDIA_FIXTURES["image"], repo_type="dataset")
        audio_url = hf_hub_url(*MEDIA_FIXTURES["audio"], repo_type="dataset")
        video_url = hf_hub_url(*MEDIA_FIXTURES["video"], repo_type="dataset")

        def content_block(modality, url):
            return {"type": f"{modality}_url", f"{modality}_url": {"url": url} if nested_url else url}

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    *[content_block("image", image_url) for _ in range(3)],
                    content_block("audio", audio_url),
                    content_block("video", video_url),
                ],
            }
        ]
        if batched:
            messages = [messages]
        original = copy.deepcopy(messages)
        with (
            patch.object(processor.image_processor, "fetch_images", return_value=[self.image] * 3) as fetch_images,
            patch.object(
                processor.video_processor,
                "fetch_videos",
                return_value=([self.video], [self.video_kwargs()["video_metadata"]]),
            ) as fetch_videos,
            patch("transformers.processing_utils.load_audio", return_value=np.zeros(24000, dtype=np.float32)) as load,
        ):
            output = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"videos_kwargs": {"do_sample_frames": False, "input_data_format": "channels_last"}},
            )
        self.assertEqual(messages, original)
        self.assertEqual(output.image_grid_thw.shape[0], 3)
        self.assertEqual(output.video_grid_thw.shape[0], 1)
        self.assertEqual(output.input_features.shape[0], 1)
        self.assertEqual(fetch_images.call_args_list[0].args, ([[image_url] * 3],))
        self.assertEqual(fetch_videos.call_count, 1)
        self.assertEqual(fetch_videos.call_args.args[0], [video_url])
        load.assert_called_once_with(
            audio_url, sampling_rate=processor.feature_extractor.sampling_rate, backend="auto"
        )

    def test_chat_template_merges_processor_kwargs(self):
        p = make_processor(chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}")
        options = {"text_kwargs": {"padding": "max_length", "max_length": 12}}
        original = copy.deepcopy(options)
        output = p.apply_chat_template(
            [{"role": "user", "content": "text"}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=options,
            sampling_rate=24000,
        )
        self.assertEqual(output.input_ids.shape, (1, 12))
        self.assertEqual(options, original)

    def test_assistant_mask_after_image_expansion(self):
        p = self.processor
        template = (
            "{% for message in messages %}{% if message['role'] == 'assistant' %}"
            "{% generation %}{{ message['content'] }}{% endgeneration %}"
            "{% else %}<|vision_start|><|image_pad|><|vision_end|> {% endif %}{% endfor %}"
        )
        messages = [
            {"role": "user", "content": [{"type": "image", "image": self.image}]},
            {"role": "assistant", "content": "after"},
        ]
        output = p.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
        )
        selected = output.input_ids[output.assistant_masks.bool()].tolist()
        self.assertEqual(selected, p.tokenizer.encode("after", add_special_tokens=False))

    def test_mixed_media_batch_processor_round_trip(self):
        processor = make_processor(chat_template=MULTIMODAL_CHAT_TEMPLATE)
        kwargs = {
            "text": [processor.image_token + processor.audio_token + processor.video_token, "text"],
            "images": self.image,
            "audio": np.zeros(24000, dtype=np.float32),
            "videos": self.video,
            "sampling_rate": 24000,
            "padding": True,
            "return_tensors": "pt",
            **self.video_kwargs(),
        }
        expected = processor(**kwargs)
        self.assertEqual(expected.input_ids.shape[0], 2)
        self.assertEqual(expected.input_ids[0].eq(processor.image_token_id).sum().item(), 2)
        self.assertEqual(expected.input_ids[0].eq(processor.video_token_id).sum().item(), 6)
        self.assertEqual(expected.input_ids[0].eq(processor.audio_token_id).sum().item(), 7)
        self.assertLess(expected.attention_mask[1].sum(), expected.attention_mask[0].sum())
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            restored = AutoProcessor.from_pretrained(directory, local_files_only=True)
            self.assertIsInstance(restored, MiMoV2Processor)
            self.assertEqual(restored.chat_template, processor.chat_template)
            actual = restored(**kwargs)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@slow
@require_torch
@require_torchvision
@require_vision
@require_torchcodec
class MiMoV2ProcessorIntegrationTest(unittest.TestCase):
    """Exercise real Hub downloads and media decoders without downloading model weights."""

    model_id = "XiaomiMiMo/MiMo-V2.6-Flash-RL"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.processor = AutoProcessor.from_pretrained(cls.model_id, trust_remote_code=False)
        cls.media_paths = {
            modality: hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            for modality, (repo_id, filename) in MEDIA_FIXTURES.items()
        }

    @parameterized.expand([("image",), ("audio",), ("video",)])
    def test_chat_template_single_modality(self, modality):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": modality, "path": self.media_paths[modality]},
                    {"type": "text", "text": "Describe this input."},
                ],
            }
        ]
        output = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True
        )
        feature_key = {"image": "pixel_values", "audio": "input_features", "video": "pixel_values_videos"}[modality]
        self.assertEqual(output.input_ids.shape[0], 1)
        self.assertGreater(output.input_ids.eq(getattr(self.processor, f"{modality}_token_id")).sum().item(), 0)
        self.assertGreater(output[feature_key].numel(), 0)
        self.assertTrue(torch.isfinite(output[feature_key]).all())

    @parameterized.expand([("url", False), ("url", True), ("alias", False), ("nested_alias", False)])
    def test_mixed_media_urls_match_files_and_saved_processor(self, content_format, batched):
        self.assertIsInstance(self.processor, MiMoV2Processor)
        url_content = []
        for modality, (repo_id, filename) in MEDIA_FIXTURES.items():
            url = hf_hub_url(repo_id, filename, repo_type="dataset")
            if content_format == "url":
                url_content.append({"type": modality, "url": url})
            else:
                value = {"url": url} if content_format == "nested_alias" else url
                url_content.append({"type": f"{modality}_url", f"{modality}_url": value})
        url_messages = [
            {
                "role": "user",
                "content": [
                    *url_content,
                    {"type": "text", "text": "Describe what you can see and hear."},
                ],
            }
        ]
        path_messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": modality, "path": path} for modality, path in self.media_paths.items()],
                    {"type": "text", "text": "Describe what you can see and hear."},
                ],
            }
        ]
        if batched:
            text_messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
            url_messages = [url_messages, text_messages]
            path_messages = [path_messages, copy.deepcopy(text_messages)]
        original = copy.deepcopy(url_messages)
        kwargs = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
            "processor_kwargs": {"text_kwargs": {"padding": True}},
        }
        actual = self.processor.apply_chat_template(url_messages, **kwargs)
        expected = self.processor.apply_chat_template(path_messages, **kwargs)
        self.assertEqual(url_messages, original)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(actual.input_ids.shape[0], 2 if batched else 1)
        self.assertEqual(actual.image_grid_thw.shape[0], 1)
        self.assertEqual(actual.video_grid_thw.shape[0], 1)
        self.assertEqual(actual.input_features.shape[0], 1)
        self.assertTrue(torch.isfinite(actual.pixel_values).all())
        self.assertTrue(torch.isfinite(actual.pixel_values_videos).all())
        self.assertTrue(torch.isfinite(actual.input_features).all())
        self.assertEqual(actual.input_ids.eq(self.processor.audio_token_id).sum(), actual.audio_feature_lens.sum())
        merge_size = self.processor.image_processor.merge_size
        self.assertEqual(
            actual.input_ids.eq(self.processor.image_token_id).sum(),
            actual.image_grid_thw.prod(-1).sum() // merge_size**2,
        )
        merge_size = self.processor.video_processor.merge_size
        self.assertEqual(
            actual.input_ids.eq(self.processor.video_token_id).sum(),
            actual.video_grid_thw.prod(-1).sum() // merge_size**2,
        )

        with tempfile.TemporaryDirectory() as directory:
            self.processor.save_pretrained(directory)
            restored = AutoProcessor.from_pretrained(directory, local_files_only=True)
            self.assertIsInstance(restored, MiMoV2Processor)
            self.assertEqual(restored.chat_template, self.processor.chat_template)
            reloaded = restored.apply_chat_template(path_messages, **kwargs)
            self.assertEqual(set(reloaded), set(expected))
            for key in expected:
                torch.testing.assert_close(reloaded[key], expected[key], rtol=0, atol=0)
