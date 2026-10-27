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

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from transformers import (
    AutoProcessor,
    Lfm3AudioFeatureExtractor,
    Lfm3OmniProcessor,
    Lfm3VlVideoProcessor,
    LiquidCodecConfig,
    LiquidCodecModel,
    Qwen2VLImageProcessor,
    is_torch_available,
)
from transformers.testing_utils import require_torch, require_torchcodec, require_vision

from ...test_processing_common import ProcessorTesterMixin
from ..lfm3_audio.test_processing_lfm3_audio import (
    GENERATION_TEMPLATE,
    assert_offsets_rebuild_the_prompts,
    expected_assistant_mask,
    tiny_codec,
    tiny_tokenizer,
)


if is_torch_available():
    import torch


TINY_CODEC_BACKBONE = {
    "vocab_size": 1,
    "pad_token_id": None,
    "bos_token_id": None,
    "eos_token_id": None,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "use_cache": False,
}


# the hub chat template reduced to what the tests render: ChatML turns, one bare placeholder per media item (the hub
# template wraps them in the vision span pair, which the processor strips before expanding; see the H5 parity report)
CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['content'] is string -%}{{- message['content'] -}}"
    "{%- else -%}{%- for content in message['content'] -%}"
    "{%- if content['type'] == 'image' -%}{{- '<|image|>' -}}"
    "{%- elif content['type'] == 'video' -%}{{- '<|video|>' -}}"
    "{%- elif content['type'] == 'audio' -%}{{- '<|audio|>' -}}"
    "{%- elif content['type'] == 'audio_out' -%}{{- '<|audio_out|>' -}}"
    "{%- elif content['type'] == 'text' -%}{{- content['text'] -}}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)


def tiny_image_processor():
    # the released patch, merge and normalization values with a small pixel budget
    return Qwen2VLImageProcessor(
        size={"shortest_edge": 1024, "longest_edge": 4096},
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


@require_torch
@require_vision
class Lfm3OmniProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Lfm3OmniProcessor

    @classmethod
    def _setup_tokenizer(cls):
        return tiny_tokenizer()

    @classmethod
    def _setup_image_processor(cls):
        return tiny_image_processor()

    @classmethod
    def _setup_video_processor(cls):
        return Lfm3VlVideoProcessor(size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64 * 4})

    @classmethod
    def _setup_feature_extractor(cls):
        return Lfm3AudioFeatureExtractor()

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "expected_dim": 0, "output_length": 32},
            {"num_frames": None, "fps": 18, "expected_dim": 0, "output_length": 12},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 24},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 24},
            {"expected_dim": 0, "output_length": 16},
        ]

    # LFM3-Omni prompts carry frame timestamps, so decoded frames are refused unless `video_metadata` gives the clock
    # (a path or URL is decoded with its own); the shared tests below pass bare frame arrays
    DECODED_FRAMES_REASON = "LFM3-Omni refuses decoded frames without `video_metadata` (the prompt carries timestamps)"

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_apply_chat_template_decoded_video_0(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_apply_chat_template_video_frame_sampling(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_flat_kwarg_applied_when_modality_dict_lacks_it(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_kwargs_overrides_default_subprocessor_kwargs_1_videos(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_model_input_names(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_processor_with_multiple_inputs(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_replacement_offsets(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_structured_kwargs_nested_from_dict_1_videos(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_subprocessor_defaults_preserved_by_kwargs_1_videos(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_unstructured_kwargs_1_videos(self):
        pass

    @unittest.skip(DECODED_FRAMES_REASON)
    def test_unstructured_kwargs_batched_1_videos(self):
        pass

    @unittest.skip(
        "`use_audio_in_video` lives in `videos_kwargs` but is a processor option, not a video-processor one"
    )
    def test_subprocessor_defaults_2_videos(self):
        pass

    @staticmethod
    def prepare_processor_dict():
        return {"chat_template": CHAT_TEMPLATE}

    def setUp(self):
        self.processor = Lfm3OmniProcessor(
            tiny_image_processor(),
            Lfm3AudioFeatureExtractor(),
            tiny_tokenizer(),
            video_processor=Lfm3VlVideoProcessor(size={"shortest_edge": 1024, "longest_edge": 4096}),
            chat_template=CHAT_TEMPLATE,
        )
        self.image = Image.new("RGB", (64, 32))
        self.audio = np.zeros(4000, dtype=np.float32)

    def test_interleaved_modalities_keep_order_and_counts(self):
        proc = self.processor
        inputs = proc(
            text=["<|image|> Hello <|audio|>", "<|audio|> Hello <|image|>"],
            images=[self.image, self.image],
            audio=[self.audio, self.audio],
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
        )
        image_count = inputs.image_grid_thw[0].prod() // proc.image_processor.merge_size**2
        audio_count = proc.feature_extractor.num_encoder_tokens(proc.feature_extractor.num_frames(len(self.audio)))
        for row in inputs.input_ids:
            self.assertEqual(int((row == proc.image_token_id).sum()), int(image_count))
            self.assertEqual(int((row == proc.audio_token_id).sum()), audio_count)
        for row, first, last in (
            (0, proc.image_token_id, proc.audio_token_id),
            (1, proc.audio_token_id, proc.image_token_id),
        ):
            ids = inputs.input_ids[row]
            self.assertLess(int((ids == first).nonzero()[0]), int((ids == last).nonzero()[0]))
        self.assertTrue(set(inputs).issubset(proc.model_input_names))
        self.assertIn("audio_out_codes", proc.model_input_names)

    def test_chat_template_keeps_multimodal_content(self):
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hello"}, {"type": "audio"}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertIn("<|image|>Hello<|audio|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        # the override keeps the shared positional signature (conversation, chat_template, tools, documents, ...)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        self.assertEqual(
            self.processor.apply_chat_template(messages, None, [], None, True),
            self.processor.apply_chat_template(messages, tools=[], add_generation_prompt=True),
        )

    def test_invalid_media_rejected(self):
        for kwargs in (
            {"text": "<|image|>"},
            {"text": "<|audio|>", "sampling_rate": 16000},
            {"text": "<|image|><|image|>", "images": [self.image]},
            {"text": "Hello", "videos": [np.zeros((2, 32, 32, 3), dtype=np.uint8)]},
        ):
            with self.subTest(kwargs=kwargs.keys()), self.assertRaises(ValueError):
                self.processor(**kwargs)

    def test_video_path_accepts_the_same_audio_inputs_as_the_shared_path(self):
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        video = {
            "videos": [frames],
            "video_metadata": [VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])],
        }
        # a clip given as a path is decoded by the shared layout step, whether or not a video is in the prompt
        with patch("transformers.feature_extraction_sequence_utils.load_audio", return_value=self.audio) as loader:
            from_path = self.processor(text="<|video|> <|audio|>", audio=["clip.wav"], sampling_rate=16000, **video)
            self.assertEqual(loader.call_count, 1)
        from_array = self.processor(text="<|video|> <|audio|>", audio=[self.audio], sampling_rate=16000, **video)
        self.assertEqual(from_path["input_ids"], from_array["input_ids"])
        self.assertEqual(from_path["input_features"].shape, from_array["input_features"].shape)
        # the deprecated kwarg is refused instead of silently dropping the clip
        with self.assertRaisesRegex(ValueError, "deprecated"):
            self.processor(text="<|video|>", audios=[self.audio], **video)
        with self.assertRaisesRegex(ValueError, "Found 1 <\\|audio\\|> tokens in the text but received 2 clips"):
            self.processor(text="<|video|> <|audio|>", audio=[self.audio, self.audio], **video)
        with self.assertRaisesRegex(ValueError, "Found 1 <\\|image\\|> tokens in the text but received 0 images"):
            self.processor(text="<|video|> <|image|>", **video)

    def test_video_metadata_without_fps_is_refused(self):
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        # the shared sampler would otherwise assume 24 fps and build the prompt timestamps from it
        for metadata in (
            VideoMetadata(total_num_frames=4, fps=None, frames_indices=[0, 1, 2, 3]),
            {"total_num_frames": 4, "frames_indices": [0, 1, 2, 3]},
        ):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "must hold the clip's fps"):
                self.processor(text="<|video|>", videos=[frames], video_metadata=[metadata])
            # a media-only call takes the shared path and is held to the same clock
            with self.subTest(metadata=metadata, text=None), self.assertRaisesRegex(ValueError, "the clip's fps"):
                self.processor(videos=[frames], video_metadata=[metadata])
        with self.assertRaisesRegex(ValueError, "pass decoded frames with `video_metadata`"):
            self.processor(videos=[frames])
        for text in ("<|video|>", None):
            outputs = self.processor(text=text, videos=[frames], video_metadata={"total_num_frames": 4, "fps": 2.0})
            self.assertIn("video_grid_thw", outputs)

    def test_nested_images_are_counted_per_prompt(self):
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])
        message = "Found \\[2, 0\\] <\\|image\\|> tokens per prompt but received \\[1, 1\\] images per prompt"
        for video, media in (("", {}), ("<|video|>", {"videos": [frames], "video_metadata": [metadata]})):
            with self.subTest(video=video), self.assertRaisesRegex(ValueError, message):
                self.processor(
                    text=[video + "<|image|><|image|>", "Hello"],
                    images=[[self.image], [self.image]],
                    padding=True,
                    **media,
                )
            inputs = self.processor(
                text=[video + "<|image|>", "Hello"], images=[[self.image], []], padding=True, **media
            )
            self.assertEqual(len(inputs["image_grid_thw"]), 1)

    def test_media_rows_are_matched_per_prompt(self):
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])
        image = np.asarray(self.image)
        for video, media in (("", {}), ("<|video|>", {"videos": [frames], "video_metadata": [metadata]})):
            for images in ([image[None], image[None]], [[self.image], [self.image], []]):
                with self.subTest(video=video, images=len(images)), self.assertRaisesRegex(ValueError, "per prompt"):
                    self.processor(text=[video + "<|image|><|image|>", "Hello"], images=images, padding=True, **media)
        # the shape the batched `apply_chat_template` passes: one row of videos per conversation
        message = "Found \\[2, 0\\] <\\|video\\|> tokens per prompt but received \\[1, 1\\] videos per prompt"
        with self.assertRaisesRegex(ValueError, message):
            self.processor(
                text=["<|video|><|video|>", "Hello"],
                videos=[[frames], [frames]],
                video_metadata=[metadata, metadata],
                padding=True,
            )
        inputs = self.processor(
            text=["<|video|>", "<|video|> Hello"],
            videos=[[frames], [frames]],
            video_metadata=[metadata, metadata],
            padding=True,
        )
        self.assertEqual(len(inputs["video_grid_thw"]), 2)

    def test_video_metadata_needs_one_entry_per_video(self):
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = {"total_num_frames": 4, "fps": 2.0}
        # the shared sampler pairs clips with metadata by `zip`: a media-only call would return one clip's features
        for text, videos, video_metadata in (
            ("<|video|><|video|>", [frames, frames], metadata),
            ("<|video|><|video|>", [frames, frames], [metadata]),
            ("<|video|>", [frames], [metadata, metadata]),
            ("<|video|>", [frames], []),
        ):
            for prompt in (text, None):
                with (
                    self.subTest(prompt=prompt, video_metadata=video_metadata),
                    self.assertRaisesRegex(ValueError, "`video_metadata` entries; pass one entry per video"),
                ):
                    self.processor(text=prompt, videos=videos, video_metadata=video_metadata)

    def test_truncated_speech_placeholders_are_rejected(self):
        from transformers.video_utils import VideoMetadata

        self.processor.audio_tokenizer = tiny_codec()
        clips = [np.zeros(1920 * 3, dtype=np.float32)]
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])
        for video, media in (("", {}), ("<|video|>", {"videos": [frames], "video_metadata": [metadata]})):
            prompt = video + "<|audio_out|> Hello"
            ids = self.processor(text=prompt, audio_out=clips, **media)["input_ids"][0]
            # keep the video and one of the 4 speech placeholders
            max_length = ids.index(self.processor.audio_out_token_id) + 1
            with self.subTest(video=video), self.assertRaisesRegex(ValueError, "after tokenization"):
                self.processor(text=prompt, audio_out=clips, truncation=True, max_length=max_length, **media)

    def test_empty_media_lists_are_refused(self):
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        video = {
            "videos": [frames],
            "video_metadata": [VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])],
        }
        for kwargs in (
            {"text": "Hello", "videos": []},
            {"text": "Hello", "images": []},
            {"text": "<|video|>", "audio": [], **video},
            {"text": "Hello", "audio_out": []},
            {"images": []},
        ):
            with self.subTest(kwargs=kwargs.keys()), self.assertRaisesRegex(ValueError, "must hold at least one"):
                self.processor(**kwargs)

    def test_media_only_calls_return_the_features(self):
        outputs = self.processor(audio=[self.audio], sampling_rate=16000, return_tensors="pt")
        self.assertEqual(set(outputs), {"input_features", "feature_attention_mask"})
        outputs = self.processor(images=[self.image], return_tensors="pt")
        self.assertIn("pixel_values", outputs)
        with self.assertRaises(ValueError):
            self.processor(audio_out=[self.audio])

    def test_chat_template_renders_videos(self):
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Hello"}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.assertIn("<|video|>Hello", prompt)

    def test_processor_codec_roundtrip_after_move(self):
        self.processor.audio_tokenizer = LiquidCodecModel(
            LiquidCodecConfig(
                encoder_config=TINY_CODEC_BACKBONE,
                decoder_config=TINY_CODEC_BACKBONE,
                encoder_conv_dim=8,
                quantizer_dim=16,
                n_q=4,
                fsq_levels=[3, 3],
            )
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            original, moved = Path(directory, "original"), Path(directory, "moved")
            self.processor.save_pretrained(original)
            original.rename(moved)
            reloaded = AutoProcessor.from_pretrained(moved, local_files_only=True)
        self.assertIsInstance(reloaded, Lfm3OmniProcessor)
        self.assertEqual(reloaded.chat_template, self.processor.chat_template)
        self.assertEqual(reloaded.model_input_names, self.processor.model_input_names)
        original = self.processor(
            text="<|image|><|audio_out|>", images=[self.image], audio_out=[self.audio], return_tensors="pt"
        )
        restored = reloaded(
            text="<|image|><|audio_out|>", images=[self.image], audio_out=[self.audio], return_tensors="pt"
        )
        for key in original:
            torch.testing.assert_close(original[key], restored[key], rtol=0, atol=0)

    def test_media_spans_wrap_images_and_videos_like_the_sft_data(self):
        inputs = self.processor(
            text="<|image|> and <|audio|>", images=[self.image], audio=[self.audio], return_tensors="pt"
        )
        ids = inputs["input_ids"][0].tolist()
        tokenizer = self.processor.tokenizer
        start, end = (
            tokenizer.convert_tokens_to_ids(self.processor.vision_start_token),
            tokenizer.convert_tokens_to_ids(self.processor.vision_end_token),
        )
        image_positions = [i for i, token in enumerate(ids) if token == self.processor.image_token_id]
        self.assertEqual(ids[image_positions[0] - 1], start)
        self.assertEqual(ids[image_positions[-1] + 1], end)
        audio_positions = [i for i, token in enumerate(ids) if token == self.processor.audio_token_id]
        self.assertEqual(ids[audio_positions[0] - 1], start)
        self.assertEqual(ids[audio_positions[-1] + 1], end)
        # a video is one timestamped, wrapped group per temporal patch, wrapped again as a whole
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        from transformers.video_utils import VideoMetadata

        metadata = VideoMetadata(
            total_num_frames=4, fps=2.0, duration=2.0, video_backend="np", frames_indices=[0, 1, 2, 3]
        )
        outputs = self.processor(text="<|video|>", videos=[frames], video_metadata=[metadata], return_tensors="pt")
        ids = outputs["input_ids"][0].tolist()
        grid = outputs["video_grid_thw"][0]
        per_group = int(grid[1] * grid[2]) // self.processor.video_processor.merge_size**2
        self.assertEqual(ids.count(self.processor.video_token_id), int(grid[0]) * per_group)
        self.assertEqual(ids.count(start), 1 + int(grid[0]))
        self.assertEqual(ids.count(end), 1 + int(grid[0]))
        # the clip opens with the outer span, then a timestamp precedes the first group's inner span
        self.assertEqual(ids[0], start)
        self.assertNotIn(ids[1], (start, end, self.processor.video_token_id))
        self.assertEqual(self.processor._calculate_timestamps([0, 1, 2, 3], 2.0, 2), [0.25, 1.25])
        with self.assertRaises(ValueError):
            self.processor(text="no placeholder", videos=[frames], video_metadata=[metadata])

    def test_video_soundtrack_is_split_across_frame_groups(self):
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(
            total_num_frames=4, fps=2.0, duration=2.0, video_backend="np", frames_indices=[0, 1, 2, 3]
        )
        soundtrack = np.zeros(16000 * 2, dtype=np.float32)  # 2 s = 200 mel frames = 25 encoder tokens
        video = {"videos": [frames], "video_metadata": [metadata]}
        audio_id = self.processor.audio_token_id

        def audio_runs(ids):
            runs, run = [], 0
            for token in ids:
                run += token == audio_id
                if token != audio_id and run:
                    runs.append(run)
                    run = 0
            return runs

        with self.assertRaises(ValueError):  # an image placeholder without an image
            self.processor(text="<|image|><|video|><|audio|>", audio=[soundtrack], use_audio_in_video=True, **video)
        with self.assertRaises(ValueError):  # no clip after the video to take as its soundtrack
            self.processor(text="<|audio|><|video|>", audio=[soundtrack], use_audio_in_video=True, **video)
        # the soundtrack placeholder may sit after the text, where `apply_chat_template(load_audio_from_video=True)`
        # appends it
        text = "<|video|> what is said? <|audio|> and <|audio|>"
        outputs = self.processor(
            text=text, audio=[soundtrack, self.audio], return_tensors="pt", use_audio_in_video=True, **video
        )
        ids = outputs["input_ids"][0].tolist()
        grid = outputs["video_grid_thw"][0]
        per_group = int(grid[1] * grid[2]) // self.processor.video_processor.merge_size**2
        self.assertEqual(ids.count(self.processor.video_token_id), int(grid[0]) * per_group)
        soundtrack_tokens = self.processor.feature_extractor.num_encoder_tokens(
            int(outputs["feature_attention_mask"][0].sum())
        )
        standalone_tokens = self.processor.feature_extractor.num_encoder_tokens(
            int(outputs["feature_attention_mask"][1].sum())
        )
        # the first frame group starts at 0 s and the second at 1 s: 1 s / 0.08 s per token = 12 tokens, then the rest
        self.assertEqual(audio_runs(ids), [12, soundtrack_tokens - 12, standalone_tokens])
        self.assertEqual(outputs["input_features"].shape[0], 2)
        # without the flag every clip is its own media span, even right after a video
        outputs = self.processor(text=text, audio=[soundtrack, self.audio], return_tensors="pt", **video)
        self.assertEqual(audio_runs(outputs["input_ids"][0].tolist()), [soundtrack_tokens, standalone_tokens])
        for use_audio_in_video, runs in ((False, [soundtrack_tokens]), (True, [12, soundtrack_tokens - 12])):
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"}, {"type": "audio"}]}], tokenize=False
            )
            outputs = self.processor(
                text=prompt, audio=[soundtrack], return_tensors="pt", use_audio_in_video=use_audio_in_video, **video
            )
            self.assertEqual(audio_runs(outputs["input_ids"][0].tolist()), runs)

    def test_video_output_options_and_replacement_offsets(self):
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])
        prompts = ["Hello <|image|> <|video|> and <|audio|>", "Hello <|video|>"]
        media = {
            "images": [self.image],
            "audio": [self.audio],
            "videos": [frames, frames],
            "video_metadata": [metadata, metadata],
        }
        for tensor_type in ("pt", "np"):
            for structured in (False, True):
                with self.subTest(tensor_type=tensor_type, structured=structured):
                    options = {
                        "return_tensors": tensor_type,
                        "padding": True,
                        "return_mm_token_type_ids": True,
                        "return_text_replacement_offsets": True,
                    }
                    kwargs = (
                        {"text_kwargs": options, "videos_kwargs": {"return_metadata": True}}
                        if structured
                        else dict(options, return_metadata=True)
                    )
                    outputs = self.processor(text=prompts, **media, **kwargs)
                    self.assertEqual(len(outputs["video_metadata"]), 2)
                    self.assertEqual(outputs["video_metadata"][0].fps, 2.0)
                    offsets = outputs["text_replacement_offsets"]
                    self.assertEqual(
                        [[o["type"] for o in row] for row in offsets], [["image", "video", "audio"], ["video"]]
                    )
                    expanded = []
                    for prompt, row in zip(prompts, offsets):
                        rebuilt = prompt
                        for replacement in reversed(row):
                            start, end = replacement["span"]
                            self.assertEqual(prompt[start:end], replacement["text"])
                            rebuilt = rebuilt[:start] + replacement["replacement"] + rebuilt[end:]
                        for replacement in row:
                            start, end = replacement["new_span"]
                            self.assertEqual(rebuilt[start:end], replacement["replacement"])
                        expanded.append(rebuilt)
                    expected = self.processor.tokenizer(expanded, padding=True)["input_ids"]
                    self.assertEqual(outputs["input_ids"].tolist(), expected)
                    ids = np.asarray(outputs["input_ids"])
                    expected_types = np.zeros_like(ids)
                    for token, kind in (
                        (self.processor.image_token_id, 1),
                        (self.processor.video_token_id, 2),
                        (self.processor.audio_token_id, 3),
                    ):
                        expected_types[ids == token] = kind
                    np.testing.assert_array_equal(outputs["mm_token_type_ids"], expected_types)
        video = {"videos": [frames], "video_metadata": [metadata]}
        for kwargs in ({}, {"return_metadata": False}, {"videos_kwargs": {"return_metadata": False}}):
            outputs = self.processor(text="<|video|>", **video, **kwargs)
            self.assertNotIn("video_metadata", outputs)
        for kwargs in ({"truncation": True, "max_length": 2}, {"text_kwargs": {"truncation": True, "max_length": 2}}):
            with self.assertRaisesRegex(ValueError, "Mismatch in `video` token count"):
                self.processor(text="<|video|>", **video, **kwargs)

    def test_video_soundtrack_replacement_offsets_name_both_placeholders(self):
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])
        prompt = "Hello <|video|><|audio|> and <|audio|>"
        outputs = self.processor(
            text=prompt,
            videos=[frames],
            video_metadata=[metadata],
            audio=[self.audio, self.audio],
            use_audio_in_video=True,
            return_text_replacement_offsets=True,
            return_tensors="pt",
        )
        video, soundtrack, audio = outputs["text_replacement_offsets"][0]
        self.assertEqual((video["type"], video["text"]), ("video", "<|video|>"))
        self.assertEqual(video["span"], (6, 6 + len(video["text"])))
        # the soundtrack's tokens sit inside the video expansion, so its own placeholder expands to nothing
        self.assertEqual(
            (soundtrack["type"], soundtrack["text"], soundtrack["replacement"]), ("audio", "<|audio|>", "")
        )
        self.assertEqual((audio["type"], audio["text"]), ("audio", "<|audio|>"))
        self.assertGreater(video["replacement"].count("<|audio|>"), 0)
        expanded = "Hello " + video["replacement"] + " and " + audio["replacement"]
        for replacement in (video, soundtrack, audio):
            start, end = replacement["new_span"]
            self.assertEqual(expanded[start:end], replacement["replacement"])
        self.assertEqual(outputs["input_ids"].tolist(), self.processor.tokenizer([expanded])["input_ids"])

    def test_output_audio_codes_follow_structured_tensor_options(self):
        from transformers.video_utils import VideoMetadata

        self.processor.audio_tokenizer = LiquidCodecModel(
            LiquidCodecConfig(
                encoder_config=TINY_CODEC_BACKBONE,
                decoder_config=TINY_CODEC_BACKBONE,
                encoder_conv_dim=8,
                quantizer_dim=16,
                n_q=4,
                fsq_levels=[3, 3],
            )
        ).eval()
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])
        for with_video in (False, True):
            media = {"videos": [frames], "video_metadata": [metadata]} if with_video else {}
            prompt = ("<|video|>" if with_video else "") + "<|audio_out|>"
            expected = self.processor(text=prompt, audio_out=[self.audio], return_tensors="np", **media)
            for kwargs in ({"text_kwargs": {"return_tensors": "np"}}, {"common_kwargs": {"return_tensors": "np"}}):
                with self.subTest(with_video=with_video, kwargs=kwargs):
                    outputs = self.processor(text=prompt, audio_out=[self.audio], **media, **kwargs)
                    self.assertIsInstance(outputs["input_ids"], np.ndarray)
                    self.assertIsInstance(outputs["audio_out_codes"], np.ndarray)
                    np.testing.assert_array_equal(outputs["audio_out_codes"], expected["audio_out_codes"])

    def test_template_provided_spans_are_not_doubled(self):
        start = self.processor.vision_start_token
        end = self.processor.vision_end_token
        wrapped = self.processor(
            text=f"{start}<|image|>{end} {start}<|audio|>{end}",
            images=[self.image],
            audio=[self.audio],
            return_tensors="pt",
        )["input_ids"][0].tolist()
        bare = self.processor(
            text="<|image|> <|audio|>", images=[self.image], audio=[self.audio], return_tensors="pt"
        )["input_ids"][0].tolist()
        self.assertEqual(wrapped, bare)
        start_id = self.processor.tokenizer.convert_tokens_to_ids(start)
        self.assertEqual(wrapped.count(start_id), 2)

    def test_soundtrack_parts_follow_their_video(self):
        rewrite = Lfm3OmniProcessor._with_soundtrack_parts
        video, question = {"type": "video", "video": "clip.mp4"}, {"type": "audio", "audio": "question.wav"}
        track, text = {"type": "audio", "audio": "clip.mp4"}, {"type": "text", "text": "Hi"}

        def user(*parts):
            return [{"role": "user", "content": list(parts)}]

        # a lone video gets its track where the shared implementation appended it; a clip of its own is kept
        self.assertEqual(rewrite(user(video)), user(video, track))
        self.assertEqual(rewrite(user(video, question)), user(video, track, question))
        self.assertEqual(rewrite(user(video, text)), user(video, track, text))
        url = {"type": "video", "url": "https://example.org/clip.mp4"}
        self.assertEqual(rewrite(user(url)), user(url, {"type": "audio", "audio": url["url"]}))
        self.assertEqual(rewrite(user({"type": "video"})), user({"type": "video"}))
        system = [{"role": "system", "content": "Be brief."}]
        self.assertEqual(rewrite(system + user(video)), system + user(video, track))
        self.assertEqual(rewrite([user(video), user(question)]), [user(video, track), user(question)])
        conversation = user(video, question)
        rewrite(conversation)
        self.assertEqual(conversation, user(video, question))

    @require_torchcodec
    def test_load_audio_from_video_keeps_standalone_clips(self):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("raushan-testing-hf/videos-test", "sample_demo_1.mp4", repo_type="dataset")
        video, clip, text = (
            {"type": "video", "video": path},
            {"type": "audio", "audio": path},
            {"type": "text", "text": "Hi"},
        )
        soundtrack = {"processor_kwargs": {"use_audio_in_video": True}}

        def encode(content, **kwargs):
            out = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **kwargs,
            )
            return out["input_ids"][0].tolist(), out["feature_attention_mask"].shape[0]

        # a video with its soundtrack and a spoken question: the question stays a clip of its own
        with_track = encode([video, clip], load_audio_from_video=True, **soundtrack)
        self.assertEqual(with_track[1], 2)
        self.assertEqual(with_track, encode([video, clip, clip], **soundtrack))
        # a lone video renders as before, with or without the pairing
        self.assertEqual(
            encode([video], load_audio_from_video=True, **soundtrack), encode([video, clip], **soundtrack)
        )
        self.assertEqual(encode([video], load_audio_from_video=True), encode([video, clip]))
        # the track follows its video, not the trailing text
        self.assertEqual(encode([video, text], load_audio_from_video=True), encode([video, clip, text]))

    def test_replacement_offsets_index_the_caller_text_through_every_rewrite(self):
        from transformers.video_utils import VideoMetadata

        self.processor.audio_tokenizer = tiny_codec()
        start, end = self.processor.vision_start_token, self.processor.vision_end_token
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        video = {
            "videos": [frames],
            "video_metadata": [VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])],
        }
        speech = [np.zeros(1920 * 2, dtype=np.float32)]
        soundtrack = np.zeros(16000, dtype=np.float32)
        for name, text, kwargs in (
            ("wrapped image", f"{start}<|image|>{end} look <|audio_out|> x <|audio|>", {"images": [self.image]}),
            ("video and speech", "<|video|> look <|audio_out|> end", video),
            # the soundtrack placeholder moves to its video, and the speech frames sit between them
            (
                "distant soundtrack",
                "<|video|> look at it <|audio_out|> now <|audio|> end",
                {**video, "use_audio_in_video": True},
            ),
            (
                "wrapped video and soundtrack",
                f"{start}<|video|>{end} a <|audio|>",
                {**video, "use_audio_in_video": True},
            ),
        ):
            with self.subTest(name):
                if "<|audio|>" in text:
                    kwargs = {**kwargs, "audio": [soundtrack]}
                if "<|audio_out|>" in text:
                    kwargs = {**kwargs, "audio_out": speech}
                outputs = self.processor(text=text, return_text_replacement_offsets=True, **kwargs)
                assert_offsets_rebuild_the_prompts(self, self.processor, [text], outputs)

    def test_assistant_mask_covers_the_answer_around_speech_frames(self):
        from transformers.video_utils import VideoMetadata

        self.processor.audio_tokenizer = tiny_codec()
        template = GENERATION_TEMPLATE.replace(
            "{%- elif content['type'] == 'audio' -%}",
            "{%- elif content['type'] == 'video' -%}{{- '<|video|>' -}}{%- elif content['type'] == 'audio' -%}",
        )
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        answer = [{"type": "text", "text": "Hello ."}]
        speech = {"type": "audio_out"}
        for name, user, assistant, extra in (
            ("speech in the user turn", [speech, {"type": "text", "text": "Hello"}], answer, {}),
            ("speech in the assistant turn", [{"type": "text", "text": "Hello"}], [*answer, speech], {}),
            (
                "speech after a video",
                [{"type": "video", "video": frames}, speech],
                answer,
                {"video_metadata": [metadata]},
            ),
        ):
            with self.subTest(name):
                outputs = self.processor.apply_chat_template(
                    [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                    chat_template=template,
                    tokenize=True,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                    processor_kwargs={"audio_out": [np.zeros(1920 * 2, dtype=np.float32)], **extra},
                )
                ids = list(outputs["input_ids"][0])
                self.assertEqual(list(outputs["assistant_masks"][0]), expected_assistant_mask(self.processor, ids))
