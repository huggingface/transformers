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
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoVideoProcessor,
    Dots3NoteFeatureExtractor,
    Dots3NoteImageProcessor,
    Dots3NoteProcessor,
    Dots3NoteVideoProcessor,
    PreTrainedTokenizerFast,
    Qwen2VLVideoProcessor,
    is_torch_available,
)
from transformers.models.dots3_note import video_processing_dots3_note
from transformers.testing_utils import require_torch, require_torchcodec, require_torchvision, slow


if is_torch_available():
    import torch


IMAGE = "<|imgpad|>"
VIDEO = "<|video_pad|>"
AUDIO_START = "<|audio_comp_start|>"
AUDIO_PAD = "<|audio_comp_pad|>"
AUDIO_END = "<|audio_comp_end|>"


def get_tiny_tokenizer():
    backend = Tokenizer(
        WordLevel(
            vocab={
                "[UNK]": 0,
                IMAGE: 1,
                VIDEO: 2,
                AUDIO_START: 3,
                AUDIO_PAD: 4,
                AUDIO_END: 5,
                "describe": 6,
                "plain": 7,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=[IMAGE, VIDEO, AUDIO_START, AUDIO_PAD, AUDIO_END],
    )


def get_tiny_processor():
    vision_kwargs = {
        "size": {"shortest_edge": 16, "longest_edge": 64},
        "patch_size": 2,
        "temporal_patch_size": 1,
        "merge_size": 2,
    }
    return Dots3NoteProcessor(
        image_processor=Dots3NoteImageProcessor(**vision_kwargs),
        tokenizer=get_tiny_tokenizer(),
        video_processor=Dots3NoteVideoProcessor(**vision_kwargs),
        feature_extractor=Dots3NoteFeatureExtractor(
            feature_size=8,
            sampling_rate=32,
            n_fft=16,
            hop_length=4,
            chunk_seconds=2,
            conv_temporal_stride=8,
        ),
    )


@require_torch
@require_torchvision
class Dots3NoteProcessorTest(unittest.TestCase):
    def test_legacy_checkpoint_defaults_without_processor_config(self):
        processor = Dots3NoteProcessor(
            image_processor=Dots3NoteImageProcessor(),
            tokenizer=get_tiny_tokenizer(),
            video_processor=Qwen2VLVideoProcessor(),
            feature_extractor=Dots3NoteFeatureExtractor(),
        )

        expected_size = {"shortest_edge": 3136, "longest_edge": 1016064}
        self.assertEqual(dict(processor.image_processor.size), expected_size)
        self.assertEqual(dict(processor.video_processor.size), expected_size)
        self.assertEqual(processor.image_processor.temporal_patch_size, 1)
        self.assertEqual(processor.video_processor.temporal_patch_size, 1)
        self.assertEqual(processor.feature_extractor.sampling_rate, 16000)
        self.assertEqual(processor.feature_extractor.feature_size, 128)

    def test_rgba_images_are_composited_on_white(self):
        processor = get_tiny_processor().image_processor
        array = np.zeros((8, 12, 4), dtype=np.uint8)
        array[..., :3] = [200, 40, 10]
        array[..., 3] = np.arange(12, dtype=np.uint8)[None] * 20
        rgba = Image.fromarray(array, "RGBA")
        white = Image.new("RGB", rgba.size, (255, 255, 255))
        white.paste(rgba, mask=rgba.getchannel("A"))

        actual = processor(rgba, return_tensors="pt")
        expected = processor(white, return_tensors="pt")

        torch.testing.assert_close(actual.pixel_values, expected.pixel_values, rtol=0, atol=0)
        torch.testing.assert_close(actual.image_grid_thw, expected.image_grid_thw, rtol=0, atol=0)

    def test_sglang_image_numerical_golden(self):
        processor = get_tiny_processor().image_processor
        image = Image.fromarray((np.arange(48, dtype=np.uint8).reshape(4, 4, 3) * 5), "RGB")
        output = processor(image, return_tensors="pt")

        self.assertEqual(output.image_grid_thw.tolist(), [[1, 2, 2]])
        selected = output.pixel_values.flatten()[torch.tensor([0, 1, 4, 11, 12, 23, 24, 47])]
        expected = torch.tensor(
            [
                -1.7922625542,
                -1.5732861757,
                -1.6770582199,
                -0.2715141475,
                -1.3543097973,
                0.1550878286,
                -0.0404513702,
                1.8614956141,
            ]
        )
        torch.testing.assert_close(selected, expected, rtol=0, atol=1e-6)

    def test_expands_audio_placeholder(self):
        processor = get_tiny_processor()
        output = processor(
            text=f"{AUDIO_START}{AUDIO_PAD}{AUDIO_END} describe",
            audio=[torch.zeros(33)],
            sampling_rate=32,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = output.input_ids[0].tolist()
        self.assertEqual(input_ids.count(processor.audio_start_token_id), 1)
        self.assertEqual(input_ids.count(processor.audio_token_id), 2)
        self.assertEqual(input_ids.count(processor.audio_end_token_id), 1)
        self.assertEqual(output.audio_token_lengths.tolist(), [2])

    def test_expands_image_placeholder(self):
        processor = get_tiny_processor()
        image = torch.zeros(3, 4, 4)
        output = processor(
            text=f"{IMAGE} describe",
            images=[image],
            do_rescale=False,
            do_normalize=False,
            add_special_tokens=False,
            return_tensors="pt",
        )

        expected_image_tokens = int(output.image_grid_thw[0].prod()) // processor.image_processor.merge_size**2
        self.assertEqual(output.input_ids[0].tolist().count(processor.image_token_id), expected_image_tokens)

    def test_expands_video_placeholder(self):
        processor = get_tiny_processor()
        video = [torch.zeros(3, 4, 4), torch.ones(3, 4, 4)]
        output = processor(
            text=f"{VIDEO} describe",
            videos=[video],
            audio_sr=32,
            add_special_tokens=False,
            return_tensors="pt",
        )

        expected_image_tokens = sum(
            int(grid.prod()) // processor.image_processor.merge_size**2 for grid in output.image_grid_thw
        )
        self.assertEqual(output.input_ids[0].tolist().count(processor.video_token_id), 0)
        self.assertEqual(output.input_ids[0].tolist().count(processor.image_token_id), expected_image_tokens)
        self.assertIn("pixel_values", output)
        self.assertNotIn("pixel_values_videos", output)
        self.assertTrue(all(grid[0] == 1 for grid in output.image_grid_thw))

    def test_native_video_sequence_length_defaults_and_override(self):
        processor = get_tiny_processor()
        for model_max_length, expected in ((524_288, 524_288), (262_144, 262_144), (int(1e30), 524_288)):
            with self.subTest(model_max_length=model_max_length):
                processor.tokenizer.model_max_length = model_max_length
                self.assertEqual(
                    video_processing_dots3_note._resolve_video_budget(processor.tokenizer, None, None, 0)[0], expected
                )
        self.assertEqual(
            video_processing_dots3_note._resolve_video_budget(processor.tokenizer, 131_072, None, 0)[0], 131_072
        )

    def test_native_video_sequence_budget(self):
        processor = get_tiny_processor()
        processor.tokenizer.model_max_length = 524_288
        video = [torch.zeros(3, 4, 4), torch.ones(3, 4, 4)]
        sequence_length = 4096
        max_new_tokens = 1024
        video_inputs = {
            "text": f"{VIDEO} describe",
            "videos": [video],
            "seq": sequence_length,
            "audio_sr": 32,
            "add_special_tokens": False,
        }
        output = processor(**video_inputs, max_new_tokens=max_new_tokens, return_tensors="pt")

        self.assertLessEqual(output.input_ids.shape[-1] + max_new_tokens, sequence_length)
        processor.tokenizer.model_max_length = sequence_length
        with self.assertRaisesRegex(ValueError, "must not exceed tokenizer.model_max_length"):
            processor(**(video_inputs | {"seq": sequence_length + 1}))
        with self.assertRaisesRegex(ValueError, "exceeding the sequence length"):
            processor(**(video_inputs | {"text": f"{VIDEO} " + "plain " * sequence_length}))
        with self.assertRaisesRegex(ValueError, "must leave room for video input"):
            processor(**(video_inputs | {"seq": None}), max_new_tokens=524_288)
        with self.assertRaisesRegex(ValueError, "output_reserve must be non-negative"):
            processor(**video_inputs, output_reserve=-1)

    def test_native_video_warns_when_audio_exceeds_budget(self):
        processor = get_tiny_processor()
        frames = [(0.0, Image.new("RGB", (4, 4)))] * 4
        with (
            patch.object(
                video_processing_dots3_note,
                "_open_video",
                return_value=SimpleNamespace(metadata=SimpleNamespace(duration_seconds=10)),
            ),
            patch.object(
                video_processing_dots3_note,
                "_decode_audio",
                return_value=(np.zeros(320, dtype=np.int16), 10.0),
            ),
            patch.object(video_processing_dots3_note, "_decode_frames", return_value=(frames, 10.0)),
            self.assertLogs(video_processing_dots3_note.logger, level="WARNING") as logs,
        ):
            processor(
                text=f"{VIDEO} describe",
                videos=[b"video"],
                seq=4096,
                audio_cap=0.0001,
                audio_sr=32,
                add_special_tokens=False,
            )
        self.assertIn("audio_cap", " ".join(logs.output))

    @slow
    @require_torchcodec
    def test_native_video_with_audio_torchcodec(self):
        processor = get_tiny_processor()
        sequence_length = 32_768
        max_new_tokens = 128
        video = hf_hub_download(
            repo_id="merve/vlm_test_images",
            filename="concert.mp4",
            repo_type="dataset",
        )
        output = processor(
            text=f"{VIDEO} describe",
            videos=[video],
            seq=sequence_length,
            max_new_tokens=max_new_tokens,
            audio_sr=32,
            add_special_tokens=False,
            return_tensors="pt",
        )

        self.assertIn("pixel_values", output)
        self.assertIn("input_features", output)
        self.assertGreater(output.input_ids.eq(processor.image_token_id).sum().item(), 0)
        self.assertGreater(output.input_ids.eq(processor.audio_token_id).sum().item(), 0)
        self.assertLessEqual(output.input_ids.shape[-1] + max_new_tokens, sequence_length)

    def test_rejects_video_transform_overrides(self):
        processor = get_tiny_processor()
        with self.assertRaisesRegex(ValueError, "fixed SGLang-aligned transform"):
            processor(
                text=f"{VIDEO} describe",
                videos=[[torch.zeros(3, 4, 4)]],
                audio_sr=32,
                do_resize=False,
            )

    def test_rejects_audio_placeholder_mismatch(self):
        processor = get_tiny_processor()
        with self.assertRaisesRegex(ValueError, "audio/placeholder count mismatch"):
            processor(
                text="plain",
                audio=[torch.zeros(32)],
                sampling_rate=32,
                add_special_tokens=False,
            )

    def test_preserves_text_only_inputs(self):
        output = get_tiny_processor()(text="plain", add_special_tokens=False)
        self.assertEqual(output.input_ids, [[7]])
        self.assertNotIn("input_features", output)

    def test_save_and_reload(self):
        processor = get_tiny_processor()
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            reloaded = AutoProcessor.from_pretrained(directory)
            image_processor = AutoImageProcessor.from_pretrained(directory)
            video_processor = AutoVideoProcessor.from_pretrained(directory)
            feature_extractor = AutoFeatureExtractor.from_pretrained(directory)

        self.assertIsInstance(reloaded.image_processor, Dots3NoteImageProcessor)
        self.assertIsInstance(reloaded.video_processor, Dots3NoteVideoProcessor)
        self.assertIsInstance(image_processor, Dots3NoteImageProcessor)
        self.assertIsInstance(video_processor, Dots3NoteVideoProcessor)
        self.assertIsInstance(feature_extractor, Dots3NoteFeatureExtractor)
        for vision_processor in (image_processor, video_processor):
            self.assertEqual(dict(vision_processor.size), {"shortest_edge": 16, "longest_edge": 64})
            self.assertEqual(vision_processor.temporal_patch_size, 1)


if __name__ == "__main__":
    unittest.main()
