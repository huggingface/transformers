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

import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from transformers import (
    Dots3NoteFeatureExtractor,
    Dots3NoteImageProcessor,
    Dots3NoteProcessor,
    Dots3NoteVideoProcessor,
    PreTrainedTokenizerFast,
    Qwen2VLVideoProcessor,
    is_torch_available,
)
from transformers.testing_utils import require_torch, require_torchvision


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
            reloaded = Dots3NoteProcessor.from_pretrained(directory)

        self.assertIsInstance(reloaded.image_processor, Dots3NoteImageProcessor)
        self.assertIsInstance(reloaded.video_processor, Dots3NoteVideoProcessor)
        self.assertEqual(reloaded.image_processor.temporal_patch_size, 1)


if __name__ == "__main__":
    unittest.main()
