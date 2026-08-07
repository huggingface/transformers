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
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from transformers import (
    Dots3NoteOmniFeatureExtractor,
    Dots3NoteOmniProcessor,
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
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
    return Dots3NoteOmniProcessor(
        image_processor=Qwen2VLImageProcessor(**vision_kwargs),
        tokenizer=get_tiny_tokenizer(),
        video_processor=Qwen2VLVideoProcessor(**vision_kwargs),
        feature_extractor=Dots3NoteOmniFeatureExtractor(
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
class Dots3NoteOmniProcessorTest(unittest.TestCase):
    def test_legacy_checkpoint_defaults_without_processor_config(self):
        processor = Dots3NoteOmniProcessor(
            image_processor=Qwen2VLImageProcessor(),
            tokenizer=get_tiny_tokenizer(),
            video_processor=Qwen2VLVideoProcessor(),
            feature_extractor=Dots3NoteOmniFeatureExtractor(),
        )

        expected_size = {"shortest_edge": 3136, "longest_edge": 1016064}
        self.assertEqual(dict(processor.image_processor.size), expected_size)
        self.assertEqual(dict(processor.video_processor.size), expected_size)
        self.assertEqual(processor.image_processor.temporal_patch_size, 1)
        self.assertEqual(processor.video_processor.temporal_patch_size, 1)
        self.assertEqual(processor.feature_extractor.sampling_rate, 16000)
        self.assertEqual(processor.feature_extractor.feature_size, 128)

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
            do_rescale=False,
            do_normalize=False,
            add_special_tokens=False,
            return_tensors="pt",
        )

        expected_video_tokens = int(output.video_grid_thw[0].prod()) // processor.video_processor.merge_size**2
        self.assertEqual(output.input_ids[0].tolist().count(processor.video_token_id), expected_video_tokens)

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


if __name__ == "__main__":
    unittest.main()
