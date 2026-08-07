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

import numpy as np

from transformers import Ovis2_5Processor
from transformers.testing_utils import require_tokenizers, require_torch, require_torchvision, require_vision
from transformers.utils import is_tokenizers_available, is_torch_available, is_torchvision_available

from ...test_processing_common import ProcessorTesterMixin


if is_tokenizers_available():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from transformers import PreTrainedTokenizerFast

if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers import Ovis2_5ImageProcessor, Ovis2_5VideoProcessor


VISUAL_TOKENS = (
    "<ovis_visual_atom>",
    "<ovis_image_start>",
    "<ovis_image_end>",
    "<ovis_video_start>",
    "<ovis_video_end>",
)


@require_vision
@require_torch
@require_torchvision
@require_tokenizers
class Ovis2_5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Ovis2_5Processor

    @classmethod
    def _setup_tokenizer(cls):
        vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<ovis_visual_atom>": 4,
            "<ovis_image_start>": 5,
            "<ovis_image_end>": 6,
            "<ovis_video_start>": 7,
            "<ovis_video_end>": 8,
            "lower": 9,
            "newer": 10,
            "upper": 11,
            "older": 12,
            "longer": 13,
            "string": 14,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            additional_special_tokens=list(VISUAL_TOKENS),
        )

    @classmethod
    def _setup_image_processor(cls):
        return Ovis2_5ImageProcessor(
            size={"shortest_edge": 64 * 64, "longest_edge": 64 * 1024},
        )

    @classmethod
    def _setup_video_processor(cls):
        return Ovis2_5VideoProcessor(
            size={"shortest_edge": 64 * 64, "longest_edge": 64 * 1024},
        )

    def prepare_text_inputs(self, batch_size: int | None = None, modalities: str | list | None = None):
        if isinstance(modalities, str):
            modalities = [modalities]
        modalities = modalities or []
        placeholder = "<image>" if "image" in modalities else "<video>" if "video" in modalities else ""
        text = f"{placeholder} lower newer"
        if batch_size is None:
            return text
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return [text] * batch_size

    def prepare_image_inputs(self, batch_size: int | None = None, nested: bool = False):
        image = np.zeros((3, 64, 64), dtype=np.uint8)
        if batch_size is None:
            return image
        images = [image.copy() for _ in range(batch_size)]
        return [[item] for item in images] if nested else images

    def prepare_video_inputs(self, batch_size: int | None = None):
        video = np.zeros((4, 3, 64, 64), dtype=np.uint8)
        if batch_size is None:
            return video
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return [video.copy() for _ in range(batch_size)]

    def test_image_prompt_expansion_matches_patch_grid(self):
        processor = self.get_processor()
        inputs = processor(
            images=self.prepare_image_inputs(),
            text="<image> lower newer",
            return_tensors="pt",
        )

        num_visual_tokens = int(inputs.image_grid_thw[0].prod()) // processor.image_processor.merge_size**2
        input_ids = inputs.input_ids[0].tolist()
        image_start = input_ids.index(processor.image_start_token_id)
        expected_image_ids = (
            [processor.image_start_token_id]
            + [processor.visual_atom_token_id] * num_visual_tokens
            + [processor.image_end_token_id]
        )
        self.assertListEqual(input_ids[image_start : image_start + num_visual_tokens + 2], expected_image_ids)

    def test_multiple_images(self):
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        inputs = processor(
            images=[image, image.copy()],
            text="<image><image> lower newer",
            return_tensors="pt",
        )

        self.assertEqual(inputs.image_grid_thw.shape[0], 2)
        self.assertEqual((inputs.input_ids == processor.image_start_token_id).sum().item(), 2)
        self.assertEqual((inputs.input_ids == processor.image_end_token_id).sum().item(), 2)

    def test_video_prompt_expansion(self):
        processor = self.get_processor()
        inputs = processor(
            videos=self.prepare_video_inputs(),
            text="<video> lower newer",
            do_sample_frames=False,
            return_tensors="pt",
        )

        num_visual_tokens = int(inputs.video_grid_thw[0].prod()) // processor.video_processor.merge_size**2
        input_ids = inputs.input_ids[0].tolist()
        video_start = input_ids.index(processor.video_start_token_id)
        expected_video_ids = (
            [processor.video_start_token_id]
            + [processor.visual_atom_token_id] * num_visual_tokens
            + [processor.video_end_token_id]
        )
        self.assertListEqual(input_ids[video_start : video_start + num_visual_tokens + 2], expected_video_ids)

    def test_special_mm_token_truncation(self):
        processor = self.get_processor()

        with self.assertRaisesRegex(ValueError, "Visual tokens were likely truncated"):
            processor(
                images=self.prepare_image_inputs(),
                text="<image> lower newer",
                truncation=True,
                max_length=3,
                return_tensors="pt",
            )

    def test_model_input_names(self):
        processor = self.get_processor()

        image_inputs = processor(
            text=self.prepare_text_inputs(modalities="image"),
            images=self.prepare_image_inputs(),
            return_tensors="pt",
        )
        self.assertSetEqual(
            set(image_inputs),
            {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"},
        )

        video_inputs = processor(
            text=self.prepare_text_inputs(modalities="video"),
            videos=self.prepare_video_inputs(),
            do_sample_frames=False,
            return_tensors="pt",
        )
        self.assertSetEqual(
            set(video_inputs),
            {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw"},
        )
        self.assertSetEqual(set(processor.model_input_names), set(image_inputs) | set(video_inputs))

    def test_processor_with_multiple_inputs(self):
        processor = self.get_processor()

        with self.assertRaisesRegex(ValueError, "one visual modality at a time"):
            processor(
                text="<image><video> lower newer",
                images=self.prepare_image_inputs(),
                videos=self.prepare_video_inputs(),
                do_sample_frames=False,
                return_tensors="pt",
            )

    def test_unstructured_kwargs_batched_video(self):
        processor = self.get_processor()
        with self.assertRaisesRegex(ValueError, "exactly one video"):
            processor(
                text=self.prepare_text_inputs(batch_size=2, modalities="video"),
                videos=self.prepare_video_inputs(batch_size=2),
                do_sample_frames=False,
                return_tensors="pt",
            )

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        processor = self.get_processor()
        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        images = [np.zeros((height, width, 3), dtype=np.uint8) for height, width in image_sizes]
        inputs = processor(
            text=[f"{processor.image_token} lower newer"] * len(images),
            images=images,
            padding=True,
            return_tensors="pt",
        )

        visual_atom_counts = (inputs.input_ids == processor.visual_atom_token_id).sum(-1).tolist()
        helper_counts = processor._get_num_multimodal_tokens(image_sizes=image_sizes)["num_image_tokens"]
        self.assertListEqual(visual_atom_counts, helper_counts)

    def test_flat_kwarg_applied_when_modality_dict_lacks_it(self):
        processor = self.get_processor()
        inputs = processor(
            text=self.prepare_text_inputs(modalities="image"),
            images=self.prepare_image_inputs(),
            text_kwargs={},
            return_tensors="pt",
        )

        for key, value in inputs.items():
            self.assertIsInstance(value, torch.Tensor, msg=f"{key} should be a torch.Tensor")

    def test_processor_text_has_no_visual(self):
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        inputs = processor(
            text=["lower newer", "<image> lower newer", "<image> lower newer"],
            images=[[], [image], [image.copy()]],
            padding=True,
            return_tensors="pt",
        )

        self.assertEqual(inputs.input_ids.shape[0], 3)
        self.assertEqual(inputs.image_grid_thw.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
