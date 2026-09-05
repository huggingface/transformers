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

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from transformers import Ovis2_5Config, Ovis2_5Processor
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

NAMED_VISUAL_TOKENS = {
    "image_token": "<ovis_visual_atom>",
    "video_token": "<ovis_visual_atom>",
    "image_start_token": "<ovis_image_start>",
    "image_end_token": "<ovis_image_end>",
    "video_start_token": "<ovis_video_start>",
    "video_end_token": "<ovis_video_end>",
}


@require_vision
@require_torch
@require_torchvision
@require_tokenizers
class Ovis2_5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Ovis2_5Processor

    @classmethod
    def _build_tokenizer(cls, named_visual_tokens=True, include_visual_tokens=True):
        tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
        if include_visual_tokens:
            tokens.extend(VISUAL_TOKENS)
        tokens.extend(["lower", "newer", "upper", "older", "longer", "string"])
        vocab = {token: index for index, token in enumerate(tokens)}
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer_kwargs = {}
        if include_visual_tokens:
            tokenizer_kwargs["extra_special_tokens"] = (
                NAMED_VISUAL_TOKENS if named_visual_tokens else list(VISUAL_TOKENS)
            )
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            **tokenizer_kwargs,
        )

    @classmethod
    def _setup_tokenizer(cls):
        return cls._build_tokenizer()

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

    def test_visual_tokens_use_tokenizer_attributes(self):
        processor = self.get_processor()

        for token_attribute in NAMED_VISUAL_TOKENS:
            self.assertEqual(getattr(processor, token_attribute), getattr(processor.tokenizer, token_attribute))
        self.assertEqual(processor.image_token, processor.video_token)
        self.assertEqual(processor.image_token_id, processor.video_token_id)

    def test_visual_tokens_fall_back_to_ovis_tokens(self):
        tokenizer = self._build_tokenizer(named_visual_tokens=False)
        processor = self.processor_class(
            image_processor=self.get_component("image_processor"),
            tokenizer=tokenizer,
            video_processor=self.get_component("video_processor"),
        )

        for token_attribute, expected_token in NAMED_VISUAL_TOKENS.items():
            self.assertEqual(getattr(processor, token_attribute), expected_token)
        self.assertEqual(processor.image_token_id, processor.video_token_id)

    def test_visual_tokens_survive_processor_reload(self):
        for named_visual_tokens in (True, False):
            processor = self.processor_class(
                image_processor=self.get_component("image_processor"),
                tokenizer=self._build_tokenizer(named_visual_tokens=named_visual_tokens),
                video_processor=self.get_component("video_processor"),
            )

            with tempfile.TemporaryDirectory() as tmpdirname:
                processor.save_pretrained(tmpdirname)
                reloaded_processor = self.processor_class.from_pretrained(tmpdirname)

            for token_attribute in NAMED_VISUAL_TOKENS:
                self.assertEqual(getattr(reloaded_processor, token_attribute), getattr(processor, token_attribute))
                self.assertNotIn(token_attribute, processor.to_dict())
            self.assertEqual(reloaded_processor.image_token_id, reloaded_processor.video_token_id)

    def test_processor_loads_legacy_hub_metadata(self):
        tokenizer = self._build_tokenizer()
        legacy_preprocessor_config = {
            "do_convert_rgb": None,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "SiglipImageProcessor",
            "image_std": [0.5, 0.5, 0.5],
            "processor_class": "SiglipProcessor",
            "resample": 2,
            "rescale_factor": 1 / 255,
            "size": {"height": 512, "width": 512},
        }

        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer.save_pretrained(tmpdirname)
            Ovis2_5Config().save_pretrained(tmpdirname)
            Path(tmpdirname, "preprocessor_config.json").write_text(json.dumps(legacy_preprocessor_config))
            processor = self.processor_class.from_pretrained(tmpdirname)

        self.assertIsInstance(processor.image_processor, Ovis2_5ImageProcessor)
        self.assertIsInstance(processor.video_processor, Ovis2_5VideoProcessor)
        for visual_processor in (processor.image_processor, processor.video_processor):
            self.assertEqual(visual_processor.size.shortest_edge, 448 * 448)
            self.assertEqual(visual_processor.size.longest_edge, 1344 * 1792)
        for token in VISUAL_TOKENS:
            self.assertNotEqual(processor.tokenizer.convert_tokens_to_ids(token), processor.tokenizer.unk_token_id)

        image_inputs = processor(
            images=np.zeros((64, 64, 3), dtype=np.uint8), text="<image> lower", return_tensors="pt"
        )
        self.assertSetEqual(set(image_inputs), {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"})

        video_inputs = processor(
            videos=self.prepare_video_inputs(), text="<video> lower", do_sample_frames=False, return_tensors="pt"
        )
        self.assertSetEqual(
            set(video_inputs), {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw"}
        )

    def test_image_prompt_expansion_matches_patch_grid(self):
        """An image placeholder expands to boundary tokens and one visual atom per merged patch."""
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
            + [processor.image_token_id] * num_visual_tokens
            + [processor.image_end_token_id]
        )
        self.assertListEqual(input_ids[image_start : image_start + num_visual_tokens + 2], expected_image_ids)

    def test_prepare_inputs_layout_normalizes_visual_inputs(self):
        """Raw placeholders and nested visual inputs are normalized before validation."""
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        video = self.prepare_video_inputs()

        images, image_text, _, image_audio = processor.prepare_inputs_layout(images=[[image]], text="<image>")
        _, video_text, videos, video_audio = processor.prepare_inputs_layout(videos=video, text="<video>")

        self.assertEqual(len(images), 1)
        self.assertEqual(len(videos), 1)
        self.assertListEqual(image_text, [processor.image_token])
        self.assertListEqual(video_text, [processor.video_token])
        self.assertIsNone(image_audio)
        self.assertIsNone(video_audio)

    def test_multiple_images(self):
        """Each image placeholder gets its own start and end boundary tokens."""
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
        """A video placeholder expands to boundary tokens and one visual atom per merged patch."""
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
            + [processor.video_token_id] * num_visual_tokens
            + [processor.video_end_token_id]
        )
        self.assertListEqual(input_ids[video_start : video_start + num_visual_tokens + 2], expected_video_ids)

    def test_special_mm_token_truncation(self):
        """Truncation raises instead of silently dropping part of an expanded visual sequence."""
        processor = self.get_processor()

        with self.assertRaisesRegex(ValueError, "Mismatch in `image` token count"):
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

        visual_atom_counts = (inputs.input_ids == processor.image_token_id).sum(-1).tolist()
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
