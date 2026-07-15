# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Focused processor tests for Cosmos3 Edge packed vision inputs."""

import unittest
from types import SimpleNamespace

import numpy as np

from transformers import (
    Cosmos3EdgeImageProcessor,
    Cosmos3EdgeProcessor,
    Cosmos3EdgeVideoProcessor,
    PreTrainedTokenizerFast,
)
from transformers.testing_utils import (
    require_tokenizers,
    require_torch,
    require_torchcodec,
    require_torchvision,
    require_vision,
)
from transformers.utils import (
    is_tokenizers_available,
    is_torch_available,
    is_torchcodec_available,
    is_vision_available,
)
from transformers.video_utils import VideoMetadata

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_tokenizers_available():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
@require_torchvision
@require_tokenizers
class Cosmos3EdgeProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Cosmos3EdgeProcessor

    @classmethod
    def _setup_tokenizer(cls):
        """Build a manual `tokenizers` WordLevel tokenizer for Edge placeholders."""
        vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<|vision_start|>": 4,
            "<|vision_end|>": 5,
            "<|image_pad|>": 6,
            "<|video_pad|>": 7,
            "lower": 8,
            "newer": 9,
            "upper": 10,
            "older": 11,
            "longer": 12,
            "string": 13,
            "Describe": 14,
            "this": 15,
            "user": 16,
            "assistant": 17,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            extra_special_tokens={
                "vision_start_token": "<|vision_start|>",
                "vision_end_token": "<|vision_end|>",
                "image_token": "<|image_pad|>",
                "video_token": "<|video_pad|>",
            },
            return_mm_token_type_ids=True,
        )

    @classmethod
    def _setup_image_processor(cls):
        """Use lightweight resize bounds aligned to the 32-pixel patch-merging factor."""
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(
            size={"shortest_edge": 32 * 32, "longest_edge": 96 * 96},
            patch_size=16,
            merge_size=2,
        )

    @classmethod
    def _setup_video_processor(cls):
        """Use lightweight aligned videos while retaining Edge's frame-wise packing."""
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        return video_processor_class(
            size={"shortest_edge": 32 * 32, "longest_edge": 8 * 96 * 96},
            patch_size=16,
            temporal_patch_size=1,
            merge_size=2,
            do_sample_frames=False,
            return_metadata=True,
        )

    @classmethod
    def _setup_test_attributes(cls, processor):
        """Expose the Edge placeholder tokens expected by the shared processor tests."""
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    @staticmethod
    def prepare_processor_dict():
        """Provide a minimal chat template containing Edge's image and video wrappers."""
        return {
            "chat_template": (
                "{% for message in messages %}{{ message['role'] + ': ' }}"
                "{% for content in message['content'] %}"
                "{% if content['type'] == 'image' %}"
                "<|vision_start|><|image_pad|><|vision_end|>"
                "{% elif content['type'] == 'video' %}"
                "<|vision_start|><|video_pad|><|vision_end|>"
                "{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
                "{% endfor %}{% endfor %}"
                "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
            )
        }

    def prepare_image_inputs(self, batch_size: int | None = None, nested: bool = False):
        """Create small 64x96 inputs aligned to patch_size * merge_size (32).

        The fixed size keeps the processor tests lightweight and valid for patch
        merging; it is unrelated to testing per-image keyword arguments.
        """
        image = Image.fromarray(np.random.randint(255, size=(64, 96, 3), dtype=np.uint8))
        if batch_size is None:
            return image
        if nested:
            return [[image] for _ in range(batch_size)]
        return [image] * batch_size

    def prepare_video_inputs(self, batch_size: int | None = None):
        """Create four 64x96 frames aligned to patch_size * merge_size (32).

        The fixed shape keeps frame-wise packing tests lightweight and valid; it
        is unrelated to testing per-video keyword arguments.
        """
        video = np.random.randint(255, size=(4, 64, 96, 3), dtype=np.uint8)
        if batch_size is None:
            return video
        return [video] * batch_size

    @require_torch
    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list,
    ):
        """Adapt shared chat-template coverage to Edge's packed patch outputs."""
        if modality == "video" and any(isinstance(item, str) for item in input_data[:batch_size]):
            if not is_torchcodec_available():
                self.skipTest("torchcodec is required to decode video URLs")

        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")
        if processor_name not in self.processor_class.get_attributes():
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

        batch_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": "Describe this."}]},
            ]
            for _ in range(batch_size)
        ]

        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        formatted_prompt_tokenized = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors=return_tensors
        )
        tokenized_prompt = processor.tokenizer(formatted_prompt, return_tensors=return_tensors)
        self.assertListEqual(tokenized_prompt.input_ids.tolist(), formatted_prompt_tokenized.tolist())

        tokenized_prompt_max_length = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=return_tensors,
            processor_kwargs={
                "padding": "max_length",
                "truncation": True,
                "max_length": self.chat_template_max_length,
            },
        )
        self.assertEqual(len(tokenized_prompt_max_length[0]), self.chat_template_max_length)

        out_dict_text = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        self.assertTrue(all(key in out_dict_text for key in ["input_ids", "attention_mask"]))
        self.assertEqual(len(out_dict_text["input_ids"]), batch_size)
        self.assertEqual(len(out_dict_text["attention_mask"]), batch_size)

        for index, item in enumerate(input_data[:batch_size]):
            batch_messages[index][1]["content"] = [
                batch_messages[index][1]["content"][0],
                {"type": modality, "url": item},
            ]

        processor_kwargs = {"num_frames": 2, "fps": None} if modality == "video" else None
        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            processor_kwargs=processor_kwargs,
        )
        input_name = getattr(self, input_name)
        grid_name = "video_grid_thw" if modality == "video" else "image_grid_thw"
        expected_num_patches = int(out_dict[grid_name].prod(dim=-1).sum())

        self.assertIn(input_name, out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)
        self.assertEqual(len(out_dict[input_name]), expected_num_patches)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for value in out_dict.values():
            self.assertIsInstance(value, return_tensor_to_type[return_tensors])

        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is the sound of"}],
        }
        for index in range(batch_size):
            batch_messages[index] = batch_messages[index] + [assistant_message]
        continue_prompt = processor.apply_chat_template(batch_messages, continue_final_message=True, tokenize=False)
        for prompt in continue_prompt:
            self.assertTrue(prompt.endswith("It is the sound of"))

    @require_torchcodec
    def test_apply_chat_template_video_frame_sampling(self):
        """Adapt the shared frame-sampling assertions to Edge's packed video patches."""
        processor = self.get_processor()

        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": url_to_local_path(
                                "https://huggingface.co/datasets/hf-internal-testing/test-videos/resolve/main/tiny_video_320x240.mp4"
                            ),
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        def assert_packed_video(output, expected_num_frames):
            self.assertIn(self.videos_input_name, output)
            self.assertEqual(tuple(output["video_grid_thw"].shape), (1, 3))
            self.assertEqual(output["video_grid_thw"][0, 0].item(), expected_num_frames)
            expected_num_patches = int(output["video_grid_thw"].prod(dim=-1).sum())
            self.assertEqual(len(output[self.videos_input_name]), expected_num_patches)
            self.assertEqual(
                output[self.videos_input_name].shape[-1],
                len(processor.video_processor.image_mean) * processor.video_processor.patch_size**2,
            )

        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"num_frames": num_frames, "fps": None, "do_sample_frames": True},
        )
        assert_packed_video(out_dict_with_video, expected_num_frames=num_frames)

        # The fixture would yield three frames at 10 FPS, which Edge clamps to its four-frame minimum.
        fps = 10
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"fps": fps, "num_frames": None, "do_sample_frames": True},
        )
        assert_packed_video(out_dict_with_video, expected_num_frames=processor.video_processor.min_frames)

        # Disabling sampling retains all eleven frames in the fixture even when an FPS is supplied.
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"do_sample_frames": False, "fps": fps, "return_tensors": "pt"},
        )
        assert_packed_video(out_dict_with_video, expected_num_frames=11)

        with self.assertRaises(ValueError):
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                processor_kwargs={"fps": fps, "num_frames": num_frames, "do_sample_frames": True},
            )

    def test_video_processor_defaults(self):
        """Ensure the processor wrapper preserves all video-processor defaults."""
        video_processor = self.get_component("video_processor")
        components = self.prepare_components()
        processor = self.processor_class(**components)
        video_input = self.prepare_video_inputs()
        video_metadata = [VideoMetadata(total_num_frames=4, fps=2, duration=2.0, frames_indices=[0, 1, 2, 3])]

        video_processor_output = video_processor(
            video_input,
            video_metadata=video_metadata,
            do_sample_frames=False,
            return_metadata=True,
            return_tensors="pt",
        )
        processor_output = processor(
            videos=video_input,
            video_metadata=[VideoMetadata(total_num_frames=4, fps=2, duration=2.0, frames_indices=[0, 1, 2, 3])],
            do_sample_frames=False,
            return_metadata=True,
            return_tensors="pt",
        )

        for key in video_processor_output:
            if key == "video_metadata":
                self.assertEqual(video_processor_output[key], processor_output[key])
            else:
                torch.testing.assert_close(video_processor_output[key], processor_output[key])

    def test_image_processor_emits_packed_patches_and_thw_grid(self):
        """Verify that an image is flattened into patches with its THW grid metadata."""
        processor = Cosmos3EdgeImageProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
        )
        image = np.zeros((4, 8, 3), dtype=np.uint8)

        processed = processor(image, return_tensors="pt")

        # 4 x 8 pixels with 2 x 2 patches produces a 2 x 4 patch grid.
        self.assertEqual(tuple(processed["pixel_values"].shape), (8, 12))
        self.assertEqual(processed["image_grid_thw"].tolist(), [[1, 2, 4]])

    def test_image_processor_uses_projector_block_major_patch_order(self):
        """Protect the 2x2 block-major patch order expected by the Edge projector."""
        processor = Cosmos3EdgeImageProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
        )
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        for height_index in range(2):
            for width_index in range(4):
                patch_index = height_index * 4 + width_index
                image[height_index * 2 : (height_index + 1) * 2, width_index * 2 : (width_index + 1) * 2] = patch_index

        processed = processor(image, return_tensors="pt")

        # The 2×2 groups must be contiguous for the checkpoint projector: the
        # first group is (0, 0), (0, 1), (1, 0), (1, 1), followed by the next group.
        self.assertEqual(processed["pixel_values"][:, 0].tolist(), [0, 1, 4, 5, 2, 3, 6, 7])

    def test_video_processor_emits_packed_patches_and_thw_grid(self):
        """Verify frame-wise packed patches and the corresponding video THW grid."""
        processor = Cosmos3EdgeVideoProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
            temporal_patch_size=1,
            return_metadata=True,
        )
        video = np.zeros((2, 4, 8, 3), dtype=np.uint8)
        metadata = [{"fps": 2, "total_num_frames": 2, "duration": 1.0}]

        processed = processor(video, video_metadata=metadata, return_tensors="pt")

        # Two 4 x 8 frames yield two 2 x 4 patch grids. Temporal patches stay
        # unmerged because Edge encodes one timestamped vision span per frame.
        self.assertEqual(tuple(processed["pixel_values_videos"].shape), (16, 12))
        self.assertEqual(processed["video_grid_thw"].tolist(), [[2, 2, 4]])
        self.assertIn("video_metadata", processed)

    def test_video_processor_uses_projector_block_major_patch_order_per_frame(self):
        """Protect projector block-major ordering independently within every frame."""
        processor = Cosmos3EdgeVideoProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=2,
            merge_size=2,
            temporal_patch_size=1,
        )
        video = np.zeros((2, 4, 8, 3), dtype=np.uint8)
        for frame_index in range(2):
            for height_index in range(2):
                for width_index in range(4):
                    patch_index = frame_index * 10 + height_index * 4 + width_index
                    video[
                        frame_index,
                        height_index * 2 : (height_index + 1) * 2,
                        width_index * 2 : (width_index + 1) * 2,
                    ] = patch_index

        processed = processor(
            video,
            video_metadata=[{"fps": 2, "total_num_frames": 2, "duration": 1.0}],
            return_tensors="pt",
        )

        self.assertEqual(
            processed["pixel_values_videos"][:, 0].tolist(),
            [0, 1, 4, 5, 2, 3, 6, 7, 10, 11, 14, 15, 12, 13, 16, 17],
        )

    def test_public_processor_name_is_cosmos_specific(self):
        """Guard the public class name stored in released processor configurations."""
        self.assertEqual(Cosmos3EdgeProcessor.__name__, "Cosmos3EdgeProcessor")

    def test_processor_returns_multimodal_token_types_by_default(self):
        """Check the Edge default while allowing an explicit tokenizer override."""
        processor = object.__new__(Cosmos3EdgeProcessor)
        processor.tokenizer = SimpleNamespace()
        merged_kwargs = processor._merge_kwargs(
            Cosmos3EdgeProcessor.valid_processor_kwargs,
            tokenizer_init_kwargs={"return_mm_token_type_ids": True},
        )
        overridden_kwargs = processor._merge_kwargs(
            Cosmos3EdgeProcessor.valid_processor_kwargs,
            tokenizer_init_kwargs={"return_mm_token_type_ids": True},
            text_kwargs={"return_mm_token_type_ids": False},
        )

        self.assertTrue(merged_kwargs["text_kwargs"]["return_mm_token_type_ids"])
        self.assertFalse(overridden_kwargs["text_kwargs"]["return_mm_token_type_ids"])

    def test_video_placeholder_uses_one_timestamped_vision_span_per_frame(self):
        """Require one timestamped vision wrapper for each unmerged video frame."""
        processor = object.__new__(Cosmos3EdgeProcessor)
        processor.video_token = "<|video_pad|>"
        processor.vision_start_token = "<|vision_start|>"
        processor.vision_end_token = "<|vision_end|>"
        processor.video_processor = SimpleNamespace(merge_size=2, temporal_patch_size=1)
        video_inputs = {
            "video_grid_thw": np.asarray([[2, 2, 4]]),
            "video_metadata": [
                VideoMetadata(
                    total_num_frames=3,
                    fps=2,
                    duration=1.5,
                    frames_indices=[0, 2],
                )
            ],
        }

        replacement = processor.replace_video_token(video_inputs, video_idx=0)

        frame_span = "<|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
        self.assertEqual(replacement, f"<0.0 seconds>{frame_span}<1.0 seconds>{frame_span}")

    def test_video_replacement_consumes_the_template_vision_wrapper_as_one_unit(self):
        """Ensure frame spans replace the full template wrapper without nested markers."""
        processor = object.__new__(Cosmos3EdgeProcessor)
        processor.image_token = "<|image_pad|>"
        processor.video_token = "<|video_pad|>"
        processor.vision_start_token = "<|vision_start|>"
        processor.vision_end_token = "<|vision_end|>"

        frame_span = "<|vision_start|><|video_pad|><|video_pad|><|vision_end|>"
        replacement = f"<0.0 seconds>{frame_span}<1.0 seconds>{frame_span}"
        template_text = "before<|vision_start|><|video_pad|><|vision_end|>after"
        text, replacement_offsets = processor.get_text_with_replacements(
            [template_text], videos_replacements=[replacement]
        )

        self.assertEqual(text, [f"before{replacement}after"])
        self.assertEqual(replacement_offsets[0][0]["text"], "<|vision_start|><|video_pad|><|vision_end|>")
