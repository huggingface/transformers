# Copyright 2025 the HuggingFace Team. All rights reserved.
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

import inspect
import unittest

import numpy as np
from PIL import Image

from transformers.testing_utils import require_av, require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import VideoLlama3Processor
if is_torch_available():
    pass


def prepare_image_inputs():
    """This function prepares a list of PIL images"""
    image_inputs = [np.random.randint(255, size=(3, 15, 50), dtype=np.uint8)]
    image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
    return image_inputs


@require_vision
@require_torch
@require_torchvision
class VideoLlama3ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VideoLlama3Processor
    # Tiny processor created with make_tiny_processor.py from "lkhl/VideoLLaMA3-2B-Image-HF"
    tiny_model_id = "hf-internal-testing/tiny-processor-video_llama_3"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(model_id, patch_size=4, max_pixels=56 * 56, min_pixels=28 * 28, **kwargs)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    def prepare_image_inputs(self, batch_size: int | None = None):
        """This function prepares a list of PIL images for testing"""
        if batch_size is None:
            return prepare_image_inputs()[0]
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return prepare_image_inputs() * batch_size

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    @require_av
    def test_apply_chat_template_video_frame_sampling(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest("Processor doesn't accept videos at input")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": url_to_local_path(
                                "https://huggingface.co/datasets/hf-internal-testing/test-videos/resolve/main/big_buck_bunny_320x240_10s.mp4"
                            ),
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 144)

        # Load with `fps` arg
        fps = 1
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            fps=fps,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 192)

        # Load with `fps` and `num_frames` args, should raise an error
        with self.assertRaises(ValueError):
            out_dict_with_video = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                fps=fps,
                num_frames=num_frames,
            )

        # Load without any arg should load the whole video
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 120)

        # Load video as a list of frames (i.e. images). NOTE: each frame should have same size
        # because we assume they come from one video
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": [
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
            ],
        }
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 192)

        # When the inputs are frame URLs/paths we expect that those are already
        # sampled and will raise an error is asked to sample again.
        with self.assertRaisesRegex(
            ValueError, "Sampling frames from a list of images is not supported! Set `do_sample_frames=False`"
        ):
            out_dict_with_video = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                do_sample_frames=True,
            )

    def test_kwargs_overrides_custom_image_processor_kwargs(self):
        processor = self.get_processor()
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 52)
        inputs = processor(text=input_str, images=image_input, max_pixels=56 * 56 * 4, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 52)

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
        image_input = self.prepare_image_inputs(batch_size=2)

        _ = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            truncation=None,
            padding=True,
        )

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=20,
            )
