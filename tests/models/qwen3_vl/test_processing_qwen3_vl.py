# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Qwen3VLProcessor


@require_vision
@require_torch
@require_torchvision
class Qwen3VLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3VLProcessor
    # Use tiny repos to avoid loading the full 151k-vocab tokenizer (~327 MB)
    # Tiny processor created with make_tiny_processor.py from "Qwen/Qwen3-VL-235B-A22B-Instruct"
    tiny_model_id = "hf-internal-testing/tiny-processor-qwen3_vl"
    videos_unstructured_max_length = 870
    videos_text_kwargs_max_length = 870
    videos_text_kwargs_override_max_length = 870

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 96},
            {"num_frames": None, "fps": 18, "expected_dim": 0, "output_length": 72},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 48},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 48},
            {"expected_dim": 0, "output_length": 96},
        ]

    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    def test_return_metadata_false_is_refused(self):
        processor = self.get_processor()
        video = self.prepare_videos_inputs()
        text = f"Video: {self.video_token}"
        inputs = processor(text=text, videos=[video], do_sample_frames=False, return_metadata=True)
        self.assertIn("video_metadata", inputs)
        with self.assertRaisesRegex(ValueError, "return_metadata=False"):
            processor(text=text, videos=[video], do_sample_frames=False, return_metadata=False)

    def test_video_chat_template_wrapper_is_replaced(self):
        # Regression test for https://github.com/huggingface/transformers/issues/51132: the chat template's
        # `<|vision_start|><|video_pad|><|vision_end|>` must be replaced as a whole, so the expanded video
        # has exactly one vision start/end pair per frame and no extra outer pair
        processor = self.get_processor()
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe this video."}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertIn(processor.vision_start_token + processor.video_token + processor.vision_end_token, prompt)

        inputs = processor(
            text=prompt,
            videos=[self.prepare_videos_inputs()],
            do_sample_frames=False,
            return_tensors="pt",
            return_text_replacement_offsets=True,
        )
        # Offsets point into the original prompt, which `return_assistant_tokens_mask` relies on
        offset = inputs["text_replacement_offsets"][0][0]
        self.assertEqual(prompt[slice(*offset["span"])], offset["text"])
        self.assertEqual(
            offset["text"], processor.vision_start_token + processor.video_token + processor.vision_end_token
        )

        num_frames = inputs["video_grid_thw"][0][0].item()
        input_ids = inputs["input_ids"][0].tolist()
        self.assertEqual(input_ids.count(processor.vision_start_token_id), num_frames)
        self.assertEqual(input_ids.count(processor.vision_end_token_id), num_frames)

        # The first frame's timestamp directly follows the user turn, not an outer `<|vision_start|>`
        decoded = processor.decode(input_ids)
        self.assertRegex(decoded, r"^<\|im_start\|>user\n<\d+\.\d seconds><\|vision_start\|>")

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image", "video"])
        image_input = self.prepare_images_inputs()
        video_inputs = self.prepare_videos_inputs()
        inputs_dict = {"text": text, "images": image_input, "videos": video_inputs}
        inputs = processor(**inputs_dict, return_tensors="pt", do_sample_frames=False)

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_kwargs_overrides_custom_image_processor_kwargs(self):
        processor = self.get_processor()

        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_images_inputs()
        inputs = processor(text=input_str, images=image_input, max_pixels=56 * 56 * 4, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 612)
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 100)

    @unittest.skip("qwen3_vl can't sample frames from image frames directly, user can use `qwen-vl-utils`")
    def test_apply_chat_template_video_1(self):
        pass

    @unittest.skip("qwen3_vl can't sample frames from image frames directly, user can use `qwen-vl-utils`")
    def test_apply_chat_template_video_2(self):
        pass
