# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_av, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import Glm46VProcessor

if is_torch_available():
    pass


@require_vision
@require_torch
class Glm46VProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Glm46VProcessor
    # Tiny processor created with make_tiny_processor.py from "THUDM/GLM-4.1V-9B-Thinking"
    tiny_model_id = "hf-internal-testing/tiny-processor-glm4v"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(
            model_id,
            do_sample_frames=False,
            patch_size=4,
            size={"shortest_edge": 12 * 12, "longest_edge": 18 * 18},
            **kwargs,
        )

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
                                "https://huggingface.co/datasets/hf-internal-testing/test-videos/resolve/main/tiny_video_320x240.mp4"
                            ),
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        # Load with `video_fps` arg
        video_fps = 10
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            fps=video_fps,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 8)

        # Load the whole video
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            do_sample_frames=False,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 24)

        # Load video as a list of frames (i.e. images). NOTE: each frame should have same size
        # because we assume they come from one video
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": [
                url_to_local_path(
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
                ),
                url_to_local_path(
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
                ),
            ],
        }
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            do_sample_frames=False,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 4)

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

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image", "video"])
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        inputs_dict = {"text": text, "images": image_input, "videos": video_inputs}
        inputs = processor(**inputs_dict, return_tensors="pt", do_sample_frames=False)

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))
