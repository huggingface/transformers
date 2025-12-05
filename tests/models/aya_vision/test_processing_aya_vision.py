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

import unittest

from transformers import AyaVisionProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_torch_available():
    import torch


@require_vision
class AyaVisionProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = AyaVisionProcessor
    model_id = "hf-internal-testing/namespace-CohereForAI-repo_name_aya-vision-8b"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained(cls.model_id, padding_side="left")

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(
            do_resize=True,
            size={"height": 20, "width": 20},
            max_patches=2,
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            do_convert_rgb=True,
        )

    @staticmethod
    def prepare_processor_dict():
        return {"patch_size": 10, "img_size": 20}

    @unittest.skip(reason="Text needs image tokens, tested in other tests")
    def test_processor_with_multiple_inputs(self):
        pass

    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    @require_torch
    def test_process_interleaved_images_videos(self):
        processor = self.get_processor()

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": url_to_local_path(
                                "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                            ),
                        },
                        {
                            "type": "image",
                            "url": url_to_local_path(
                                "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"
                            ),
                        },
                        {"type": "text", "text": "What are the differences between these two images?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": url_to_local_path("https://llava-vl.github.io/static/images/view.jpg"),
                        },
                        {"type": "text", "text": "Write a haiku for this image"},
                    ],
                }
            ],
        ]

        inputs_batched = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        # Process non batched inputs to check if the pixel_values and input_ids are reconstructed in the correct order when batched together
        images_patches_index = 0
        for i, message in enumerate(messages):
            inputs = processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            # We slice with [-inputs["input_ids"].shape[1] :] as the input_ids are left padded
            torch.testing.assert_close(
                inputs["input_ids"][0], inputs_batched["input_ids"][i][-inputs["input_ids"].shape[1] :]
            )
            torch.testing.assert_close(
                inputs["pixel_values"],
                inputs_batched["pixel_values"][
                    images_patches_index : images_patches_index + inputs["pixel_values"].shape[0]
                ],
            )
            images_patches_index += inputs["pixel_values"].shape[0]
