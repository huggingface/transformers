# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import shutil
import tempfile
import unittest

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        AutoProcessor,
        LlavaOnevisionImageProcessor,
        LlavaOnevisionProcessor,
        LlavaOnevisionVideoProcessor,
        Qwen2TokenizerFast,
    )


@require_vision
class LlavaOnevisionProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaOnevisionProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = LlavaOnevisionImageProcessor()
        video_processor = LlavaOnevisionVideoProcessor()
        tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        processor = LlavaOnevisionProcessor(
            video_processor=video_processor, image_processor=image_processor, tokenizer=tokenizer
        )
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_Video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
        expected_prompt = "<|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>\n<|im_start|>assistant\n"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        # Rewrite as llava-next image processor return pixel values with an added dimesion for image patches
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", size=(234, 234))
        video_processor = self.get_component("video_processor", size=(234, 234))
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)
        # added dimension for image patches
        self.assertEqual(len(inputs["pixel_values"][0][0][0]), 234)

    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", crop_size=(234, 234))
        video_processor = self.get_component("video_processor", size=(234, 234))
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, size=[224, 224])
        # added dimension for image patches
        self.assertEqual(len(inputs["pixel_values"][0][0][0]), 224)

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")
        tokenizer = self.get_component("tokenizer")
        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            padding="max_length",
            max_length=76,
        )

        # added dimension for image patches
        self.assertEqual(inputs["pixel_values"].shape[3], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")
        tokenizer = self.get_component("tokenizer")
        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )
        self.assertEqual(inputs["pixel_values"].shape[3], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 5)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")
        tokenizer = self.get_component("tokenizer")
        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"size": {"height": 214, "width": 214}},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["pixel_values"].shape[3], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"size": {"height": 214, "width": 214}},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[3], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_doubly_passed_kwargs(self):
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                images_kwargs={"size": {"height": 222, "width": 222}},
                size={"height": 214, "width": 214},
            )

    @require_vision
    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt", max_length=112)
        self.assertEqual(len(inputs["input_ids"][0]), 112)

    @require_vision
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor
        )
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(len(inputs["input_ids"][0]), 117)
