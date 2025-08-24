# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import json
import random
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download

from transformers.models.auto.processing_auto import processor_class_from_name
from transformers.processing_utils import Unpack
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_av,
    require_torch,
    require_vision,
)
from transformers.utils import is_torch_available, is_vision_available


global_rng = random.Random()

if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch


def prepare_image_inputs():
    """This function prepares a list of PIL images"""
    image_inputs = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]
    image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]
    return image_inputs


# Copied from tests.models.whisper.test_feature_extraction_whisper.floats_list
def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


@require_torch
@require_vision
class ProcessorTesterMixin:
    processor_class = None
    text_input_name = "input_ids"
    images_input_name = "pixel_values"
    videos_input_name = "pixel_values_videos"

    def prepare_processor_dict(self):
        return {}

    def get_component(self, attribute, **kwargs):
        assert attribute in self.processor_class.attributes
        component_class_name = getattr(self.processor_class, f"{attribute}_class")
        if isinstance(component_class_name, tuple):
            component_class_name = component_class_name[0]

        component_class = processor_class_from_name(component_class_name)
        component = component_class.from_pretrained(self.tmpdirname, **kwargs)  # noqa
        if "tokenizer" in attribute and not component.pad_token:
            component.pad_token = "[TEST_PAD]"
            if component.pad_token_id is None:
                component.pad_token_id = 0

        return component

    def prepare_components(self):
        components = {}
        for attribute in self.processor_class.attributes:
            component = self.get_component(attribute)
            components[attribute] = component

        return components

    def get_processor(self):
        components = self.prepare_components()
        processor = self.processor_class(**components, **self.prepare_processor_dict())
        return processor

    def prepare_text_inputs(self, batch_size: Optional[int] = None):
        if batch_size is None:
            return "lower newer"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return ["lower newer"]
        return ["lower newer", "upper older longer string"] + ["lower newer"] * (batch_size - 2)

    @require_vision
    def prepare_image_inputs(self, batch_size: Optional[int] = None):
        """This function prepares a list of PIL images for testing"""
        if batch_size is None:
            return prepare_image_inputs()[0]
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return prepare_image_inputs() * batch_size

    @require_vision
    def prepare_video_inputs(self, batch_size: Optional[int] = None):
        """This function prepares a list of numpy videos."""
        video_input = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)] * 8
        if batch_size is None:
            return video_input
        return [video_input] * batch_size

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            # Chat template is saved as a separate file
            if key not in "chat_template":
                self.assertEqual(obj[key], value)
                self.assertEqual(getattr(processor, key, None), value)

    def test_processor_from_and_save_pretrained(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_files = processor_first.save_pretrained(tmpdirname)
            if len(saved_files) > 0:
                check_json_file_has_correct_format(saved_files[0])
                processor_second = self.processor_class.from_pretrained(tmpdirname)

                self.assertEqual(processor_second.to_dict(), processor_first.to_dict())

                for attribute in processor_first.attributes:
                    attribute_first = getattr(processor_first, attribute)
                    attribute_second = getattr(processor_second, attribute)

                    # tokenizer repr contains model-path from where we loaded
                    if "tokenizer" not in attribute:
                        self.assertEqual(repr(attribute_first), repr(attribute_second))

    # These kwargs-related tests ensure that processors are correctly instantiated.
    # they need to be applied only if an image_processor exists.

    def skip_processor_without_typed_kwargs(self, processor):
        # TODO this signature check is to test only uniformized processors.
        # Once all are updated, remove it.
        is_kwargs_typed_dict = False
        call_signature = inspect.signature(processor.__call__)
        for param in call_signature.parameters.values():
            if param.kind == param.VAR_KEYWORD and param.annotation != param.empty:
                is_kwargs_typed_dict = (
                    hasattr(param.annotation, "__origin__") and param.annotation.__origin__ == Unpack
                )
        if not is_kwargs_typed_dict:
            self.skipTest(f"{self.processor_class} doesn't have typed kwargs.")

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 117)

    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        """
        We use do_rescale=True, rescale_factor=-1 to ensure that image_processor kwargs are preserved in the processor.
        We then check that the mean of the pixel_values is less than or equal to 0 after processing.
        Since the original pixel_values are in [0, 255], this is a good indicator that the rescale_factor is indeed applied.
        """
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=-1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str, images=image_input, return_tensors="pt", max_length=112, padding="max_length"
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], 112)

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, do_rescale=True, rescale_factor=-1, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="max_length",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2)
        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="longest",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < 76
        )

    def test_doubly_passed_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = [self.prepare_text_inputs()]
        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                images_kwargs={"do_rescale": True, "rescale_factor": -1},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    #  text + audio kwargs testing
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer(max_length=117, padding="max_length")
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")
        else:
            self.assertTrue(False, "Processor doesn't have get_tokenizer or get_component defined")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        raw_speech = floats_list((3, 1000))
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt")
        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 117)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 117)

    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer(max_length=117)
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer", max_length=117)
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = "lower newer"
        raw_speech = floats_list((3, 1000))
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt", max_length=112, padding="max_length")
        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 112)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 112)

    @require_torch
    def test_unstructured_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer(max_length=117)
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer", max_length=117)
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        raw_speech = floats_list((3, 1000))
        inputs = processor(
            text=input_str,
            audio=raw_speech,
            return_tensors="pt",
            padding="max_length",
            max_length=76,
        )

        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 76)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 76)

    @require_torch
    def test_doubly_passed_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer()
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        raw_speech = floats_list((3, 1000))
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                audio=raw_speech,
                audio_kwargs={"padding": "max_length"},
                padding="max_length",
            )

    @require_torch
    @require_vision
    def test_structured_kwargs_audio_nested(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        feature_extractor = self.get_component("feature_extractor")
        if hasattr(self, "get_tokenizer"):
            tokenizer = self.get_tokenizer()
        elif hasattr(self, "get_component"):
            tokenizer = self.get_component("tokenizer")
        if not tokenizer.pad_token:
            tokenizer.pad_token = "[TEST_PAD]"
        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer"]
        raw_speech = floats_list((3, 1000))

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
            "audio_kwargs": {"padding": "max_length", "max_length": 66},
        }

        inputs = processor(text=input_str, audio=raw_speech, **all_kwargs)
        if "input_ids" in inputs:
            self.assertEqual(len(inputs["input_ids"][0]), 76)
        elif "labels" in inputs:
            self.assertEqual(len(inputs["labels"][0]), 76)

    def test_tokenizer_defaults_preserved_by_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        video_input = self.prepare_video_inputs()
        inputs = processor(text=input_str, videos=video_input, return_tensors="pt")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 117)

    def test_video_processor_defaults_preserved_by_video_kwargs(self):
        """
        We use do_rescale=True, rescale_factor=-1 to ensure that image_processor kwargs are preserved in the processor.
        We then check that the mean of the pixel_values is less than or equal to 0 after processing.
        Since the original pixel_values are in [0, 255], this is a good indicator that the rescale_factor is indeed applied.
        """
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["video_processor"] = self.get_component(
            "video_processor", do_rescale=True, rescale_factor=-1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        video_input = self.prepare_video_inputs()

        inputs = processor(text=input_str, videos=video_input, return_tensors="pt")
        self.assertLessEqual(inputs[self.videos_input_name][0][0][0].mean(), 0)

    def test_kwargs_overrides_default_tokenizer_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        video_input = self.prepare_video_inputs()
        inputs = processor(
            text=input_str, videos=video_input, return_tensors="pt", max_length=112, padding="max_length"
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], 112)

    def test_kwargs_overrides_default_video_processor_kwargs(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["video_processor"] = self.get_component(
            "video_processor", do_rescale=True, rescale_factor=1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        video_input = self.prepare_video_inputs()

        inputs = processor(text=input_str, videos=video_input, do_rescale=True, rescale_factor=-1, return_tensors="pt")
        self.assertLessEqual(inputs[self.videos_input_name][0][0][0].mean(), 0)

    def test_unstructured_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        video_input = self.prepare_video_inputs()
        inputs = processor(
            text=input_str,
            videos=video_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="max_length",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_unstructured_kwargs_batched_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2)
        video_input = self.prepare_video_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            videos=video_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="longest",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0][0][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < 76
        )

    def test_doubly_passed_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = [self.prepare_text_inputs()]
        video_input = self.prepare_video_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                videos=video_input,
                videos_kwargs={"do_rescale": True, "rescale_factor": -1},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_structured_kwargs_nested_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        video_input = self.prepare_video_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "videos_kwargs": {"do_rescale": True, "rescale_factor": -1},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, videos=video_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertLessEqual(inputs[self.videos_input_name][0][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_structured_kwargs_nested_from_dict_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        video_input = self.prepare_video_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "videos_kwargs": {"do_rescale": True, "rescale_factor": -1},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, videos=video_input, **all_kwargs)
        self.assertLessEqual(inputs[self.videos_input_name][0][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    # TODO: the same test, but for audio + text processors that have strong overlap in kwargs
    # TODO (molbap) use the same structure of attribute kwargs for other tests to avoid duplication
    def test_overlapping_text_kwargs_handling(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                padding="max_length",
                text_kwargs={"padding": "do_not_pad"},
            )

    def test_prepare_and_validate_optional_call_args(self):
        processor = self.get_processor()
        optional_call_args_name = getattr(processor, "optional_call_args", [])
        num_optional_call_args = len(optional_call_args_name)
        if num_optional_call_args == 0:
            self.skipTest("No optional call args")
        # test all optional call args are given
        optional_call_args = processor.prepare_and_validate_optional_call_args(
            *(f"optional_{i}" for i in range(num_optional_call_args))
        )
        self.assertEqual(
            optional_call_args, {arg_name: f"optional_{i}" for i, arg_name in enumerate(optional_call_args_name)}
        )
        # test only one optional call arg is given
        optional_call_args = processor.prepare_and_validate_optional_call_args("optional_1")
        self.assertEqual(optional_call_args, {optional_call_args_name[0]: "optional_1"})
        # test no optional call arg is given
        optional_call_args = processor.prepare_and_validate_optional_call_args()
        self.assertEqual(optional_call_args, {})
        # test too many optional call args are given
        with self.assertRaises(ValueError):
            processor.prepare_and_validate_optional_call_args(
                *(f"optional_{i}" for i in range(num_optional_call_args + 1))
            )

    def test_chat_template_save_loading(self):
        processor = self.get_processor()
        signature = inspect.signature(processor.__init__)
        if "chat_template" not in {*signature.parameters.keys()}:
            self.skipTest("Processor doesn't accept chat templates at input")

        existing_tokenizer_template = getattr(processor.tokenizer, "chat_template", None)
        processor.chat_template = "test template"
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.save_pretrained(tmpdirname)
            self.assertTrue(Path(tmpdirname, "chat_template.json").is_file())
            self.assertFalse(Path(tmpdirname, "chat_template.jinja").is_file())
            reloaded_processor = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
            # When we don't use single-file chat template saving, processor and tokenizer chat templates
            # should remain separate
            self.assertEqual(getattr(reloaded_processor.tokenizer, "chat_template", None), existing_tokenizer_template)

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.save_pretrained(tmpdirname, save_raw_chat_template=True)
            self.assertTrue(Path(tmpdirname, "chat_template.jinja").is_file())
            self.assertFalse(Path(tmpdirname, "chat_template.json").is_file())
            reloaded_processor = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
            # When we save as single files, tokenizers and processors share a chat template, which means
            # the reloaded tokenizer should get the chat template as well
            self.assertEqual(reloaded_processor.chat_template, reloaded_processor.tokenizer.chat_template)

    def test_chat_template_single(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors=None
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        expected_output = processor.tokenizer(
            formatted_prompt, return_tensors=None, add_special_tokens=add_special_tokens
        ).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertTrue(all(key in out_dict for key in ["input_ids", "attention_mask"]))

        # Now test the ability to return dict
        messages[0][0]["content"].append(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertTrue(self.images_input_name in out_dict)

        # should always have input_ids and attention_mask
        self.assertEqual(len(out_dict["input_ids"]), 1)
        self.assertEqual(len(out_dict["attention_mask"]), 1)
        self.assertEqual(len(out_dict[self.images_input_name]), 1)

    def test_chat_template_batched(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        batched_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see?"},
                    ],
                },
            ],
        ]

        formatted_prompt = processor.apply_chat_template(batched_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 2)

        formatted_prompt_tokenized = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, tokenize=True, padding=True, return_tensors=None
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        expected_output = processor.tokenizer(
            formatted_prompt,
            return_tensors=None,
            padding=True,
            add_special_tokens=add_special_tokens,
        ).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(
            batched_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
        )
        self.assertTrue(all(key in out_dict for key in ["input_ids", "attention_mask"]))

        # Now test the ability to return dict
        batched_messages[0][0]["content"].append(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
        batched_messages[1][0]["content"].append(
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"}
        )
        out_dict = processor.apply_chat_template(
            batched_messages, add_generation_prompt=True, tokenize=True, return_dict=True, padding=True
        )
        self.assertTrue(self.images_input_name in out_dict)

        # should always have input_ids and attention_mask
        self.assertEqual(len(out_dict["input_ids"]), 2)
        self.assertEqual(len(out_dict["attention_mask"]), 2)
        self.assertEqual(len(out_dict[self.images_input_name]), 2)

    def test_chat_template_accepts_processing_kwargs(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ]
        ]

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            truncation=True,
            max_length=50,
        )
        self.assertEqual(len(formatted_prompt_tokenized[0]), 50)

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            truncation=True,
            max_length=5,
        )
        self.assertEqual(len(formatted_prompt_tokenized[0]), 5)

        # Now test the ability to return dict
        messages[0][0]["content"].append(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
        out_dict = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            do_rescale=True,
            rescale_factor=-1,
            return_tensors="np",
        )
        self.assertLessEqual(out_dict[self.images_input_name][0][0].mean(), 0)

    @require_torch
    def test_chat_template_dict_torch(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        out_dict_tensors = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertTrue(self.images_input_name in out_dict_tensors)
        for k in out_dict_tensors:
            self.assertIsInstance(out_dict_tensors[k], torch.Tensor)

    @require_av
    def test_chat_template_video(self):
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
                        {"type": "video"},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors=None
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        expected_output = processor.tokenizer(
            formatted_prompt,
            return_tensors=None,
            add_special_tokens=add_special_tokens,
        ).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertTrue(all(key in out_dict for key in ["input_ids", "attention_mask"]))

        # Add video URL for return dict and load with `num_frames` arg
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4",
        }
        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), num_frames)

        # Load with `video_fps` arg
        video_fps = 1
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            video_fps=video_fps,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), video_fps * 10)

        # Load with `video_fps` and `num_frames` args, should raise an error
        with self.assertRaises(ValueError):
            out_dict_with_video = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                video_fps=video_fps,
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
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 300)

        # Load video as a list of frames (i.e. images). NOTE: each frame should have same size
        # because we assume they come from one video
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": [
                "https://www.ilankelman.org/stopsigns/australia.jpg",
                "https://www.ilankelman.org/stopsigns/australia.jpg",
            ],
        }
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 2)

    @require_av
    def test_chat_template_video_custom_sampling(self):
        """
        Tests that models can pass their custom callables to sample video indices.
        """
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest("Processor doesn't accept videos at input")

        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "path": video_file_path,
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        def dummy_sample_indices_fn(metadata, **fn_kwargs):
            # sample only the first two frame always
            return [0, 1]

        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            sample_indices_fn=dummy_sample_indices_fn,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 2)

    @require_av
    def test_chat_template_video_special_processing(self):
        """
        Tests that models can use their own preprocessing to preprocess conversations.
        """
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest("Processor doesn't accept videos at input")

        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": video_file_path},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        def _process_messages_for_chat_template(
            conversation,
            batch_images,
            batch_videos,
            batch_video_metadata,
            **chat_template_kwargs,
        ):
            # Let us just always return a dummy prompt
            new_msg = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},  # no need to use path, video is loaded already by this moment
                            {"type": "text", "text": "Dummy prompt for preprocess testing"},
                        ],
                    },
                ]
            ]
            return new_msg

        processor._process_messages_for_chat_template = _process_messages_for_chat_template
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)

        # Check with `in` because we don't know how each template formats the prompt with BOS/EOS/etc
        formatted_text = processor.batch_decode(out_dict_with_video["input_ids"], skip_special_tokens=True)[0]
        self.assertTrue("Dummy prompt for preprocess testing" in formatted_text)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 243)
