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
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers.models.auto.processing_auto import processor_class_from_name
from transformers.processing_utils import Unpack
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_av,
    require_librosa,
    require_torch,
    require_vision,
)
from transformers.utils import is_torch_available, is_vision_available


sys.path.append(".")
from utils.fetch_hub_objects_for_ci import url_to_local_path


global_rng = random.Random()

if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

MODALITY_INPUT_DATA = {
    "images": [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
    ],
    "videos": [
        "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/Big_Buck_Bunny_720_10s_10MB.mp4",
        "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
    ],
    "audio": [
        "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/glass-breaking-151256.mp3",
        "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav",
    ],
}


for modality, urls in MODALITY_INPUT_DATA.items():
    MODALITY_INPUT_DATA[modality] = [url_to_local_path(url) for url in urls]


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
    audio_input_name = "input_features"

    @staticmethod
    def prepare_processor_dict():
        return {}

    def get_component(self, attribute, **kwargs):
        assert attribute in self.processor_class.attributes
        component_class_name = getattr(self.processor_class, f"{attribute}_class")
        if isinstance(component_class_name, tuple):
            if attribute == "image_processor":
                # TODO: @yoni, change logic in v4.52 (when use_fast set to True by default)
                component_class_name = component_class_name[0]
            else:
                component_class_name = component_class_name[-1]

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

    def prepare_text_inputs(self, batch_size: Optional[int] = None, modalities: Optional[Union[str, list]] = None):
        if isinstance(modalities, str):
            modalities = [modalities]

        special_token_to_add = ""
        if modalities is not None:
            for modality in modalities:
                special_token_to_add += getattr(self, f"{modality}_token", "")

        if batch_size is None:
            return f"lower newer {special_token_to_add}"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return [f"lower newer {special_token_to_add}"]
        return [f"lower newer {special_token_to_add}", f" {special_token_to_add} upper older longer string"] + [
            f"lower newer {special_token_to_add}"
        ] * (batch_size - 2)

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
        video_input = np.array(video_input)
        if batch_size is None:
            return video_input
        return [video_input] * batch_size

    def prepare_audio_inputs(self, batch_size: Optional[int] = None):
        """This function prepares a list of numpy audio."""
        raw_speech = floats_list((1, 1000))
        raw_speech = [np.asarray(audio) for audio in raw_speech]
        if batch_size is None:
            return raw_speech
        return raw_speech * batch_size

    def test_processor_to_json_string(self):
        processor = self.get_processor()
        obj = json.loads(processor.to_json_string())
        for key, value in self.prepare_processor_dict().items():
            # Chat template is saved as a separate file
            if key not in "chat_template":
                # json converts dict keys to str, but some processors force convert back to int when init
                if (
                    isinstance(obj[key], dict)
                    and isinstance(list(obj[key].keys())[0], str)
                    and isinstance(list(value.keys())[0], int)
                ):
                    obj[key] = {int(k): v for k, v in obj[key].items()}
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

    def test_processor_from_and_save_pretrained_as_nested_dict(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save with `legacy_serialization=False` so that all attrbiutes are saved in one json file
            saved_files = processor_first.save_pretrained(tmpdirname, legacy_serialization=False)
            check_json_file_has_correct_format(saved_files[0])

            # Load it back and check if loaded correctly
            processor_second = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor_second.to_dict(), processor_first.to_dict())

            # Try to load each attribute separately from saved directory
            for attribute in processor_first.attributes:
                attribute_class_name = getattr(processor_first, f"{attribute}_class")
                if isinstance(attribute_class_name, tuple):
                    if attribute == "image_processor":
                        # TODO: @yoni, change logic in v4.52 (when use_fast set to True by default)
                        attribute_class_name = attribute_class_name[0]
                    else:
                        attribute_class_name = attribute_class_name[-1]

                attribute_class = processor_class_from_name(attribute_class_name)
                attribute_reloaded = attribute_class.from_pretrained(tmpdirname)
                attribute_first = getattr(processor_first, attribute)

                # tokenizer repr contains model-path from where we loaded
                if "tokenizer" not in attribute:
                    self.assertEqual(repr(attribute_first), repr(attribute_reloaded))

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image", "video", "audio"])
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        audio_inputs = self.prepare_audio_inputs()
        inputs_dict = {"text": text, "images": image_input, "videos": video_inputs, "audio": audio_inputs}

        call_signature = inspect.signature(processor.__call__)
        input_args = [param.name for param in call_signature.parameters.values()]
        inputs_dict = {k: v for k, v in inputs_dict.items() if k in input_args}

        inputs = processor(**inputs_dict, return_tensors="pt")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_processor_text_has_no_visual(self):
        """
        Tests that multimodal models can process batch of inputs where samples can
        be with images/videos or without. See https://github.com/huggingface/transformers/issues/40263
        """
        processor = self.get_processor()
        call_signature = inspect.signature(processor.__call__)
        input_args = [param.name for param in call_signature.parameters.values() if param.annotation != param.empty]

        if not ("text" in input_args and ("images" in input_args and "videos" in input_args)):
            self.skipTest(f"{self.processor_class} doesn't support several vision modalities with text.")

        # Prepare inputs and filter by input signature. Make sure to use a high batch size, we'll set some
        # samples to text-only later
        text = self.prepare_text_inputs(batch_size=3, modalities=["image", "video"])
        image_inputs = self.prepare_image_inputs(batch_size=3)
        video_inputs = self.prepare_video_inputs(batch_size=3)
        inputs_dict = {"text": text, "images": image_inputs, "videos": video_inputs}
        inputs_dict = {k: v for k, v in inputs_dict.items() if k in input_args}

        processing_kwargs = {"return_tensors": "pt", "padding": True}
        if "videos" in inputs_dict:
            processing_kwargs["do_sample_frames"] = False

        # Firts call processor with all inputs and use nested input type, which is the format supported by all multimodal processors
        image_inputs_nested = [[image] if not isinstance(image, list) else image for image in image_inputs]
        video_inputs_nested = [[video] for video in video_inputs]
        inputs_dict_nested = {"text": text, "images": image_inputs_nested, "videos": video_inputs_nested}
        inputs_dict_nested = {k: v for k, v in inputs_dict_nested.items() if k in input_args}
        inputs = processor(**inputs_dict_nested, **processing_kwargs)
        self.assertTrue(self.text_input_name in inputs)

        # Now call with one of the samples with no associated vision input. Let's set the first input to be a plain text
        # with no placeholder tokens and no images/videos. The final format would be `images = [[], [image2], [image3]]`
        plain_text = "lower newer"
        image_inputs_nested[0] = []
        video_inputs_nested[0] = []
        text[0] = plain_text
        inputs_dict_no_vision = {"text": text, "images": image_inputs_nested, "videos": video_inputs_nested}
        inputs_dict_no_vision = {k: v for k, v in inputs_dict_no_vision.items() if k in input_args}
        inputs_nested = processor(**inputs_dict_no_vision, **processing_kwargs)

        # Check that text samples are same and are expanded with placeholder tokens correctly. First sample
        # has no vision input associated, so we skip it and check it has no vision
        self.assertListEqual(
            inputs[self.text_input_name][1:].tolist(), inputs_nested[self.text_input_name][1:].tolist()
        )

        # Now test if we can apply chat templates with no vision inputs in one of the samples
        # NOTE: we don't skip the test as we want the above to be checked even if process has to chat template
        if processor.chat_template is not None:
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is the capital of France?"},
                        ],
                    },
                ],
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is the capital of France?"},
                            {
                                "type": "image",
                                "url": url_to_local_path(
                                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
                                ),
                            },
                        ],
                    },
                ],
            ]

            inputs_chat_template = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            self.assertTrue(self.text_input_name in inputs_chat_template)

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
        input_str = self.prepare_text_inputs(modalities="image")
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

        input_str = self.prepare_text_inputs(modalities="image")
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
        input_str = self.prepare_text_inputs(modalities="image")
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

        input_str = self.prepare_text_inputs(modalities="image")
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

        input_str = self.prepare_text_inputs(modalities="image")
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

        input_str = self.prepare_text_inputs(batch_size=2, modalities="image")
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

        input_str = [self.prepare_text_inputs(modalities="image")]
        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                images_kwargs={"do_rescale": True, "rescale_factor": -1},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_args_overlap_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_first = self.get_processor()
        image_processor = processor_first.image_processor
        image_processor.is_override = True

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_first.save_pretrained(tmpdirname)
            processor_second = self.processor_class.from_pretrained(tmpdirname, image_processor=image_processor)
            self.assertTrue(processor_second.image_processor.is_override)

    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(modalities="image")
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
        input_str = self.prepare_text_inputs(modalities="image")
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

    # text + audio kwargs testing
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer", max_length=300, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=3, modalities="audio")
        raw_speech = self.prepare_audio_inputs(batch_size=3)
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt")
        self.assertEqual(len(inputs[self.text_input_name][0]), 300)

    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer", max_length=117)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=3, modalities="audio")
        raw_speech = self.prepare_audio_inputs(batch_size=3)
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt", max_length=300, padding="max_length")

        self.assertEqual(len(inputs[self.text_input_name][0]), 300)

    @require_torch
    def test_unstructured_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=3, modalities="audio")
        raw_speech = self.prepare_audio_inputs(batch_size=3)
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt", max_length=300, padding="max_length")

        self.assertEqual(len(inputs[self.text_input_name][0]), 300)

    @require_torch
    def test_doubly_passed_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=3, modalities="audio")
        raw_speech = self.prepare_audio_inputs(batch_size=3)
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                audio=raw_speech,
                text_kwargs={"padding": "max_length"},
                padding="max_length",
            )

    @require_torch
    @require_vision
    def test_structured_kwargs_audio_nested(self):
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        feature_extractor = self.get_component("feature_extractor")
        tokenizer = self.get_component("tokenizer", max_length=117)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(tokenizer=tokenizer, feature_extractor=feature_extractor, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=3, modalities="audio")
        raw_speech = self.prepare_audio_inputs(batch_size=3)

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
            "audio_kwargs": {"padding": "max_length", "max_length": 300},
        }

        inputs = processor(text=input_str, audio=raw_speech, **all_kwargs)
        self.assertEqual(len(inputs[self.text_input_name][0]), 76)

    def test_tokenizer_defaults_preserved_by_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=167, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()
        inputs = processor(text=input_str, videos=video_input, do_sample_frames=False, return_tensors="pt")
        self.assertEqual(inputs[self.text_input_name].shape[-1], 167)

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
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=167, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()

        inputs = processor(text=input_str, videos=video_input, do_sample_frames=False, return_tensors="pt")
        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)

    def test_kwargs_overrides_default_tokenizer_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()
        inputs = processor(
            text=input_str,
            videos=video_input,
            do_sample_frames=False,
            return_tensors="pt",
            max_length=162,
            padding="max_length",
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], 162)

    def test_kwargs_overrides_default_video_processor_kwargs(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["video_processor"] = self.get_component(
            "video_processor", do_rescale=True, rescale_factor=1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=167, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()

        inputs = processor(
            text=input_str,
            videos=video_input,
            do_sample_frames=False,
            do_rescale=True,
            rescale_factor=-1,
            return_tensors="pt",
        )
        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)

    def test_unstructured_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()
        inputs = processor(
            text=input_str,
            videos=video_input,
            do_sample_frames=False,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="max_length",
            max_length=176,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 176)

    def test_unstructured_kwargs_batched_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2, modalities="video")
        video_input = self.prepare_video_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            videos=video_input,
            do_sample_frames=False,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="longest",
            max_length=176,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < 176
        )

    def test_doubly_passed_kwargs_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = [self.prepare_text_inputs(modalities="video")]
        video_input = self.prepare_video_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                videos=video_input,
                do_sample_frames=False,
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

        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "videos_kwargs": {"do_rescale": True, "rescale_factor": -1, "do_sample_frames": False},
            "text_kwargs": {"padding": "max_length", "max_length": 176},
        }

        inputs = processor(text=input_str, videos=video_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 176)

    def test_structured_kwargs_nested_from_dict_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs(modalities="video")
        video_input = self.prepare_video_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "videos_kwargs": {"do_rescale": True, "rescale_factor": -1, "do_sample_frames": False},
            "text_kwargs": {"padding": "max_length", "max_length": 176},
        }

        inputs = processor(text=input_str, videos=video_input, **all_kwargs)
        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 176)

    # TODO: the same test, but for audio + text processors that have strong overlap in kwargs
    # TODO (molbap) use the same structure of attribute kwargs for other tests to avoid duplication
    def test_overlapping_text_image_kwargs_handling(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(modalities="image")
        image_input = self.prepare_image_inputs()

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                padding="max_length",
                text_kwargs={"padding": "do_not_pad"},
            )

    def test_overlapping_text_audio_kwargs_handling(self):
        """
        Checks that `padding`, or any other overlap arg between audio extractor and tokenizer
        is be passed to only text and ignored for audio for BC purposes
        """
        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=3, modalities="audio")
        audio_lengths = [4000, 8000, 16000, 32000]
        raw_speech = [np.asarray(audio)[:length] for audio, length in zip(floats_list((3, 32_000)), audio_lengths)]

        # padding = True should not raise an error and will if the audio processor popped its value to None
        _ = processor(text=input_str, audio=raw_speech, padding=True, return_tensors="pt")

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
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        signature = inspect.signature(processor.__init__)
        if "chat_template" not in {*signature.parameters.keys()}:
            self.skipTest("Processor doesn't accept chat templates at input")

        existing_tokenizer_template = getattr(processor.tokenizer, "chat_template", None)
        processor.chat_template = "test template"
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.save_pretrained(tmpdirname, save_jinja_files=False)
            self.assertTrue(Path(tmpdirname, "chat_template.json").is_file())
            self.assertFalse(Path(tmpdirname, "chat_template.jinja").is_file())
            reloaded_processor = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
            # When we don't use single-file chat template saving, processor and tokenizer chat templates
            # should remain separate
            self.assertEqual(getattr(reloaded_processor.tokenizer, "chat_template", None), existing_tokenizer_template)

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.save_pretrained(tmpdirname)
            self.assertTrue(Path(tmpdirname, "chat_template.jinja").is_file())
            self.assertFalse(Path(tmpdirname, "chat_template.json").is_file())
            self.assertFalse(Path(tmpdirname, "additional_chat_templates").is_dir())
            reloaded_processor = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
            # When we save as single files, tokenizers and processors share a chat template, which means
            # the reloaded tokenizer should get the chat template as well
            self.assertEqual(reloaded_processor.chat_template, reloaded_processor.tokenizer.chat_template)

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.chat_template = {"default": "a", "secondary": "b"}
            processor.save_pretrained(tmpdirname)
            self.assertTrue(Path(tmpdirname, "chat_template.jinja").is_file())
            self.assertFalse(Path(tmpdirname, "chat_template.json").is_file())
            self.assertTrue(Path(tmpdirname, "additional_chat_templates").is_dir())
            reloaded_processor = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)
            # When we save as single files, tokenizers and processors share a chat template, which means
            # the reloaded tokenizer should get the chat template as well
            self.assertEqual(reloaded_processor.chat_template, reloaded_processor.tokenizer.chat_template)

        with self.assertRaises(ValueError):
            # Saving multiple templates in the legacy format is not permitted
            with tempfile.TemporaryDirectory() as tmpdirname:
                processor.chat_template = {"default": "a", "secondary": "b"}
                processor.save_pretrained(tmpdirname, save_jinja_files=False)

    @require_torch
    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list[str],
    ):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        if processor_name not in self.processor_class.attributes:
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

        # some models have only Fast image processor
        if getattr(processor, processor_name).__class__.__name__.endswith("Fast"):
            return_tensors = "pt"

        batch_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": "Describe this."}]},
            ]
        ] * batch_size

        # Test that jinja can be applied
        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        # Test that tokenizing with template and directly with `self.tokenizer` gives same output
        formatted_prompt_tokenized = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors=return_tensors
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        tok_output = processor.tokenizer(
            formatted_prompt, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )
        expected_output = tok_output.input_ids
        self.assertListEqual(expected_output.tolist(), formatted_prompt_tokenized.tolist())

        # Test that kwargs passed to processor's `__call__` are actually used
        tokenized_prompt_100 = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
            max_length=100,
        )
        self.assertEqual(len(tokenized_prompt_100[0]), 100)

        # Test that `return_dict=True` returns text related inputs in the dict
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

        # Test that with modality URLs and `return_dict=True`, we get modality inputs in the dict
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx][1]["content"] = [batch_messages[idx][1]["content"][0], {"type": modality, "url": url}]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            num_frames=2,  # by default no more than 2 frames, otherwise too slow
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)
        self.assertEqual(len(out_dict[input_name]), batch_size)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

        # Test continue from final message
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is the sound of"}],
        }
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx] = batch_messages[idx] + [assistant_message]
        continue_prompt = processor.apply_chat_template(batch_messages, continue_final_message=True, tokenize=False)
        for prompt in continue_prompt:
            self.assertTrue(prompt.endswith("It is the sound of"))  # no `eos` token at the end

    @require_librosa
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    def test_apply_chat_template_audio(self, batch_size: int, return_tensors: str):
        self._test_apply_chat_template(
            "audio", batch_size, return_tensors, "audio_input_name", "feature_extractor", MODALITY_INPUT_DATA["audio"]
        )

    @require_av
    @parameterized.expand([(1, "pt")])
    def test_apply_chat_template_decoded_video(self, batch_size: int, return_tensors: str):
        dummy_preloaded_video = np.array(self.prepare_video_inputs())
        input_data = [dummy_preloaded_video]
        self._test_apply_chat_template(
            "video", batch_size, return_tensors, "videos_input_name", "video_processor", input_data
        )

    @require_av
    @parameterized.expand([(1, "pt"), (2, "pt")])  # video processor supports only torchvision
    def test_apply_chat_template_video(self, batch_size: int, return_tensors: str):
        self._test_apply_chat_template(
            "video", batch_size, return_tensors, "videos_input_name", "video_processor", MODALITY_INPUT_DATA["videos"]
        )

    @parameterized.expand([(1, "pt"), (2, "pt")])  # fast image processors supports only torchvision
    def test_apply_chat_template_image(self, batch_size: int, return_tensors: str):
        self._test_apply_chat_template(
            "image", batch_size, return_tensors, "images_input_name", "image_processor", MODALITY_INPUT_DATA["images"]
        )

    @require_torch
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
                                "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/tiny_video.mp4"
                            ),
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
            return_tensors="pt",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), num_frames)

        # Load with `fps` arg
        fps = 10
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            fps=fps,
            return_tensors="pt",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        # 3 frames are inferred from input video's length and FPS, so can be hardcoded
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 3)

        # Whan `do_sample_frames=False` no sampling is done and whole video is loaded, even if number of frames is passed
        fps = 10
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            do_sample_frames=False,
            fps=fps,
            return_tensors="pt",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 11)

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
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 11)

        # Load video as a list of frames (i.e. images).
        # NOTE: each frame should have same size because we assume they come from one video
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": [
                url_to_local_path(
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
                )
            ]
            * 2,
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

    @require_librosa
    @require_av
    def test_chat_template_audio_from_video(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest(f"{self.processor_class} does not support video inputs")

        if "feature_extractor" not in self.processor_class.attributes:
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")

        video_file_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_file_path},
                    {"type": "text", "text": "Which of these animals is making the sound?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is a cow."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me all about this animal."},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)  # batch size=1

        out_dict = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
            load_audio_from_video=True,
        )
        self.assertTrue(self.audio_input_name in out_dict)
        self.assertTrue(self.videos_input_name in out_dict)

        # should always have input_ids and attention_mask
        self.assertEqual(len(out_dict["input_ids"]), 1)  # batch-size=1
        self.assertEqual(len(out_dict["attention_mask"]), 1)  # batch-size=1
        self.assertEqual(len(out_dict[self.audio_input_name]), 1)  # 1 audio in the conversation
        self.assertEqual(len(out_dict[self.videos_input_name]), 1)  # 1 video in the conversation

    def test_chat_template_jinja_kwargs(self):
        """Tests that users can pass any kwargs and they will be used in jinja templates."""
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which of these animals is making the sound?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is a cow."}],
            },
        ]

        dummy_template = (
            "{% for message in messages %}"
            "{% if add_system_prompt %}"
            "{{'You are a helpful assistant.'}}"
            "{% endif %}"
            "{% if (message['role'] != 'assistant') %}"
            "{{'<|special_start|>' + message['role'] + '\n' + message['content'][0]['text'] + '<|special_end|>' + '\n'}}"
            "{% elif (message['role'] == 'assistant')%}"
            "{{'<|special_start|>' + message['role'] + '\n'}}"
            "{{message['content'][0]['text'] + '<|special_end|>' + '\n'}}"
            "{% endif %}"
            "{% endfor %}"
        )

        formatted_prompt = processor.apply_chat_template(
            messages, add_system_prompt=True, tokenize=False, chat_template=dummy_template
        )
        expected_prompt = "You are a helpful assistant.<|special_start|>user\nWhich of these animals is making the sound?<|special_end|>\nYou are a helpful assistant.<|special_start|>assistant\nIt is a cow.<|special_end|>\n"
        self.assertEqual(formatted_prompt, expected_prompt)

    @require_torch
    def test_apply_chat_template_assistant_mask(self):
        processor = self.get_processor()

        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is the capital of France?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The capital of France is Paris."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What about Italy?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The capital of Italy is Rome."},
                    ],
                },
            ]
        ]

        dummy_template = (
            "{% for message in messages %}"
            "{% if (message['role'] != 'assistant') %}"
            "{{'<|special_start|>' + message['role'] + '\n' + message['content'][0]['text'] + '<|special_end|>' + '\n'}}"
            "{% elif (message['role'] == 'assistant')%}"
            "{{'<|special_start|>' + message['role'] + '\n'}}"
            "{% generation %}"
            "{{message['content'][0]['text'] + '<|special_end|>' + '\n'}}"
            "{% endgeneration %}"
            "{% endif %}"
            "{% endfor %}"
        )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            chat_template=dummy_template,
        )
        self.assertTrue("assistant_masks" in inputs)
        self.assertEqual(len(inputs["assistant_masks"]), len(inputs["input_ids"]))

        mask = inputs["assistant_masks"].bool()
        assistant_ids = inputs["input_ids"][mask]

        assistant_text = (
            "The capital of France is Paris.<|special_end|>\nThe capital of Italy is Rome.<|special_end|>\n"
        )

        # Some tokenizers add extra spaces which aren't then removed when decoding, so we need to check token ids
        # if we can't get identical text outputs
        text_is_same = assistant_text == processor.decode(assistant_ids, clean_up_tokenization_spaces=True)
        ids_is_same = processor.tokenizer.encode(assistant_text, add_special_tokens=False), assistant_ids.tolist()
        self.assertTrue(text_is_same or ids_is_same)
