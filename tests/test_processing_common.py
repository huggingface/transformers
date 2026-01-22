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
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers.processing_utils import (
    MODALITY_TO_AUTOPROCESSOR_MAPPING,
    Unpack,
)
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_av,
    require_librosa,
    require_torch,
    require_vision,
)
from transformers.utils import is_torch_available, is_vision_available


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(parent_dir, "utils"))
from fetch_hub_objects_for_ci import url_to_local_path  # noqa: E402


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
    model_id = (
        None  # Optional: set this to load from a specific pretrained model instead of creating generic components
    )
    text_input_name = "input_ids"
    images_input_name = "pixel_values"
    videos_input_name = "pixel_values_videos"
    audio_input_name = "input_features"

    @classmethod
    def setUpClass(cls):
        """
        Automatically set up the processor test by creating and saving all required components.
        Individual test classes only need to set processor_class and optionally:
        - model_id: to load components from a specific pretrained model
        - prepare_processor_dict(): to provide custom kwargs for processor initialization
        """
        if cls.processor_class is None:
            raise ValueError(
                f"{cls.__name__} must define 'processor_class' attribute. Example: processor_class = MyProcessor"
            )

        cls.tmpdirname = tempfile.mkdtemp()

        # If model_id is specified, load components from that model
        if cls.model_id is not None:
            processor = cls._setup_from_pretrained(cls.model_id)
        else:
            # Otherwise, create generic components
            processor = cls._setup_from_components()

        # setup test attributes
        cls._setup_test_attributes(processor)
        processor.save_pretrained(cls.tmpdirname)

    @classmethod
    def _setup_test_attributes(cls, processor):
        # to override in the child class to define class attributes
        # such as image_token, video_token, audio_token, etc.
        pass

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        """Load all components from a pretrained model."""

        # check if there are any custom components to setup
        custom_components = {}
        for attribute in cls.processor_class.get_attributes():
            if hasattr(cls, f"_setup_{attribute}"):
                custom_method = getattr(cls, f"_setup_{attribute}")
                custom_components[attribute] = custom_method()
        # if there is one custom component, we need to add all the other ones (with from_pretrained)
        if custom_components:
            for attribute in cls.processor_class.get_attributes():
                if attribute not in custom_components:
                    component_class = cls._get_component_class_from_processor(attribute)
                    custom_components[attribute] = component_class.from_pretrained(model_id)

        kwargs.update(cls.prepare_processor_dict())
        processor = cls.processor_class.from_pretrained(model_id, **custom_components, **kwargs)
        return processor

    @classmethod
    def _setup_from_components(cls):
        """Create all required components for the processor and save the complete processor."""
        # Get all required attributes for this processor
        attributes = cls.processor_class.get_attributes()

        # Create each component (but don't save them individually)
        components = {}
        for attribute in attributes:
            components[attribute] = cls._setup_component(attribute)

        processor_kwargs = cls.prepare_processor_dict()
        processor = cls.processor_class(**components, **processor_kwargs)
        return processor

    @classmethod
    def _setup_component(cls, attribute):
        """
        Create and return a component.

        This method first checks for a custom setup method (_setup_{attribute}).
        If not found, it tries to get the component class from the processor's Auto mappings
        and instantiate it without arguments.
        If that fails, it raises an error telling the user to override the setup method.

        Individual test classes should override _setup_{attribute}() for custom component setup.
        Custom methods should return the created component.

        Returns:
            The created component instance.
        """
        # Check if there's a custom setup method for this specific attribute
        custom_method = getattr(cls, f"_setup_{attribute}", None)
        if custom_method is not None:
            return custom_method()

        # Get the component class from processor's Auto mappings
        component_class = cls._get_component_class_from_processor(attribute)

        # Get the base class name for the component to provide helpful error messages
        component_type = attribute.replace("_", " ")

        # Try to instantiate the component without arguments
        try:
            component = component_class()
        except Exception as e:
            raise TypeError(
                f"Failed to instantiate {component_type} ({component_class}) without arguments.\n"
                f"Error: {e}\n\n"
                f"To fix this, override the setup method in your test class:\n\n"
                f"    @classmethod\n"
                f"    def _setup_{attribute}(cls):\n"
                f"        # Create your custom {component_type}\n"
                f"        from transformers import {component_class}\n"
                f"        component = {component_class}(...)\n"
                f"        return component\n"
            ) from e

        return component

    @classmethod
    def _get_component_class_from_processor(cls, attribute, use_fast: bool = True):
        """
        Get the component class for a given attribute from the processor's Auto mappings.

        This extracts the model type from the test file name and uses that to look up
        the config class, which is then used to find the appropriate component class.
        """
        import inspect
        import re

        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING,
            CONFIG_MAPPING_NAMES,
            SPECIAL_MODEL_TYPE_TO_MODULE_NAME,
        )

        # Extract model_type from the test file name
        # Test files are named like test_processing_align.py or test_processor_align.py
        test_file = inspect.getfile(cls)
        match = re.search(r"test_process(?:ing|or)_(\w+)\.py$", test_file)
        if not match:
            raise ValueError(
                f"Could not extract model type from test file name: {test_file}. "
                f"Please override _setup_{attribute}() in your test class."
            )

        model_type = match.group(1)
        if model_type not in CONFIG_MAPPING_NAMES:
            # check if the model type is a special model type
            for special_model_type, special_module_name in SPECIAL_MODEL_TYPE_TO_MODULE_NAME.items():
                if model_type == special_module_name:
                    model_type = special_model_type
                    break

        # Get the config class for this model type
        if model_type not in CONFIG_MAPPING_NAMES:
            raise ValueError(
                f"Model type '{model_type}' not found in CONFIG_MAPPING_NAMES. "
                f"Please override _setup_{attribute}() in your test class."
            )

        config_class = CONFIG_MAPPING[model_type]

        # Now get the component class from the appropriate Auto mapping
        if attribute in MODALITY_TO_AUTOPROCESSOR_MAPPING:
            mapping_name = attribute
        elif "tokenizer" in attribute:
            mapping_name = "tokenizer"
        else:
            raise ValueError(
                f"Unknown attribute type: '{attribute}'. "
                f"Please override _setup_{attribute}() in your test class to provide custom setup."
            )

        # Get the appropriate Auto mapping for this component type
        if mapping_name == "tokenizer":
            from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
            from transformers.utils import is_tokenizers_available

            component_class = TOKENIZER_MAPPING.get(config_class, None)
            if component_class is None and is_tokenizers_available():
                from transformers.tokenization_utils_tokenizers import TokenizersBackend

                component_class = TokenizersBackend
        elif mapping_name == "image_processor":
            from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING

            component_class = IMAGE_PROCESSOR_MAPPING.get(config_class, None)
        elif mapping_name == "feature_extractor":
            from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING

            component_class = FEATURE_EXTRACTOR_MAPPING.get(config_class, None)
        elif mapping_name == "video_processor":
            from transformers.models.auto.video_processing_auto import VIDEO_PROCESSOR_MAPPING

            component_class = VIDEO_PROCESSOR_MAPPING.get(config_class, None)
        else:
            raise ValueError(f"Unknown mapping for attribute: {attribute}")

        if component_class is None:
            raise ValueError(
                f"Could not find {mapping_name} class for config {config_class.__name__}. "
                f"Please override _setup_{attribute}() in your test class."
            )

        # Handle tuple case (some mappings return tuples of classes)
        if isinstance(component_class, tuple):
            if use_fast:
                component_class = component_class[-1] if component_class[-1] is not None else component_class[0]
            else:
                component_class = component_class[0] if component_class[0] is not None else component_class[1]

        return component_class

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory."""
        if hasattr(cls, "tmpdirname"):
            shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @staticmethod
    def prepare_processor_dict():
        """Override this method to provide custom kwargs for processor initialization."""
        return {}

    def get_component(self, attribute, **kwargs):
        if attribute not in MODALITY_TO_AUTOPROCESSOR_MAPPING and "tokenizer" in attribute:
            auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING["tokenizer"]
            component = auto_processor_class.from_pretrained(self.tmpdirname, subfolder=attribute, **kwargs)  # noqa
        else:
            auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[attribute]
            component = auto_processor_class.from_pretrained(self.tmpdirname, **kwargs)  # noqa
        if "tokenizer" in attribute and not component.pad_token:
            component.pad_token = "[TEST_PAD]"
            if component.pad_token_id is None:
                component.pad_token_id = 0

        return component

    def prepare_components(self, **kwargs):
        components = {}
        for attribute in self.processor_class.get_attributes():
            component = self.get_component(attribute)
            components[attribute] = component

        return components

    def get_processor(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        return processor

    def prepare_text_inputs(self, batch_size: int | None = None, modalities: str | list | None = None):
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
    def prepare_image_inputs(self, batch_size: int | None = None, nested: bool = False):
        """This function prepares a list of PIL images for testing"""
        if batch_size is None:
            return prepare_image_inputs()[0]
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        if nested:
            return [prepare_image_inputs()] * batch_size
        return prepare_image_inputs() * batch_size

    @require_vision
    def prepare_video_inputs(self, batch_size: int | None = None):
        """This function prepares a list of numpy videos."""
        video_input = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)] * 8
        video_input = np.array(video_input)
        if batch_size is None:
            return video_input
        return [video_input] * batch_size

    def prepare_audio_inputs(self, batch_size: int | None = None):
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

                for attribute in processor_first.get_attributes():
                    attribute_first = getattr(processor_first, attribute)
                    attribute_second = getattr(processor_second, attribute)

                    # tokenizer repr contains model-path from where we loaded
                    if "tokenizer" not in attribute:
                        # We don't store/load `_processor_class` for subprocessors.
                        # The `_processor_class` is saved once per config, at general level
                        self.assertFalse(hasattr(attribute_second, "_processor_class"))
                        self.assertFalse(hasattr(attribute_first, "_processor_class"))

                        self.assertFalse(hasattr(attribute_second, "processor_class"))
                        self.assertFalse(hasattr(attribute_first, "processor_class"))

                        self.assertEqual(repr(attribute_first), repr(attribute_second))

    def test_processor_from_and_save_pretrained_as_nested_dict(self):
        processor_first = self.get_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_files = processor_first.save_pretrained(tmpdirname)
            check_json_file_has_correct_format(saved_files[0])

            # Load it back and check if loaded correctly
            processor_second = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor_second.to_dict(), processor_first.to_dict())

            # Try to load each attribute separately from saved directory
            for attribute in processor_first.get_attributes():
                if attribute not in MODALITY_TO_AUTOPROCESSOR_MAPPING and "tokenizer" in attribute:
                    auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING["tokenizer"]
                    attribute_reloaded = auto_processor_class.from_pretrained(tmpdirname, subfolder=attribute)
                else:
                    auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[attribute]
                    attribute_reloaded = auto_processor_class.from_pretrained(tmpdirname)
                attribute_first = getattr(processor_first, attribute)

                # tokenizer repr contains model-path from where we loaded
                if "tokenizer" not in attribute:
                    self.assertEqual(repr(attribute_first), repr(attribute_reloaded))

    def test_save_load_pretrained_additional_features(self):
        """
        Tests that additional kwargs passed to from_pretrained are correctly applied to components.
        """
        attributes = self.processor_class.get_attributes()

        if not any(
            attr in ["tokenizer", "image_processor", "feature_extractor", "video_processor"] for attr in attributes
        ):
            self.skipTest("Processor has no tokenizer or image_processor to test additional features")
        additional_kwargs = {}

        has_tokenizer = "tokenizer" in attributes
        if has_tokenizer:
            additional_kwargs["cls_token"] = "(CLS)"
            additional_kwargs["sep_token"] = "(SEP)"

        has_image_processor = "image_processor" in attributes
        if has_image_processor:
            additional_kwargs["do_normalize"] = False
        has_video_processor = "video_processor" in attributes
        if has_video_processor:
            additional_kwargs["do_normalize"] = False

        processor_second = self.processor_class.from_pretrained(self.tmpdirname, **additional_kwargs)
        if has_tokenizer:
            self.assertEqual(processor_second.tokenizer.cls_token, "(CLS)")
            self.assertEqual(processor_second.tokenizer.sep_token, "(SEP)")
        if has_image_processor:
            self.assertEqual(processor_second.image_processor.do_normalize, False)
        if has_video_processor:
            self.assertEqual(processor_second.video_processor.do_normalize, False)

    def test_processor_from_pretrained_vs_from_components(self):
        """
        Tests that loading a processor fully with from_pretrained produces the same result as
        loading each component individually with from_pretrained and building the processor from them.
        """
        # Load processor fully with from_pretrained
        processor_full = self.get_processor()

        # Load each component individually with from_pretrained
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        # Build processor from components + prepare_processor_dict() kwargs
        processor_kwargs = self.prepare_processor_dict()
        processor_from_components = self.processor_class(**components, **processor_kwargs)

        self.assertEqual(processor_from_components.to_dict(), processor_full.to_dict())

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

    def test_image_processor_defaults(self):
        """
        Tests that image processor is called correctly when passing images to the processor.
        This test verifies that processor(images=X) produces the same output as image_processor(X).
        """
        # Skip if processor doesn't have image_processor
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        image_processor = self.get_component("image_processor")

        # Get all required components for processor
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        processor = self.processor_class(**components)

        image_input = self.prepare_image_inputs()

        input_image_proc = image_processor(image_input, return_tensors="pt")
        try:
            input_processor = processor(images=image_input, return_tensors="pt")
        except Exception:
            # The processor does not accept image only input, so we can skip this test
            self.skipTest("Processor does not accept image-only input.")

        # Verify outputs match
        for key in input_image_proc:
            torch.testing.assert_close(input_image_proc[key], input_processor[key])

    def test_tokenizer_defaults(self):
        """
        Tests that tokenizer is called correctly when passing text to the processor.
        This test verifies that processor(text=X) produces the same output as tokenizer(X).
        """
        # Skip if processor doesn't have tokenizer
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")

        # Get all required components for processor
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        processor = self.processor_class(**components)
        tokenizer = components["tokenizer"]

        input_str = ["lower newer"]

        # Process with both tokenizer and processor (disable padding to ensure same output)
        try:
            encoded_processor = processor(text=input_str, padding=False, return_tensors="pt")
        except Exception:
            # The processor does not accept text only input, so we can skip this test
            self.skipTest("Processor does not accept text-only input.")
        encoded_tok = tokenizer(input_str, padding=False, return_tensors="pt")

        # Verify outputs match (handle processors that might not return token_type_ids)
        for key in encoded_tok:
            if key in encoded_processor:
                self.assertListEqual(encoded_tok[key].tolist(), encoded_processor[key].tolist())

    def test_feature_extractor_defaults(self):
        """
        Tests that feature extractor is called correctly when passing audio to the processor.
        This test verifies that processor(audio=X) produces the same output as feature_extractor(X).
        """
        # Skip if processor doesn't have feature_extractor
        if (
            "feature_extractor" not in self.processor_class.get_attributes()
            and "audio_processor" not in self.processor_class.get_attributes()
        ):
            self.skipTest(f"feature_extractor or audio_processor attribute not present in {self.processor_class}")

        if "feature_extractor" in self.processor_class.get_attributes():
            feature_extractor = self.get_component("feature_extractor")
        else:
            feature_extractor = self.get_component("audio_processor")

        # Get all required components for processor
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        processor = self.processor_class(**components)

        audio_input = self.prepare_audio_inputs()

        # Process with both feature_extractor and processor
        input_feat_extract = feature_extractor(audio_input, return_tensors="pt")
        try:
            input_processor = processor(audio=audio_input, return_tensors="pt")
        except Exception:
            # The processor does not accept audio only input, so we can skip this test
            self.skipTest("Processor does not accept audio-only input.")

        # Verify outputs match
        for key in input_feat_extract:
            torch.testing.assert_close(input_feat_extract[key], input_processor[key])

    def test_video_processor_defaults(self):
        """
        Tests that video processor is called correctly when passing videos to the processor.
        This test verifies that processor(videos=X) produces the same output as video_processor(X).
        """
        # Skip if processor doesn't have video_processor
        if "video_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")

        video_processor = self.get_component("video_processor")

        # Get all required components for processor
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        processor = self.processor_class(**components)

        video_input = self.prepare_video_inputs()

        # Process with both video_processor and processor
        input_video_proc = video_processor(video_input, return_tensors="pt")
        try:
            input_processor = processor(videos=video_input, return_tensors="pt")
        except Exception:
            # The processor does not accept video only input, so we can skip this test
            self.skipTest("Processor does not accept video-only input.")

        # Verify outputs match
        for key in input_video_proc:
            torch.testing.assert_close(input_video_proc[key], input_processor[key])

    def test_tokenizer_decode_defaults(self):
        """
        Tests that processor.batch_decode() correctly forwards to tokenizer.batch_decode().
        """
        # Skip if processor doesn't have tokenizer
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")

        # Get all required components for processor
        components = {}
        for attribute in self.processor_class.get_attributes():
            components[attribute] = self.get_component(attribute)

        processor = self.processor_class(**components)
        tokenizer = components["tokenizer"]

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        # Test batch_decode
        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_processor_with_multiple_inputs(self):
        """
        Tests that processor correctly handles multiple modality inputs together.
        Verifies that the output contains expected keys and raises error when no input is provided.
        """
        # Skip if processor doesn't have multiple attributes (not multimodal)
        attributes = self.processor_class.get_attributes()
        if len(attributes) <= 1:
            self.skipTest(f"Processor only has {len(attributes)} attribute(s), test requires multimodal processor")

        processor = self.get_processor()

        # Map attributes to input parameter names, prepare methods, and output key names
        attr_to_input_param = {
            "tokenizer": ("text", "prepare_text_inputs", "text_input_name"),
            "image_processor": ("images", "prepare_image_inputs", "images_input_name"),
            "video_processor": ("videos", "prepare_video_inputs", "videos_input_name"),
            "feature_extractor": ("audio", "prepare_audio_inputs", "audio_input_name"),
        }

        # Prepare inputs dynamically based on processor attributes
        processor_inputs = {}
        expected_output_keys = []

        for attr in attributes:
            if attr in attr_to_input_param:
                param_name, prepare_method_name, output_key_attr = attr_to_input_param[attr]
                # Call the prepare method
                prepare_method = getattr(self, prepare_method_name)
                if param_name == "text":
                    modalities = []
                    if "image_processor" in attributes:
                        modalities.append("image")
                    if "video_processor" in attributes:
                        modalities.append("video")
                    if "audio_processor" in attributes or "feature_extractor" in attributes:
                        modalities.append("audio")
                    processor_inputs[param_name] = prepare_method(modalities=modalities)
                else:
                    processor_inputs[param_name] = prepare_method()
                # Track expected output keys
                expected_output_keys.append(getattr(self, output_key_attr))

        # Test combined processing
        inputs = processor(**processor_inputs, return_tensors="pt")

        # Verify output contains all expected keys
        for key in expected_output_keys:
            self.assertIn(key, inputs)

        # Test that it raises error when no input is passed
        with self.assertRaises((TypeError, ValueError)):
            processor()

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

        # First call processor with all inputs and use nested input type, which is the format supported by all multimodal processors
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
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
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
        We use do_rescale=True, rescale_factor=-1.0 to ensure that image_processor kwargs are preserved in the processor.
        We then check that the mean of the pixel_values is less than or equal to 0 after processing.
        Since the original pixel_values are in [0, 255], this is a good indicator that the rescale_factor is indeed applied.
        """
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=-1.0
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
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
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
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
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

        inputs = processor(
            text=input_str, images=image_input, do_rescale=True, rescale_factor=-1.0, return_tensors="pt"
        )
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.get_attributes():
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
            rescale_factor=-1.0,
            padding="max_length",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.get_attributes():
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
            rescale_factor=-1.0,
            padding="longest",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < 76
        )

    def test_doubly_passed_kwargs(self):
        if "image_processor" not in self.processor_class.get_attributes():
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
                images_kwargs={"do_rescale": True, "rescale_factor": -1.0},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_args_overlap_kwargs(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_first = self.get_processor()
        image_processor = processor_first.image_processor
        image_processor.is_override = True

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_first.save_pretrained(tmpdirname)
            processor_second = self.processor_class.from_pretrained(tmpdirname, image_processor=image_processor)
            self.assertTrue(processor_second.image_processor.is_override)

    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.get_attributes():
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
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1.0},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.get_attributes():
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
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1.0},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    # text + audio kwargs testing
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.get_attributes():
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=300, padding="max_length")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=3, modalities="audio")
        raw_speech = self.prepare_audio_inputs(batch_size=3)
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt")
        self.assertEqual(len(inputs[self.text_input_name][0]), 300)

    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.get_attributes():
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=3, modalities="audio")
        raw_speech = self.prepare_audio_inputs(batch_size=3)
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt", max_length=300, padding="max_length")

        self.assertEqual(len(inputs[self.text_input_name][0]), 300)

    @require_torch
    def test_unstructured_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.get_attributes():
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=3, modalities="audio")
        raw_speech = self.prepare_audio_inputs(batch_size=3)
        inputs = processor(text=input_str, audio=raw_speech, return_tensors="pt", max_length=300, padding="max_length")

        self.assertEqual(len(inputs[self.text_input_name][0]), 300)

    @require_torch
    def test_doubly_passed_kwargs_audio(self):
        if "feature_extractor" not in self.processor_class.get_attributes():
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
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
        if "feature_extractor" not in self.processor_class.get_attributes():
            self.skipTest(f"feature_extractor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
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
        if "video_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
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
        We use do_rescale=True, rescale_factor=-1.0 to ensure that image_processor kwargs are preserved in the processor.
        We then check that the mean of the pixel_values is less than or equal to 0 after processing.
        Since the original pixel_values are in [0, 255], this is a good indicator that the rescale_factor is indeed applied.
        """
        if "video_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["video_processor"] = self.get_component(
            "video_processor", do_rescale=True, rescale_factor=-1.0
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
        if "video_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
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
        if "video_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        if "tokenizer" not in self.processor_class.get_attributes():
            self.skipTest(f"tokenizer attribute not present in {self.processor_class}")
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
            rescale_factor=-1.0,
            return_tensors="pt",
        )
        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)

    def test_unstructured_kwargs_video(self):
        if "video_processor" not in self.processor_class.get_attributes():
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
            rescale_factor=-1.0,
            padding="max_length",
            max_length=176,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 176)

    def test_unstructured_kwargs_batched_video(self):
        if "video_processor" not in self.processor_class.get_attributes():
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
            rescale_factor=-1.0,
            padding="longest",
            max_length=176,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertTrue(
            len(inputs[self.text_input_name][0]) == len(inputs[self.text_input_name][1])
            and len(inputs[self.text_input_name][1]) < 176
        )

    def test_doubly_passed_kwargs_video(self):
        if "video_processor" not in self.processor_class.get_attributes():
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
                videos_kwargs={"do_rescale": True, "rescale_factor": -1.0},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_structured_kwargs_nested_video(self):
        if "video_processor" not in self.processor_class.get_attributes():
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
            "videos_kwargs": {"do_rescale": True, "rescale_factor": -1.0, "do_sample_frames": False},
            "text_kwargs": {"padding": "max_length", "max_length": 176},
        }

        inputs = processor(text=input_str, videos=video_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 176)

    def test_structured_kwargs_nested_from_dict_video(self):
        if "video_processor" not in self.processor_class.get_attributes():
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
            "videos_kwargs": {"do_rescale": True, "rescale_factor": -1.0, "do_sample_frames": False},
            "text_kwargs": {"padding": "max_length", "max_length": 176},
        }

        inputs = processor(text=input_str, videos=video_input, **all_kwargs)
        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(inputs[self.text_input_name].shape[-1], 176)

    # TODO: the same test, but for audio + text processors that have strong overlap in kwargs
    # TODO (molbap) use the same structure of attribute kwargs for other tests to avoid duplication
    def test_overlapping_text_image_kwargs_handling(self):
        if "image_processor" not in self.processor_class.get_attributes():
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
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
        if "feature_extractor" not in self.processor_class.get_attributes():
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

    def test_chat_template_save_loading(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        signature = inspect.signature(processor.__init__)
        if "chat_template" not in {*signature.parameters.keys()}:
            self.skipTest("Processor doesn't accept chat templates at input")

        processor.chat_template = "test template"
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.save_pretrained(tmpdirname)
            with open(Path(tmpdirname, "chat_template.json"), "w") as fp:
                json.dump({"chat_template": processor.chat_template}, fp)
            os.remove(Path(tmpdirname, "chat_template.jinja"))

            reloaded_processor = self.processor_class.from_pretrained(tmpdirname)
            self.assertEqual(processor.chat_template, reloaded_processor.chat_template)

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

        if processor_name not in self.processor_class.get_attributes():
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

        # When `do_sample_frames=False` no sampling is done and whole video is loaded, even if number of frames is passed
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

        if "feature_extractor" not in self.processor_class.get_attributes():
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
            return_tensors="pt",
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
