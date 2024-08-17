import tempfile
import unittest

from transformers import ViltProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


class ViltProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "dandelin/vilt-b32-mlm"
    processor_class = ViltProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", size=(234, 234), crop_size=(234, 234), size_divisor=32
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        # VILT resizes images to dims divisible by size_divisor
        vilt_compatible_image_size = (32, 384)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], vilt_compatible_image_size[-1])

    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component("image_processor", size=(234, 234))
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        # VILT resizes images to dims divisible by size_divisor
        vilt_compatible_image_size = (32, 352)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(
            text=input_str, images=image_input, size=[224, 224], crop_size=(224, 224), return_tensors="pt"
        )

        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], vilt_compatible_image_size[-1])

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        # VILT resizes images to dims divisible by size_divisor
        vilt_compatible_image_size = (32, 352)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {
                "size": {"height": 214, "width": 214},
                "crop_size": {"height": 214, "width": 214},
            },
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], vilt_compatible_image_size[-1])
        self.assertEqual(inputs[self.text_data_arg_name].shape[-1], 76)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        # VILT resizes images to dims divisible by size_divisor
        vilt_compatible_image_size = (32, 352)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {
                "size": {"height": 214, "width": 214},
                "crop_size": {"height": 214, "width": 214},
            },
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], vilt_compatible_image_size[-1])
        self.assertEqual(inputs[self.text_data_arg_name].shape[-1], 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        # VILT resizes images to dims divisible by size_divisor
        vilt_compatible_image_size = (32, 352)

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            crop_size={"height": 214, "width": 214},
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], vilt_compatible_image_size[-1])
        self.assertEqual(inputs[self.text_data_arg_name].shape[-1], 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        # VILT resizes images to dims divisible by size_divisor
        vilt_compatible_image_size = (32, 352)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            crop_size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )

        self.assertEqual(inputs[self.images_data_arg_name].shape[-1], vilt_compatible_image_size[-1])
        self.assertEqual(len(inputs[self.text_data_arg_name][0]), len(inputs[self.text_data_arg_name][1]))
