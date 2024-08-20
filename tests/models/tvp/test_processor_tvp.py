import inspect
import tempfile
import unittest

from transformers import TvpProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


class TvpProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "Jiqing/tiny-random-tvp"
    processor_class = TvpProcessor
    videos_data_arg_name = "pixel_values"

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)

    @require_torch
    @require_vision
    def test_video_processor_defaults_preserved_by_kwargs(self):
        if "video_processor" not in self.processor_class.attributes and (
            "videos" not in inspect.signature(self.processor_class.__call__).parameters
            or inspect.signature(self.processor_class.__call__).parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", size=(234, 234), crop_size=(234, 234), do_pad=False
        )
        if "video_processor" in self.processor_class.attributes:
            processor_components["video_processor"] = self.get_component(
                "video_processor", size=(234, 234), crop_size=(234, 234), do_pad=False
            )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = "lower newer"
        video_input = self.prepare_video_inputs()

        inputs = processor(text=input_str, videos=video_input, return_tensors="pt")
        self.assertEqual(inputs[self.videos_data_arg_name].shape[-1], 234)

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        self.skipTest("TVP does not process images")

    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        self.skipTest("TVP does not process images")

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        self.skipTest("TVP does not process images")

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        self.skipTest("TVP does not process images")

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        self.skipTest("TVP does not process images")

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        self.skipTest("TVP does not process images")
