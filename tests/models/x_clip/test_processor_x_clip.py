import tempfile
import unittest

import numpy as np

from transformers import XCLIPProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


class XCLIPProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "microsoft/xclip-base-patch32"
    processor_class = XCLIPProcessor
    videos_data_arg_name = "pixel_values"

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)

    @require_vision
    def prepare_video_inputs(self):
        return [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)]

    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        self.skipTest("XCLIP does not process images")

    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        self.skipTest("XCLIP does not process images")

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        self.skipTest("XCLIP does not process images")

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        self.skipTest("XCLIP does not process images")

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        self.skipTest("XCLIP does not process images")

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        self.skipTest("XCLIP does not process images")
