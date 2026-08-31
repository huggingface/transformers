import unittest

import pytest

from transformers.testing_utils import (
    require_tokenizers,
    require_vision,
)
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import TrOCRProcessor


@require_tokenizers
@require_vision
class TrOCRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    text_input_name = "labels"
    processor_class = TrOCRProcessor
    # Tiny processor created with make_tiny_processor.py from "microsoft/trocr-base-handwritten"
    tiny_model_id = "hf-internal-testing/tiny-processor-trocr"

    def test_processor_text(self):
        processor = self.get_processor()
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["pixel_values", "labels"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()
