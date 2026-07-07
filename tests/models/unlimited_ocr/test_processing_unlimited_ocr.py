import unittest

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import UnlimitedOcrProcessor


@require_vision
class UnlimitedOcrProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = UnlimitedOcrProcessor
    # TODO: Change before merge
    model_id = "guarin/Unlimited-OCR"

    # Defaults from mixin are too small as a single image expands to 273 image tokens
    # for this checkpoint (size=1024)
    image_text_kwargs_max_length = 320
    image_text_kwargs_override_max_length = 310
    image_unstructured_max_length = 300
