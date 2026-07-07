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

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor = image_processor_class()
        image_processor.size = {"height": 64, "width": 64}
        image_processor.tile_size = 512
        return image_processor
