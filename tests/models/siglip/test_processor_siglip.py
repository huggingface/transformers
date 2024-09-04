import tempfile
import unittest

from transformers import SiglipProcessor

from ...test_processing_common import ProcessorTesterMixin


class SiglipProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/siglip-base-patch16-224"
    processor_class = SiglipProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)
