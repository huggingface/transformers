import tempfile
import unittest

from transformers import AltCLIPProcessor

from ...test_processing_common import ProcessorTesterMixin


class AltCLIPProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "BAAI/AltCLIP"
    processor_class = AltCLIPProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)
