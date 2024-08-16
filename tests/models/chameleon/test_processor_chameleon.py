import tempfile
import unittest

from transformers import ChameleonProcessor

from ...test_processing_common import ProcessorTesterMixin


class ChameleonProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "leloy/Anole-7b-v0.1-hf"
    processor_class = ChameleonProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)
