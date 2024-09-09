import tempfile
import unittest

from transformers import NougatProcessor

from ...test_processing_common import ProcessorTesterMixin


class NougatProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "facebook/nougat-base"
    text_data_arg_name = "labels"
    processor_class = NougatProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)
