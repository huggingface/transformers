import tempfile
import unittest

from transformers import InstructBlipVideoProcessor

from ...test_processing_common import ProcessorTesterMixin


class InstructBlipVideoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "Salesforce/instructblip-vicuna-7b"
    processor_class = InstructBlipVideoProcessor

    def prepare_components(self):
        components = super().prepare_components()
        components["qformer_tokenizer"] = components["tokenizer"]
        return components

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)
