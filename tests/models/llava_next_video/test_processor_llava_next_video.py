import tempfile
import unittest

from transformers import LlavaNextVideoProcessor

from ...test_processing_common import ProcessorTesterMixin


class LlavaNextVideoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    processor_class = LlavaNextVideoProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)
