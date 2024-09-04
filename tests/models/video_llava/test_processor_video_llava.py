import tempfile
import unittest

from transformers.models.video_llava.processing_video_llava import VideoLlavaProcessor

from ...test_processing_common import ProcessorTesterMixin


class VideoLlavaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "LanguageBind/Video-LLaVA-7B-hf"
    processor_class = VideoLlavaProcessor
    images_data_arg_name = "pixel_values_images"

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)
