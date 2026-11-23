import unittest

from transformers import Owlv2Processor
from transformers.testing_utils import require_scipy

from ...test_processing_common import ProcessorTesterMixin


@require_scipy
class Owlv2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Owlv2Processor
    # Tiny processor created with make_tiny_processor.py from "google/owlv2-base-patch16-ensemble"
    tiny_model_id = "hf-internal-testing/tiny-processor-owlv2"

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        # Default size=960×960 produces ~11 MB pixel_values per image. Use 64×64 for tests.
        return image_processor_class.from_pretrained(cls.tiny_model_id, size={"height": 64, "width": 64})

    def test_processor_images_kwargs_query_images(self):
        """Test for BC. Can be removed once DeprecatedOwlv2ImagesKwargs is removed."""
        processor = self.get_processor()

        image_input = self.prepare_images_inputs()
        query_input = self.prepare_images_inputs()

        inputs = processor(images=image_input, images_kwargs={"query_images": query_input})
        expected = processor(images=image_input, query_images=query_input)

        self.assertListEqual(list(inputs.keys()), ["query_pixel_values", "pixel_values"])
        for key in expected:
            self.assertTrue((inputs[key] == expected[key]).all())
