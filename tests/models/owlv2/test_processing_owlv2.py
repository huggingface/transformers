import unittest

from transformers import Owlv2Processor
from transformers.testing_utils import require_scipy

from ...test_processing_common import ProcessorTesterMixin


@require_scipy
class Owlv2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Owlv2Processor
    # Tiny processor created with make_tiny_processor.py from "google/owlv2-base-patch16-ensemble"
    tiny_model_id = "hf-internal-testing/tiny-processor-owlv2"
