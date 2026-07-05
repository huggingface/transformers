import unittest

from transformers import Owlv2Processor
from transformers.testing_utils import require_scipy

from ...test_processing_common import ProcessorTesterMixin


@require_scipy
class Owlv2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Owlv2Processor
    tiny_model_id = "hf-internal-testing/tiny-processor-owlv2"
    model_id = "google/owlv2-base-patch16-ensemble"
