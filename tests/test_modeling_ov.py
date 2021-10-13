import unittest

from transformers.testing_utils import require_ov
from transformers import is_ov_available

if is_ov_available():
    from transformers import OVAutoModel


@require_ov
class OVAutoModelIntegrationTest(unittest.TestCase):
    model = OVAutoModel.from_pretrained("dkurt/test_openvino")
