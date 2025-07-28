import unittest

from transformers import LlasaForCausalLM, is_torch_available
from transformers.testing_utils import require_torch

from ...generation.test_utils import GenerationTesterMixin
from ...test_modeling_common import ModelTesterMixin


@require_torch
class LlasaModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (LlasaForCausalLM,) if is_torch_available() else ()
