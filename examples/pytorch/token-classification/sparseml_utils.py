from typing import Any

import numpy

from sparseml.pytorch.utils import ModuleExporter
from transformers.modeling_outputs import TokenClassifierOutput


class TokenClassificationModuleExporter(ModuleExporter):
    """
    Module exporter class for Token Classification
    """

    @classmethod
    def get_output_names(self, out: Any):
        if not isinstance(out, TokenClassifierOutput):
            raise ValueError(f"Expected TokenClassifierOutput, got {type(out)}")
        expected = ["logits"]
        if numpy.any([name for name in expected if name not in out]):
            raise ValueError("Expected output names not found in model output")
        return expected
