from typing import TYPE_CHECKING

from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .exporter_dynamo import DynamoExporter


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__file__)


class OnnxExporter(DynamoExporter):
    export_config: OnnxConfig

    required_packages = ["torch", "onnx", "onnxscript"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not is_torch_greater_or_equal("2.9.0"):
            raise ImportError(f"{self.__class__.__name__} requires torch>=2.9.0 for Dynamo based ONNX export.")

    def export(self, model: "PreTrainedModel"):
        from torch.export import ExportedProgram
        from torch.onnx import ONNXProgram

        exported_program: ExportedProgram = super().export(model)
        onnx_program: ONNXProgram = torch.onnx.export(
            exported_program,
            f=self.export_config.f,
            optimize=self.export_config.optimize,
            do_constant_folding=self.export_config.do_constant_folding,
        )
        model.exported_model = onnx_program
        return onnx_program
