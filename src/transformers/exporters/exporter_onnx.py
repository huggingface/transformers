from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .exporter_dynamo import DynamoExporter


if is_torch_available():
    import torch

    if is_torch_greater_or_equal("2.6.0"):
        from torch.export import ExportedProgram
        from torch.onnx import ONNXProgram

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__file__)


class OnnxExporter(DynamoExporter):
    export_config: OnnxConfig

    required_packages = ["torch", "onnx", "onnxscript"]

    def export(self, model: "PreTrainedModel", sample_inputs: dict[str, Any]):
        """Exports a model to ONNX format using TorchDynamo.
        Args:
            model (`PreTrainedModel`):
                The model to export.
            sample_inputs (`Dict[str, Any]`):
                The sample inputs to use for the export.
        Returns:
            `ONNXProgram`: The exported model.
        """
        exported_program: ExportedProgram = super().export(model, sample_inputs)
        onnx_program: ONNXProgram = torch.onnx.export(
            exported_program,
            f=self.export_config.f,
            optimize=self.export_config.optimize,
            opset_version=self.export_config.opset_version,
            do_constant_folding=self.export_config.do_constant_folding,
            external_data=self.export_config.external_data,
            export_params=self.export_config.export_params,
        )
        return onnx_program
