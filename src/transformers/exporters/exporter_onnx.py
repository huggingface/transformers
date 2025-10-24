from typing import TYPE_CHECKING

from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .base import HfExporter


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


class OnnxExporter(HfExporter):
    """
    Exports a HuggingFace model to ONNX format.
    """

    export_config: OnnxConfig

    required_packages = ["torch", "onnx", "onnxscript"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not is_torch_greater_or_equal("2.9.0"):
            raise ImportError("OnnxExporter requires torch>=2.9.0 for Dynamo based ONNX export.")

    def export(self, model: "PreTrainedModel"):
        from torch.onnx import ONNXProgram

        if self.export_config.sample_inputs is None:
            raise NotImplementedError(
                "OnnxExporter does not generate inptus for now. Please provide sample_inputs in the exporter_config."
            )

        args = ()
        kwargs = None

        if isinstance(self.export_config.sample_inputs, tuple):
            args = self.export_config.sample_inputs
        elif isinstance(self.export_config.sample_inputs, dict):
            kwargs = self.export_config.sample_inputs
        else:
            raise ValueError(
                "sample_inputs should be either a tuple of positional arguments or a dict of keyword arguments."
            )

        # some input validation can be done here (like using pytree)

        exported_model: ONNXProgram = torch.onnx.export(
            model,
            args=args,
            kwargs=kwargs,
            f=self.export_config.output_file,
            opset_version=self.export_config.opset_version,
        )
        model._exported_model = exported_model
        model._exported = True
