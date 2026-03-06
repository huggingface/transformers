import copy
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_torch_available
from .exporter_dynamo import DynamoExporter
from .patch_utils import patch_torch_for_onnx_export
from .utils import get_inputs_outputs_names, prepare_for_export


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.onnx import ONNXProgram

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__file__)


ONNX_UNSUPPORTED_MODEL_TYPES: set[str] = {
    # --- FX graph / torch.export failures ---
    "bigbird_pegasus",  # CUDA device-side assert triggered during export
    "doge",  # FX decomposition failure (InsertTypePromotion pass fails at step 2/3)
    "wavlm",  # FX graph decomposition failure (non-contiguous tensor view in attention reshape)
    # --- SDPA: 5D tensors not supported ---
    "granite_speech",  # SDPA only supports 4D tensors; model uses 5D attention (grouped convolution)
    # --- SDPA: attention mechanism not supported ---
    "falcon_mamba",  # does not support SDPA attention during FX export
    # --- ONNX Runtime runtime / graph errors ---
    "fine_acoustics",  # BarkFineModel: attention_mask exported as rank-3 but ORT expects rank-2
    "dia",  # Squeeze dimension error in ONNX Runtime (node_squeeze: dim must be 1 not 7)
    "flava",  # ForPreTraining: Where node provider type not set in ORT; FlavaModel: optimization renames outputs
    "higgs_audio_v2",  # Where node condition cannot broadcast (shape mismatch {3,14,1} vs {20,32})
    "mllama",  # cross-attention q/kv size mismatch in test inputs (tensor a 1808 vs tensor b 904)
}


ONNX_DISABLED_OPTIMIZATION_MODEL_TYPES: set[str] = {
    # Optimization renames past_key_values outputs
    "gemma3n_text",  # optimization renames past_key_values outputs (shared_layers instead of layers)
    "moshi",  # optimization renames past_key_values outputs (depth_ prefix added to layer names)
}


class OnnxExporter(DynamoExporter):
    export_config: OnnxConfig

    required_packages = ["torch", "onnx", "onnxscript"]

    def export(self, model: "PreTrainedModel", sample_inputs: dict[str, Any]) -> "ONNXProgram":
        """Exports a model to ONNX format using TorchDynamo.
        Args:
            model (`PreTrainedModel`):
                The model to export.
            sample_inputs (`Dict[str, Any]`):
                The sample inputs to use for the export.
        Returns:
            `ONNXProgram`: The exported model.
        """
        if model.config.model_type in ONNX_UNSUPPORTED_MODEL_TYPES:
            raise NotImplementedError(
                f"{self.__class__.__name__} is not supported for model type '{model.config.model_type}'."
            )

        optimize = self.export_config.optimize
        if model.config.model_type in ONNX_DISABLED_OPTIMIZATION_MODEL_TYPES and optimize:
            logger.warning(
                f"Disabling optimization for model type '{model.config.model_type}' as it results in an invalid ONNX model."
            )
            optimize = False

        # we use a copy to avoid side effects
        inputs = copy.deepcopy(sample_inputs)
        model, inputs, outputs = prepare_for_export(model, inputs)
        inputs_names, outputs_names = get_inputs_outputs_names(inputs, outputs)

        with patch_torch_for_onnx_export():
            exported_program: ExportedProgram = super().export(model, inputs)
            onnx_program: ONNXProgram = torch.onnx.export(
                exported_program,
                args=(),
                kwargs=inputs,
                f=self.export_config.f,
                input_names=inputs_names,
                output_names=outputs_names,
                opset_version=self.export_config.opset_version,
                external_data=self.export_config.external_data,
                export_params=self.export_config.export_params,
                optimize=optimize,
            )

        # Verify that the exported model has the expected output names
        onnx_outptuts = [node.name for node in onnx_program.model_proto.graph.output]
        if onnx_outptuts != outputs_names:
            logger.warning(
                f"The exported ONNX model has different output names than expected. Expected: {outputs_names}, got: {onnx_outptuts}."
                "This is a known side effect when model outputs the same tensor multiple times under different names. "
                "This might also be due to optimizations (constant folding) removing some outputs. "
            )

        return onnx_program
