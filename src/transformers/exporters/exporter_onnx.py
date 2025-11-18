import copy
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .exporter_dynamo import DYNAMO_UNSUPPORTED_MODEL_TYPES, DynamoExporter
from .utils import get_inputs_outputs_names, patch_torch_for_onnx_export, prepare_for_export


if is_torch_available():
    import torch

    if is_torch_greater_or_equal("2.6.0"):
        from torch.export import ExportedProgram
        from torch.onnx import ONNXProgram

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__file__)

DISABLED_OPTIMIZATION_MODEL_TYPES: set[str] = {
    "conditional_detr",  # optimization breaks the model (missing outputs)
    "fuyu",  # optimization breaks the model (missing outputs)
    "helium",  # nan outputs after optimization
    "t5gemma",  # optimization breaks the model (duplicate outputs)
}

ONNX_UNSUPPORTED_MODEL_TYPES: set[str] = {
    *DYNAMO_UNSUPPORTED_MODEL_TYPES,
    "flava_image_codebook",  # the onnx model returns nothing
    "hifigan",  # the onnx model returns nothing
    "kosmos-2",  # export fails due to advanced tensor slicing / indexing
    "kosmos-2.5",  # export fails due to advanced tensor slicing / indexing
    "vits",  # export fails due to advanced tensor slicing / indexing
    # TODO: check if these models can be fixed easily or add a comment explaining why not
    "chameleon",
    "bros",
    "aria",
    "eomt",
    "clvp",
    "falcon_h1",
    "data2vec-audio",
    "bamba",
    "autoformer",
    "longt5",
    "granitemoehybrid",
    "granitemoeshared",
    "granite_speech",
    "flava",
    "longformer",
    "mllama",
    "idefics3",
    "idefics2",
    "hubert",
    "mlcd_vision_model",
    "granitemoe",
    "informer",
    "efficientloftr",
    "maskformer-swin",
    "mamba2",
    "mimi",
    "jetmoe",
    "led",
    "nemotron",
    "janus",
    "patchtsmixer",
    "speech_to_text",
    "sew-d",
    "prophetnet",
    "smolvlm",
    "speecht5",
    "tapas",
    "unispeech-sat",
    "swin2sr",
    "mm-grounding-dino",
    "grounding-dino",
    "qwen3_next",
    "perceiver",
    "sew",
    "videomae",
    "splinter",
    "wav2vec2-bert",
    "wav2vec2",
    "timm_backbone",
    "wav2vec2-conformer",
    "unispeech",
    "time_series_transformer",
    "wavlm",
    "zamba2",
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
        if optimize and model.config.model_type in DISABLED_OPTIMIZATION_MODEL_TYPES:
            logger.warning(
                f"Disabling optimization for model type '{model.config.model_type}' as it results in an invalid ONNX model."
            )
            optimize = False

        # we use a copy to avoid side effects
        inputs = copy.deepcopy(sample_inputs)
        # we need to compute the outputs before the dynamo export because torch.onnx.export
        # might end up modifying the model and making it unsable afterwards ;-;
        # we only need the output names, so we can use torch.device("meta")
        # but it doesn't work with all models
        with torch.no_grad():
            outputs = model(**copy.deepcopy(inputs))
        model, inputs = prepare_for_export(model, inputs, outputs=outputs)
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
        return onnx_program
