import copy
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .exporter_dynamo import DYNAMO_UNSUPPORTED_MODEL_TYPES, DynamoExporter
from .patch_utils import patch_torch_for_onnx_export
from .utils import get_inputs_outputs_names, prepare_for_export


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
    # Known issues during ONNX export
    "flava_image_codebook",  # the onnx model returns nothing
    "hifigan",  # the onnx model returns nothing
    "kosmos-2",  # export fails due to advanced tensor slicing / indexing
    "kosmos-2.5",  # export fails due to advanced tensor slicing / indexing
    "vits",  # export fails due to advanced tensor slicing / indexing
    # TODO: check if these models can be fixed, patched or add a comment explaining why not
    "aria",
    "autoformer",
    "bamba",
    "bros",
    "chameleon",
    "clvp",
    "data2vec-audio",
    "efficientloftr",
    "eomt",
    "falcon_h1",
    "flava",
    "granite_speech",
    "granitemoe",
    "granitemoehybrid",
    "granitemoeshared",
    "grounding-dino",
    "hubert",
    "idefics2",
    "idefics3",
    "informer",
    "janus",
    "jetmoe",
    "longformer",
    "longt5",
    "mamba2",
    "maskformer-swin",
    "mimi",
    "mlcd_vision_model",
    "mm-grounding-dino",
    "nemotron",
    "patchtsmixer",
    "perceiver",
    "prophetnet",
    "qwen3_next",
    "sew",
    "sew-d",
    "smolvlm",
    "speech_to_text",
    "speecht5",
    "splinter",
    "swin2sr",
    "tapas",
    "time_series_transformer",
    "timm_backbone",
    "unispeech",
    "unispeech-sat",
    "videomae",
    "wav2vec2",
    "wav2vec2-bert",
    "wav2vec2-conformer",
    "wavlm",
    "zamba2",
}

# The following are models that can be exported but
# their outputs don't match the expected outputs
# Q: should we error on these models ?
ONNX_INACCURATE_MODEL_TYPES: set[str] = {
    "audioflamingo3",
    "bit",
    "blt",
    "clvp",
    "clvp_encoder",
    "clvp_decoder",
    "d_fine",
    "flaubert",
    "janus_vqgan",
    "parakeet_ctc",
    "parakeet_encoder",
    "patchtst",
    "rt_detr",
    "rt_detr_v2",
    "superglue",
    "vit_mae",
    "voxtral",
    "xlm",
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

        if model.config.model_type in ONNX_INACCURATE_MODEL_TYPES:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not produce accurate outputs for model type '{model.config.model_type}'."
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
        with torch.no_grad():
            outputs = model(**copy.deepcopy(inputs))
        model, inputs = prepare_for_export(model, inputs, outputs)
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
