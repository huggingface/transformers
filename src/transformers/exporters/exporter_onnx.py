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
    # --- FX graph / torch.export failures (SymInt not tracked with proxy) ---
    "colmodernvbert",  # SymInt not tracked with proxy (runtime assert in FX graph)
    "d_fine",  # SymInt not tracked with proxy (runtime assert in FX graph)
    "doge",  # FX decomposition failure (InsertTypePromotion pass fails at step 2/3)
    "grounding-dino",  # SymInt not tracked with proxy (runtime assert in FX graph)
    "idefics3",  # aot_autograd: detach_ (in-place op) found in graph
    "mm-grounding-dino",  # SymInt not tracked with proxy (runtime assert in FX graph)
    "modernvbert",  # SymInt not tracked with proxy (runtime assert in FX graph)
    "rt_detr_v2",  # SymInt not tracked with proxy (runtime assert in FX graph)
    "smolvlm",  # SymInt not tracked with proxy (runtime assert in FX graph)
    "videomae",  # GuardOnDataDependentSymNode in mse_loss (dynamic masking at export time)
    "wavlm",  # FX graph decomposition failure (non-contiguous tensor view in attention reshape)
    # --- FX step 3 / ONNX translation failures ---
    "maskformer-swin",  # 'int' object has no attribute 'name' in ONNX return node translation
    "swin2sr",  # Key 'b_mean' does not match value name 'type_as' in graph builder
    # --- Missing ONNX ops ---
    "patchtsmixer",  # aten.randperm — no ONNX function registered
    "splinter",  # aten.bincount — no ONNX function registered
    # --- SDPA: 5D tensors not supported ---
    "granite_speech",  # SDPA only supports 4D tensors; model uses 5D attention (grouped convolution)
    # --- SDPA: attention mechanism not supported ---
    "falcon_mamba",  # does not support SDPA attention during FX export
    # --- ONNX Runtime runtime / graph errors ---
    "fine_acoustics",  # BarkFineModel: attention_mask exported as rank-3 but ORT expects rank-2
    "dia",  # Squeeze dimension error in ONNX Runtime (node_squeeze: dim must be 1 not 7)
    "flava",  # ForPreTraining: Where node provider type not set in ORT; FlavaModel: optimization renames outputs
    "gemma3n_text",  # past_key_values.layers.0 aliased to shared_layers.0 at ONNX export time (not ORT optimizer)
    "higgs_audio_v2",  # Where node condition cannot broadcast (shape mismatch {3,14,1} vs {20,32})
    "idefics2",  # Invalid ONNX graph: tensor(float) input to Gather node expects tensor(int64)
    "kosmos-2",  # Where node (index_put): incompatible dimensions in shape inference
    "kosmos-2.5",  # Where node (index_put): incompatible dimensions in shape inference
    "mllama",  # cross-attention q/kv size mismatch in test inputs (tensor a 1808 vs tensor b 904)
    "moshi",  # past_key_values key mismatch and sliding_window_tensor numerical mismatch even with optimization disabled
    "pe_audio",  # text_outputs.last_hidden_state aliased with hidden_states.2 at ONNX export time (not ORT optimizer)
    "pe_audio_encoder",  # output_mask aliased with internal mask node at ONNX export time
    "pe_video",  # text_outputs.last_hidden_state aliased with hidden_states.2 at ONNX export time
    "pp_doclayout_v2",  # TopK node provider type not set in ORT (graph optimizer leaves empty provider)
    "t5gemma",  # optimization renames decoder outputs; missing last_hidden_state even with optimization disabled
}

# The following are models that can be exported but their outputs
# are extremely inaccurate compared to the original model.
ONNX_EXTREMELY_INACCURATE_MODEL_TYPES: set[str] = {
    "blt",  # 94.3% mismatch in last_hidden_state
    "flaubert",  # 40% mismatch in end_top_index (top-k beam search non-determinism)
    "parakeet_ctc",  # 100% NaN in logits
    "parakeet_encoder",  # 100% NaN in last_hidden_state
    "patchtst",  # NaN loss output
    "pp_doclayout_v3",  # 68.3% mismatch in enc_topk_bboxes
    "rt_detr",  # 43.3% mismatch in enc_topk_bboxes
    "siglip2",  # 100% mismatch in logits
    "siglip2_vision_model",  # 73.8% mismatch in last_hidden_state
    "vit_mae",  # 99.3% mismatch in ids_restore (random masking)
    "xlm",  # 6.2% mismatch in end_top_index
}


ONNX_DISABLED_OPTIMIZATION_MODEL_TYPES: set[str] = {}


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

        if model.config.model_type in ONNX_EXTREMELY_INACCURATE_MODEL_TYPES:
            raise NotImplementedError(
                f"Exporting a model of type '{model.config.model_type}' results in an ONNX model with extremely inaccurate outputs."
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
