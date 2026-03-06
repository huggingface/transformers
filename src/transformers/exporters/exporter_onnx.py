# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_torch_available
from .exporter_dynamo import DynamoExporter
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
                optimize=self.export_config.optimize,
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


@contextmanager
def patch_torch_for_onnx_export():
    # ONNX export patcher context
    # This context manager monkey-patches PyTorch ops that are unsupported or buggy in ONNX export.
    # The following ops are patched with fallback implementations or workarounds:
    #   - torch.unsqueeze: supports complex tensors
    #   - torch.where / torch.Tensor.where: handles dtype mismatches
    #   - torch.nn.RMSNorm.forward: bypasses aten._fused_rms_norm when elementwise_affine=False
    #   - torch.nn.functional.scaled_dot_product_attention: handles equal q/kv heads (MHA) when enable_gqa=True
    # These patches are only active during export and are reverted afterwards.
    original_torch_where = torch.where
    original_tensor_where = torch.Tensor.where
    original_torch_unsqueeze = torch.unsqueeze
    original_tensor_unsqueeze = torch.Tensor.unsqueeze
    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    original_rms_norm_forward = torch.nn.RMSNorm.forward

    def _torch_where(condition, x=None, y=None):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype != y.dtype:
            y = y.to(x.dtype)
        elif isinstance(x, torch.Tensor) and isinstance(y, (int, float, bool)):
            y = torch.tensor(y, dtype=x.dtype, device=x.device)
        elif isinstance(y, torch.Tensor) and isinstance(x, (int, float, bool)):
            x = torch.tensor(x, dtype=y.dtype, device=y.device)
        if x is None and y is None:
            return original_torch_where(condition)
        elif y is None:
            return original_torch_where(condition, x)
        else:
            return original_torch_where(condition, x, y)

    def _tensor_where(self, condition, other):
        return _torch_where(condition, self, other)

    def _unsqueeze(self_or_input, dim):
        if torch.is_complex(self_or_input):
            real = original_torch_unsqueeze(self_or_input.real, dim)
            imag = original_torch_unsqueeze(self_or_input.imag, dim)
            return torch.complex(real, imag)
        else:
            return original_torch_unsqueeze(self_or_input, dim)

    def _rms_norm_forward(self, x):
        if not self.elementwise_affine:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return original_rms_norm_forward(self, x)

    def _scaled_dot_product_attention(query, key, *args, enable_gqa: bool = False, **kwargs):
        if enable_gqa and query.shape[1] == key.shape[1]:
            enable_gqa = False
        return original_scaled_dot_product_attention(query, key, *args, enable_gqa=enable_gqa, **kwargs)

    torch.where = _torch_where
    torch.Tensor.where = _tensor_where
    torch.unsqueeze = _unsqueeze
    torch.Tensor.unsqueeze = _unsqueeze
    torch.nn.RMSNorm.forward = _rms_norm_forward
    torch.nn.functional.scaled_dot_product_attention = _scaled_dot_product_attention

    try:
        yield
    finally:
        torch.where = original_torch_where
        torch.Tensor.where = original_tensor_where
        torch.unsqueeze = original_torch_unsqueeze
        torch.Tensor.unsqueeze = original_tensor_unsqueeze
        torch.nn.RMSNorm.forward = original_rms_norm_forward
        torch.nn.functional.scaled_dot_product_attention = original_scaled_dot_product_attention
