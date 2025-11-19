# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import DynamoConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .base import HfExporter
from .patch_utils import patch_model_for_export
from .utils import (
    get_auto_dynamic_shapes,
    prepare_for_export,
    register_dynamic_cache_for_export,
    register_encoder_decoder_cache_for_export,
)


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

    if is_torch_greater_or_equal("2.6.0"):
        from torch.export import ExportedProgram

logger = logging.get_logger(__file__)

DYNAMO_UNSUPPORTED_MODEL_TYPES: set[str] = {
    "clvp",  # many data-dependent branches that add bos/eos tokens
    "colqwen2",  # Uses Qwen2VLModel which uses get_rope_index that is data-dependent
    "emu3",  # Emu3VQVAE.encode is data-dependent
    "encodec",  # torch.export struggles with torch.nn.functional.pad with "reflect" mode
    "esm",  # uses compute_tm function which data-dependent
    "fastspeech2_conformer",  # Even after making parts of it exportable, dynamo still struggle with the convolutions in FastSpeech2ConformerMultiLayeredConv1d
    "fastspeech2_conformer_with_hifigan",  # Even after making parts of it exportable, dynamo still struggle with the convolutions in FastSpeech2ConformerMultiLayeredConv1d
    "funnel",  # torch.export struggles with torch.einsum in FunnelRelMultiheadAttention
    "glm4v",  # Glm4vVisionAttention implementation is highly data-dependent
    "glm4v_moe",  # Glm4vMoeVisionAttention implementation is highly data-dependent
    "hiera",  # torch.export struggles with a reshape operation in HieraEncoder.reroll
    "ibert",  # Uses numpy arrays and decimal.Decimal in batch_frexp
    "led",  # global attention implementation is data-dependent
    "lightglue",  # torch.export struggles with sigmoid_log_double_softmax
    "llava_next",  # All three have the same unexplicable error during export
    "llava_next_video",  # All three have the same unexplicable error during export
    "llava_onevision",  # All three have the same unexplicable error during export
    "longformer",  # torch.export is struggling with the global attention implementation
    "mistral3",  # PixtralVisionModel uses some data-dependent truncation
    "modernbert",  # Uses torch.compile directly on some module forward methods
    "nllb-moe",  # TODO: Moe implementation needs to be patched for export
    "omdet-turbo",  # cryptic error AssertionError: assert len(kp) > 0
    "oneformer",  # torch.export is failing on multiple torch methods like torch.linspace and torch.meshgrid
    "pixtral",  # PixtralModel.forward does some data-dependent truncation
    "phi4_multimodal",  # I guess the model is just broken in its current state
    "qwen2_5_omni_thinker",  # already made many parts exportable but still has some non-exportable ops
    "qwen2_5_vl",  # Qwen2_5_VisionTransformerPretrainedModel.get_window_index is data-dependent
    "qwen2_vl",  # Qwen2VLModel.get_rope_index is data-dependent
    "qwen3_omni_moe_thinker",  # Qwen3OmniMoeAudioEncoder.forward does data-dependent chunking
    "qwen3_vl",  # fast_pos_embed_interpolate is data-dependent
    "qwen3_vl_moe",  # fast_pos_embed_interpolate is highly data-dependent
    "superpoint",  # torch.export is failing on torch.nn.functional.grid_sample
    "video_llama_3",  # VideoLlama3VisionAttention implementation is highly data-dependent
    "video_llama_3_vision",  # VideoLlama3VisionAttention implementation is highly data-dependent
    "vilt",  # torch.export is failing on torch.nn.functional.interpolate
    "xmod",  # XmodOutput.lang_adapter is data-dependent
}


class DynamoExporter(HfExporter):
    export_config: DynamoConfig

    required_packages = ["torch"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not is_torch_greater_or_equal("2.6.0"):
            raise ImportError(f"{self.__class__.__name__} requires torch>=2.6.0 for stable Dynamo based export.")

    def export(self, model: "PreTrainedModel", sample_inputs: dict[str, Any]) -> "ExportedProgram":
        """Exports a model to a TorchDynamo ExportedProgram.
        Args:
            model (`PreTrainedModel`):
                The model to export.
            sample_inputs (`Dict[str, Any]`):
                The sample inputs to use for the export.
        Returns:
            `ExportedProgram`: The exported model.
        """
        if model.config.model_type in DYNAMO_UNSUPPORTED_MODEL_TYPES:
            raise NotImplementedError(
                f"{self.__class__.__name__} is not supported for model type '{model.config.model_type}'."
            )

        # we use a copy to avoid side effects
        inputs = copy.deepcopy(sample_inputs)
        model, inputs = prepare_for_export(model, inputs)

        dynamic_shapes = self.export_config.dynamic_shapes
        if self.export_config.dynamic and dynamic_shapes is None:
            dynamic_shapes = get_auto_dynamic_shapes(inputs)

        register_dynamic_cache_for_export()
        register_encoder_decoder_cache_for_export()

        with patch_model_for_export(model):
            exported_program: ExportedProgram = torch.export.export(
                model,
                args=(),
                kwargs=inputs,
                dynamic_shapes=dynamic_shapes,
                strict=self.export_config.strict,
                prefer_deferred_runtime_asserts_over_guards=self.export_config.prefer_deferred_runtime_asserts_over_guards,
            )

        return exported_program
