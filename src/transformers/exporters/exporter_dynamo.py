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
import inspect
from typing import TYPE_CHECKING

from ..cache_utils import DynamicCache, EncoderDecoderCache
from ..generation.utils import GenerationMixin
from ..utils import logging
from ..utils.export_config import DynamoConfig
from ..utils.import_utils import is_torch_available, is_torch_greater_or_equal
from .base import HfExporter
from .utils import (
    get_auto_dynamic_shapes,
    patch_masks_for_export,
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


class DynamoExporter(HfExporter):
    export_config: DynamoConfig

    required_packages = ["torch"]

    def validate_environment(self, *args, **kwargs):
        super().validate_environment(*args, **kwargs)

        if not is_torch_greater_or_equal("2.6.0"):
            raise ImportError(f"{self.__class__.__name__} requires torch>=2.6.0 for stable Dynamo based export.")

    def export(self, model: "PreTrainedModel"):
        if self.export_config.sample_inputs is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} can't automatically generate export inptus. Please provide sample_inputs in the exporter_config as a dictionary. "
                "You can do so by using the tokenizer/processor to prepare a batch of inputs as you would do for a normal forward pass. "
                f"{self.__class__.__name__} can automatically generate past_key_values and its dynamic shapes if the model is "
                "auto-regressive and model.config.use_cache is set to True."
            )

        sample_inputs = copy.deepcopy(self.export_config.sample_inputs)

        register_dynamic_cache_for_export()
        register_encoder_decoder_cache_for_export()
        if (
            isinstance(model, GenerationMixin)
            and getattr(model.config, "use_cache", False)
            and "past_key_values" in inspect.signature(model.forward).parameters
        ):
            if "past_key_values" not in sample_inputs:
                logger.info(
                    f"{self.__class__.__name__} detected an auto-regressive model with use_cache=True but no past_key_values in sample_inputs. "
                    "Generating a dummy past_key_values for export requires running a forward pass which may be time-consuming. "
                    "You can also provide past_key_values in sample_inputs to avoid this step."
                )
                self.prepare_cache_inputs_for_export(model, sample_inputs)

        dynamic_shapes = self.export_config.dynamic_shapes
        if self.export_config.dynamic and dynamic_shapes is None:
            # assigns AUTO to all axes to let torch.onnx decide
            dynamic_shapes = get_auto_dynamic_shapes(sample_inputs)

        with patch_masks_for_export():
            exported_program: ExportedProgram = torch.export.export(
                model,
                args=(),
                kwargs=sample_inputs,
                dynamic_shapes=dynamic_shapes,
                strict=self.export_config.strict,
            )

        model.exported_model = exported_program

        return exported_program

    @staticmethod
    def prepare_cache_inputs_for_export(model: "PreTrainedModel", sample_inputs: dict):
        with torch.no_grad():
            dummy_outputs = model(**copy.deepcopy(sample_inputs))

        if hasattr(dummy_outputs, "past_key_values"):
            if isinstance(dummy_outputs.past_key_values, DynamicCache):
                sample_inputs["past_key_values"] = dummy_outputs.past_key_values
                if model.config.model_type not in {"qwen2_vl", "qwen2_5_vl"}:
                    seq_length = sample_inputs["input_ids"].shape[1]
                    past_length = sample_inputs["past_key_values"].get_seq_length()
                    sample_inputs["attention_mask"] = torch.ones(
                        (sample_inputs["input_ids"].shape[0], past_length + seq_length),
                        device=model.device,
                        dtype=torch.long,
                    )
            elif isinstance(dummy_outputs.past_key_values, EncoderDecoderCache):
                logger.warning(
                    "The model seems to be returning an EncoderDecoderCache as past_key_values. "
                    "DynamoExporter does not yet support cache in inputs for encoder-decoder models. "
                    "Please provide past_key_values in sample_inputs manually if needed."
                )
