# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from importlib import metadata
from types import SimpleNamespace
from typing import TYPE_CHECKING

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_gptqmodel_available, is_optimum_available, is_torch_available, logging
from ..utils.quantization_config import GPTQConfig, QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


MIN_GPTQ_VERSION = "1.4.3"
MIN_OPTIMUM_VERSION = "1.24.0"


class GptqHfQuantizer(HfQuantizer):
    """
    Quantizer of the GPTQ method - for GPTQ the quantizer support calibration of the model through
    the GPT-QModel package (Python import name `gptqmodel`). Quantization is done under the hood for users if they
    load a non-prequantized model.
    """

    requires_calibration = False
    quantization_config: "GPTQConfig"

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        if self.pre_quantized and self.quantization_config.dequantize:
            return  # dequantizing is pure torch

        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")
        from optimum.gptq import GPTQQuantizer

        self.optimum_quantizer = GPTQQuantizer.from_dict(self.quantization_config.to_dict_optimum())

    def validate_environment(self, *args, **kwargs):
        if self.pre_quantized and self.quantization_config.dequantize:
            return  # dequantizing is pure torch
        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")

        gptq_supports_cpu = is_gptqmodel_available()
        if not gptq_supports_cpu and not torch.cuda.is_available():
            raise RuntimeError("GPU is required to quantize or run quantize model.")
        elif not is_gptqmodel_available():
            raise ImportError("Loading a GPTQ quantized model requires gptqmodel (`pip install gptqmodel`) library.")
        elif is_gptqmodel_available() and (
            version.parse(metadata.version("gptqmodel")) < version.parse(MIN_GPTQ_VERSION)
            or version.parse(metadata.version("optimum")) < version.parse(MIN_OPTIMUM_VERSION)
        ):
            raise ImportError(
                f"The gptqmodel version should be >= {MIN_GPTQ_VERSION}, optimum version should >= {MIN_OPTIMUM_VERSION}"
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype != torch.float16:
            logger.info("We suggest you to set `dtype=torch.float16` for better efficiency with GPTQ.")
        return dtype

    def update_device_map(self, device_map):
        if device_map is None:
            device_map = {"": torch.device("cpu")}
        return device_map

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized and self.quantization_config.dequantize:
            # The model stays dense: the `dequantize` load operation runs on every checkpoint weight
            return
        if model.__class__.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")

        if self.pre_quantized:
            # compat: latest optimum has gptqmodel refactor
            if version.parse(metadata.version("optimum")) < version.parse(MIN_OPTIMUM_VERSION):
                model = self.optimum_quantizer.convert_model(model)
            else:
                model = self.optimum_quantizer.convert_model(model, **kwargs)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized and self.quantization_config.dequantize:
            return
        if self.pre_quantized:
            # What optimum's `post_init_model` does that is not per layer, the rest ran in `GptqDeserialize`
            from ..integrations.gptq import gptqmodel_post_init

            model.quantize_config = SimpleNamespace(desc_act=self.quantization_config.desc_act)
            gptqmodel_post_init(model, use_act_order=self.quantization_config.desc_act)
        else:
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path

            self.optimum_quantizer.quantize_model(model, self.quantization_config.tokenizer)
            model.config.quantization_config = GPTQConfig.from_dict(self.optimum_quantizer.to_dict())

    @property
    def is_trainable(self) -> bool:
        return True

    def is_serializable(self):
        return True

    serialized_names = {"weight": ["qweight", "qzeros", "scales", "g_idx"]}

    def get_load_operations(self):
        from ..integrations.gptq import GptqDequantize, GptqDeserialize

        return {"deserialize": GptqDeserialize, "dequantize": GptqDequantize}

    def get_weight_conversions(self):
        if self.pre_quantized and not self.quantization_config.dequantize:
            from ..core_model_loading import WeightConverter
            from ..integrations.gptq import GptqDeserialize

            # The gptqmodel layers have no `weight`: their tensors gather under `qweight`
            return [WeightConverter(self.serialized_names["weight"], "qweight", [GptqDeserialize(self)])]
        return super().get_weight_conversions()
