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


import os
import re
from typing import List

from ..utils import is_compressed_tensors_available, is_torch_available, logging
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package.  Loads and restores models to
    quantized state with compressed_tensors
    """

    requires_calibration = True
    required_packages = ["compressed_tensors"]

    def __init__(self, quantization_config: CompressedTensorsConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )

        # Call post_init here to ensure proper config setup when `run_compressed`
        # is provided directly via CompressedTensorsConfig, and to avoid duplicate logging.

        quantization_config.post_init()
        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)
        self.run_compressed = quantization_config.run_compressed
        self.quantization_config = quantization_config

    def update_missing_keys_after_loading(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        """
        Update missing keys after loading the model. This is necessary for compressed tensors
        to load the model correctly. We expect weights to be present in missing keys.
        The weight's are re-constructed by ModelCompressor in _process_model_after_weight_loading

        This function cleans up expected missing keys and returns the remaining missing keys
        """

        if self.run_compressed:
            return missing_keys

        # We expect some keys to be missing for
        # compressed models
        # This is fine as the weights are reconstructed by ModelCompressor
        # in _process_model_after_weight_loading

        expected_missing_keys = self.compressor.get_missing_module_keys(model)
        return [
            key for key in missing_keys if not any(re.match(f".*{pattern}", key) for pattern in expected_missing_keys)
        ]

    def update_unexpected_keys(self, model, unexpected_keys: List[str], prefix: str) -> List[str]:
        """
        Override this method if you want to adjust the `unexpected_keys`.

        Args:
            unexpected_keys (`List[str]`, *optional*):
                The list of unexpected keys in the checkpoint compared to the state dict of the model
        """

        if self.run_compressed:
            return unexpected_keys

        # We expect some unexpected keys in model
        # safetensors file for compressed models
        keys_to_ignore = self.compressor.get_unexpected_file_keys(model)
        return [key for key in unexpected_keys if not any(re.match(f".*{pattern}", key) for pattern in keys_to_ignore)]

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )
        if not is_torch_available():
            # torch already should be installed as part of compressed tensors
            raise ImportError("torch is required for using compressed-tensors quantization")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            logger.info("Loading model using torch.float16 for compressed-tensors quantization")
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.info(
                "We suggest you to set `torch_dtype=torch.float16` for better efficiency with compressed_tensors."
            )
        return torch_dtype

    def _process_model_before_weight_loading(self, model, **kwargs):
        from compressed_tensors.quantization import apply_quantization_config

        ct_quantization_config = self.compressor.quantization_config

        if self.run_compressed:
            apply_quantization_config(model, ct_quantization_config, run_compressed=True)
        elif not self.quantization_config.is_quantization_compressed:
            apply_quantization_config(model, ct_quantization_config)

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Decompress loaded model if necessary - need for qat"""

        if (
            self.quantization_config.is_quantization_compressed and not self.run_compressed
        ) or self.quantization_config.is_sparsification_compressed:
            config = kwargs.get("config", None)
            cache_path = config._name_or_path

            if not os.path.exists(cache_path):
                from transformers.utils import cached_file

                config_file_path = cached_file(cache_path, "config.json")
                cache_path = os.path.sep.join(config_file_path.split(os.path.sep)[:-1])

            if self.quantization_config.is_quantization_compressed and not self.run_compressed:
                from compressed_tensors.quantization import QuantizationStatus

                self.compressor.quantization_config.quantization_status = QuantizationStatus.FROZEN
            self.compressor.decompress(model_path=cache_path, model=model)

    def update_tp_plan(self, config):
        additional_plan = {
            "layers.*.feed_forward.experts.*.gate_proj.weight": "local_colwise",
            "layers.*.feed_forward.experts.*.gate_proj.weight_scale": "local_colwise",
            "layers.*.feed_forward.experts.*.up_proj.weight": "local_colwise",
            "layers.*.feed_forward.experts.*.up_proj.weight_scale": "local_colwise",
            "layers.*.feed_forward.experts.*.down_proj.weight": "local_rowwise",
        }
        if config.get_text_config() is not None and config.get_text_config().base_model_tp_plan is not None:
            config.get_text_config().base_model_tp_plan.update(additional_plan)

        return config

    @property
    def is_trainable(self):
        return True

    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        # models need to be decompressed carry out qat
        return not self.run_compressed or not self.quantization_config.is_quantization_compressed

    def is_serializable(self, safe_serialization=None) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True
