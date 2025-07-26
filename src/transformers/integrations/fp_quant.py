# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"FP-Quant integration file"

from ..utils import (
    is_fp_quant_available,
)


if is_fp_quant_available():
    from fp_quant import FPQuantConfig as FPQuantLinearConfig
    from fp_quant import FPQuantDtype

from transformers.utils.quantization_config import FPQuantConfig


def adapt_fp_quant_config(config: FPQuantConfig):
    if config.forward_dtype == "mxfp4":
        forward_dtype = FPQuantDtype.MXFP4
    else:
        raise ValueError(f"Unsupported forward dtype: {config.forward_dtype}")

    if config.backward_dtype == "bf16":
        backward_dtype = FPQuantDtype.BF16
    else:
        raise ValueError(f"Unsupported backward dtype: {config.backward_dtype}")

    return FPQuantLinearConfig(
        forward_dtype=forward_dtype,
        forward_method=config.forward_method,
        backward_dtype=backward_dtype,
        store_master_weights=config.store_master_weights,
        hadamard_group_size=config.hadamard_group_size,
        pseudoquantization=config.pseudoquantization,
        modules_to_not_convert=config.modules_to_not_convert,
    )
