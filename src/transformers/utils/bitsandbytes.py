# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import warnings


warnings.warn(
    "transformers.utils.bitsandbytes module is deprecated and will be removed in a future version. Please import bitsandbytes modules directly from transformers.integrations",
    FutureWarning,
)

from ..integrations import (  # noqa
    get_keys_to_not_convert,
    replace_8bit_linear,
    replace_with_bnb_linear,
    set_module_8bit_tensor_to_device,
    set_module_quantized_tensor_to_device,
)
