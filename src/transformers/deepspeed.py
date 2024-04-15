# Copyright 2020 The HuggingFace Team. All rights reserved.
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
"""
Integration with Deepspeed - kept for backward compatiblity, if you plan to make any edit, make sure to modify the file
in `integrations/deepspeed` instead.

Check: https://github.com/huggingface/transformers/pull/25599
"""
import warnings


warnings.warn(
    "transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations",
    FutureWarning,
)

# Backward compatibility imports, to make sure all those objects can be found in integrations/deepspeed
from .integrations.deepspeed import (  # noqa
    HfDeepSpeedConfig,
    HfTrainerDeepSpeedConfig,
    deepspeed_config,
    deepspeed_init,
    deepspeed_load_checkpoint,
    deepspeed_optim_sched,
    is_deepspeed_available,
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)
