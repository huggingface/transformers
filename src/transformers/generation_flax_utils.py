# coding=utf-8
# Copyright 2021 The Google AI Flax Team Authors, and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from .generation import FlaxGenerationMixin


class FlaxGenerationMixin(FlaxGenerationMixin):
    # warning at import time
    warnings.warn(
        "Importing `FlaxGenerationMixin` from `src/transformers/generation_flax_utils.py` is deprecated and will "
        "be removed in Transformers v5. Import as `from transformers import FlaxGenerationMixin` instead.",
        FutureWarning,
    )
