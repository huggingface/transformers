# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""One quantizer per checkpoint PRODUCER. `base` serves the plain `weight` / `weight_scale_inv`
pair; an arm exists where a producer's key layout needs one, and says which format each module
is in through the config's groups, not through the arm."""

from .base import FineGrainedHfQuantizer
from .blockfp8 import FineGrainedBlockFp8HfQuantizer
from .mxfp4 import FineGrainedMxfp4HfQuantizer
from .mxfp8 import FineGrainedMxfp8HfQuantizer
from .nvfp4 import FineGrainedNvfp4HfQuantizer


__all__ = [
    "FineGrainedBlockFp8HfQuantizer",
    "FineGrainedHfQuantizer",
    "FineGrainedMxfp4HfQuantizer",
    "FineGrainedMxfp8HfQuantizer",
    "FineGrainedNvfp4HfQuantizer",
]
