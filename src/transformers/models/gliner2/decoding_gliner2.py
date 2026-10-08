# Copyright 2026 The HuggingFace Inc. team and the GLiNER2 authors. All rights reserved.
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
"""Public GLiNER2 decoders.

Constrained classification and joint IE are split out of this module. Both
consume logits or score tensors and return plain result dictionaries.
"""

from .decoding_constraints import decode_constrained_classification
from .decoding_joint import decode_joint, resolve_overlaps


decode_classification = decode_constrained_classification
