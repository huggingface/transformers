# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING

# --- Utility Imports ---
from ...utils import _LazyModule, is_torch_available
from ...utils.import_utils import define_import_structure


# --- Structure Definition ---
# Define what modules exist and what classes they contain for TYPE_CHECKING
# This helps linters and IDEs, even if imports seem "unused" at runtime.
if TYPE_CHECKING:
    # Conditionally import modeling classes based on PyTorch availability
    if is_torch_available():
        pass
    # Note: No separate HindiCausalLMModel in the final flattened version


# --- Lazy Loading Setup ---
# This part handles the actual runtime loading efficiently.
else:
    import sys

    _file = globals()["__file__"]
    # Let define_import_structure automatically create the structure
    # based on the files found in this directory.
    _import_structure = define_import_structure(_file)

    # Set up the lazy module loading for this package
    sys.modules[__name__] = _LazyModule(__name__, _file, _import_structure, module_spec=__spec__)
