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

from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_sam3_1_tracker_video import *
    from .modeling_sam3_1_tracker_video import *
else:
    import sys

    _file = globals()["__file__"]
    _import_structure = define_import_structure(_file)
    # Sam31ViTConfig uses a different naming convention (no module prefix), so we add it explicitly.
    _no_dep_key = frozenset()
    _import_structure[_no_dep_key] = dict(_import_structure.get(_no_dep_key, {}))
    _import_structure[_no_dep_key].setdefault("configuration_sam3_1_tracker_video", set()).add("Sam31ViTConfig")
    sys.modules[__name__] = _LazyModule(__name__, _file, _import_structure, module_spec=__spec__)
