# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.
from typing import TYPE_CHECKING

from .import_utils import _LazyModule, define_import_structure


if TYPE_CHECKING:
    from .backbone_utils import *
    from .chat_template_utils import *
    from .constants import *
    from .doc import *
    from .generic import *
    from .hub import *
    from .import_utils import *
    from .logging import *
    from .peft_utils import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
