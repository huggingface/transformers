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
from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .cpmant import *
    from .decision_transformer import *
    from .glpn import *
    from .groupvit import *
    from .imagegpt import *
    from .lxmert import *
    from .mobilevit import *
    from .mobilevitv2 import *
    from .mra import *
    from .nystromformer import *
    from .pegasus_x import *
    from .regnet import *
    from .roberta_prelayernorm import *
    from .switch_transformers import *
    from .univnet import *
    from .visual_bert import *
    from .vit_msn import *
    from .yoso import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
