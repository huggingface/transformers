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

from typing import TYPE_CHECKING

from ..utils import _LazyModule
from ..utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .aqlm import *
    from .awq import *
    from .bitnet import *
    from .bitsandbytes import *
    from .deepspeed import *
    from .eetq import *
    from .executorch import *
    from .fbgemm_fp8 import *
    from .finegrained_fp8 import *
    from .flex_attention import *
    from .fsdp import *
    from .ggml import *
    from .higgs import *
    from .hqq import *
    from .hub_kernels import *
    from .integration_utils import *
    from .peft import *
    from .quanto import *
    from .spqr import *
    from .vptq import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
