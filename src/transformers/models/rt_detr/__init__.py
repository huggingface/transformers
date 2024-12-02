# Copyright 2024 The HuggingFace Team. All rights reserved.
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

<<<<<<< HEAD
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_rt_detr": ["RTDetrConfig"],
    "configuration_rt_detr_resnet": ["RTDetrResNetConfig"],
}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_rt_detr"] = ["RTDetrImageProcessor"]
    _import_structure["image_processing_rt_detr_fast"] = ["RTDetrImageProcessorFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_rt_detr"] = [
        "RTDetrForObjectDetection",
        "RTDetrModel",
        "RTDetrPreTrainedModel",
    ]
    _import_structure["modeling_rt_detr_resnet"] = [
        "RTDetrResNetBackbone",
        "RTDetrResNetPreTrainedModel",
    ]
=======
from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure
>>>>>>> a09860d758302d61d4d1b73a791329e94f762b0e


if TYPE_CHECKING:
    from .configuration_rt_detr import *
    from .configuration_rt_detr_resnet import *
    from .image_processing_rt_detr import *
    from .image_processing_rt_detr_fast import *
    from .modeling_rt_detr import *
    from .modeling_rt_detr_resnet import *
else:
    import sys

<<<<<<< HEAD
    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
=======
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
>>>>>>> a09860d758302d61d4d1b73a791329e94f762b0e
