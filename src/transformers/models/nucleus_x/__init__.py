# MIT License
#
# Copyright (c) 2023  NucleusAI and The HuggingFace Inc. team and Sehyun Choi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_nucleus_x": ["NUCLEUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP", "NucleusXConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_nucleus_x"] = [
        "NUCLEUS_X_PRETRAINED_MODEL_ARCHIVE_LIST",
        "NucleusXForCausalLM",
        "NucleusXForSequenceClassification",
        "NucleusXModel",
        "NucleusXPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_nucleus_x import NUCLEUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP, NucleusXConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_nucleus_x import (
            NUCLEUS_X_PRETRAINED_MODEL_ARCHIVE_LIST,
            NucleusXForCausalLM,
            NucleusXForSequenceClassification,
            NucleusXModel,
            NucleusXPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
