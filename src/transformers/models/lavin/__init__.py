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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_essentia_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_torch_available,
)


_import_structure = {
    "configuration_lavin": ["LAVIN_PRETRAINED_CONFIG_ARCHIVE_MAP", "LavinConfig"],
}

try:
  if not is_sentencepiece_available():
    raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
  pass
else:
  _import_structure["lavin_llama"] = ["LavinTokenizer"]

try:
   if not is_tokenizers_available():
      raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
  pass
else:
  _import_structure["tokenization_llama_fast"] = ["LlamaTokenizerFast"]

try:
  if not is_torch_available():
    raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
  pass
else:
  _import_structure["modeling_lavin"] = [
      "LavinForCausalLM",
      "LavinModel",
      "LavinPreTrainedModel",
      "LavinForSequenceClassification",
  ]

if TYPE_CHECKING:
  from .configuration_lavin import LAVIN_PRETRAINED_CONFIG_ARCHIVE_MAP, LavinConfig

  try:
    if not is_sentencepiece_available():
      raise OptionalDependencyNotAvailable()
  except OptionalDependencyNotAvailable:
    pass
  else:
    from .tokenization_lavin import LavinTokenizer

  try:
    if not is_tokenizers_available():
      raise OptionalDependencyNotAvailable()
  except OptionalDependencyNotAvailable:
    pass
  else:
    from .tokenization_llama import LavinTokenizer

  try:
    if not is_torch_available():
      raise OptionalDependencyNotAvailable()
  except OptionalDependencyNotAvailable:
    pass
  else:
    from .modeling_lavin import LavinForCausalLM, LavinForSequenceClassification, LavinModel, LavinPreTrainedModel

else:
  import sys

  sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
