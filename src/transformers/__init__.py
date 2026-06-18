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

__version__ = "4.51.0.dev0"

# flake8: noqa
# There's no way to override "standard" names without breaking backward compatibility.
# It would be possible to list all public symbols in __all__, but we don't.
from . import (
    agents,
    audio_utils,
    benchmarks,
    commands,
    data,
    debug_utils,
    dependency_versions_check,
    dependency_versions_table,
    dynamic_module_utils,
    feature_extraction_utils,
    generation,
    graph_optimization,
    huggingface_tools,
    integrations,
    keras_callbacks,
    model_debugging,
    modelcard,
    modeling_gguf_pytorch_utils,
    modeling_outputs,
    modeling_tf_utils,
    modeling_utils,
    onnx,
    outputs,
    pipelines,
    pytorch_utils,
    quantization_utils,
    quantizers,
    safetensors,
    testing_utils,
    tf_utils,
    time_series_utils,
    tokenization_utils,
    tokenization_utils_base,
    trainer_utils,
    utils,
)

# DECLARATION: Expose PhiRecursiveGenerator
from .generation.phi_utils import (
    PhiRecursiveGenerator,
    phi_max_tokens,
    phi_repetition_penalty,
    phi_temperature,
    phi_top_k,
    phi_top_p,
)
