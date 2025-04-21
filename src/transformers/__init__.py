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

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "4.52.0.dev0"


from typing import TYPE_CHECKING

from .utils import logging
from .utils.import_utils import _LazyModule, define_import_structure


if TYPE_CHECKING:
    from .activations import *
    from .activations_tf import *
    from .audio_utils import *
    from .cache_utils import *
    from .commands import *
    from .configuration_utils import *
    from .convert_graph_to_onnx import *
    from .convert_slow_tokenizer import *
    from .convert_slow_tokenizers_checkpoints_to_fast import *
    from .convert_tf_hub_seq_to_seq_bert_to_pytorch import *
    from .data import *
    from .data.data_collator import *
    from .data.datasets import *
    from .data.metrics import *
    from .data.processors import *
    from .debug_utils import *
    from .dependency_versions_check import *
    from .dependency_versions_table import *
    from .dynamic_module_utils import *
    from .feature_extraction_sequence_utils import *
    from .feature_extraction_utils import *
    from .file_utils import *
    from .generation import *
    from .hf_argparser import *
    from .hyperparameter_search import *
    from .image_processing_base import *
    from .image_processing_utils import *
    from .image_processing_utils_fast import *
    from .image_transforms import *
    from .image_utils import *
    from .keras_callbacks import *
    from .loss import *
    from .model_debugging_utils import *
    from .modelcard import *
    from .modeling_flash_attention_utils import *
    from .modeling_flax_outputs import *
    from .modeling_flax_utils import *
    from .modeling_layers import *
    from .modeling_outputs import *
    from .modeling_rope_utils import *
    from .modeling_tf_outputs import *
    from .modeling_tf_pytorch_utils import *
    from .modeling_tf_utils import *
    from .modeling_utils import *
    from .models import *
    from .onnx import *
    from .optimization import *
    from .optimization_tf import *
    from .pipelines import *
    from .processing_utils import *
    from .pytorch_utils import *
    from .quantizers import *
    from .sagemaker import *
    from .testing_utils import *
    from .tf_utils import *
    from .time_series_utils import *
    from .tokenization_utils import *
    from .tokenization_utils_base import *
    from .tokenization_utils_fast import *
    from .trainer import *
    from .trainer_callback import *
    from .trainer_pt_utils import *
    from .trainer_seq2seq import *
    from .trainer_utils import *
    from .training_args import *
    from .training_args_seq2seq import *
    from .training_args_tf import *
    from .utils import *
    from .utils.quantization_config import *
else:
    import sys

    _file = globals()["__file__"]

    sys.modules[__name__] = _LazyModule(
        __name__,
        _file,
        define_import_structure(_file),
        module_spec=__spec__,
        # To authorize 'from transformers import logging' additionally to 'from transformers.utils import logging'
        explicit_import_shortcut={"utils.logging": ["logging"]},
        extra_objects={"__version__": __version__},
    )
