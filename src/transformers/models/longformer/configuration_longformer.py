# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
""" Longformer configuration"""
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union

from ...onnx import OnnxConfig
from ...utils import TensorType, logging
from ..roberta.configuration_roberta import RobertaConfig


if TYPE_CHECKING:
    from ...configuration_utils import PretrainedConfig
    from ...onnx.config import PatchingSpec
    from ...tokenization_utils_base import PreTrainedTokenizerBase


logger = logging.get_logger(__name__)

LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/config.json",
    "allenai/longformer-large-4096": "https://huggingface.co/allenai/longformer-large-4096/resolve/main/config.json",
    "allenai/longformer-large-4096-finetuned-triviaqa": (
        "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/config.json"
    ),
    "allenai/longformer-base-4096-extra.pos.embd.only": (
        "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/config.json"
    ),
    "allenai/longformer-large-4096-extra.pos.embd.only": (
        "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/config.json"
    ),
}


class LongformerConfig(RobertaConfig):
    r"""
    This is the configuration class to store the configuration of a [`LongformerModel`] or a [`TFLongformerModel`]. It
    is used to instantiate a Longformer model according to the specified arguments, defining the model architecture.

    This is the configuration class to store the configuration of a [`LongformerModel`]. It is used to instantiate an
    Longformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LongFormer
    [allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096) architecture with a sequence
    length 4,096.

    The [`LongformerConfig`] class directly inherits [`RobertaConfig`]. It reuses the same defaults. Please check the
    parent class for more information.

    Args:
        attention_window (`int` or `List[int]`, *optional*, defaults to 512):
            Size of an attention window around each token. If an `int`, use the same size for all layers. To specify a
            different window size for each layer, use a `List[int]` where `len(attention_window) == num_hidden_layers`.

    Example:

    ```python
    >>> from transformers import LongformerConfig, LongformerModel

    >>> # Initializing a Longformer configuration
    >>> configuration = LongformerConfig()

    >>> # Initializing a model from the configuration
    >>> model = LongformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "longformer"

    def __init__(
        self, attention_window: Union[List[int], int] = 512, sep_token_id: int = 2, onnx_export: bool = False, **kwargs
    ):
        super().__init__(sep_token_id=sep_token_id, **kwargs)
        self.attention_window = attention_window
        self.onnx_export = onnx_export


class LongformerOnnxConfig(OnnxConfig):
    def __init__(self, config: "PretrainedConfig", task: str = "default", patching_specs: "List[PatchingSpec]" = None):
        super().__init__(config, task, patching_specs)
        config.onnx_export = True

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("global_attention_mask", dynamic_axis),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        outputs = super().outputs
        if self.task == "default":
            outputs["pooler_output"] = {0: "batch"}
        return outputs

    @property
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        return 1e-4

    @property
    def default_onnx_opset(self) -> int:
        # needs to be >= 14 to support tril operator
        return max(super().default_onnx_opset, 14)

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        inputs = super().generate_dummy_inputs(
            preprocessor=tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )
        import torch

        inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
        # make every second token global
        inputs["global_attention_mask"][:, ::2] = 1
        return inputs
