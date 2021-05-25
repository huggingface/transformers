# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" XLM-RoBERTa configuration """
from ...onnx import OnnxConfig, OnnxVariable
from ...utils import logging
from ..roberta.configuration_roberta import RobertaConfig


logger = logging.get_logger(__name__)

XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base/resolve/main/config.json",
    "xlm-roberta-large": "https://huggingface.co/xlm-roberta-large/resolve/main/config.json",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/config.json",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/config.json",
    "xlm-roberta-large-finetuned-conll03-english": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/config.json",
    "xlm-roberta-large-finetuned-conll03-german": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/config.json",
}


class XLMRobertaConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    model_type = "xlm-roberta"


XLM_ROBERTA_ONNX_CONFIG = OnnxConfig(
    inputs=[
        OnnxVariable("input_ids", {0: "batch", 1: "sequence"}, repeated=1, value=None),
        OnnxVariable("attention_mask", {0: "batch", 1: "sequence"}, repeated=1, value=None),
    ],
    outputs=[
        OnnxVariable("last_hidden_state", {0: "batch", 1: "sequence"}, repeated=1, value=None),
        OnnxVariable("pooler_output", {0: "batch"}, repeated=1, value=None),
    ],
    runtime_config_overrides=None,
    use_external_data_format=False,
    minimum_required_onnx_opset=12,
    optimizer="bert",
    optimizer_features={
        "enable_gelu": True,
        "enable_layer_norm": True,
        "enable_attention": True,
        "enable_skip_layer_norm": True,
        "enable_embed_layer_norm": True,
        "enable_bias_skip_layer_norm": True,
        "enable_bias_gelu": True,
        "enable_gelu_approximation": False,
    },
    optimizer_additional_args={
        "num_heads": "$config.num_attention_heads",
        "hidden_size": "$config.hidden_size"
    }
)
