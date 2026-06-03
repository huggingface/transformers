# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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


import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import (
    auto_docstring,
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..idefics.modeling_idefics import IdeficsBaseModelOutputWithPast, IdeficsCausalLMOutputWithPast
from ..janus.image_processing_janus import JanusImageProcessor
from ..janus.image_processing_pil_janus import JanusImageProcessorPil
from ..janus.modeling_janus import JanusForConditionalGeneration, JanusModel, JanusPreTrainedModel


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-community/deepseek-vl-1.3b-chat")
@strict
class DeepseekVLConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import DeepseekVLConfig, DeepseekVLModel

    >>> # Initializing a DeepseekVL deepseek-community/deepseek-vl-1.3b-chat style configuration
    >>> configuration = DeepseekVLConfig()

    >>> # Initializing a model (with random weights) from the deepseek-community/deepseek-vl-1.3b-chat style configuration
    >>> model = DeepseekVLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 100015
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = {}
            logger.info("`text_config` is `None`. Initializing the `LlamaConfig` with default values.")
        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)

        if self.vision_config is None:
            self.vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `SiglipVisionConfig` with default values.")
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)

        super().__post_init__(**kwargs)


class DeepseekVLBaseModelOutputWithPast(IdeficsBaseModelOutputWithPast):
    pass


class DeepseekVLCausalLMOutputWithPast(IdeficsCausalLMOutputWithPast):
    pass


class DeepseekVLAligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        in_features = config.vision_config.hidden_size
        out_features = config.text_config.hidden_size

        self.linear1 = nn.Linear(in_features, out_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, vision_encodings: torch.Tensor) -> torch.Tensor:
        x = self.linear1(vision_encodings)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class DeepseekVLPreTrainedModel(JanusPreTrainedModel):
    _no_split_modules = ["LlamaDecoderLayer"]

    def _init_weights(self, module):
        raise AttributeError("No need to inherit!")


@auto_docstring
class DeepseekVLModel(JanusModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.vision_model = AutoModel.from_config(config.vision_config)
        self.aligner = DeepseekVLAligner(config)

        self.language_model = AutoModel.from_config(config=config.text_config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing.
        self.post_init()

        del self.vqmodel
        del self.generation_embeddings
        del self.generation_aligner
        del self.generation_head


class DeepseekVLForConditionalGeneration(JanusForConditionalGeneration):
    output_modalities = ("text",)

    def prepare_embeddings_for_image_generation(self):
        raise AttributeError("Not needed for DeepseekVL")

    def decode_image_tokens(self):
        raise AttributeError("Not needed for DeepseekVL")

    def generate(self):
        raise AttributeError("Not needed for DeepseekVL")


class DeepseekVLImageProcessorPil(JanusImageProcessorPil):
    def postprocess(self):
        raise AttributeError("Not needed for DeepseekVL")

    def unnormalize(self):
        raise AttributeError("Not needed for DeepseekVL")


class DeepseekVLImageProcessor(JanusImageProcessor):
    def postprocess(self):
        raise AttributeError("Not needed for DeepseekVL")


class DeepseekVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
            "return_text_replacement_offsets": False,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class DeepseekVLProcessor(ProcessorMixin):
    valid_processor_kwargs = DeepseekVLProcessorKwargs

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        num_image_tokens=576,
    ):
        r"""
        num_image_tokens (`int`, *optional*, defaults to 576):
            The number of special image tokens used as placeholders for visual content in text sequences.
        """
        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.encode(self.image_token, add_special_tokens=False)[0]
        self.num_image_tokens = num_image_tokens

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        return self.image_token * self.num_image_tokens


__all__ = [
    "DeepseekVLConfig",
    "DeepseekVLPreTrainedModel",
    "DeepseekVLModel",
    "DeepseekVLForConditionalGeneration",
    "DeepseekVLImageProcessor",
    "DeepseekVLImageProcessorPil",
    "DeepseekVLProcessor",
]
