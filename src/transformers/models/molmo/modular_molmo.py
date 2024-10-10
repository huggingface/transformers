# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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


from torch import nn

from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.llava.configuration_llava import (
    LlavaConfig,
)

from ...utils import logging
from ..clip.modeling_clip import (
    CLIPEncoder,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionTransformer,
)
from ..llava.modeling_llava import (
    LlavaForConditionalGeneration,
    LlavaMultiModalProjector,
)
from ..qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2MLP, Qwen2Model


logger = logging.get_logger(__name__)


class MolmoVisionConfig(CLIPVisionConfig):
    pass


class MolmoConfig(LlavaConfig):
    pass


class MolmoMLP(Qwen2MLP):
    def __init__(self, config):
        super().__init__()
        self.down_proj = nn.Linear(config.intermediate_size // 2, config.hidden_size, bias=False)


class MolmoDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.mlp = MolmoMLP(config)


class MolmoModel(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size + config.additional_vocab_size,
            config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [MolmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class MolmoForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = MolmoModel(config)
        self.post_init()


class MolmoMultiModalProjector(LlavaMultiModalProjector):
    def __init__(self, config: MolmoConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.intermediate_size // 2,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            config.text_config.intermediate_size // 2,
            config.text_config.hidden_size,
            bias=False,
        )
        self.linear_3 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.intermediate_size // 2,
            bias=False,
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        intermediate_states = self.linear_3(image_features)
        hidden_states = self.linear_2(hidden_states, intermediate_states)
        return hidden_states


"""
class MolmoImagePooling2D(nn.Module):
    self.image_pooling_2d = MultiHeadDotProductAttention(config, is_vit_layer=False)
"""


# This needs to be in caps for some reason in the modular renaming
class MolmoVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config):
        super().__init__()
        self.position_embedding = nn.Embedding(config.num_image_positions, config.hidden_size)


class MolmoVisionTransformer(CLIPVisionTransformer):
    def __init__(self, config):
        super().__init__()
        self.embeddings = MolmoVisionEmbeddings(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=True)


class MolmoEncoder(CLIPEncoder):
    pass


class MolmoVisionModel(CLIPVisionModel):
    def __init__(self, config):
        super().__init__()
        self.vision_model = MolmoVisionTransformer(config)
        self.encoder = MolmoEncoder(config)


class MolmoForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: MolmoConfig):
        super().__init__(config)
        self.multi_modal_projector = MolmoMultiModalProjector(config)

        self.language_model = MolmoForCausalLM._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vision_tower = MolmoVisionModel._from_config(config.vision_config)
        self.post_init()


__all__ = [
    "MolmoConfig",
    "MolmoVisionConfig",
    "MolmoVisionEmbeddings",
    "MolmoVisionModel",
    "MolmoModel",
    "MolmoForConditionalGeneration",
]
