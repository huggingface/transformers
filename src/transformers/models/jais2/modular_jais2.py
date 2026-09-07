# Copyright 2025 the HuggingFace Team. All rights reserved.
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


import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring, can_return_tuple
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from ..nemotron.modeling_nemotron import NemotronMLP


@auto_docstring(checkpoint="inceptionai/Jais-2-8B-Chat")
@strict
class Jais2Config(LlamaConfig):
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 150272
    hidden_size: int = 3328
    intermediate_size: int = 26624
    num_attention_heads: int = 26
    hidden_act: str = "relu2"
    max_position_embeddings: int = 8192
    layer_norm_eps: float = 1e-5
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 150024
    attention_bias: bool = True
    mlp_bias: bool = True
    rms_norm_eps = AttributeError()
    pretraining_tp = AttributeError()


class Jais2MLP(NemotronMLP):
    pass


class Jais2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Jais2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class Jais2PreTrainedModel(LlamaPreTrainedModel):
    pass


class Jais2Model(LlamaModel):
    def __init__(self, config: Jais2Config):
        super().__init__(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class Jais2ForCausalLM(LlamaForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Jais2ForCausalLM

        >>> model = Jais2ForCausalLM.from_pretrained("inceptionai/Jais-2-8B-Chat")
        >>> tokenizer = AutoTokenizer.from_pretrained("inceptionai/Jais-2-8B-Chat")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "Jais2Config",
    "Jais2Model",
    "Jais2ForCausalLM",
    "Jais2PreTrainedModel",
]
