# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...modeling_layers import GradientCheckpointingLayer
from ...utils import auto_docstring
from ..kolibri1.configuration_kolibri1 import Kolibri1Config
from ..kolibri1.modeling_kolibri1 import (
    Kolibri1Attention,
    Kolibri1DecoderLayer,
    Kolibri1ForCausalLM,
    Kolibri1MLP,
    Kolibri1Model,
    Kolibri1PreTrainedModel,
    Kolibri1RMSNorm,
    Kolibri1SparseMoeBlock,
)


@auto_docstring(checkpoint="Aleph-Alpha/Kolibri-1.1")
@strict
class Kolibri1_1Config(Kolibri1Config):
    r"""
    mlp_only_layers (`list[int]`, *optional*):
        Indices of the layers with a dense MLP instead of a mixture of experts. Defaults to none.
    mlp_only_intermediate_size (`int`, *optional*):
        Intermediate size of the dense MLPs of `mlp_only_layers`.

    ```python
    >>> from transformers import Kolibri1_1Model, Kolibri1_1Config

    >>> configuration = Kolibri1_1Config()
    >>> model = Kolibri1_1Model(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "kolibri1_1"

    mlp_only_layers: list[int] | None = None
    mlp_only_intermediate_size: int | None = None

    def __post_init__(self, **kwargs):
        if self.mlp_only_layers is None:
            self.mlp_only_layers = []
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if not set(self.mlp_only_layers) <= set(range(self.num_hidden_layers)):
            raise ValueError(f"`mlp_only_layers` must be layer indices, got {self.mlp_only_layers}.")
        if self.mlp_only_layers and (self.mlp_only_intermediate_size or 0) <= 0:
            raise ValueError("`mlp_only_layers` need a positive `mlp_only_intermediate_size`.")


class Kolibri1_1Attention(Kolibri1Attention):
    pass


class Kolibri1_1MLP(Kolibri1MLP):
    pass


class Kolibri1_1RMSNorm(Kolibri1RMSNorm):
    pass


class Kolibri1_1SparseMoeBlock(Kolibri1SparseMoeBlock):
    pass


class Kolibri1_1DecoderLayer(Kolibri1DecoderLayer):
    def __init__(self, config: Kolibri1_1Config, layer_idx: int):
        # Not the parent's `__init__`, which would build the experts of the dense MLP layers only to discard them
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Kolibri1_1Attention(config=config, layer_idx=layer_idx)
        if layer_idx in config.mlp_only_layers:
            self.mlp = Kolibri1_1MLP(config, intermediate_size=config.mlp_only_intermediate_size)
        else:
            self.mlp = Kolibri1_1SparseMoeBlock(config)
        self.input_layernorm = Kolibri1_1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Kolibri1_1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = Kolibri1_1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = Kolibri1_1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Kolibri1_1PreTrainedModel(Kolibri1PreTrainedModel):
    pass


class Kolibri1_1Model(Kolibri1Model):
    pass


class Kolibri1_1ForCausalLM(Kolibri1ForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Kolibri1_1ForCausalLM

        >>> model = Kolibri1_1ForCausalLM.from_pretrained("Aleph-Alpha/Kolibri-1.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("Aleph-Alpha/Kolibri-1.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "Kolibri1_1Config",
    "Kolibri1_1ForCausalLM",
    "Kolibri1_1Model",
    "Kolibri1_1PreTrainedModel",
]
