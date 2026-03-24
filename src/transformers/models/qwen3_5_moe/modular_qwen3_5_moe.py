# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3.5Moe model."""

import torch
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ..qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from ..qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5Model,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
    Qwen3_5VisionRotaryEmbedding,
)
from ..qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from ..qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextDynamicCache,
    Qwen3NextExperts,
    Qwen3NextForCausalLM,
    Qwen3NextPreTrainedModel,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
)
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from ..qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeCausalLMOutputWithPast,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModelOutputWithPast,
    Qwen3VLMoeTextTopKRouter,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Qwen/Qwen3.5-35B-A3B")
@strict
class Qwen3_5MoeTextConfig(Qwen3NextConfig):
    r"""
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the convolution used in linear attention layers.
    linear_key_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each key head in linear attention.
    linear_value_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each value head in linear attention.
    linear_num_key_heads (`int`, *optional*, defaults to 16):
        Number of key heads used in linear attention layers.
    linear_num_value_heads (`int`, *optional*, defaults to 32):
        Number of value heads used in linear attention layers.

    ```python
    >>> from transformers import Qwen3_5MoeTextModel, Qwen3_5MoeTextConfig

    >>> # Initializing a Qwen3.5-MoE style configuration
    >>> configuration =  Qwen3_5MoeTextConfig()

    >>> # Initializing a model from the Qwen3.5-35B-A3B style configuration
    >>> model = Qwen3_5MoeTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "qwen3_5_moe_text"
    base_config_key = "text_config"

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_experts_per_tok: int = 8
    num_experts: int = 256
    intermediate_size = AttributeError()
    decoder_sparse_step = AttributeError()
    norm_topk_prob = AttributeError()
    mlp_only_layers = AttributeError()

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        del self.mlp_only_layers


@auto_docstring(checkpoint="Qwen/Qwen3.5-35B-A3B")
@strict
class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):
    pass


@auto_docstring(checkpoint="Qwen/Qwen3.5-35B-A3B")
@strict
class Qwen3_5MoeConfig(Qwen3VLConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeConfig

    >>> # Initializing a Qwen3.5-MoE style configuration
    >>> configuration = Qwen3_5MoeConfig()

    >>> # Initializing a model from the Qwen3.5-35B-A3B style configuration
    >>> model = Qwen3_5MoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054


class Qwen3_5MoeVisionRotaryEmbedding(Qwen3_5VisionRotaryEmbedding):
    pass


class Qwen3_5MoeTextRotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    pass


class Qwen3_5MoeDynamicCache(Qwen3NextDynamicCache):
    pass


class Qwen3_5MoeGatedDeltaNet(Qwen3_5GatedDeltaNet):
    pass


class Qwen3_5MoeAttention(Qwen3NextAttention):
    pass


class Qwen3_5MoeMLP(Qwen3_5MLP):
    pass


class Qwen3_5MoeExperts(Qwen3NextExperts):
    pass


class Qwen3_5MoeTopKRouter(Qwen3VLMoeTextTopKRouter):
    pass


class Qwen3_5MoeSparseMoeBlock(Qwen3NextSparseMoeBlock):
    pass


class Qwen3_5MoeRMSNorm(Qwen3NextRMSNorm):
    pass


class Qwen3_5MoeDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(config, layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3_5MoePreTrainedModel(Qwen3NextPreTrainedModel):
    _no_split_modules = ["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Qwen3_5MoeGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, Qwen3_5MoeRMSNorm):
            init.zeros_(module.weight)
        elif isinstance(module, Qwen3_5MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3_5MoeSparseMoeBlock):
            init.normal_(module.gate.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3_5MoeVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Qwen3_5MoeVisionModel(Qwen3_5VisionModel):
    pass


class Qwen3_5MoeModelOutputWithPast(Qwen3VLMoeModelOutputWithPast):
    router_logits: tuple[torch.FloatTensor] | None = None


class Qwen3_5MoeCausalLMOutputWithPast(Qwen3VLMoeCausalLMOutputWithPast):
    pass


class Qwen3_5MoeTextModel(Qwen3_5TextModel):
    pass


class Qwen3_5MoeModel(Qwen3_5Model):
    pass


class Qwen3_5MoeForCausalLM(Qwen3NextForCausalLM):
    config: Qwen3_5MoeTextConfig
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5MoeTextModel(config)


class Qwen3_5MoeForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

        >>> model = Qwen3_5MoeForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-35B-A3B-Instruct", dtype="auto", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-35B-A3B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image in short."},
                ],
            }
        ]

        >>> # Preparation for inference
        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        >>> inputs = inputs.to(model.device)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=128)
        >>> generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        >>> processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "A woman in a plaid shirt sits on a sandy beach at sunset, smiling as she gives a high-five to a yellow Labrador Retriever wearing a harness. The ocean waves roll in the background."
        ```"""
        super().forward(**super_kwargs)

    def get_video_features(
        self,
        **super_kwargs,
    ) -> tuple | BaseModelOutputWithPooling:
        return super().get_video_features(**super_kwargs)

    def get_image_features(
        self,
        **super_kwargs,
    ) -> tuple | BaseModelOutputWithPooling:
        return super().get_image_features(**super_kwargs)


__all__ = [
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoeVisionModel",
    "Qwen3_5MoeTextModel",
    "Qwen3_5MoeModel",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoePreTrainedModel",
]
