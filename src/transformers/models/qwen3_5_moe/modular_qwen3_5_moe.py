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

from ... import initialization as init
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...utils import logging
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


class Qwen3_5MoeTextConfig(Qwen3NextConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3_5MoeTextModel`]. It is used to instantiate a
    Qwen3.5-MoE model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of
    Qwen3.5-35B-A3B-Instruct [Qwen/Qwen3.5-35B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 248320):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 256):
            Projection weights dimension in multi-head attention.
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
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the routed expert.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the shared expert.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        layer_types (`list[str]`, *optional*):
            Types of each layer (attention or linear).
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.

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
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=256,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        layer_types=None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        kwargs["ignore_keys_at_rope_validation"] = {"mrope_section", "mrope_interleaved"}
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        del self.intermediate_size
        del self.decoder_sparse_step
        del self.norm_topk_prob
        del self.mlp_only_layers


class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):
    pass


class Qwen3_5MoeConfig(Qwen3VLConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3_5MoeModel`]. It is used to instantiate a
    Qwen3.5-MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3.5-35B-A3B-Instruct [Qwen/Qwen3.5-35B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3_5TextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen3_5VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 248056):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 248057):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 248053):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 248054):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.

    ```python
    >>> from transformers import Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeConfig

    >>> # Initializing a Qwen3.5-MoE style configuration
    >>> configuration = Qwen3_5MoeConfig()

    >>> # Initializing a model from the Qwen3.5-35B-A3B style configuration
    >>> model = Qwen3_5MoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_5_moe"
    sub_configs = {"vision_config": Qwen3_5MoeVisionConfig, "text_config": Qwen3_5MoeTextConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


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
