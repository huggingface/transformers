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
"""PyTorch Qwen3-VL-MOE model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, can_return_tuple, logging
from ..qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeExperts,
    Qwen3MoePreTrainedModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeSparseMoeBlock,
    load_balancing_loss_func,
)
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLVisionConfig
from ..qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
    Qwen3VLVisionRotaryEmbedding,
)


logger = logging.get_logger(__name__)


class Qwen3VLMoeTextConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3VLMoeTextModel`]. It is used to instantiate a
    Qwen3-VL-MOE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-VL-30B-A3B-Instruct [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2MoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2MoeModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 128000):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 1408):
            Intermediate size of the routed expert.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 60):
            Number of routed experts.
        mlp_only_layers (`List[int]`, *optional*, defaults to `[]`):
            Indicate which layers use Qwen3VLMoeMLP rather than Qwen3VLMoeSparseMoeBlock
            The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
            If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        head_dim (`int`, *optional*):
            The dimension of the head. If not specified, will default to `hidden_size // num_attention_heads`.

    ```python
    >>> from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLMoeConfig

    >>> # Initializing a Qwen3VLMoe style configuration
    >>> configuration = Qwen3VLMoeConfig()

    >>> # Initializing a model from the Qwen3-VL-30B-A3B style configuration
    >>> model = Qwen3VLMoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_vl_moe_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    # Default tensor parallel plan for base model `Qwen3VLMoe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 151936,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 5632,
        num_hidden_layers: int | None = 24,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 16,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 128000,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        decoder_sparse_step: int | None = 1,
        moe_intermediate_size: int | None = 1408,
        num_experts_per_tok: int | None = 4,
        num_experts: int | None = 60,
        mlp_only_layers: list[int] | None = None,
        rope_parameters: RopeParameters | None = None,
        head_dim: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.rope_parameters = rope_parameters

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        super().__init__(
            ignore_keys_at_rope_validation={"mrope_section", "mrope_interleaved"},
            **kwargs,
        )


class Qwen3VLMoeVisionConfig(Qwen3VLVisionConfig):
    pass


class Qwen3VLMoeConfig(Qwen3VLConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3VLMoeModel`]. It is used to instantiate a
    Qwen3-VL-MOE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-VL-30B-A3B-Instruct [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen3VLMoeTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen3VLMoeVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The start token index to encode the image prompt.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The end token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    ```python
    >>> from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLMoeConfig

    >>> # Initializing a Qwen3-VL-MOE style configuration
    >>> configuration = Qwen3VLMoeConfig()

    >>> # Initializing a model from the Qwen3-VL-30B-A3B style configuration
    >>> model = Qwen3VLMoeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_vl_moe"
    sub_configs = {"vision_config": Qwen3VLMoeVisionConfig, "text_config": Qwen3VLMoeTextConfig}


class Qwen3VLMoeTextRMSNorm(Qwen3MoeRMSNorm):
    pass


class Qwen3VLMoeTextExperts(Qwen3MoeExperts):
    pass


class Qwen3VLMoeTextTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen3VLMoeTextSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    pass


class Qwen3VLMoeTextAttention(Qwen3VLTextAttention):
    pass


class Qwen3VLMoeTextDecoderLayer(Qwen3MoeDecoderLayer):
    pass


class Qwen3VLMoePreTrainedModel(Qwen3MoePreTrainedModel):
    config: Qwen3VLMoeConfig
    _no_split_modules = ["Qwen3VLMoeTextDecoderLayer", "Qwen3VLMoeVisionBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights."""
        PreTrainedModel._init_weights(self, module)
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)
        if isinstance(module, Qwen3VLMoeTextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Qwen3VLMoeTextTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, Qwen3VLMoeVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Qwen3VLMoeVisionRotaryEmbedding(Qwen3VLVisionRotaryEmbedding):
    pass


class Qwen3VLMoeVisionModel(Qwen3VLVisionModel):
    pass


class Qwen3VLMoeTextModel(Qwen3VLTextModel):
    pass


class Qwen3VLMoeCausalLMOutputWithPast(Qwen3VLCausalLMOutputWithPast):
    aux_loss: torch.FloatTensor | None = None


class Qwen3VLMoeModel(Qwen3VLModel):
    pass


class Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
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
        >>> from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

        >>> model = Qwen3VLMoeForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

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

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        return Qwen3VLMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


__all__ = [
    "Qwen3VLMoeConfig",
    "Qwen3VLMoeTextConfig",
    "Qwen3VLMoeVisionModel",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3VLMoeModel",
    "Qwen3VLMoePreTrainedModel",
    "Qwen3VLMoeTextModel",
]
