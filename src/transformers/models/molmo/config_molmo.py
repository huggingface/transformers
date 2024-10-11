# coding=utf-8
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

from typing import List, Tuple

from transformers import PretrainedConfig, AutoTokenizer


class MolmoVisionConfig(PretrainedConfig):
    def __init__(
        self,
        image_default_input_size: Tuple[int, int] = (336, 336),
        image_patch_size: int = 14,
        image_pos_patch_size: int = 14,
        image_emb_dim: int = 1024,
        image_num_heads: int = 16,
        image_num_key_value_heads: int = 16,
        image_num_layers: int = 23,
        image_head_dim: int = 64,
        image_mlp_dim: int = 4096,
        image_mlp_activations: str = "quick_gelu",
        residual_dropout: float = 0,
        image_num_pos: int = 577,
        image_norm_eps: float = 1e-5,
        float32_attention: bool = True,
        attention_type: str = "spda",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_default_input_size = image_default_input_size
        self.image_patch_size = image_patch_size
        self.image_pos_patch_size = image_pos_patch_size
        self.image_emb_dim = image_emb_dim
        self.image_num_heads = image_num_heads
        self.image_num_key_value_heads = image_num_key_value_heads
        self.image_num_layers = image_num_layers
        self.image_head_dim = image_head_dim
        self.image_mlp_dim = image_mlp_dim
        self.image_mlp_activations = image_mlp_activations
        self.residual_dropout = residual_dropout
        self.image_num_pos = image_num_pos
        self.image_norm_eps = image_norm_eps
        self.float32_attention = float32_attention

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


class MolmoConfig(PretrainedConfig):
    model_type = "molmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        embedding_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        float32_attention=True,
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        layer_norm_eps: float = 1e-5,
        rope_theta=10000.0,
        activation_type="silu",
        qkv_bias: bool = False,
        tie_word_embeddings: bool=True,
        bias_for_layer_norm: bool=False,
        qk_layer_norm: bool=False,
        norm_after: bool = False,
        layer_norm_type: str="rms",
        vision_config: MolmoVisionConfig=None,
        vit_layers=(-2, -9),
        residual_dropout: float=0.0,
        embedding_dropout: float=0.0,
        attention_dropout: float=0.0,
        image_feature_dropout: float=0.0,
        additional_vocab_size=128,
        attention_type: str = "sdpa",
        image_padding_embed="pad_and_partial_pad",
        moe_num_experts=None,
        moe_top_k=None,
        normalize_input_embeds: bool=False,
        scale_logits: bool=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = MolmoVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = MolmoVisionConfig()
        else:
            self.vision_config = vision_config

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.qk_layer_norm = qk_layer_norm
        self.num_key_value_heads = num_key_value_heads
        self.float32_attention= float32_attention
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.activation_type = activation_type
        self.qkv_bias = qkv_bias
        self.norm_after = norm_after
        self.tie_word_embeddings = tie_word_embeddings
        self.layer_norm_type = layer_norm_type
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.vit_layers = vit_layers
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.image_feature_dropout = image_feature_dropout
        self.image_padding_embed = image_padding_embed
        self.bias_for_layer_norm = bias_for_layer_norm
        self.additional_vocab_size = additional_vocab_size
        self.attention_type = attention_type
        self.normalize_input_embeds = normalize_input_embeds
        self.scale_logits = scale_logits

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def effective_num_key_value_heads(self) -> int:
        if self.num_key_value_heads is None:
            return self.num_attention_heads
        else:
            return self.num_key_value_heads

    @property
    def image_num_patch(self):
        assert self.vision_config is not None
        return self.vision_config.image_num_patch


MolmoVisionConfig.register_for_auto_class()
MolmoConfig.register_for_auto_class()