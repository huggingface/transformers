from transformers.models.llama.modeling_llama import *
import torch.nn as nn
from .configuration_persimmon import PersimmonConfig
from transformers.utils import ModelConverter

PersimmonConverter = ModelConverter(__file__)

PersimmonConverter.register("PersimmonRotaryEmbedding", LlamaRotaryEmbedding)
PersimmonConverter.register("PersimmonMLP", LlamaMLP) 

class PersimmonAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PersimmonConfig, layer_idx: Optional[int] = None):
        super().__init__()
        ... # copy before? add the line? how to best support this
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        self.qk_layernorm = config.qk_layernorm

        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self._init_rope()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # [batch_size, seq_length, 3 x hidden_size]
        fused_qkv = self.query_key_value(hidden_states)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_states, key_states, value_states) = self._split_heads(fused_qkv)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        os, sin = self.rotary_emb(value_states, seq_len=None)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = self.rotary_emb(query_rot, key_rot, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)
        ... # TODO copy the rest of the function? if we do this it's unusable



PersimmonSdpaAttention = PersimmonConverter.register("PersimmonSdpaAttention", LlamaAttention) 
PersimmonFlashAttention2 = PersimmonConverter.register("PersimmonFlashAttention2", LlamaAttention) 

COHERE_ATTENTION_CLASSES = {"eager": PersimmonAttention, "flash_attention_2": PersimmonFlashAttention2, "sdpa": PersimmonSdpaAttention}

PersimmonConverter.register("PersimmonDecoderLayer", LlamaDecoderLayer) 
PersimmonConverter.register("PersimmonPreTrainedModel", LlamaPreTrainedModel)

PersimmonConverter.register("PersimmonModel", LlamaModel)
PersimmonConverter.register("PersimmonForCausalLM", LlamaForCausalLM)