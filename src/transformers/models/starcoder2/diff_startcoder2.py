from typing import List, Tuple
from torch import FloatTensor, LongTensor, Tensor
from torch._C import FloatTensor, LongTensor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import *
import torch.nn as nn
from transformers import Starcoder2Config
from transformers.utils import ModelConverter

Starcoder2Converter = ModelConverter(__file__)

Starcoder2RMSNorm = Starcoder2Converter.register("Starcoder2RMSNorm", LlamaRMSNorm)
StarcoderRotaryEmbedding = Starcoder2Converter.register("StarcoderRotaryEmbedding", LlamaRotaryEmbedding)

class Starcoder2MLP(nn.Module):
    def __init__(self, config: Starcoder2Config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, config.intermediate_size, bias=config.use_bias)
        self.c_proj = nn.Linear(config.intermediate_size, embed_dim, bias=config.use_bias)
        self.act = ACT2FN[config.hidden_act]
        self.residual_dropout = config.residual_dropout

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.residual_dropout, training=self.training)
        return hidden_states

# TODO either we support this, or we don't allow call to super?
# if part of the super is used, then we are fucked. Let's restrict this to init?

# TODO if a class is not registered, the original should be copied with replaces?
# Copied form where? No.
# But then how do we check the architecture etc.

# TODO do we support multiple inheritance? 
# This will depend on whether we usually copy from more than one module
# Mixtral for example? 

class Starcoder2Attention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx) # here call to super means
                                            # we should copy super
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self.rotary_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        attn_output = nn.functional.dropout(attn_output, p=self.residual_dropout, training=self.training)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

Starcoder2SdpaAttention = Starcoder2Converter.register("Starcoder2SdpaAttention", LlamaAttention) 
Starcoder2FlashAttention2 = Starcoder2Converter.register("Starcoder2FlashAttention2", LlamaAttention) 

STARCODER2_ATTENTION_CLASSES = {"eager": Starcoder2Attention, "flash_attention_2": Starcoder2FlashAttention2, "sdpa": Starcoder2SdpaAttention}


Starcoder2DecoderLayer = Starcoder2Converter.register("Starcoder2DecoderLayer", LlamaDecoderLayer) 
Starcoder2PreTrainedModel = Starcoder2Converter.register("Starcoder2PreTrainedModel", LlamaPreTrainedModel)

class Starcoder2Model(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.embedding_dropout = config.embedding_dropout

    def forward(self, input_ids: LongTensor = None, attention_mask: Tensor | None = None, position_ids: LongTensor | None = None, past_key_values: List[FloatTensor] | None = None, inputs_embeds: FloatTensor | None = None, use_cache: bool | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None, cache_position: LongTensor | None = None) -> Tuple | BaseModelOutputWithPast:
        if inputs_embeds is None: 
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.embedding_dropout, training=self.training)
        return super().forward(None, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict, cache_position)

Starcoder2ForCausalLM = Starcoder2Converter.register("Starcoder2ForCausalLM", LlamaForCausalLM)
Starcoder2ForSequenceClassification = Starcoder2Converter.register("Starcoder2ForSequenceClassification", LlamaForSequenceClassification)