# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HindiCausalLM model."""

import math
from typing import List, Optional, Tuple, Union

from ...utils import is_torch_available, logging
from .configuration_hindi_causal_lm import HindiCausalLMConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"

HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "convaiinnovations/hindi-foundational-model-base",
    # See all HindiCausalLM models at https://huggingface.co/models?filter=hindi_causal_lm
]

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import CrossEntropyLoss

    from ...activations import ACT2FN
    from ...modeling_outputs import (
        BaseModelOutputWithPast,
        CausalLMOutputWithPast,
    )
    from ...modeling_utils import PreTrainedModel
    from ...utils import (
        add_code_sample_docstrings,
        add_start_docstrings,
        add_start_docstrings_to_model_forward,
    )

    def create_sinusoidal_positions(num_pos, dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        sinusoid_inp = torch.einsum("i,j->ij", torch.arange(num_pos, dtype=torch.float), inv_freq).to(torch.get_default_dtype())
        return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization"""
        def __init__(self, hidden_size, eps=1e-12):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps
            
        def forward(self, hidden_states):
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            
            # Scale by weight
            return self.weight * hidden_states


    class SwiGLU(nn.Module):
        """SwiGLU activation function"""
        def forward(self, x):
            x1, x2 = x.chunk(2, dim=-1)
            return F.silu(x1) * x2


    class CausalSelfAttention(nn.Module):
        """Causal self-attention layer"""
        def __init__(self, config):
            super().__init__()
            assert config.hidden_size % config.num_attention_heads == 0
            
            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = config.hidden_size // config.num_attention_heads
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            
            # Query, Key, Value projections
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
            
            # Output projection
            self.output = nn.Sequential(
                nn.Linear(self.all_head_size, config.hidden_size),
                nn.Dropout(config.attention_probs_dropout_prob)
            )
            
            # Causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "causal_mask",
                torch.triu(
                    torch.ones(config.max_position_embeddings, config.max_position_embeddings) * -1e10, 
                    diagonal=1
                )
            )
            self.register_buffer("masked_bias", torch.tensor(-1e9))
            
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
            
        def transpose_for_scores(self, x):
            # Reshape from [batch_size, seq_length, hidden_size] to [batch_size, seq_length, num_heads, head_size]
            new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_shape)
            # Transpose to [batch_size, num_heads, seq_length, head_size]
            return x.permute(0, 2, 1, 3)
        
        def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            batch_size, seq_length = hidden_states.size()[:2]
            
            # Project inputs to queries, keys, and values
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            
            # Take the dot product between "query" and "key" to get the raw attention scores
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            
            # Scale attention scores
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            
            # Apply causal mask - prevents attending to future tokens
            causal_mask = self.causal_mask[:seq_length, :seq_length]
            attention_scores = attention_scores + causal_mask
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Apply the attention mask
                attention_scores = attention_scores + attention_mask
            
            # Normalize the attention scores to probabilities
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)
            
            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            
            # Apply attention to values
            context_layer = torch.matmul(attention_probs, value_layer)
            
            # Reshape back to [batch_size, seq_length, hidden_size]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_shape)
            
            # Final output projection
            output = self.output(context_layer)
            
            outputs = (output, attention_probs) if output_attentions else (output,)
            
            return outputs


    class TransformerBlock(nn.Module):
        """Transformer block with causal attention for language modeling"""
        def __init__(self, config):
            super().__init__()
            self.attention = CausalSelfAttention(config)
            
            # Choose normalization type based on config
            if getattr(config, "normalization_layer", "layernorm").lower() == "rmsnorm":
                self.attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
                self.ffn_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            else:
                self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                self.ffn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
            # Feed-forward network with different activation functions based on config
            hidden_act = getattr(config, "hidden_act", "gelu")
            if hidden_act == "silu":
                act_fn = F.silu
            elif hidden_act == "swiglu":
                # SwiGLU activation function is gated SiLU
                self.ffn = nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size * 2),
                    SwiGLU(),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.Dropout(config.hidden_dropout_prob)
                )
                return
            else:
                act_fn = ACT2FN[hidden_act]
            
            # Standard feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                act_fn,
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            )
        
        def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            # Self-attention block with pre-layer norm
            residual = hidden_states
            hidden_states = self.attention_layernorm(hidden_states)
            
            # Self-attention
            attn_outputs = self.attention(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            outputs = attn_outputs[1:]
            
            # Residual connection
            hidden_states = attn_output + residual
            
            # Feed-forward block with pre-layer norm
            residual = hidden_states
            hidden_states = self.ffn_layernorm(hidden_states)
            
            # Feed-forward
            feed_forward_hidden_states = self.ffn(hidden_states)
            
            # Residual connection
            hidden_states = residual + feed_forward_hidden_states
            
            if use_cache:
                outputs = (hidden_states,) + outputs
            else:
                outputs = (hidden_states,) + outputs[1:]
            
            return outputs  # hidden_states, present, (attentions)


    class HindiCausalLMPreTrainedModel(PreTrainedModel):
        """
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
        """
        config_class = HindiCausalLMConfig
        base_model_prefix = "transformer"
        supports_gradient_checkpointing = True
        _no_split_modules = ["TransformerBlock"]
        
        def __init__(self, *inputs, **kwargs):
            super().__init__(*inputs, **kwargs)
        
        def _init_weights(self, module):
            """Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)


    class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
        """
        Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`TransformerBlock`].
        
        Args:
            config: HindiCausalLMConfig
        """
        def __init__(self, config):
            super().__init__(config)
            
            self.embed_dim = config.hidden_size
            
            # Embeddings
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
            self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.drop = nn.Dropout(config.hidden_dropout_prob)
            
            # Transformer layers
            self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
            
            # Final normalization layer
            if getattr(config, "normalization_layer", "layernorm").lower() == "rmsnorm":
                self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            else:
                self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            
            # Initialize weights and apply final processing
            self.post_init()
        
        def get_input_embeddings(self):
            return self.wte
        
        def set_input_embeddings(self, new_embeddings):
            self.wte = new_embeddings
        
        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            """
            for layer, heads in heads_to_prune.items():
                self.h[layer].attention.prune_heads(heads)
        
        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                batch_size = input_ids.shape[0]
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size = inputs_embeds.shape[0]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])
            
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            else:
                position_ids = position_ids.view(-1, input_shape[-1])
            
            if past_key_values is None:
                past_length = 0
                past_key_values = tuple([None] * len(self.h))
            else:
                past_length = past_key_values[0][0].size(-2)
            
            if position_ids is not None:
                position_ids = position_ids[:, past_length:]
            
            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape batch_size x num_heads x N x N
            # head_mask has shape n_layer x batch x n_head x N x N
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
            
            # Embedding
            if inputs_embeds is None:
                inputs_embeds = self.wte(input_ids)
            
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            
            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids)
                hidden_states = hidden_states + token_type_embeds
            
            hidden_states = self.drop(hidden_states)
            
            # Prepare attention mask
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                
                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and -10000.0 for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * -10000.0
            
            # By default, past_key_values contains only one element which is None
            # if not use_cache or tuple of tensors
            encoder_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None
            
            for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
                if output_hidden_states:
                    encoder_hidden_states = encoder_hidden_states + (hidden_states,)
                
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
                hidden_states = outputs[0]
                
                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[1],)
            
            # Add last hidden state
            hidden_states = self.ln_f(hidden_states)
            
            if output_hidden_states:
                encoder_hidden_states = encoder_hidden_states + (hidden_states,)
            
            if not return_dict:
                return tuple(
                    v for v in [hidden_states, None, encoder_hidden_states, all_self_attentions] if v is not None
                )
            
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=None,
                hidden_states=encoder_hidden_states,
                attentions=all_self_attentions,
            )


    class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel):
        """
        Hindi Causal LM for causal language modeling.
        """
        _keys_to_ignore_on_load_missing = [r"h\.\d+\.attention\.masked_bias", r"lm_head.weight"]
        
        def __init__(self, config):
            super().__init__(config)
            self.transformer = HindiCausalLMModel(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
            # Initialize weights and apply final processing
            self.post_init()
        
        def get_output_embeddings(self):
            return self.lm_head
        
        def set_output_embeddings(self, new_embeddings):
            self.lm_head = new_embeddings
        
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
            token_type_ids = kwargs.get("token_type_ids", None)
            # only last token for inputs_ids if past is defined in kwargs
            if past_key_values:
                input_ids = input_ids[:, -1].unsqueeze(-1)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)
            
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -1].unsqueeze(-1)
            else:
                position_ids = None
            
            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        
        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
            transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            hidden_states = transformer_outputs[0]
            lm_logits = self.lm_head(hidden_states)
            
            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            
            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
