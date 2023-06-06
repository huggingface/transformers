""" Pytorch Bark model"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, Tuple, Union

import math

from ...modeling_utils import PreTrainedModel
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MaskedLMOutput
)

from .configuration_bark import BarkConfig
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging


logger = logging.get_logger(__name__)


# DONE: ask about _init_weights from https://huggingface.co/docs/transformers/add_new_model#2-next-prepare-your-environment

# DONE: import modules
# DONE: should I change the naming of the Bark modules? YES
# DONE: should I comment my thoughts? YES
# DONE: j'enlève toutes mentions du flash attention - YES
# DONE: do I change parameters name ? YES
# DONE: merge causal and total attention layers? YES
# DONE: GPTNEO seems not to divide by sqrt(dim)

# TODO: at the end of the day, we need to
# 1. keep the handmade LayerNorm because the config used by Bark author uses no biases
# 2. update the causal modules with this layer norm
# 3. left the non-causal module with the classic layer norm


class BarkSelfAttention(nn.Module):
    # adapted from GPTNeoSelfAttention and Bark code
    # BarkSelfAttention can have two attention type, i.e full attention or causal attention
            
    # dic
    # self.n_head -> self.num_heads
    # config.n_head -> config.num_heads
    # self.n_embd -> self.embed_dim
    # config.n_embd -> config.hidden_size
    # c_attn -> att_proj
    # c_proj -> out_proj
    # past_kv <- layer_past
    # block_size <- max_position_embeddings
    # n_layer -> num_layers
    
    def __init__(self, config, is_causal=False):
        super().__init__()
        
        

        # regularization
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)   
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
          
        assert config.hidden_size % config.num_heads == 0, f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"f" {self.num_heads})."

        # key, query, value projections for all heads, but in a batch
        self.att_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        # output projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)        

        self.is_causal = is_causal
        if is_causal:
            block_size = config.block_size
            bias = torch.tril(torch.ones((block_size, block_size), dtype=bool)).view(
                1, 1, block_size, block_size
            )
            self.register_buffer("bias", bias)



    
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        # (batch, seq_len, num_heads*attn_head_size) -> (batch, num_heads, seq_len, attn_head_size)
        tensor = tensor.view(tensor.size()[:-1] + (num_heads, attn_head_size))
        tensor = tensor.transpose(1,2)

        return tensor # (batch, num_heads, seq_len, attn_head_size)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """

        # re-assemble all head outputs side by side
        # (batch, num_heads, seq_len, attn_head_size) -> (batch, seq_len, num_heads*attn_head_size)
        tensor = tensor.transpose(1, 2).contiguous()
        tensor = tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))        

        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):


        # unlike GPTNeo's SelfAttention, divide by the square root of the dimension of the query and the key
        attn_weights = torch.matmul(query, key.transpose(-1, -2))* (1.0 / math.sqrt(key.size(-1)))


        if self.is_causal:
            query_length, key_length = query.size(-2), key.size(-2)
            

            # fill the upper left part of the attention weights with inf
            attn_weights = attn_weights.masked_fill(self.bias[:,:,key_length - query_length : key_length, :key_length] == 0, torch.finfo(attn_weights.dtype).min)
            
        
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        # (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, attn_head_size) 
        # -> (batch, num_heads, seq_len, attn_head_size)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states,
        attention_mask=None, 
        past_kv=None,
        head_mask=None, 
        use_cache=False,
        output_attentions=False, 
    ):

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query, key, value  = self.att_proj(hidden_states).split(self.embed_dim, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if past_kv is not None:
            past_key = past_kv[0]
            past_value = past_kv[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)



# Same as model.py
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

# TODO: move
# c_fc -> in_proj
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.in_proj    = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.bias)
        self.out_proj  = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.in_proj(x)
        x = self.gelu(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x
    
    

class BarkBlock(nn.Module):

    def __init__(self, config, layer_idx, is_causal = False):
        super().__init__()
        
        if is_causal:
            # if causal, uses handmade LayerNorm, so that the layerNorm bias is optional
            # this handmade layerNorm is used to stick with Bark choice of leaving optional bias in AutoRegressive models
            # (corresponding to the "Text" and the "Coarse" modules)
            self.ln_1 = LayerNorm(config.hidden_size, bias=config.bias)
            self.ln_2 = LayerNorm(config.hidden_size, bias=config.bias)
        else:
            self.ln_1 = nn.LayerNorm(config.hidden_size)
            self.ln_2 = nn.LayerNorm(config.hidden_size)
        
        
        self.attn = BarkSelfAttention(config, is_causal=is_causal)

        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        past_kv=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
   ):
        
        attn_outputs = self.attn(
            self.ln_1(hidden_states), 
            past_kv=past_kv, 
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions)
         
        attn_output = attn_outputs[0] #output_attn: output, present_kv, (attn_weights)
        outputs = attn_outputs
        
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, ((present), attentions)
    
    
# Done: Block and FineBlock might be fused
# TODO: deal with PreTrained models. Note that there should be two main classes models
# CausalGPT, NonCausalGPT -> BarkSemanticModel , BarkCoarseAcousticsModel, BarkFineAcousticsModel
# CausalModel?

class BarkModulePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BarkConfig # TODO: do this
    base_model_prefix = "transformer" # TODO: verify this is the right base_model
    # supports_gradient_checkpointing = True
    _no_split_modules = ["BarkBlock"] # TODO: what to do with this?
    
    # TODO: ask about module.padding_idx
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BarkModule):
            module.gradient_checkpointing = value    
    


# GPT -> BarkModule


class BarkModule(BarkModulePreTrainedModel):
    # TODO: what do I do with that here?    
    #@add_start_docstrings_to_model_forward(BARK_INPUTS_DOCSTRING)
    #@add_code_sample_docstrings(
    #    checkpoint=_CHECKPOINT_FOR_DOC,
    #    output_type=BaseModelOutputWithPast,
    #    config_class=_CONFIG_FOR_DOC,
    #)
    
    def __init__(self, config):
        super().__init__(config)
        # TODO: verify if the assert here are needed
        assert config.input_vocab_size is not None
        assert config.output_vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self._initialize_modules(config)

        
        self.gradient_checkpointing = False # TODO: verify if that is good here
        # Initialize weights and apply final processing
        self.post_init()
      
    
    def _initialize_modules(self, config):
        # initialize as an autoregressive GPT-like model
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.input_vocab_size, config.hidden_size),
            wpe = nn.Embedding(config.block_size, config.hidden_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([BarkBlock(config, idx, is_causal=True) for idx in range(config.num_layers)]),
            ln_f = LayerNorm(config.hidden_size, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.output_vocab_size, bias=False)
        
      
    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings
        

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    
    def _get_and_check_input_embeddings(self, input_ids, input_embeds, past_key_values):
        # Verify if input_embeds already exists, and check sequence_lengths are plausible
        # then compute embeddings.
        # In a separate function because the Semantic model computes it differently. 

        if input_ids is not None and input_embeds is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        elif input_ids is not None:
            _, t = input_ids.size() # (batch_size, seq_len)
            if past_key_values is not None:
                # in that case, embeddings for past tokens have already been computed, so only need to compute the most
                # recent token embedding
                assert t == 1
            else:
                 assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
                 
                    
        elif input_embeds is None:
            raise ValueError("You have to specify either input_ids or input_embeds")
    
        input_embeds = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        
        return input_embeds
    
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            attention_mask: Optional[torch.Tensor] = None, 
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None, 
            labels: Optional[torch.LongTensor] = None,
            input_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        

        input_embeds = self._get_and_check_input_embeddings(input_ids, input_embeds, past_key_values)
        
        input_shape = input_embeds.size()[:-1]
        batch_size = input_embeds.shape[0]
        seq_length = input_shape[-1]
        
        device = input_ids.device if input_ids is not None else input_embeds.device
        
        

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.transformer.h))
        else:
            past_length = past_key_values[0][0].size(-2)
            

        # TODO: verify if it needs to be asserted (maybe use GPTNeo's way)
        if position_ids is None:
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0) # shape (1, seq_length)
            assert position_ids.shape == (1, seq_length)

        position_embeds = self.transformer.wpe(position_ids) # position embeddings of shape (1, t, n_embd)
                        
        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
            
            
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_heads x N x N
        # head_mask has shape num_layers x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        

        hidden_states = self.transformer.drop(input_embeds + position_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
                
        
        present_key_values = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, (block, past_layer_kv) in enumerate(zip(self.transformer.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)
                
                    return custom_forward
                
                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    past_kv=past_layer_kv,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
            hidden_states = outputs[0]

            if use_cache:
                present_key_values = present_key_values + (outputs[1],)
                
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.transformer.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(hidden_states[:, [-1], :]) # note: using list [-1] to preserve the time dim

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            return tuple(v for v in [None, logits, present_key_values, all_hidden_states, all_self_attentions] if v is not None)

        return CausalLMOutputWithPast(
            loss = loss,
            logits = logits,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            )






######################################################################################

class BarkSemanticModel(BarkModule):
    def __init__(self, config):
        # Same architecture than an autoregressive gpt-like model except for an hacky context merging at the very beginning of the generation
        super().__init__(config)

    def _get_and_check_input_embeddings(self, input_ids, input_embeds, past_key_values):    
        # Hack From Bark original repository to sum text and history prompt embeddings
        # It sums the text embeddings and the history prompt embeddings once at the very beginning of the generation
        # (merge_context) 
        # TODO: verify if that can be clearer
        
        if input_ids is not None and input_embeds is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        elif input_ids is not None:
            _, t = input_ids.size() # (batch_size, seq_len)

            if past_key_values is not None:
                # in that case, embeddings for past tokens have already been computed, so only need to compute the most
                # recent token embedding
                assert t == 1
                input_embeds = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
            else:
                assert(input_ids.shape[1] >= 256+256+1)
                t = input_ids.shape[1] - 256

                input_embeds = torch.cat([
                    self.transformer.wte(input_ids[:,:256]) + self.transformer.wte(input_ids[:,256:256+256]),
                    self.transformer.wte(input_ids[:,256+256:])
                ], dim=1)

        elif input_embeds is None:
            raise ValueError("You have to specify either input_ids or input_embeds")
    
    
        return input_embeds

#BarkSemanticModel , BarkCoarseAcousticsModel
    

class BarkFineAcousticsModel(BarkModule):
    def __init__(self, config):
        # non-causal gpt-like model with one embedding layer and one lm_head for each codebook of Encodec
        super().__init__(config)
        self.n_codes_total = config.n_codes_total

   
    def get_input_embeddings(self):
        # one embedding layers for each codebook
        return self.transformer.wtes

    def set_input_embeddings(self, new_embeddings):
        # one embedding layers for each codebook
        self.transformer.wtes = new_embeddings         
            
    def _initialize_modules(self, config):
        # initialize a modified non causal GPT-like model
        # note that for there is one embedding layer and one lm_head for each codebook of Encodec
        self.transformer = nn.ModuleDict(
            dict(
                wtes=nn.ModuleList(
                    [
                        nn.Embedding(config.input_vocab_size, config.hidden_size)
                        for _ in range(config.n_codes_total)
                    ]
                ),
                wpe=nn.Embedding(config.block_size, config.hidden_size),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([BarkBlock(config, layer_idx, is_causal=False) for layer_idx in range(config.num_layers)]),
                ln_f=nn.LayerNorm(config.hidden_size),
            )
        )
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.output_vocab_size, bias=False)
                for _ in range(config.n_codes_given, config.n_codes_total)
            ]
        )
        for i in range(config.n_codes_total - config.n_codes_given):
            self.transformer.wtes[i + 1].weight = self.lm_heads[i].weight
            # TODO: check what it does
            
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            for wte in self.transformer.wtes:
                n_params -= wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params     
    
    
    def _get_and_check_input_embeddings(self, input_ids, input_embeds, pred_idx):
        # the input_embeddings are the sum of the j previous codebooks embeddings before the current pred_idx codebook

        if input_ids is not None and input_embeds is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_len, codes = input_ids.size()
            assert (
                batch_size <= self.config.block_size
            ), f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"  
            
            assert codes == self.n_codes_total, (batch_size, seq_len, codes)
            
            # forward the GPT model itself
            input_embeds = [
                wte(input_ids[:, :, i]).unsqueeze(-1) for i, wte in enumerate(self.transformer.wtes)
            ]  # token embeddings of shape (b, t, n_embd)
            input_embeds = torch.cat(input_embeds, dim=-1)
            input_embeds = input_embeds[:, :, :, : pred_idx + 1].sum(dim=-1)            
                        
        elif input_embeds is not None:
            input_embeds = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        else:
            raise ValueError("You have to specify either input_ids or input_embeds")
    
        
        
        return input_embeds

            
    # contrary to its base class (BarkModule), it is non-causal, so no need for past key values
    # And there is an additionnal idx corresponding to the id of the codebook that will be predicted      
    def forward(
            self,
            pred_idx: int,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None, 
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None, 
            labels: Optional[torch.LongTensor] = None,
            input_embeds: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    
        assert pred_idx > 0, "cannot predict 0th codebook"
        input_embeds = self._get_and_check_input_embeddings(input_ids, input_embeds, pred_idx)
        
        input_shape = input_embeds.size()[:-1]
        batch_size = input_embeds.shape[0]
        seq_length = input_shape[1]
        
        device = input_ids.device if input_ids is not None else input_embeds.device
        
        
        # TODO: verify if it needs to be asserted (maybe use GPTNeo's way)
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0) # shape (1, seq_length)
            assert position_ids.shape == (1, seq_length)

        position_embeds = self.transformer.wpe(position_ids) # position embeddings of shape (1, t, n_embd)
            
            
        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
            
            
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        
        

        hidden_states = self.transformer.drop(input_embeds + position_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)
                
        
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, block in enumerate(self.transformer.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
                
            hidden_states = outputs[0]
                
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

        hidden_states = self.transformer.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_heads[pred_idx - self.config.n_codes_given](hidden_states)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            return tuple(v for v in [None, logits, all_hidden_states, all_self_attentions] if v is not None)

        return MaskedLMOutput(
            loss = loss,
            logits = logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            )
            
