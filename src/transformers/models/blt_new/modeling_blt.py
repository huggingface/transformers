# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BLT model."""

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
from .configuration_blt import BLTConfig

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from ...integrations.flex_attention import make_flex_block_causal_mask

logger = logging.get_logger(__name__)

# Constants for special tokens
BOE_ID = 0  # Beginning of encoding
BOS_ID = 1  # Beginning of sequence
PAD_ID = 2  # Padding
EOS_ID = 3  # End of sequence

def get_blt_input(
    tokens: torch.Tensor,
    enforce_patch_size_multiple: bool,
    nb_boe: torch.Tensor,
    patch_size: int,
    boe_id: int,
):
    """Get input tensors for BLT model."""
    bs, N = tokens.shape
    
    # Add beginning of encoding tokens
    if nb_boe > 0:
        boe_tokens = tokens.new(bs, nb_boe).fill_(boe_id)
        local_encoder_tokens = torch.cat([boe_tokens, tokens], dim=1)
    else:
        local_encoder_tokens = tokens
        
    # Create decoder tokens
    local_decoder_tokens = tokens
    
    return local_encoder_tokens, None, local_decoder_tokens

def patch_ids_from_lengths(patch_lengths, seq_len):
    """Generate patch IDs from patch lengths."""
    bs, num_patches = patch_lengths.shape
    patch_ids = torch.zeros(bs, seq_len, dtype=torch.long, device=patch_lengths.device)
    
    for i in range(bs):
        start = 0
        for j in range(num_patches):
            length = patch_lengths[i, j]
            if length > 0:
                patch_ids[i, start:start + length] = j
                start += length
                
    return patch_ids

def decoder_patch_ids_from_lengths(patch_lengths, nb_boe, seq_len):
    """Generate decoder patch IDs from patch lengths."""
    bs, num_patches = patch_lengths.shape
    decoder_patch_ids = torch.zeros(bs, seq_len, dtype=torch.long, device=patch_lengths.device)
    
    for i in range(bs):
        start = 0
        for j in range(num_patches):
            length = patch_lengths[i, j]
            if length > 0:
                decoder_patch_ids[i, start:start + length] = j
                start += length
                
    return decoder_patch_ids

def cross_attn_mask(
    patch_ids,
    patch_lengths,
    N,
    patches_as_queries=False,
    cross_attn_k=1,
    window=None,
    block_mask=True,
):
    """Create cross attention mask."""
    bs, seq_len = patch_ids.shape
    num_patches = patch_lengths.shape[1]
    
    if patches_as_queries:
        mask = torch.zeros(bs, num_patches, seq_len, device=patch_ids.device)
        for i in range(bs):
            for j in range(num_patches):
                if patch_lengths[i, j] > 0:
                    mask[i, j, patch_ids[i] == j] = 1
    else:
        mask = torch.zeros(bs, seq_len, num_patches, device=patch_ids.device)
        for i in range(bs):
            for j in range(seq_len):
                patch_id = patch_ids[i, j]
                if patch_id < num_patches:
                    mask[i, j, patch_id] = 1
                    
    return mask

def downsample(
    h,
    num_patches,
    patch_lengths,
    patch_ids,
    downsampling_by_pooling=None,
    patch_size=None,
):
    """Downsample hidden states based on patch lengths."""
    bs, seq_len, dim = h.shape
    
    if downsampling_by_pooling == "mean":
        h_downsampled = torch.zeros(bs, num_patches, dim, device=h.device)
        for i in range(bs):
            for j in range(num_patches):
                if patch_lengths[i, j] > 0:
                    mask = patch_ids[i] == j
                    h_downsampled[i, j] = h[i, mask].mean(dim=0)
    else:
        # Default to taking first token of each patch
        h_downsampled = torch.zeros(bs, num_patches, dim, device=h.device)
        for i in range(bs):
            for j in range(num_patches):
                if patch_lengths[i, j] > 0:
                    mask = patch_ids[i] == j
                    h_downsampled[i, j] = h[i, mask][0]
                    
    return h_downsampled

def compute_hash_embeddings(
    local_encoder_tokens: torch.Tensor,
    local_encoder,
    encoder_hash_tok_embedding: nn.ModuleList,
    encoder_hash_byte_group_nb_functions: int,
    encoder_hash_byte_group_size: list,
    encoder_hash_byte_group_vocab: int,
) -> torch.Tensor:
    """Compute hash embeddings for tokens."""
    if encoder_hash_tok_embedding is None:
        return None
        
    bs, seq_len = local_encoder_tokens.shape
    embeddings = []
    
    for i in range(encoder_hash_byte_group_nb_functions):
        group_size = encoder_hash_byte_group_size[i]
        hash_values = byte_group_hash_function(
            local_encoder_tokens,
            group_size=group_size,
            hash_func_nb=i,
            max_hash=encoder_hash_byte_group_vocab,
        )
        embedding = encoder_hash_tok_embedding[i](hash_values)
        embeddings.append(embedding)
        
    return sum(embeddings)

def byte_group_hash_function(
    x: torch.Tensor,
    group_size: int = 2,
    hash_func_nb: int = 0,
    max_hash: int = 30000
):
    """Compute hash values for byte groups."""
    with torch.no_grad():
        bs, seq_len = x.shape
        prefix = torch.zeros(bs, group_size - 1, dtype=torch.int64, device=x.device)
        x = torch.cat([prefix, x], dim=1)
        windows = x.unfold(1, group_size, 1)
        hashes = rolling_polynomial_hash(windows, hash_func_nb)
        hash_values_range = hashes % max_hash
    hash_values_range.requires_grad = False
    return hash_values_range

def rolling_polynomial_hash(t, hash_func_nb: int = 0):
    """Compute rolling polynomial hash."""
    if hash_func_nb == 0:
        return t.sum(dim=-1)
    elif hash_func_nb == 1:
        return (t * torch.arange(t.shape[-1], device=t.device)).sum(dim=-1)
    else:
        return (t * torch.arange(t.shape[-1], device=t.device) ** 2).sum(dim=-1)

@auto_docstring
class BLTPreTrainedModel(PreTrainedModel):
    config_class = BLTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BLTLocalEncoder", "BLTGlobalTransformer", "BLTLocalDecoder"]
    _supports_cache_class = True
    _supports_static_cache = False
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_quantized_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, MllamaTextRMSNorm):
            module.weight.data.fill_(1.0)

@auto_docstring
class BLTModel(BLTPreTrainedModel):
    """
    The Byte Latent Transformer (BLT) model which consists of a local encoder, global transformer, and local decoder.
    This model is designed to process byte sequences efficiently by using a hierarchical architecture with patch-based processing.
    
    The model architecture consists of:
    1. Local Encoder: Processes input tokens in patches
    2. Global Transformer: Processes the encoded patches globally
    3. Local Decoder: Decodes the global representation back to token level
    """
    
    def __init__(self, config: BLTConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize components
        self.local_encoder = BLTLocalEncoder(config)
        self.global_transformer = BLTGlobalTransformer(config)
        self.local_decoder = BLTLocalDecoder(config)
        
        # Initialize hash embeddings if configured
        self._init_hash_embeddings()
        
        # Initialize weights
        self.post_init()
        
    def _init_hash_embeddings(self):
        """Initialize hash embeddings based on configuration."""
        if self.config.use_hash_embeddings:
            self.encoder_hash_tok_embedding = nn.ModuleList([
                nn.Embedding(self.config.encoder_hash_byte_group_vocab, self.config.hidden_size)
                for _ in range(self.config.encoder_hash_byte_group_nb_functions)
            ])
        else:
            self.encoder_hash_tok_embedding = None
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_lengths: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass of the BLT model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Optional attention mask of shape (batch_size, sequence_length)
            patch_lengths: Optional tensor of patch lengths of shape (batch_size, num_patches)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a ModelOutput object
            
        Returns:
            ModelOutput or tuple containing:
            - last_hidden_state: Final hidden states
            - hidden_states: All hidden states if output_hidden_states=True
            - attentions: All attention weights if output_attentions=True
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get input tensors
        local_encoder_tokens, _, local_decoder_tokens = get_blt_input(
            tokens=input_ids,
            enforce_patch_size_multiple=self.config.enforce_patch_size_multiple,
            nb_boe=self.config.nb_boe,
            patch_size=self.config.patch_size,
            boe_id=BOE_ID,
        )
        
        # Generate patch IDs if not provided
        if patch_lengths is None:
            patch_lengths = torch.ones(
                (input_ids.shape[0], input_ids.shape[1] // self.config.patch_size),
                dtype=torch.long,
                device=input_ids.device
            ) * self.config.patch_size
            
        patch_ids = patch_ids_from_lengths(patch_lengths, local_encoder_tokens.shape[1])
        decoder_patch_ids = decoder_patch_ids_from_lengths(
            patch_lengths, self.config.nb_boe, local_decoder_tokens.shape[1]
        )
        
        # Compute hash embeddings if configured
        hash_embeddings = None
        if self.config.use_hash_embeddings:
            hash_embeddings = compute_hash_embeddings(
                local_encoder_tokens=local_encoder_tokens,
                local_encoder=self.local_encoder,
                encoder_hash_tok_embedding=self.encoder_hash_tok_embedding,
                encoder_hash_byte_group_nb_functions=self.config.encoder_hash_byte_group_nb_functions,
                encoder_hash_byte_group_size=self.config.encoder_hash_byte_group_size,
                encoder_hash_byte_group_vocab=self.config.encoder_hash_byte_group_vocab,
            )
            
        # Local encoder forward pass
        local_encoder_outputs = self.local_encoder(
            input_ids=local_encoder_tokens,
            attention_mask=attention_mask,
            patch_ids=patch_ids,
            hash_embeddings=hash_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # Downsample local encoder outputs
        local_encoder_hidden_states = downsample(
            h=local_encoder_outputs.last_hidden_state,
            num_patches=patch_lengths.shape[1],
            patch_lengths=patch_lengths,
            patch_ids=patch_ids,
            downsampling_by_pooling=self.config.downsampling_by_pooling,
            patch_size=self.config.patch_size,
        )
        
        # Global transformer forward pass
        global_transformer_outputs = self.global_transformer(
            hidden_states=local_encoder_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        # Local decoder forward pass
        local_decoder_outputs = self.local_decoder(
            input_ids=local_decoder_tokens,
            hidden_states=global_transformer_outputs[0],
            attention_mask=attention_mask,
            patch_ids=decoder_patch_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        if not return_dict:
            return tuple(v for v in [
                local_decoder_outputs.last_hidden_state,
                local_decoder_outputs.hidden_states,
                local_decoder_outputs.attentions
            ] if v is not None)
            
        return BaseModelOutput(
            last_hidden_state=local_decoder_outputs.last_hidden_state,
            hidden_states=local_decoder_outputs.hidden_states,
            attentions=local_decoder_outputs.attentions,
        )

class BLTLocalEncoder(nn.Module):
    """
    Local encoder component of the BLT model that processes input tokens in patches.
    
    This encoder consists of:
    1. Token embeddings
    2. Position embeddings
    3. Transformer layers with self-attention
    """
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BLTLocalEncoderLayer(config) for _ in range(config.num_local_encoder_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_ids: Optional[torch.Tensor] = None,
        hash_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass of the local encoder.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Optional attention mask of shape (batch_size, sequence_length)
            patch_ids: Optional tensor of patch IDs of shape (batch_size, sequence_length)
            hash_embeddings: Optional hash embeddings of shape (batch_size, sequence_length, hidden_size)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a ModelOutput object
            
        Returns:
            ModelOutput or tuple containing:
            - last_hidden_state: Final hidden states
            - hidden_states: All hidden states if output_hidden_states=True
            - attentions: All attention weights if output_attentions=True
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get sequence length
        batch_size, seq_length = input_ids.shape
        
        # Get position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        
        # Add hash embeddings if provided
        if hash_embeddings is not None:
            hidden_states = hidden_states + hash_embeddings
            
        # Apply layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Initialize output containers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                patch_ids=patch_ids,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                
        # Add final hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
            
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

class BLTLocalEncoderLayer(nn.Module):
    """
    A single layer of the local encoder in the BLT model.
    
    This layer consists of:
    1. Self-attention mechanism
    2. Feed-forward network
    3. Layer normalization
    """
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attention = BLTSelfAttention(config)
        self.self_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = BLTFeedForward(config)
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of a single local encoder layer.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, sequence_length)
            patch_ids: Optional tensor of patch IDs of shape (batch_size, sequence_length)
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple containing:
            - hidden_states: Output hidden states
            - attention_weights: Optional attention weights if output_attentions=True
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attention_layer_norm(hidden_states)
        hidden_states, attention_weights = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            patch_ids=patch_ids,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.feed_forward_layer_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs

class BLTSelfAttention(nn.Module):
    """
    Self-attention mechanism used in the BLT model.
    
    This attention mechanism:
    1. Projects input to query, key, and value
    2. Computes attention scores
    3. Applies attention mask if provided
    4. Computes weighted sum of values
    """
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Projection layers
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Number of attention heads
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose input tensor for attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the self-attention mechanism.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, sequence_length)
            patch_ids: Optional tensor of patch IDs of shape (batch_size, sequence_length)
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple containing:
            - hidden_states: Output hidden states
            - attention_weights: Optional attention weights if output_attentions=True
        """
        # Project input to query, key, and value
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply patch-based attention if patch_ids provided
        if patch_ids is not None:
            # Create patch attention mask
            patch_attention_mask = (patch_ids.unsqueeze(1) == patch_ids.unsqueeze(2)).float()
            patch_attention_mask = patch_attention_mask.unsqueeze(1)
            attention_scores = attention_scores + (1.0 - patch_attention_mask) * -10000.0
            
        # Normalize attention scores
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Compute weighted sum of values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Project to output
        output = self.output(context_layer)
        
        outputs = (output,)
        if output_attentions:
            outputs += (attention_probs,)
            
        return outputs

class BLTFeedForward(nn.Module):
    """
    Feed-forward network used in the BLT model.
    
    This network consists of:
    1. First linear layer with activation
    2. Second linear layer
    """
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # First linear layer
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # Activation function
        self.activation = ACT2FN[config.hidden_act]
        
        # Second linear layer
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, sequence_length, hidden_size)
            
        Returns:
            Output hidden states of shape (batch_size, sequence_length, hidden_size)
        """
        # First linear layer with activation
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.activation(hidden_states)
        
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # Second linear layer
        hidden_states = self.output(hidden_states)
        
        return hidden_states

class BLTGlobalTransformer(nn.Module):
    """
    Global transformer component of the BLT model.
    
    This component processes the global context of the input sequence using:
    1. Position embeddings
    2. Multiple transformer layers with self-attention
    """
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([BLTGlobalTransformerLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Forward pass of the global transformer.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, sequence_length)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            
        Returns:
            If output_attentions or output_hidden_states is True:
                Tuple of (last_hidden_state, all_hidden_states, all_attentions)
            Otherwise:
                Last hidden state of shape (batch_size, sequence_length, hidden_size)
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Add position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if not output_attentions and not output_hidden_states:
            return hidden_states
            
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

class BLTGlobalTransformerLayer(nn.Module):
    """
    Single layer of the global transformer in the BLT model.
    
    This layer consists of:
    1. Self-attention mechanism
    2. Feed-forward network
    3. Layer normalization
    """
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attention = BLTSelfAttention(config)
        self.self_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = BLTFeedForward(config)
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Forward pass of the global transformer layer.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, sequence_length)
            output_attentions: Whether to output attention weights
            
        Returns:
            If output_attentions is True:
                Tuple of (hidden_states, attention_weights)
            Otherwise:
                Hidden states of shape (batch_size, sequence_length, hidden_size)
        """
        # Self-attention
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        attention_output = self_attention_outputs[0]
        attention_output = self.self_attention_layer_norm(hidden_states + attention_output)
        
        # Feed-forward network
        feed_forward_output = self.feed_forward(attention_output)
        feed_forward_output = self.feed_forward_layer_norm(attention_output + feed_forward_output)
        
        if output_attentions:
            return (feed_forward_output, self_attention_outputs[1])
            
        return (feed_forward_output,)

class BLTLocalDecoder(nn.Module):
    """
    Local decoder component of the BLT model.
    
    This component processes the local context of the input sequence using:
    1. Token embeddings
    2. Position embeddings
    3. Multiple transformer layers with self-attention and cross-attention
    """
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([BLTLocalDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Forward pass of the local decoder.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            hidden_states: Encoder hidden states of shape (batch_size, encoder_sequence_length, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, sequence_length)
            patch_ids: Optional patch IDs of shape (batch_size, sequence_length)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            
        Returns:
            If output_attentions or output_hidden_states is True:
                Tuple of (last_hidden_state, all_hidden_states, all_attentions)
            Otherwise:
                Last hidden state of shape (batch_size, sequence_length, hidden_size)
        """
        batch_size, seq_length = input_ids.shape
        
        # Get token embeddings
        token_embeddings = self.token_embeddings(input_ids)
        
        # Add position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = token_embeddings + position_embeddings
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                encoder_hidden_states=hidden_states,
                attention_mask=attention_mask,
                patch_ids=patch_ids,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        if not output_attentions and not output_hidden_states:
            return hidden_states
            
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

class BLTLocalDecoderLayer(nn.Module):
    """
    Single layer of the local decoder in the BLT model.
    
    This layer consists of:
    1. Self-attention mechanism
    2. Cross-attention mechanism
    3. Feed-forward network
    4. Layer normalization
    """
    
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attention = BLTSelfAttention(config)
        self.self_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Cross-attention
        self.cross_attention = BLTCrossAttention(config)
        self.cross_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.feed_forward = BLTFeedForward(config)
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Forward pass of the local decoder layer.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, sequence_length, hidden_size)
            encoder_hidden_states: Encoder hidden states of shape (batch_size, encoder_sequence_length, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, sequence_length)
            patch_ids: Optional patch IDs of shape (batch_size, sequence_length)
            output_attentions: Whether to output attention weights
            
        Returns:
            If output_attentions is True:
                Tuple of (hidden_states, attention_weights)
            Otherwise:
                Hidden states of shape (batch_size, sequence_length, hidden_size)
        """
        # Self-attention
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        
        attention_output = self_attention_outputs[0]
        attention_output = self.self_attention_layer_norm(hidden_states + attention_output)
        
        # Cross-attention
        cross_attention_outputs = self.cross_attention(
            attention_output,
            encoder_hidden_states,
            attention_mask=attention_mask,
            patch_ids=patch_ids,
            output_attentions=output_attentions,
        )
        
        cross_attention_output = cross_attention_outputs[0]
        cross_attention_output = self.cross_attention_layer_norm(attention_output + cross_attention_output)
        
        # Feed-forward network
        feed_forward_output = self.feed_forward(cross_attention_output)
        feed_forward_output = self.feed_forward_layer_norm(cross_attention_output + feed_forward_output)
        
        if output_attentions:
            return (feed_forward_output, self_attention_outputs[1], cross_attention_outputs[1])
            
        return (feed_forward_output,)
