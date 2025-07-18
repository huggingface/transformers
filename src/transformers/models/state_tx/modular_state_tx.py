# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Any, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_utils import PreTrainedModel
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import logging

from ..llama.modeling_llama import LlamaModel
from ..llama.configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

class LlamaBidirectionalConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
        )

class StateTxConfig(PretrainedConfig):
    r"""
    Configuration class for StateTx (State Transformer) model based on PertSetsPerturbationModel.
    
    This model uses a bidirectional Llama transformer backbone to process perturbation data.
    """

    model_type = "state_tx"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        gene_dim=2000,
        pert_dim=91,
        basal_dim=2000,
        hidden_dim=1440,
        num_layers=4,
        num_heads=16,
        intermediate_size=4416,
        vocab_size=32000,
        num_batches=18,
        dropout=0.1,
        rms_norm_eps=1e-6,
        use_cache=True,
        max_position_embeddings=512,
        num_key_value_heads=None,
        head_dim=None,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        layer_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rotary_dim=0,
        use_rotary_embeddings=False,
        n_positions=512,
        **kwargs,
    ):
        super().__init__(
            use_cache=use_cache,
            **kwargs,
        )
        self.gene_dim = gene_dim
        self.pert_dim = pert_dim
        self.basal_dim = basal_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_dim  # Add for Llama compatibility
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_attention_heads = num_heads  # Add for Llama compatibility
        self.num_key_value_heads = num_key_value_heads or num_heads  # Add for Llama compatibility
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.num_batches = num_batches
        self.dropout = dropout
        self.attention_dropout = attention_dropout  # Add for Llama compatibility
        self.hidden_dropout = hidden_dropout
        self.attention_bias = False  # Add for Llama compatibility
        self.mlp_bias = False  # Add for Llama compatibility
        self.hidden_act = "silu"  # Add for Llama compatibility
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rotary_dim = rotary_dim
        self.use_rotary_embeddings = use_rotary_embeddings
        self.n_positions = n_positions


class SamplesLoss(nn.Module):
    """Samples loss function for perturbation model."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        return F.mse_loss(predictions, targets)


class LatentToGeneDecoder(nn.Module):
    """Decoder that converts latent representations back to gene space."""
    
    def __init__(self, config: StateTxConfig):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(config.gene_dim, 1024, bias=True),
            nn.LayerNorm(1024, eps=1e-05),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(1024, 1024, bias=True),
            nn.LayerNorm(1024, eps=1e-05),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(1024, 512, bias=True),
            nn.LayerNorm(512, eps=1e-05),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(512, config.gene_dim, bias=True),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.decoder(x)


class NoRoPE(nn.Module):
    """
    A drop-in replacement for LlamaRotaryEmbedding that always returns:
      cos = all ones, sin = all zeros
    of shape (batch_size, seq_len, head_dim), so rotary has no effect.
    """

    def __init__(self, num_attention_heads: int, hidden_size: int):
        super().__init__()
        self.num_heads = num_attention_heads
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Create cos = ones, sin = zeros
        #   shape --> (batch_size, seq_len, head_dim)
        cos = hidden_states.new_ones(batch_size, seq_len, self.num_heads)
        sin = hidden_states.new_zeros(batch_size, seq_len, self.num_heads)
        return cos, sin


class LlamaBidirectionalModel(LlamaModel):
    """
    A drop-in replacement for LlamaModel with bidirectional attention.
    By overriding _update_causal_mask to return None, all tokens attend to each other.
    """

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = NoRoPE(
            num_attention_heads=config.head_dim,
            hidden_size=config.hidden_size,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):
        # By returning None, we disable any causal‐(look‐ahead) masking.
        # The only mask that remains is whatever “attention_mask” the user has passed
        # (e.g. padding‐mask), which will be handled by Flash/SDPA internally as non‐causal.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        kwargs["is_causal"] = False
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )


class StateTxPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = StateTxConfig
    base_model_prefix = "state_tx"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)


class StateTxModel(StateTxPreTrainedModel):
    """
    StateTx Model implementing PertSetsPerturbationModel architecture.
    
    This model processes perturbation data through encoders, a bidirectional transformer,
    and produces gene expression predictions.
    """

    def __init__(self, config: StateTxConfig):
        super().__init__(config)
        self.config = config

        # Loss function
        self.loss_fn = SamplesLoss()

        # Gene decoder
        self.gene_decoder = LatentToGeneDecoder(config)

        # Perturbation encoder
        self.pert_encoder = nn.Sequential(
            nn.Linear(config.pert_dim, config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=True),
        )

        # Basal encoder
        self.basal_encoder = nn.Sequential(
            nn.Linear(config.basal_dim, config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=True),
        )

        # Transformer backbone
        transformer_config = LlamaBidirectionalConfig(
            max_position_embeddings=512,
            hidden_size=1440,
            intermediate_size=4416,
            num_hidden_layers=4,
            num_attention_heads=12,
            num_key_value_heads=12,
            head_dim=120,
            use_cache=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            layer_norm_eps=1e-06,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rotary_dim=0,
            use_rotary_embeddings=False,
            n_positions=512,
        )
        self.transformer_backbone = LlamaBidirectionalModel(transformer_config)

        # Project out
        self.project_out = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, config.gene_dim, bias=True),
        )

        # Batch encoder
        self.batch_encoder = nn.Embedding(config.num_batches, config.hidden_dim)

        # ReLU activation
        self.relu = nn.ReLU()

        # Initialize weights
        self.post_init()

    def forward(
        self,
        pert_input: torch.Tensor,
        basal_input: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], dict]:
        """
        Forward pass of the StateTx model.

        Args:
            pert_input: Perturbation input tensor, shape (batch_size, pert_dim)
            basal_input: Basal gene expression input, shape (batch_size, basal_dim)
            batch_ids: Batch identifiers, shape (batch_size,)
            attention_mask: Attention mask for transformer
            labels: Target gene expressions for loss computation
            return_dict: Whether to return a dictionary or tuple

        Returns:
            Model outputs including gene predictions and optionally loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode inputs
        pert_encoded = self.pert_encoder(pert_input)  # (batch_size, hidden_dim)
        basal_encoded = self.basal_encoder(basal_input)  # (batch_size, hidden_dim)

        # Combine encodings - concatenate along sequence dimension
        # Shape: (batch_size, 2, hidden_dim)
        combined_input = pert_encoded.unsqueeze(1) + basal_encoded.unsqueeze(1)
        seq_input = combined_input  # (batch_size, 2, hidden_dim)

        # # Add batch embeddings if provided
        # if batch_ids is not None:
        #     batch_embeds = self.batch_encoder(batch_ids)  # (batch_size, hidden_dim)
        #     # Add batch embedding to each position
        #     combined_input = combined_input + batch_embeds.unsqueeze(1)
        batch_embeddings = self.batch_encoder(torch.zeros([512]).long()).unsqueeze(1)

        seq_input = seq_input + batch_embeddings
        seq_input = seq_input.transpose(0, 1)

        # Pass through transformer backbone
        transformer_output = self.transformer_backbone(inputs_embeds=seq_input)
        transformer_output = transformer_output.last_hidden_state

        # Project to gene space
        control_cells = basal_encoded.unsqueeze(1).transpose(0, 1)
        out_pred = self.project_out(transformer_output + control_cells)

        # Apply final ReLU
        gene_predictions = self.relu(out_pred).squeeze(0)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(gene_predictions, labels)

        if not return_dict:
            output = (gene_predictions,)
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "gene_predictions": gene_predictions,
            "transformer_output": transformer_output,
        }


__all__ = [
    "StateTxConfig",
    "StateTxPreTrainedModel", 
    "StateTxModel",
]
