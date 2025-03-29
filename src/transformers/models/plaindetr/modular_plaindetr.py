import torch
import torch.nn as nn
import torch.nn.functional as F

from ..deformable_detr.modeling_deformable_detr import DeformableDetrSinePositionEmbedding, DeformableDetrLearnedPositionEmbedding



class PlainDetrSinePositionEmbedding(DeformableDetrSinePositionEmbedding):
    pass

class PlainDetrLearnedPositionEmbedding(DeformableDetrLearnedPositionEmbedding):
    pass

def build_position_encoding(config):
    if config.position_embedding_type == "sine":
        position_embedding = PlainDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = PlainDetrLearnedPositionEmbedding(n_steps)
    elif config.position_embedding_type == "sine":
        position_embedding = PlainDetrSinePositionEmbedding(n_steps, normalize=False)
    else:
        raise ValueError(f"Unknown position embedding type: {config.position_embedding_type}")
    
    return position_embedding


class PlainDetrGlobalAttentionWithPositionEmbedding(nn.Module):
    """
    Global cross-attention module used in PlainDETR.
    
    This module implements cross-attention between query embeddings and 
    flattened feature maps, incorporating position information.
    """
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.query_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.value_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attn_drop)
        self.output_proj = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        query_vectors,
        key_input_flatten,
        value_input_flatten,
        attention_mask=None,
    ):
        batch_size, seq_len, embedding_dim = key_input_flatten.shape
        keys = self.key_proj(key_input_flatten).reshape(
            batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads
        )
        keys = keys.permute(0, 2, 1, 3)
        values = self.value_proj(value_input_flatten).reshape(
            batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads
        )
        values = values.permute(0, 2, 1, 3)
        
        batch_size, seq_len, embedding_dim = query_vectors.shape
        queries = self.query_proj(query_vectors).reshape(
            batch_size, seq_len, self.num_heads, embedding_dim // self.num_heads
        )
        queries = queries.permute(0, 2, 1, 3)
        queries = queries * self.scale

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask[:, None, None] * -100
            
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, values)
        context_layer = context_layer.transpose(1, 2).reshape(batch_size, seq_len, embedding_dim)
        output = self.output_proj(context_layer)
        output = self.output_dropout(output)
        
        return output