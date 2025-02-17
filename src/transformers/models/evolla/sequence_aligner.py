import torch
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def FeedForward(dim, mult=None):
    if mult is None:
        mult = 4
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_probs_dropout_prob: float,
        enable_bias: bool,
        ffn_mult: float,
        protein_encoder_dim: int = None,
        structure_encoder_dim: int = None,
        msa_encoder_dim: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = num_attention_heads**-0.5
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        if protein_encoder_dim is not None:
            self.key_protein = nn.Linear(protein_encoder_dim, self.all_head_size)
            self.value_protein = nn.Linear(protein_encoder_dim, self.all_head_size)
        else:
            self.key_protein = None
            self.value_protein = None

        if structure_encoder_dim is not None:
            self.key_structure = nn.Linear(structure_encoder_dim, self.all_head_size)
            self.value_structure = nn.Linear(structure_encoder_dim, self.all_head_size)
        else:
            self.key_structure = None
            self.value_structure = None

        if msa_encoder_dim is not None:
            self.key_msa = nn.Linear(msa_encoder_dim, self.all_head_size)
            self.value_msa = nn.Linear(msa_encoder_dim, self.all_head_size)
        else:
            self.key_msa = None
            self.value_msa = None

        self.attention_norm = RMSNorm(self.hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=enable_bias)

        self.ff = FeedForward(self.hidden_size, ffn_mult)
        self.gate_attention = nn.Parameter(torch.tensor([0.0]))
        self.gate_ffw = nn.Parameter(torch.tensor([0.0]))

    def cross_attention(
        self,
        query_states,
        protein_key_value_states,
        structure_key_value_states,
        msa_key_value_states,
        query_attn_mask,
        protein_kv_attn_mask,
        structure_kv_attn_mask,
        msa_kv_attn_mask,
    ):
        """
        query_states: text
        key_value_states: protein
        query_states: [bs, query_seq_len, dim]
        key_value_states: [bs, kv_seq_len, dim]
        query_attn_mask: [bs, query_seq_len]
        kv_attn_mask: [bs, kv_seq_len]
        """

        # Concatenate protein and structure
        kv_attn_mask = [protein_kv_attn_mask, structure_kv_attn_mask, msa_kv_attn_mask]
        kv_attn_mask = [_ for _ in kv_attn_mask if _ is not None]
        if not kv_attn_mask:
            raise ValueError("At least one modality should be provided for cross attention.")
        kv_attn_mask = torch.cat(kv_attn_mask, dim=1)

        query_layer = self.attention_norm(query_states)

        # Warning: This place might cause issues, refers to
        # https://discuss.pytorch.org/t/cuda-error-cublas-status-not-supported-when-calling-cublasltmatmul-from-torch-nn-functional-linear/170214/13
        # Solution: add `DISABLE_ADDMM_CUDA_LT=1` as environment variable
        # Apply linear transformation to input_query, input_key, and input_value
        query_layer = self.query(query_layer)  # [bs, querylength, dim]

        if self.key_protein is not None and self.value_protein is not None:
            protein_key_value_states = protein_key_value_states.to(query_states)
            key_layer_protein = self.key_protein(protein_key_value_states)  # [bs, keylength, dim]
            value_layer_protein = self.value_protein(protein_key_value_states)  # [bs, keylength, dim]
        else:
            key_layer_protein = None
            value_layer_protein = None

        if self.key_structure is not None and self.value_structure is not None:
            structure_key_value_states = structure_key_value_states.to(query_states)
            key_layer_structure = self.key_structure(structure_key_value_states)  # [bs, keylength, dim]
            value_layer_structure = self.value_structure(structure_key_value_states)  # [bs, keylength, dim]
        else:
            key_layer_structure = None
            value_layer_structure = None

        if self.key_msa is not None and self.value_msa is not None:
            msa_key_value_states = msa_key_value_states.to(query_states)
            key_layer_msa = self.key_msa(msa_key_value_states)  # [bs, keylength, dim]
            value_layer_msa = self.value_msa(msa_key_value_states)  # [bs, keylength, dim]
        else:
            key_layer_msa = None
            value_layer_msa = None

        key_layer = [key_layer_protein, key_layer_structure, key_layer_msa]
        key_layer = [_ for _ in key_layer if _ is not None]
        key_layer = torch.cat(key_layer, dim=1)

        value_layer = [value_layer_protein, value_layer_structure, value_layer_msa]
        value_layer = [_ for _ in value_layer if _ is not None]
        value_layer = torch.cat(value_layer, dim=1)

        query_layer = self.transpose_for_scores(query_layer)  # [bs, numheads, querylength, dim/numheads]
        key_layer = self.transpose_for_scores(key_layer)  # [bs, numheads, keylength, dim/numheads]
        value_layer = self.transpose_for_scores(value_layer)  # [bs, numheads, keylength, dim/numheads]

        query_layer = query_layer * self.scale

        # attention_mask: [bs, 1, querylength, keylength]
        attention_mask = query_attn_mask[:, None, :, None] * kv_attn_mask[:, None, None, :]
        # Compute the scaled dot-product attention scores
        attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [bs, numheads, querylength, keylength]
        attn_weights = attn_weights - attn_weights.amax(dim=-1, keepdim=True).detach()  # To stablize score
        attention_scores = attn_weights.masked_fill(
            (1 - attention_mask).bool(), torch.finfo(attn_weights.dtype).min
        )  # [bs, numheads, querylength, keylength]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs_dropped = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, numheads, querylength, dim/numheads]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.out_proj(context_layer)

        return context_layer

    def forward(
        self,
        query_states,
        protein_kv_states,
        structure_kv_states,
        msa_kv_states,
        query_attn_mask,
        protein_kv_attn_mask=None,
        structure_kv_attn_mask=None,
        msa_kv_attn_mask=None,
        protein_batch_mask=None,
        structure_batch_mask=None,
        msa_batch_mask=None,
        past_key_value=None,
    ):
        """
        kv_states: protein
        query_states: text

        query_states: [bs, query_seq_len, dim]
        kv_states: [bs, kv_seq_len, dim]
        query_attn_mask: [bs, query_seq_len]
        kv_attn_mask: [bs, kv_seq_len], default None
        past_key_value: [bs, past_kv_seq_len, dim], default None
        """
        if protein_kv_states is not None:
            bs, protein_kv_seq_len, dim = protein_kv_states.shape
            if protein_kv_attn_mask is None:
                protein_kv_attn_mask = (
                    torch.ones(bs, protein_kv_seq_len).to(protein_batch_mask.device)
                    * protein_batch_mask.expand(size=(protein_kv_seq_len, bs)).T
                ).to(protein_kv_states.device)
        else:
            protein_kv_attn_mask = None

        if structure_kv_states is not None:
            bs, structure_kv_seq_len, dim = structure_kv_states.shape
            if structure_kv_attn_mask is None:
                structure_kv_attn_mask = (
                    torch.ones(bs, structure_kv_seq_len).to(protein_batch_mask.device)
                    * structure_batch_mask.expand(size=(structure_kv_seq_len, bs)).T
                ).to(structure_kv_states.device)
        else:
            structure_kv_attn_mask = None

        if msa_kv_states is not None:
            bs, msa_kv_seq_len, dim = msa_kv_states.shape
            if msa_kv_attn_mask is None:
                msa_kv_attn_mask = (
                    torch.ones(bs, msa_kv_seq_len).to(protein_batch_mask.device)
                    * msa_batch_mask.expand(size=(msa_kv_seq_len, bs)).T
                ).to(msa_kv_states.device)
        else:
            msa_kv_attn_mask = None
        hidden_states = query_states
        # only when there's at least one valid modality, crossattention will be performed
        if (
            (protein_kv_states is not None and protein_kv_attn_mask.any())
            or (structure_kv_states is not None and structure_kv_attn_mask.any())
            or (msa_kv_states is not None and msa_kv_attn_mask.any())
        ):
            residual = hidden_states
            hidden_states = self.cross_attention(
                query_states=hidden_states,
                protein_key_value_states=protein_kv_states,
                structure_key_value_states=structure_kv_states,
                msa_key_value_states=msa_kv_states,
                query_attn_mask=query_attn_mask,
                protein_kv_attn_mask=protein_kv_attn_mask,
                structure_kv_attn_mask=structure_kv_attn_mask,
                msa_kv_attn_mask=msa_kv_attn_mask,
            )  # [bs, query_seq_len, dim]
            # tanh gate
            hidden_states = torch.tanh(self.gate_attention) * hidden_states

            hidden_states = residual + hidden_states  # input_query

            residual = hidden_states
            hidden_states = self.ff(hidden_states) * torch.tanh(self.gate_ffw)
            hidden_states = residual + hidden_states

        return hidden_states

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
