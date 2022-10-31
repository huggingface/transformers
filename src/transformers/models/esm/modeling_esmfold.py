# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm


from ...utils import ContextManagers

from .modeling_esm import EsmModel, EsmPreTrainedModel
from .openfold_data_transforms import make_atom14_masks
from .openfold_np import residue_constants
from .openfold_np.protein import Protein as OFProtein
from .openfold_np.protein import to_pdb
from .openfold_utils.feats import atom14_to_atom37
from .openfold_utils.loss import compute_predicted_aligned_error, compute_tm
from .openfold_triangular_attention import TriangleAttentionEndingNode, TriangleAttentionStartingNode
from .openfold_triangular_multiplicative_update import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
from .openfold_structure import StructureModule

# TODO remove:
from einops import rearrange, repeat


def collate_dense_tensors(samples: List[torch.Tensor], pad_v: float = 0) -> torch.Tensor:
    """
    Takes a list of tensors with the following dimensions:
        [(d_11, ..., d_1K),
         (d_21, ..., d_2K), ..., (d_N1, ..., d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(f"Samples has varying dimensions: {[x.dim() for x in samples]}")
    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result


class EsmFoldAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_width, gated=False):
        super().__init__()
        assert embed_dim == num_heads * head_width

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width

        self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gated = gated
        if gated:
            self.g_proj = nn.Linear(embed_dim, embed_dim)
            torch.nn.init.zeros_(self.g_proj.weight)
            torch.nn.init.ones_(self.g_proj.bias)

        self.rescale_factor = self.head_width**-0.5

        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, x, mask=None, bias=None, indices=None):
        """
        Basic self attention with optional mask and external pairwise bias. To handle sequences of different lengths,
        use mask.

        Inputs:
          x: batch of input sequneces (.. x L x C) mask: batch of boolean masks where 1=valid, 0=padding position (.. x
          L_k). optional. bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads). optional.

        Outputs:
          sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """

        t = rearrange(self.proj(x), "... l (h c) -> ... h l c", h=self.num_heads)
        q, k, v = t.chunk(3, dim=-1)

        q = self.rescale_factor * q
        a = torch.einsum("...qc,...kc->...qk", q, k)

        # Add external attention bias.
        if bias is not None:
            a = a + rearrange(bias, "... lq lk h -> ... h lq lk")

        # Do not attend to padding tokens.
        if mask is not None:
            mask = repeat(mask, "... lk -> ... h lq lk", h=self.num_heads, lq=q.shape[-2])
            a = a.masked_fill(mask == False, -np.inf)

        a = nn.functional.softmax(a, dim=-1)

        y = torch.einsum("...hqk,...hkc->...qhc", a, v)
        y = rearrange(y, "... h c -> ... (h c)", h=self.num_heads)

        if self.gated:
            y = self.g_proj(x).sigmoid() * y
        y = self.o_proj(y)

        return y, rearrange(a, "... lq lk h -> ... h lq lk")


class EsmFoldDropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask along a particular dimension.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        super().__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        return x * self.dropout(x.new_ones(shape))


class EsmFoldSequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
        super().__init__()

        self.layernorm = nn.LayerNorm(sequence_state_dim)
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2, bias=True)
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim, bias=True)

        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, sequence_state):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim

        Output:
          pairwise_state: B x L x L x pairwise_state_dim

        Intermediate state:
          B x L x L x 2*inner_dim
        """

        assert len(sequence_state.shape) == 3

        s = self.layernorm(sequence_state)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x


class EsmFoldPairToSequence(nn.Module):
    def __init__(self, pairwise_state_dim, num_heads):
        super().__init__()

        self.layernorm = nn.LayerNorm(pairwise_state_dim)
        self.linear = nn.Linear(pairwise_state_dim, num_heads, bias=False)

    def forward(self, pairwise_state):
        """
        Inputs:
          pairwise_state: B x L x L x pairwise_state_dim

        Output:
          pairwise_bias: B x L x L x num_heads
        """
        assert len(pairwise_state.shape) == 4
        z = self.layernorm(pairwise_state)
        pairwise_bias = self.linear(z)
        return pairwise_bias


class EsmFoldResidueMLP(nn.Module):
    def __init__(self, embed_dim, inner_dim, norm=nn.LayerNorm, dropout=0):
        super().__init__()

        self.mlp = nn.Sequential(
            norm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.mlp(x)


class EsmFoldTriangularSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        sequence_state_dim,
        pairwise_state_dim,
        sequence_head_width,
        pairwise_head_width,
        dropout=0,
    ):
        super().__init__()

        if sequence_state_dim % sequence_head_width != 0:
            raise ValueError(
                f"`sequence_state_dim` (here {sequence_state_dim} needs to be round multiple of `sequence_head_dim`"
                f" (here {sequence_head_width})."
            )
        if pairwise_state_dim % pairwise_head_width != 0:
            raise ValueError(
                f"`pairwise_state_dim` (here {pairwise_state_dim} needs to be round multiple of `pairwise_head_width`"
                f" (here {pairwise_head_width})."
            )
        sequence_num_heads = sequence_state_dim // sequence_head_width
        pairwise_num_heads = pairwise_state_dim // pairwise_head_width

        if sequence_state_dim != sequence_num_heads * sequence_head_width:
            raise ValueError(
                "`sequence_state_dim` should be equal to `sequence_num_heads * sequence_head_width, got"
                f" {sequence_state_dim} != {sequence_num_heads} * {sequence_head_width}."
            )
        if pairwise_state_dim != pairwise_num_heads * pairwise_head_width:
            raise ValueError(
                "`pairwise_state_dim` should be equal to `pairwise_num_heads * pairwise_head_width, got"
                f" {pairwise_state_dim} != {pairwise_num_heads} * {pairwise_head_width}."
            )
        if pairwise_state_dim % 2 != 0:
            raise ValueError(f"`pairwise_state_dim` should be even, got {pairwise_state_dim}.")

        self.sequence_state_dim = sequence_state_dim
        self.pairwise_state_dim = pairwise_state_dim

        self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        self.sequence_to_pair = EsmFoldSequenceToPair(sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim)
        self.pair_to_sequence = EsmFoldPairToSequence(pairwise_state_dim, sequence_num_heads)

        self.seq_attention = EsmFoldAttention(sequence_state_dim, sequence_num_heads, sequence_head_width, gated=True)
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            pairwise_state_dim,
            pairwise_state_dim,
        )
        self.tri_att_start = TriangleAttentionStartingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore
        self.tri_att_end = TriangleAttentionEndingNode(
            pairwise_state_dim,
            pairwise_head_width,
            pairwise_num_heads,
            inf=1e9,
        )  # type: ignore

        self.mlp_seq = EsmFoldResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=dropout)
        self.mlp_pair = EsmFoldResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=dropout)

        if dropout >= 0.4:
            raise ValueError(f"`dropout` should not be greater than 0.4, got {dropout}.")
        self.drop = nn.Dropout(dropout)
        self.row_drop = EsmFoldDropout(dropout * 2, 2)
        self.col_drop = EsmFoldDropout(dropout * 2, 1)

        torch.nn.init.zeros_(self.tri_mul_in.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_in.linear_z.bias)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.weight)
        torch.nn.init.zeros_(self.tri_mul_out.linear_z.bias)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_start.mha.linear_o.bias)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.weight)
        torch.nn.init.zeros_(self.tri_att_end.mha.linear_o.bias)

        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.weight)
        torch.nn.init.zeros_(self.sequence_to_pair.o_proj.bias)
        torch.nn.init.zeros_(self.pair_to_sequence.linear.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.weight)
        torch.nn.init.zeros_(self.seq_attention.o_proj.bias)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_seq.mlp[-2].bias)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].weight)
        torch.nn.init.zeros_(self.mlp_pair.mlp[-2].bias)

    def forward(self, sequence_state, pairwise_state, mask=None, chunk_size=None, **__kwargs):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim mask: B x L boolean
          tensor of valid positions

        Output:
          sequence_state: B x L x sequence_state_dim pairwise_state: B x L x L x pairwise_state_dim
        """
        if len(sequence_state.shape) != 3:
            raise ValueError(f"`sequence_state` should be a 3d-tensor, got {len(sequence_state.shape)} dims.")
        if len(pairwise_state.shape) != 4:
            raise ValueError(f"`pairwise_state` should be a 4d-tensor, got {len(pairwise_state.shape)} dims.")
        if mask is not None and len(mask.shape) != 2:
            raise ValueError(f"`mask` should be a 2d-tensor, got {len(mask.shape)} dims.")

        batch_dim, seq_dim, sequence_state_dim = sequence_state.shape
        pairwise_state_dim = pairwise_state.shape[3]

        if sequence_state_dim != self.sequence_state_dim:
            raise ValueError(
                "`sequence_state` last dimension should be equal to `self.sequence_state_dim`. Got"
                f"{sequence_state_dim} != {self.sequence_state_dim}."
            )
        if pairwise_state_dim != self.pairwise_state_dim:
            raise ValueError(
                "`pairwise_state` last dimension should be equal to `self.pairwise_state_dim`. Got "
                f"{pairwise_state_dim} != {self.pairwise_state_dim}."
            )
        if batch_dim != pairwise_state.shape[0]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent batch size: {batch_dim} != "
                f"{pairwise_state.shape[0]}."
            )
        if seq_dim != pairwise_state.shape[1] or seq_dim != pairwise_state.shape[2]:
            raise ValueError(
                f"`sequence_state` and `pairwise_state` have inconsistent sequence length: {seq_dim} != "
                f"{pairwise_state.shape[1]} or {pairwise_state.shape[2]}."
            )

        # Update sequence state
        bias = self.pair_to_sequence(pairwise_state)

        # Self attention with bias + mlp.
        y = self.layernorm_1(sequence_state)
        y, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + self.drop(y)
        sequence_state = self.mlp_seq(sequence_state)

        # Update pairwise state
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        # Axial attention with triangular bias.
        tri_mask = mask.unsqueeze(2) * mask.unsqueeze(1) if mask is not None else None
        pairwise_state = pairwise_state + self.row_drop(self.tri_mul_out(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.col_drop(self.tri_mul_in(pairwise_state, mask=tri_mask))
        pairwise_state = pairwise_state + self.row_drop(
            self.tri_att_start(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )
        pairwise_state = pairwise_state + self.col_drop(
            self.tri_att_end(pairwise_state, mask=tri_mask, chunk_size=chunk_size)
        )

        # MLP over pairs.
        pairwise_state = self.mlp_pair(pairwise_state)

        return sequence_state, pairwise_state


class EsmCategoricalMixture:
    def __init__(self, param, bins=50, start=0, end=1):
        # All tensors are of shape ..., bins.
        self.logits = param
        bins = torch.linspace(start, end, bins + 1, device=self.logits.device, dtype=self.logits.dtype)
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        # Shapes are:
        #     self.probs: ... x bins
        #     true      : ...
        true_index = (true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1)
        nll = self.logits.log_softmax(-1)
        return torch.take_along_dim(nll, true_index.unsqueeze(-1), dim=-1).squeeze(-1)

    def mean(self):
        return (self.logits.softmax(-1) @ self.v_bins.unsqueeze(1)).squeeze(-1)


def categorical_lddt(logits, bins=50):
    # Logits are ..., 37, bins.
    return EsmCategoricalMixture(logits, bins=bins).mean()


def get_axial_mask(mask):
    """
    Helper to convert B x L mask of valid positions to axial mask used in row column attentions.

    Input:
      mask: B x L tensor of booleans

    Output:
      mask: B x L x L tensor of booleans
    """

    if mask is None:
        return None

    if len(mask.shape) != 2:
        raise ValueError(f"`mask` should be a 2d-tensor, got {len(mask.shape)} dims.")
    batch_dim, seq_dim = mask.shape
    m = mask.unsqueeze(1).expand(batch_dim, seq_dim, seq_dim)
    m = m.reshape(batch_dim * seq_dim, seq_dim)
    return m


class RelativePosition(nn.Module):
    def __init__(self, bins, pairwise_state_dim):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long) mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """
        if residue_index.dtype != torch.long:
            raise ValueError(f"`residue_index` has dtype {residue_index.dtype}, it should be `torch.long`.")
        if mask is not None and residue_index.shape != mask.shape:
            raise ValueError(
                f"`residue_index` and `mask` have inconsistent shapes: {residue_index.shape} != {mask.shape}."
            )

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0

        output = self.embedding(diff)
        return output


class EsmFoldingTrunk(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        c_s = self.config.sequence_state_dim
        c_z = self.config.pairwise_state_dim

        self.pairwise_positional_embedding = RelativePosition(self.config.position_bins, c_z)

        self.blocks = nn.ModuleList(
            [
                EsmFoldTriangularSelfAttentionBlock(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=self.config.sequence_head_width,
                    pairwise_head_width=self.config.pairwise_head_width,
                    dropout=self.config.dropout,
                )
                for i in range(self.config.num_blocks)
            ]
        )

        self.recycle_bins = 15
        self.recycle_s_norm = nn.LayerNorm(c_s)
        self.recycle_z_norm = nn.LayerNorm(c_z)
        self.recycle_disto = nn.Embedding(self.recycle_bins, c_z)
        self.recycle_disto.weight[0].detach().zero_()

        self.structure_module = StructureModule(**config.structure_module.to_dict())
        self.trunk2sm_s = nn.Linear(c_s, self.structure_module.c_s)
        self.trunk2sm_z = nn.Linear(c_z, self.structure_module.c_z)

        self.chunk_size = self.config.chunk_size

    def set_chunk_size(self, chunk_size):
        # This parameter means the axial attention will be computed
        # in a chunked manner. This should make the memory used more or less O(L) instead of O(L^2).
        # It's equivalent to running a for loop over chunks of the dimension we're iterative over,
        # where the chunk_size is the size of the chunks, so 128 would mean to parse 128-lengthed chunks.
        self.chunk_size = chunk_size

    def forward(self, seq_feats, pair_feats, true_aa, residx, mask):
        """
        Inputs:
          seq_feats: B x L x C tensor of sequence features pair_feats: B x L x L x C tensor of pair features residx: B
          x L long tensor giving the position in the sequence mask: B x L boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """

        device = seq_feats.device
        s_s_0 = seq_feats
        s_z_0 = pair_feats

        no_recycles = self.config.max_recycles

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return s, z

        s_s = s_s_0
        s_z = s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        for recycle_idx in range(no_recycles):
            with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
                # === Recycling ===
                recycle_s = self.recycle_s_norm(recycle_s.detach())
                recycle_z = self.recycle_z_norm(recycle_z.detach())
                recycle_z += self.recycle_disto(recycle_bins.detach())

                s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

                # === Structure module ===
                structure = self.structure_module(
                    {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                    true_aa,
                    mask.float(),
                )

                recycle_s = s_s
                recycle_z = s_z
                # Distogram needs the N, CA, C coordinates, and bin constants same as alphafold.
                recycle_bins = EsmFoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375,
                    21.375,
                    self.recycle_bins,
                )

        structure["s_s"] = s_s
        structure["s_z"] = s_z

        return structure

    @staticmethod
    def distogram(coords, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=coords.device,
        )
        boundaries = boundaries**2
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)
        bins = torch.sum(dists > boundaries, dim=-1)  # [..., L, L]
        return bins


class EsmForProteinFolding(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.distogram_bins = 64

        self.esm = EsmModel(config, add_pooling_layer=False)

        self.esm.requires_grad_(False)
        if self.config.esmfold_config.fp16_esm:
            self.esm.half()

        self.esm_feats = self.config.hidden_size
        self.esm_attns = self.config.num_hidden_layers * self.config.num_attention_heads
        self.esm_layers = self.config.num_hidden_layers
        self.register_buffer("af2_to_esm", self._af2_to_esm_from_vocab_list(config.vocab_list))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm_layers + 1))

        c_s = self.config.esmfold_config.trunk.sequence_state_dim
        c_z = self.config.esmfold_config.trunk.pairwise_state_dim
        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.esm_dict_cls_idx = self.config.vocab_list.index("<cls>")
        self.esm_dict_mask_idx = self.config.vocab_list.index("<mask>")
        self.esm_dict_eos_idx = self.config.vocab_list.index("<eos>")
        self.esm_dict_padding_idx = self.config.vocab_list.index("<pad>")
        if self.config.esmfold_config.embed_aa:
            self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        trunk_cfg_dict = self.config.esmfold_config.trunk

        self.trunk = EsmFoldingTrunk(trunk_cfg_dict)

        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        self.lddt_bins = 50
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(self.config.esmfold_config.trunk.structure_module.c_s),
            nn.Linear(
                self.config.esmfold_config.trunk.structure_module.c_s, self.config.esmfold_config.lddt_head_hid_dim
            ),
            nn.Linear(self.config.esmfold_config.lddt_head_hid_dim, self.config.esmfold_config.lddt_head_hid_dim),
            nn.Linear(self.config.esmfold_config.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    @staticmethod
    def _af2_to_esm_from_vocab_list(vocab_list: List[str]) -> torch.Tensor:
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [vocab_list.index("<pad>")] + [vocab_list.index(v) for v in residue_constants.restypes_with_x]
        return torch.tensor(esm_reorder)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_aa: bool = False,
        residx: Optional[torch.Tensor] = None,
        masking_pattern: Optional[torch.Tensor] = None,
    ):
        cfg = self.config.esmfold_config

        aa = input_ids  # B x L
        B = aa.shape[0]
        L = aa.shape[1]
        device = input_ids.device
        if residx is None:
            residx = torch.arange(L, device=device).expand_as(input_ids)

        # === ESM ===
        esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)

        if (self.training or mask_aa) and masking_pattern is not None:
            masked_aa, esmaa, mlm_targets = self.bert_mask(aa, esmaa, attention_mask, masking_pattern)
        else:
            masked_aa = aa
            mlm_targets = None

        # We get sequence and pair representations from whatever version of ESM /
        # configuration we are using. The sequence representation esm_s is always
        # present. The pair embedding esm_z may be present depending on the
        # configuration of the model. If esm_z is not used by the model then it
        # is returned as None here.
        esm_s, esm_z = self.compute_language_model_representations(esmaa)

        # Convert esm_s and esm_z, if present, to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esm_s_combine.dtype)
        esm_z = esm_z.to(self.esm_s_combine.dtype) if esm_z is not None else None

        if cfg.esm_ablate_sequence:
            esm_s = esm_s * 0

        if cfg.esm_ablate_pairwise:
            esm_z = esm_z * 0 if esm_z is not None else None

        esm_s = esm_s.detach()
        esm_z = esm_z.detach() if esm_z is not None else None

        # === preprocessing ===
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.esm_s_mlp(esm_s)

        if cfg.use_esm_attn_map:
            s_z_0 = self.esm_z_mlp(esm_z)
        else:
            s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

        if self.config.esmfold_config.embed_aa:
            s_s_0 += self.embedding(masked_aa)

        structure: dict = self.trunk(s_s_0, s_z_0, aa, residx, attention_mask)
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
            ]
        }

        # Add BERT mask for the loss to use, if available.
        if mlm_targets:
            structure["mlm_targets"] = mlm_targets

        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits

        structure["aatype"] = aa
        make_atom14_masks(structure)
        # Of course, this doesn't respect the true mask because it doesn't know about it...
        # We're not going to properly mask change of index tensors:
        #    "residx_atom14_to_atom37",
        #    "residx_atom37_to_atom14",
        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= attention_mask.unsqueeze(-1)
        structure["residue_index"] = residx

        lddt_head = self.lddt_head(structure["states"]).reshape(structure["states"].shape[0], B, L, -1, self.lddt_bins)
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
        structure["plddt"] = plddt

        ptm_logits = self.ptm_head(structure["s_z"])
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
        structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins))

        return structure

    def af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def compute_language_model_representations(self, esmaa: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        B, L = esmaa.shape  # B = batch size, L = sequence length.

        if self.config.esmfold_config.bypass_lm:
            esm_s = torch.zeros(B, L, self.esm_s_combine.size[0], -1, self.esm_feats, device=device)
            esm_z = (
                torch.zeros(B, L, L, self.esm_attns, device=device)
                if self.config.esmfold_config.use_esm_attn_map
                else None
            )
            return esm_s, esm_z

        bosi, eosi = self.esm_dict_cls_idx, self.esm_dict_eos_idx
        bos = esmaa.new_full((B, 1), bosi)
        eos = esmaa.new_full((B, 1), self.esm_dict_padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(B), (esmaa != 1).sum(1)] = eosi

        # _, esm_z, esm_s = self.esm(esmaa, return_pairs=self.config.esmfold_config.use_esm_attn_map)
        # Because we do not support use_esm_attn_map in the HF port as it is not used in any public models,
        # esm_z is always None
        esm_hidden_states = self.esm(esmaa, attention_mask=esmaa != 1, output_hidden_states=True)["hidden_states"]
        esm_s = torch.stack(esm_hidden_states, dim=2)
        esm_z = None

        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
        if esm_z is not None:
            esm_z = esm_z[:, 1:-1, 1:-1, :]  # B, L, L, C

        return esm_s, esm_z

    def bert_mask(self, aa, esmaa, mask, pattern):
        new_aa = aa.clone()
        target = aa.clone()
        new_esmaa = esmaa.clone()
        new_aa[pattern == 1] = self.mask_idx
        target[pattern != 1] = 0
        new_esmaa[pattern == 1] = self.esm_dict_mask_idx
        return new_aa, new_esmaa, target

    @torch.no_grad()
    def infer(
        self,
        seqs: Union[str, List[str]],
        residx=None,
        with_mask: Optional[torch.Tensor] = None,
    ):
        if type(seqs) is str:
            lst = [seqs]
        else:
            lst = seqs
        # Returns the raw outputs of the model given an input sequence.
        device = next(self.parameters()).device
        aatype = collate_dense_tensors(
            [
                torch.from_numpy(
                    residue_constants.sequence_to_onehot(
                        sequence=seq,
                        mapping=residue_constants.restype_order_with_x,
                        map_unknown_to_x=True,
                    )
                )
                .to(device)
                .argmax(dim=1)
                for seq in lst
            ]
        )  # B=1 x L
        mask = collate_dense_tensors([aatype.new_ones(len(seq)) for seq in lst])
        residx = (
            torch.arange(aatype.shape[1], device=device).expand(len(lst), -1) if residx is None else residx.to(device)
        )
        if residx.ndim == 1:
            residx = residx.unsqueeze(0)
        return self.forward(
            aatype,
            mask,
            mask_aa=with_mask is not None,
            masking_pattern=with_mask,
            residx=residx,
        )

    @staticmethod
    def output_to_pdb(output: Dict) -> List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        output = {k: v.to("cpu").numpy() for k, v in output.items()}
        pdbs = []
        final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
        final_atom_mask = output["atom37_atom_exists"]
        for i in range(output["aatype"].shape[0]):
            aa = output["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = output["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=output["plddt"][i],
            )
            pdbs.append(to_pdb(pred))
        return pdbs

    def infer_pdb(self, seqs, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        assert type(seqs) is str
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)[0]

    def infer_pdbs(self, seqs: List[str], *args, **kwargs) -> List[str]:
        """Returns the pdb (file) string from the model given an input sequence."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)


#
# # TODO(@ebetica): Once v0.3b is done training, let's figure out how to disseminate and release it.
# # DO NOT release these weights to the public
# def load_esmfold():
#     model = ESMFold(OmegaConf.structured(ESMFoldConfig()))
#     ckpt = torch.load(
#         "/large_experiments/protein/checkpoints/ssf/esmfold-esm2_3B-v0.3/finetune/stripped-last.pt"
#     )
#     model.load_state_dict(ckpt, strict=False)
#     return model.eval()
