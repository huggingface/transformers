# coding=utf-8
# Copyright 2026 Biohub. All rights reserved.
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
"""ESMFold2 experimental variant — older architecture from during the
development of ESMFold2.

Most users want :class:`ESMFold2Model` (in ``modeling_esmfold2``) instead;
this module retains an explicit refinement loop that re-injects the
previous pair representation through ``pair_loop_proj`` each iteration,
and exists to load checkpoints predating the standard architecture.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...modeling_utils import PreTrainedModel  # type: ignore[import]
from .configuration_esmfold2 import ESMFold2Config
from .modeling_esmfold2_common import (
    CHAR_VOCAB_SIZE,
    MAX_ATOMIC_NUMBER,
    NUM_RES_TYPES,
    DiffusionStructureHead,
    FoldingTrunk,
    InputsEmbedder,
    LanguageModelShim,
    MSAPairWeightedAveraging,
    OuterProductMean,
    ResIdxAsymIdSymIdEntityIdEncoding,
    RowAttentionPooling,
    SwiGLUMLP,
    TriangleMultiplicativeUpdate,
    _categorical_mean,
    _compute_intra_token_idx,
    _seed_context,
    compute_lm_hidden_states,
    gather_rep_atom_coords,
    gather_token_to_atom,
)

_EPS = 1e-5
_NONPOLYMER_ID: int = 3


# ===========================================================================
# ConfidenceHead
# ===========================================================================


class ConfidenceHead(nn.Module):
    """Confidence head predicting per-atom pLDDT and pairwise PAE."""

    boundaries: Tensor

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__()
        ch = config.confidence_head
        d_single = config.d_single
        d_pair = config.d_pair
        d_inputs = config.inputs.d_inputs

        # Distogram bins boundary buffer
        boundaries = torch.linspace(ch.min_dist, ch.max_dist, ch.distogram_bins - 1)
        self.register_buffer("boundaries", boundaries)

        self.dist_bin_pairwise_embed = nn.Embedding(ch.distogram_bins, d_pair)

        self.s_norm = nn.LayerNorm(d_single)

        self.s_inputs_to_single = nn.Linear(d_inputs, d_single, bias=False)

        self.s_to_z = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_transpose = nn.Linear(d_inputs, d_pair, bias=False)

        # s_to_z_prod
        self.s_to_z_prod_in1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_out = nn.Linear(d_pair, d_pair, bias=False)

        # s_input_to_s
        self.s_input_to_s = nn.Linear(d_inputs, d_single, bias=False)

        self.s_inputs_norm = nn.LayerNorm(d_inputs)
        self.z_norm = nn.LayerNorm(d_pair)

        # Row attention pooling
        self.row_attention_pooling = RowAttentionPooling(
            d_pair=d_pair, d_single=d_single
        )

        # Confidence folding trunk (4 blocks)
        pf = ch.folding_trunk
        self.folding_trunk = FoldingTrunk(
            n_layers=pf.n_layers, d_pair=d_pair, expansion_ratio=4
        )

        # pLDDT head
        self.plddt_ln = nn.LayerNorm(d_single)
        max_atoms_per_token = 23
        self.plddt_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, d_single, ch.num_plddt_bins)
        )

        # PAE head
        self.pae_head = nn.Linear(d_pair, ch.num_pae_bins, bias=False)

    # ------------------------------------------------------------------
    # Kernel / chunking configuration
    # ------------------------------------------------------------------

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _repeat_batch(x: Tensor, num_diffusion_samples: int) -> Tensor:
        if num_diffusion_samples == 1:
            return x
        return x.repeat_interleave(num_diffusion_samples, 0)

    @staticmethod
    def _flatten_sample_axis(x: Tensor) -> Tensor:
        if x.ndim == 4:
            b, mult, n, c = x.shape
            return x.reshape(b * mult, n, c)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        s_inputs: Tensor,
        z: Tensor,
        x_pred: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        asym_id: Tensor,
        mol_type: Tensor,
        num_diffusion_samples: int = 1,
        relative_position_encoding: Tensor | None = None,
        token_bonds_encoding: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run confidence head."""
        # Shared computation (batch-scale, before num_diffusion_samples expansion)
        s_inputs_normed = self.s_inputs_norm(s_inputs)

        z_base = self.z_norm(z)
        if relative_position_encoding is not None:
            z_base = z_base + relative_position_encoding
        if token_bonds_encoding is not None:
            z_base = z_base + token_bonds_encoding
        z_base = z_base + self.s_to_z(s_inputs_normed).unsqueeze(2)
        z_base = z_base + self.s_to_z_transpose(s_inputs_normed).unsqueeze(1)
        z_base = z_base + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(s_inputs_normed)[:, :, None, :]
            * self.s_to_z_prod_in2(s_inputs_normed)[:, None, :, :]
        )

        # Expand to num_diffusion_samples
        pair = self._repeat_batch(z_base, num_diffusion_samples)
        x_pred_flat = self._flatten_sample_axis(x_pred)
        atom_to_token_m = self._repeat_batch(atom_to_token, num_diffusion_samples)
        atom_mask_m = self._repeat_batch(atom_attention_mask, num_diffusion_samples)
        rep_idx_m = self._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
        mask = self._repeat_batch(token_attention_mask, num_diffusion_samples)
        Bm = pair.shape[0]

        # Distogram from predicted coords
        rep_coords = gather_rep_atom_coords(x_pred_flat, rep_idx_m)
        rep_distances = torch.cdist(
            rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist"
        )
        distogram_bins = (
            (rep_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        )
        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)

        # Expand 1-D token mask → 2-D pair mask for folding trunk
        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()  # [B*m, L, L]

        # FoldingTrunk + row attention pooling -> single
        pair = pair + self.folding_trunk(pair, pair_attention_mask=pair_mask)
        single = self.row_attention_pooling(pair, mask)

        # Per-atom pLDDT
        atom_mask_f = atom_mask_m.float()
        s_at_atoms = gather_token_to_atom(single, atom_to_token_m)
        s_at_atoms = self.plddt_ln(s_at_atoms)

        intra_idx = _compute_intra_token_idx(atom_to_token_m)
        intra_idx = intra_idx.clamp(max=self.plddt_weight.shape[0] - 1)
        w = self.plddt_weight[intra_idx]  # [B*m, A, d_single, num_bins]
        plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms, w)

        plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

        # Per-token pLDDT (scatter mean)
        L = single.shape[1]
        plddt_sum = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
        atom_count = torch.zeros(
            Bm, L, device=single.device, dtype=plddt_per_atom.dtype
        )
        atom_mask_t = atom_mask_f.to(plddt_per_atom.dtype)
        plddt_sum.scatter_add_(1, atom_to_token_m, plddt_per_atom * atom_mask_t)
        atom_count.scatter_add_(1, atom_to_token_m, atom_mask_t)
        plddt = plddt_sum / atom_count.clamp(min=1e-6)

        # Complex pLDDT (flat mean over all atoms)
        complex_plddt = (plddt_per_atom * atom_mask_f).sum(dim=-1) / (
            atom_mask_f.sum(dim=-1) + _EPS
        )

        # Complex ipLDDT (interface-weighted)
        expanded_type = self._repeat_batch(mol_type, num_diffusion_samples)
        expanded_asym = self._repeat_batch(asym_id, num_diffusion_samples)
        is_ligand = (expanded_type == _NONPOLYMER_ID).float()
        inter_chain = (
            expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)
        ).float()
        near_contact = (rep_distances < 8).float()
        interface_per_token = (
            near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)
        ).amax(dim=-1)
        iplddt_weight = torch.where(
            is_ligand.bool(),
            torch.full_like(interface_per_token, 2.0),
            interface_per_token,
        )
        iplddt_weight_atoms = gather_token_to_atom(
            iplddt_weight.unsqueeze(-1), atom_to_token_m
        ).squeeze(-1)
        atom_iplddt_w = atom_mask_f * iplddt_weight_atoms
        complex_iplddt = (plddt_per_atom * atom_iplddt_w).sum(dim=-1) / (
            atom_iplddt_w.sum(dim=-1) + _EPS
        )

        # pLDDT at CA / representative atom
        plddt_ca = plddt_per_atom.gather(1, rep_idx_m)

        # PAE
        pae_logits = self.pae_head(pair)
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0).detach()

        # pTM / ipTM / per-chain-pair ipTM derived from pae_logits.
        n_bins = pae_logits.shape[-1]
        bin_width = 32.0 / n_bins
        bin_centers = torch.arange(
            0.5 * bin_width, 32.0, bin_width, device=pae_logits.device
        )
        mask_f = mask.float()
        N_res = mask_f.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (N_res.clamp(min=19) - 15) ** (1 / 3) - 1.8  # [Bm, 1]
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)  # [Bm, n_bins]
        pae_probs = F.softmax(pae_logits, dim=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(
            dim=-1
        )  # [Bm, L, L]

        pair_mask_2d = mask_f.unsqueeze(-1) * mask_f.unsqueeze(-2)  # [Bm, L, L]

        # pTM: avg over all valid pairs per row, max over rows.
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (
            pair_mask_2d.sum(dim=-1) + _EPS
        )
        ptm = ptm_per_row.max(dim=-1).values  # [Bm]

        # ipTM: avg over inter-chain valid pairs per row, max over rows.
        inter_chain_mask = (
            expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)
        ).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (
            inter_chain_mask.sum(dim=-1) + _EPS
        )
        iptm = iptm_per_row.max(dim=-1).values  # [Bm]

        # Per-chain-pair ipTM: dense [Bm, N_chains, N_chains] padded to max chain id + 1.
        max_chain_id = int(expanded_asym.max().item()) if Bm > 0 else 0
        n_chains = max_chain_id + 1
        pair_chains_iptm = torch.zeros(
            Bm, n_chains, n_chains, device=tm_expected.device, dtype=tm_expected.dtype
        )
        for c1 in range(n_chains):
            chain_c1 = (expanded_asym == c1).float() * mask_f
            if chain_c1.sum() == 0:
                continue
            for c2 in range(n_chains):
                chain_c2 = (expanded_asym == c2).float() * mask_f
                pair_m = chain_c1.unsqueeze(-1) * chain_c2.unsqueeze(-2)
                denom = pair_m.sum(dim=(-1, -2)) + _EPS
                pair_chains_iptm[:, c1, c2] = (tm_expected * pair_m).sum(
                    dim=(-1, -2)
                ) / denom

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt.detach(),
            "plddt_per_atom": plddt_per_atom.detach(),
            "plddt_ca": plddt_ca.detach(),
            "complex_plddt": complex_plddt.detach(),
            "complex_iplddt": complex_iplddt.detach(),
            "pae_logits": pae_logits,
            "pae": pae,
            "ptm": ptm.detach(),
            "iptm": iptm.detach(),
            "pair_chains_iptm": pair_chains_iptm.detach(),
        }


# ===========================================================================
# MSA Encoder
# ===========================================================================


class _TransitionFFN(nn.Module):
    """LayerNorm + SwiGLU FFN without residual (used inside MSAEncoderBlock)."""

    def __init__(self, d_model: int, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUMLP(d_model, expansion_ratio=expansion_ratio, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn(self.norm(x))


class MSAEncoderBlock(nn.Module):
    """One block of the MSA encoder: MSA update + pair update."""

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hidden: int = 32,
        n_heads_msa: int = 8,
        msa_head_width: int = 32,
    ) -> None:
        super().__init__()
        self.outer_product_mean = OuterProductMean(
            d_msa, d_hidden, d_pair, divide_outer_before_proj=True
        )
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            d_msa, d_pair, n_heads_msa, msa_head_width
        )
        self.msa_transition = _TransitionFFN(d_msa, expansion_ratio=4)
        self.tri_mul_out = TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=False)
        self.pair_transition = _TransitionFFN(d_pair, expansion_ratio=4)

    def forward(
        self,
        msa_repr: Tensor,
        pair_repr: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
        msa_track_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            msa_repr:           [B, L, M, d_msa]
            pair_repr:          [B, L, L, d_pair]
            msa_attention_mask: [B, L, M]
            pair_attention_mask:[B, L, L]
            msa_track_mask:     [B] bool — if False for a sample, zero out its contribution
        Returns:
            (msa_repr, pair_repr)
        """
        mask4d = (
            msa_track_mask[:, None, None, None].to(dtype=msa_repr.dtype)
            if msa_track_mask is not None
            else None
        )

        def _maybe_mask(x: Tensor) -> Tensor:
            return x * mask4d if mask4d is not None else x

        msa_repr = msa_repr + _maybe_mask(
            self.msa_pair_weighted_averaging(msa_repr, pair_repr, pair_attention_mask)
        )
        msa_repr = msa_repr + _maybe_mask(self.msa_transition(msa_repr))

        pair_repr = pair_repr + _maybe_mask(
            self.outer_product_mean(msa_repr, msa_attention_mask)
        )
        pair_repr = pair_repr + _maybe_mask(
            self.tri_mul_out(pair_repr, mask=pair_attention_mask)
        )
        pair_repr = pair_repr + _maybe_mask(
            self.tri_mul_in(pair_repr, mask=pair_attention_mask)
        )
        pair_repr = pair_repr + _maybe_mask(self.pair_transition(pair_repr))

        return msa_repr, pair_repr


class MSAEncoder(nn.Module):
    """Embeds MSA features and runs encoder blocks to update the pair representation."""

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_inputs: int,
        d_hidden: int = 32,
        n_layers: int = 4,
        n_heads_msa: int = 8,
        msa_head_width: int = 32,
    ) -> None:
        super().__init__()
        # 33 aa one-hot + has_deletion + deletion_value = 35
        self.embed = nn.Linear(35, d_msa, bias=False)
        self.project_inputs = nn.Linear(d_inputs, d_msa, bias=False)
        self.blocks = nn.ModuleList(
            [
                MSAEncoderBlock(
                    d_msa=d_msa,
                    d_pair=d_pair,
                    d_hidden=d_hidden,
                    n_heads_msa=n_heads_msa,
                    msa_head_width=msa_head_width,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x_pair: Tensor,
        x_inputs: Tensor,
        msa_oh: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_attention_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            x_pair:            [B, L, L, d_pair]   current pair representation
            x_inputs:          [B, L, d_inputs]     per-token input features
            msa_oh:            [B, L, M, 33]        one-hot MSA (already transposed)
            has_deletion:      [B, L, M]
            deletion_value:    [B, L, M]
            msa_attention_mask:[B, L, M]
        Returns:
            [B, L, L, d_pair] pair update
        """
        B, L, M = msa_attention_mask.shape

        m_feat = torch.cat(
            [msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1
        )
        m = self.embed(m_feat) + self.project_inputs(x_inputs).unsqueeze(2)

        # Mask out the full update for samples with no real non-query MSA rows.
        if M > 1:
            msa_track_mask = msa_attention_mask[:, :, 1:].any(dim=(1, 2))
        else:
            msa_track_mask = torch.zeros(B, dtype=torch.bool, device=x_pair.device)

        tok_mask = msa_attention_mask[:, :, 0]
        pair_attention_mask = tok_mask.unsqueeze(2) * tok_mask.unsqueeze(1)

        for block in self.blocks:
            m, x_pair = block(
                m, x_pair, msa_attention_mask, pair_attention_mask, msa_track_mask
            )

        x_pair = x_pair * msa_track_mask[:, None, None, None].to(dtype=x_pair.dtype)
        return x_pair


# ===========================================================================
# ESMFold2ExperimentalModel — the top-level PreTrainedModel
# ===========================================================================


class ESMFold2ExperimentalModel(PreTrainedModel):
    """ESMFold2 v2 structure prediction model."""

    config_class = ESMFold2Config

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__(config)

        # InputsEmbedder
        self.inputs_embedder = InputsEmbedder(config)

        # z_init projections
        d_inputs = config.inputs.d_inputs
        d_pair = config.d_pair

        self.z_init_1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.z_init_2 = nn.Linear(d_inputs, d_pair, bias=False)

        # Trunk relative position encoding
        self.rel_pos = ResIdxAsymIdSymIdEntityIdEncoding(
            n_relative_residx_bins=config.n_relative_residx_bins,
            n_relative_chain_bins=config.n_relative_chain_bins,
            d_pair=d_pair,
        )

        # Token bonds
        self.token_bonds = nn.Linear(1, d_pair, bias=False)

        self.language_model = LanguageModelShim(
            d_z=d_pair, d_model=config.lm_d_model, num_layers=config.lm_num_layers
        )
        self._esmc: nn.Module | None = None  # lazily loaded

        # FoldingTrunk
        pf = config.folding_trunk
        self.folding_trunk = FoldingTrunk(
            n_layers=pf.n_layers, d_pair=d_pair, expansion_ratio=4
        )

        # Per-loop pair re-injection projection
        self.pair_loop_proj = nn.Sequential(
            nn.LayerNorm(d_pair), nn.Linear(d_pair, d_pair, bias=False)
        )
        nn.init.zeros_(self.pair_loop_proj[1].weight)  # type: ignore[arg-type]

        # Structure head
        self.structure_head = DiffusionStructureHead(config)

        # Distogram head
        self.distogram_head = nn.Linear(
            d_pair, config.structure_head.distogram_bins, bias=True
        )

        if config.confidence_head.enabled:
            self.confidence_head: ConfidenceHead | None = ConfidenceHead(config)
        else:
            self.confidence_head = None

        # MSA encoder (Large MSA models only)
        msa_cfg = config.msa_encoder
        if msa_cfg.enabled:
            self.msa_encoder: MSAEncoder | None = MSAEncoder(
                d_msa=msa_cfg.d_msa,
                d_pair=d_pair,
                d_inputs=d_inputs,
                d_hidden=msa_cfg.d_hidden,
                n_layers=msa_cfg.n_layers,
                n_heads_msa=msa_cfg.n_heads_msa,
                msa_head_width=msa_cfg.msa_head_width,
            )
        else:
            self.msa_encoder = None

        self.post_init()

    def set_chunk_size(self, chunk_size: int | None) -> None:
        """Set chunk size for memory-efficient triangle multiplicative updates."""
        self.folding_trunk.set_chunk_size(chunk_size)
        if self.confidence_head is not None:
            self.confidence_head.set_chunk_size(chunk_size)

    def configure_lm_dropout(
        self, lm_dropout: float, *, force_lm_dropout_during_inference: bool = True
    ) -> None:
        """Configure LM embedding dropout (binder-design / critic scoring)."""
        self.config.lm_dropout = lm_dropout
        self.config.force_lm_dropout_during_inference = (
            force_lm_dropout_during_inference
        )

    def load_esmc(self, esmc_model_path: str) -> None:
        """Load the ESMC LM backbone from a HuggingFace Hub repo ID or local directory."""
        from ..esmc.modeling_esmc import ESMCModel  # type: ignore[import]

        esmc = ESMCModel.from_pretrained(esmc_model_path)
        self._esmc = esmc.bfloat16().to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *args, load_esmc: bool = True, **kwargs
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if load_esmc:
            model.load_esmc(model.config.esmc_id)
        return model

    @torch.no_grad()
    def infer_protein(self, seq: str, **forward_kwargs) -> dict:
        from .protein_utils import OUTPUT_TO_PDB_FEATURE_KEYS, prepare_protein_features

        features = prepare_protein_features(seq)
        features = {k: v.to(self.device) for k, v in features.items()}
        output = self(**features, **forward_kwargs)
        for k in OUTPUT_TO_PDB_FEATURE_KEYS:
            output[k] = features[k]
        return output

    def infer_protein_as_pdb(self, seq: str, **forward_kwargs) -> str:
        return self.output_to_pdb(self.infer_protein(seq, **forward_kwargs))

    @torch.no_grad()
    def infer_all_atom(self, structure_input, **forward_kwargs):
        try:
            import evolutionaryscale.opensource as esm  # TODO: change to `import esm` when open sourcing
            import evolutionaryscale.opensource.models.esmfold2  # noqa: F401  # TODO: drop when open sourcing
        except ImportError as e:
            raise NotImplementedError(
                "All-atom inference requires the `esm` companion package: "
                "`pip install esm`."
            ) from e
        esmfold2 = esm.models.esmfold2  # type: ignore[attr-defined]

        if isinstance(structure_input, esmfold2.ProteinInput):
            structure_input = esmfold2.StructurePredictionInput(
                sequences=[structure_input]
            )
        processor = esmfold2.ESMFold2InputBuilder()
        features, chain_infos = processor.prepare_input(structure_input)
        features = {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in features.items()
        }
        output = self(**features, **forward_kwargs)
        return self._output_to_molecular_complex(output, features, chain_infos)

    @staticmethod
    def _output_to_molecular_complex(output: dict, features: dict, chain_infos: list):
        import evolutionaryscale.opensource as esm  # TODO: change to `import esm` when open sourcing
        import evolutionaryscale.opensource.models.esmfold2  # noqa: F401  # TODO: drop when open sourcing

        esmfold2 = esm.models.esmfold2  # type: ignore[attr-defined]

        ELEMENT_NUMBER_TO_SYMBOL = esmfold2.ELEMENT_NUMBER_TO_SYMBOL
        MolecularComplex = esmfold2.MolecularComplex
        MolecularComplexMetadata = esmfold2.MolecularComplexMetadata

        coords = output["sample_atom_coords"]
        if coords.dim() == 4:
            coords = coords[:, 0]
        coords_np = coords.detach().cpu().numpy()

        plddt = output["plddt"].detach().cpu().numpy()
        atom_to_token = features["atom_to_token"].cpu().numpy()
        ref_chars = features["ref_atom_name_chars"].cpu().numpy()
        ref_element = features["ref_element"].cpu().numpy()
        atom_mask = features["atom_attention_mask"].cpu().numpy().astype(bool)

        if atom_to_token.ndim == 1:
            atom_to_token = atom_to_token[None]
            ref_chars = ref_chars[None]
            ref_element = ref_element[None]
            atom_mask = atom_mask[None]

        b = 0
        atoms_by_token: dict[int, list[int]] = {}
        for a in range(atom_to_token.shape[1]):
            if not atom_mask[b, a]:
                continue
            atoms_by_token.setdefault(int(atom_to_token[b, a]), []).append(a)

        flat_positions: list[np.ndarray] = []
        flat_elements: list[str] = []
        flat_names: list[str] = []
        flat_hetero: list[bool] = []
        sequence_tokens: list[str] = []
        token_to_atoms: list[list[int]] = []
        chain_ids_per_token: list[int] = []
        confidence_scores: list[float] = []
        chain_lookup: dict[int, str] = {}
        entity_info: dict[int, str] = {}

        cursor = 0
        for chain in chain_infos:
            chain_lookup[chain.asym_id] = chain.chain_id
            entity_info[chain.entity_id] = (
                "polymer" if chain.mol_type != 3 else "non-polymer"
            )
            for tok in chain.tokens:
                sequence_tokens.append(tok.residue_name)
                chain_ids_per_token.append(chain.asym_id)
                confidence_scores.append(float(plddt[b, tok.token_index]))
                start = cursor
                for a in atoms_by_token.get(tok.token_index, []):
                    flat_positions.append(coords_np[b, a])
                    name = "".join(
                        chr(int(c) + 32) if int(c) != 0 else " "
                        for c in ref_chars[b, a]
                    ).strip()
                    flat_names.append(name)
                    flat_elements.append(
                        ELEMENT_NUMBER_TO_SYMBOL.get(int(ref_element[b, a]), "X")
                    )
                    flat_hetero.append(chain.mol_type == 3)
                    cursor += 1
                token_to_atoms.append([start, cursor])

        return MolecularComplex(
            id="prediction",
            sequence=sequence_tokens,
            atom_positions=np.array(flat_positions, dtype=np.float32),
            atom_elements=np.array(flat_elements, dtype=object),
            token_to_atoms=np.array(token_to_atoms, dtype=np.int32),
            chain_id=np.array(chain_ids_per_token, dtype=np.int64),
            plddt=np.array(confidence_scores, dtype=np.float32),
            atom_names=np.array(flat_names, dtype=object),
            atom_hetero=np.array(flat_hetero, dtype=bool),
            metadata=MolecularComplexMetadata(
                entity_lookup={k: str(v) for k, v in entity_info.items()},
                chain_lookup=chain_lookup,
                assembly_composition=None,
            ),
        )

    @staticmethod
    def output_to_pdb(output: dict) -> str:
        from .protein_utils import output_to_pdb as _output_to_pdb

        return _output_to_pdb(output)

    def _compute_lm_hidden_states(
        self,
        input_ids: Tensor,
        asym_id: Tensor,
        residue_index: Tensor,
        mol_type: Tensor,
        token_mask: Tensor,
    ) -> Tensor:
        """Run ESMC with BOS/EOS wrapping, return hidden states [B, L, N, D] with N=81 layers."""
        assert self._esmc is not None
        return compute_lm_hidden_states(
            self._esmc, input_ids, asym_id, residue_index, mol_type, token_mask
        )

    def forward(
        self,
        # Token features
        token_index: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        mol_type: Tensor,
        res_type: Tensor,
        token_bonds: Tensor,
        token_attention_mask: Tensor,
        # Atom features
        ref_pos: Tensor,
        ref_element: Tensor,
        ref_charge: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
        distogram_atom_idx: Tensor,
        # MSA features
        deletion_mean: Tensor | None = None,
        msa: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        # LM features (auto-computed from input_ids if ESMC loaded)
        input_ids: Tensor | None = None,
        lm_hidden_states: Tensor | None = None,
        # Used in design to provide a soft sequence input.
        res_type_soft: Tensor | None = None,
        # Inference config
        num_loops: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        early_exit: bool = False,
        seed: int | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Full ESMFold2 inference pipeline.

        Accepts tensors directly from ESMFold2InputBuilder.prepare_input().

        Returns:
            dict with sample_atom_coords, plddt, pae, distogram_logits, etc.
        """
        tok_mask = token_attention_mask
        atm_mask = atom_attention_mask
        disto_idx = distogram_atom_idx

        n_loops: int = num_loops if num_loops is not None else self.config.num_loops
        n_samples: int = (
            num_diffusion_samples
            if num_diffusion_samples is not None
            else self.config.num_diffusion_samples
        )

        # One-hot res_type for input embedder concatenation
        if res_type.dim() == 2:
            res_type_oh = F.one_hot(res_type.long(), num_classes=NUM_RES_TYPES).float()
            res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()
        else:
            res_type_oh = res_type.float()

        # Profile: masked mean over MSA depth, with res_type fallback when no MSA.
        if msa is not None:
            msa_oh_profile = F.one_hot(
                msa.long(), num_classes=NUM_RES_TYPES
            ).float()  # [B, M, L, V]
            if msa_attention_mask is not None:
                mask_f = msa_attention_mask.float().unsqueeze(-1)  # [B, M, L, 1]
                msa_oh_profile = msa_oh_profile * mask_f
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                profile = msa_oh_profile.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                profile = msa_oh_profile.mean(dim=1)
        else:
            profile = res_type_oh

        # Used in design to provide a soft sequence input.
        if res_type_soft is not None:
            res_type_oh = res_type_soft.float()
            if not getattr(self.config, "disable_msa_features", False) and kwargs.get(
                "provide_soft_sequence_to_msa_and_profile", True
            ):
                profile = res_type_oh
                msa = res_type_oh.unsqueeze(1)
                msa_attention_mask = tok_mask.unsqueeze(1)

        if deletion_mean is None:
            deletion_mean = torch.zeros(
                res_type.shape[0], res_type.shape[1], device=res_type.device
            )

        if getattr(self.config, "disable_msa_features", False):
            profile = torch.zeros_like(profile)
            deletion_mean = torch.zeros_like(deletion_mean)

        ref_element = F.one_hot(
            ref_element.long(), num_classes=MAX_ATOMIC_NUMBER
        ).float()
        ref_atom_name_chars = F.one_hot(
            ref_atom_name_chars.long(), num_classes=CHAR_VOCAB_SIZE
        ).float()
        # Bias-free downstream Linears require zeroed padding.
        atm_mask_f = atm_mask.float()
        ref_element = ref_element * atm_mask_f.unsqueeze(-1)
        ref_atom_name_chars = ref_atom_name_chars * atm_mask_f.unsqueeze(-1).unsqueeze(
            -1
        )

        atom_to_token = atom_to_token * atm_mask.long()

        use_amp = ref_pos.device.type == "cuda"
        with (
            torch.set_grad_enabled(res_type_soft is not None),
            torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16),
        ):
            # 1. Input embeddings
            x_inputs = self.inputs_embedder(
                aatype=res_type_oh,
                profile=profile.float(),
                deletion_mean=deletion_mean.float(),
                ref_pos=ref_pos,
                atom_attention_mask=atm_mask,
                ref_space_uid=ref_space_uid,
                ref_charge=ref_charge,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                atom_to_token=atom_to_token,
            )

            # 2. Initialize pair representation
            z_init = self.z_init_1(x_inputs).unsqueeze(2) + self.z_init_2(
                x_inputs
            ).unsqueeze(1)

            # 3. Positional encodings
            relative_position_encoding = self.rel_pos(
                residue_index=residue_index,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                token_index=token_index,
            )
            token_bonds_encoding = self.token_bonds(token_bonds.float())
            z_init = z_init + relative_position_encoding + token_bonds_encoding

            # 4. Language model integration
            if (
                lm_hidden_states is None
                and input_ids is not None
                and self._esmc is not None
            ):
                lm_hidden_states = self._compute_lm_hidden_states(
                    input_ids, asym_id, residue_index, mol_type, tok_mask
                )
            if lm_hidden_states is not None:
                lm_z = self.language_model(
                    lm_hidden_states.detach(), lm_dropout=self.config.lm_dropout
                )
                z_init = z_init + lm_z.to(z_init.dtype)

            # MSA tensors prepared once: encoder consumes [B, L, M, ...] layout.
            _msa_kwargs: dict | None = None
            if self.msa_encoder is not None and msa is not None:
                if msa.dim() == 4:
                    B_msa, M, L_msa, _ = msa.shape
                    msa_oh = msa.permute(0, 2, 1, 3).float()
                else:
                    B_msa, M, L_msa = msa.shape
                    msa_oh = F.one_hot(
                        msa.permute(0, 2, 1).long(), num_classes=NUM_RES_TYPES
                    ).float()  # [B, L, M, 33]
                msa_attn = (
                    msa_attention_mask.permute(0, 2, 1).float()
                    if msa_attention_mask is not None
                    else tok_mask[:, :, None].expand(-1, -1, M).float()
                )
                # Bias-free MSAEncoder.embed requires zeroed padding.
                msa_oh = msa_oh * msa_attn.unsqueeze(-1)
                hd = (
                    has_deletion.permute(0, 2, 1).float()
                    if has_deletion is not None
                    else torch.zeros(B_msa, L_msa, M, device=msa.device)
                )
                dv = (
                    deletion_value.permute(0, 2, 1).float()
                    if deletion_value is not None
                    else torch.zeros(B_msa, L_msa, M, device=msa.device)
                )
                _msa_kwargs = dict(
                    x_inputs=x_inputs,
                    msa_oh=msa_oh,
                    has_deletion=hd,
                    deletion_value=dv,
                    msa_attention_mask=msa_attn,
                )

            # Expand 1-D token mask → 2-D pair mask for folding trunk
            pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()

            # Loop: pair-only folding trunk, MSA encoder runs inside each iteration.
            z = torch.zeros_like(z_init)
            prev_pair: Tensor | None = None
            prev_disto_probs: Tensor | None = None
            for loop_num in range(n_loops + 1):
                z = z_init + self.pair_loop_proj(z)
                if _msa_kwargs is not None and self.msa_encoder is not None:
                    z = z + self.msa_encoder(x_pair=z, **_msa_kwargs).to(z.dtype)
                z = self.folding_trunk(z, pair_attention_mask=pair_mask)

                # Loop early-exit
                if early_exit and loop_num < n_loops:
                    l2_converged = False
                    if prev_pair is not None and loop_num > 0:
                        rel_l2 = (
                            z.float() - prev_pair.float()
                        ).norm() / prev_pair.float().norm().clamp(min=1e-8)
                        l2_converged = rel_l2.item() < 0.25
                    prev_pair = z.detach().clone()

                    sym_z = z.float() + z.float().transpose(-2, -3)
                    cur_probs = F.softmax(self.distogram_head(sym_z).float(), dim=-1)
                    if prev_disto_probs is not None and loop_num > 0:
                        kl_per_pair = (
                            cur_probs
                            * (
                                cur_probs.clamp(min=1e-8)
                                / prev_disto_probs.clamp(min=1e-8)
                            ).log()
                        ).sum(-1)
                        kl = (kl_per_pair + kl_per_pair.transpose(-1, -2)).mean() / 2
                        if l2_converged or kl.item() < 0.05:
                            break
                    prev_disto_probs = cur_probs.detach()

            # 6. Distogram (inside the trunk autocast so z stays bf16)
            distogram_logits = self.distogram_head(z + z.transpose(-2, -3))

        # 7. Diffusion sampling (always no_grad; optional seed for parity)
        with torch.no_grad(), _seed_context(seed):
            structure_output = self.structure_head.sample(
                z_trunk=z.float(),
                s_inputs=x_inputs,
                s_trunk=None,
                relative_position_encoding=relative_position_encoding,
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_mask=atm_mask,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_space_uid=ref_space_uid,
                tok_idx=atom_to_token,
                asym_id=asym_id,
                residue_index=residue_index,
                entity_id=entity_id,
                token_index=token_index,
                sym_id=sym_id,
                token_attention_mask=tok_mask,
                num_diffusion_samples=n_samples,
                num_sampling_steps=num_sampling_steps,
                return_atom_repr=False,
                denoising_early_exit_rmsd=(0.10 if early_exit else None),
            )

            sample_coords = structure_output["sample_atom_coords"]
            assert sample_coords is not None
            output: dict[str, Tensor] = {"distogram_logits": distogram_logits}
            output["sample_atom_coords"] = sample_coords

            if self.confidence_head is not None:
                confidence_output = self.confidence_head(
                    s_inputs=x_inputs.detach(),
                    z=z.detach().float(),
                    x_pred=sample_coords.detach(),
                    distogram_atom_idx=disto_idx,
                    token_attention_mask=tok_mask,
                    atom_to_token=atom_to_token,
                    atom_attention_mask=atm_mask,
                    asym_id=asym_id,
                    mol_type=mol_type,
                    num_diffusion_samples=n_samples,
                    relative_position_encoding=relative_position_encoding.detach(),
                    token_bonds_encoding=token_bonds_encoding.detach(),
                )
                output.update(confidence_output)

            # Pass-through tensors used by output decoders.
            output["atom_pad_mask"] = (
                atm_mask.unsqueeze(0) if atm_mask.dim() == 1 else atm_mask
            )
            output["residue_index"] = residue_index
            output["entity_id"] = entity_id

            return output


__all__ = ["ESMFold2ExperimentalModel"]
