# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from .categorical_mixture import categorical_lddt
from .configuration_esm import EsmConfig
from .esmfold_trunk import FoldingTrunk
from .modeling_esm import EsmForMaskedLM, EsmModel, EsmPreTrainedModel
from .openfold_data_transforms import make_atom14_masks
from .openfold_np import residue_constants
from .openfold_np.protein import Protein as OFProtein
from .openfold_np.protein import to_pdb
from .openfold_utils.feats import atom14_to_atom37
from .openfold_utils.loss import compute_predicted_aligned_error, compute_tm


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


class EsmForProteinFolding(EsmPreTrainedModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config)

        self.config = config if config else EsmConfig(is_folding_model=True, **kwargs)

        self.distogram_bins = 64

        self.esm = EsmModel(config=config, add_pooling_layer=False)

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

        self.trunk = FoldingTrunk(trunk_cfg_dict)

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
        print(esm_hidden_states[0])
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
