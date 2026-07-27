# Copyright 2026 BioHub and The HuggingFace Inc. team. All rights reserved.
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

"""Generation / inference loop for ESMFold2, kept out of the modeling file.

``EsmFold2Model.forward`` runs the folding trunk and returns its pair representation; this mixin owns
the diffusion sampling loop that turns that representation into 3D coordinates, the confidence head
call on the sampled structure, and the public ``infer_protein`` entry points. It deliberately does NOT
use ``GenerationConfig`` or the ``Cache`` classes (ESMFold2 emits continuous coordinates, not tokens);
sampling hyperparameters are plain kwargs falling back to the model config, and the conditioning that
is constant across denoising steps is precomputed once into an ``EsmFold2DiffusionStepInvariants`` so
the module forwards stay free of caching branches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor


if TYPE_CHECKING:
    from .modeling_esmfold2 import EsmFold2AtomInputs, EsmFold2Output


class EsmFold2GenerationMixin:
    """Diffusion sampling loop and the ``infer_protein`` entry points for ESMFold2."""

    def fold(
        self,
        token_index: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        mol_type: Tensor,
        res_type: Tensor,
        token_bonds: Tensor,
        token_attention_mask: Tensor,
        ref_pos: Tensor,
        ref_element: Tensor,
        ref_charge: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
        distogram_atom_idx: Tensor,
        deletion_mean: Tensor | None = None,
        msa: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        **kwargs,
    ) -> EsmFold2Output:
        r"""
        Predict a structure end-to-end from featurized inputs: run the trunk
        ([`EsmFold2Model.forward`], which documents the feature arguments), sample coordinates from the
        diffusion structure head, and score them with the confidence head.

        distogram_atom_idx (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Index of the representative atom (Cβ, or Cα for glycine) of each token. Used by the
            confidence head; the trunk does not need it.
        num_loops (`int`, *optional*):
            Number of trunk refinement loops. Defaults to `config.num_loops`.
        num_diffusion_samples (`int`, *optional*):
            Number of parallel structure samples to draw; the confidence head re-runs once per sample.
            Defaults to `config.num_diffusion_samples`.
        num_sampling_steps (`int`, *optional*):
            Number of diffusion sampling steps. Defaults to `config.structure_head.inference_num_steps`.
        """
        from .modeling_esmfold2 import EsmFold2Output

        n_samples: int = (
            num_diffusion_samples if num_diffusion_samples is not None else self.config.num_diffusion_samples
        )

        trunk = self(
            token_index=token_index,
            residue_index=residue_index,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            mol_type=mol_type,
            res_type=res_type,
            token_bonds=token_bonds,
            token_attention_mask=token_attention_mask,
            ref_pos=ref_pos,
            ref_element=ref_element,
            ref_charge=ref_charge,
            ref_atom_name_chars=ref_atom_name_chars,
            ref_space_uid=ref_space_uid,
            atom_attention_mask=atom_attention_mask,
            atom_to_token=atom_to_token,
            deletion_mean=deletion_mean,
            msa=msa,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            msa_attention_mask=msa_attention_mask,
            input_ids=input_ids,
            lm_hidden_states=lm_hidden_states,
            num_loops=num_loops,
        )

        sample_coords = self._sample_structure(
            pair_trunk=trunk.pair_states,
            single_inputs=trunk.single_inputs,
            relative_position_encoding=trunk.relative_position_encoding,
            atom_inputs=trunk.atom_inputs,
            token_attention_mask=token_attention_mask,
            num_diffusion_samples=n_samples,
            num_sampling_steps=num_sampling_steps,
        )

        confidence_output = self.confidence_head(
            single_inputs=trunk.single_inputs.detach(),
            z=trunk.pair_states.detach(),
            x_pred=sample_coords.detach(),
            distogram_atom_idx=distogram_atom_idx,
            token_attention_mask=token_attention_mask,
            # The trunk's copy is the one zeroed at padding.
            atom_to_token=trunk.atom_inputs.atom_to_token,
            atom_attention_mask=atom_attention_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=n_samples,
            relative_position_encoding=trunk.relative_position_encoding.detach(),
            token_bonds_encoding=trunk.token_bonds_encoding.detach(),
        )

        return EsmFold2Output(
            distogram_logits=trunk.distogram_logits,
            sample_atom_coords=sample_coords,
            **confidence_output,
        )

    def _sample_structure(
        self,
        pair_trunk: Tensor,
        single_inputs: Tensor,
        relative_position_encoding: Tensor,
        atom_inputs: EsmFold2AtomInputs,
        token_attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        num_sampling_steps: int | None = None,
    ) -> Tensor:
        """Diffusion sampling (Algorithm 18): returns the sampled atom coordinates.

        ``num_sampling_steps`` is the number of denoising steps actually run. The remaining sampling
        hyperparameters (noise/step scales, the ``max_inference_sigma`` schedule cap) are read from
        config; see ``_build_noise_schedule``.
        """
        denoiser = self.structure_head.diffusion_module
        n_atoms = atom_inputs.atom_to_token.shape[1]
        device = single_inputs.device
        target_batch = single_inputs.shape[0] * num_diffusion_samples

        # Everything that doesn't depend on the noise level or the noisy coordinates is built once
        # here, so each denoising step is a plain forward with no caching branch inside it. This is
        # also the one place the batch is expanded across diffusion samples: from here down every
        # tensor is at ``target_batch`` and no module takes ``num_diffusion_samples``.
        step_invariants = denoiser.prepare_step_invariants(
            atom_inputs=atom_inputs,
            pair_trunk=pair_trunk,
            relative_position_encoding=relative_position_encoding,
            single_inputs=single_inputs,
            num_diffusion_samples=num_diffusion_samples,
        )
        if token_attention_mask is not None:
            token_attention_mask = token_attention_mask.repeat_interleave(num_diffusion_samples, 0)

        schedule, gammas = self.structure_head._build_noise_schedule(num_sampling_steps, device)

        lam = self.config.structure_head.noise_scale
        eta = self.config.structure_head.step_scale

        x = schedule[0] * torch.randn(target_batch, n_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = atom_inputs.atom_attention_mask.repeat_interleave(num_diffusion_samples, 0).float()

        x_denoised_prev: Tensor | None = None

        step_pairs = list(zip(schedule[:-1], schedule[1:], gammas[1:]))

        for sigma_tm, sigma_t, gamma in step_pairs:
            x, x_denoised_prev = self.structure_head._center_random_augmentation(
                x, atom_mask, second_coords=x_denoised_prev
            )

            sigma_tm_val = sigma_tm.item()
            t_hat_val = sigma_tm_val * (1.0 + gamma.item())
            eps_std = lam * max(t_hat_val**2 - sigma_tm_val**2, 0.0) ** 0.5
            x_noisy = x + eps_std * torch.randn_like(x)

            # One denoising step. This is the diffusion module's forward, not the model's -- the
            # model's forward is the trunk that produced ``step_invariants``.
            x_denoised = denoiser(
                x_noisy=x_noisy,
                t_hat=torch.full((target_batch,), t_hat_val, device=device, dtype=torch.float32),
                step_invariants=step_invariants,
                token_attention_mask=token_attention_mask,
            )

            # Reverse diffusion alignment (Kabsch). Coordinates are fp32 for the whole loop: ``x``
            # starts fp32 and the denoiser's output is fp32 too (its noise-level scalars promote the
            # coordinate update), so nothing here needs a cast.
            x_noisy = self.structure_head._weighted_rigid_align(x_noisy, x_denoised, atom_mask, atom_mask)

            # ODE/SDE step
            sigma_t_val = sigma_t.item()
            denoised_over_sigma = (x_noisy - x_denoised) / t_hat_val
            x = x_noisy + eta * (sigma_t_val - t_hat_val) * denoised_over_sigma

            x_denoised_prev = x_denoised

        return x

    @torch.no_grad()
    def infer_protein(self, seq: str, **forward_kwargs) -> EsmFold2Output:
        from .protein_utils import prepare_protein_features

        features = prepare_protein_features(seq)
        features = {k: v.to(self.device) for k, v in features.items()}
        return self.fold(**features, **forward_kwargs)

    @torch.no_grad()
    def infer_protein_as_pdb(self, seq: str, **forward_kwargs) -> str:
        from .protein_utils import output_to_pdb, prepare_protein_features

        features = prepare_protein_features(seq)
        features = {k: v.to(self.device) for k, v in features.items()}
        output = self.fold(**features, **forward_kwargs)
        return output_to_pdb(output, features)

    @staticmethod
    def output_to_pdb(output: EsmFold2Output, features: dict[str, Tensor]) -> str:
        """Render a PDB string from an [`EsmFold2Output`] and the input ``features`` it was produced from."""
        from .protein_utils import output_to_pdb as _output_to_pdb

        return _output_to_pdb(output, features)
