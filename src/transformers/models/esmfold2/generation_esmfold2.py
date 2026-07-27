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

``EsmFold2Model.forward`` runs the folding trunk; this mixin owns the diffusion sampling loop that
turns its pair representation into 3D coordinates, the confidence head call, and the public
``infer_protein`` entry points. ESMFold2 emits continuous coordinates rather than tokens, so it uses
neither ``GenerationConfig`` nor the ``Cache`` classes: sampling hyperparameters are plain kwargs
falling back to the model config."""

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
        token_attention_mask: Tensor,
        asym_id: Tensor,
        mol_type: Tensor,
        distogram_atom_idx: Tensor,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        **trunk_kwargs,
    ) -> EsmFold2Output:
        r"""
        Predict a structure end-to-end from featurized inputs: run the trunk
        ([`EsmFold2Model.forward`], which documents the feature arguments), sample coordinates from the
        diffusion structure head, and score them with the confidence head.

        Only the arguments the sampler and the confidence head need are named here; the rest are
        forwarded to the trunk untouched, so ``fold(**features)`` takes the same dict as ``forward``.

        token_attention_mask (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Mask marking valid tokens (``1``) versus padding (``0``). Also forwarded to the trunk.
        asym_id (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Asymmetric-unit (chain) ID for each token. Also forwarded to the trunk.
        mol_type (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Molecule-type code for each token (``0`` = protein). Also forwarded to the trunk.
        distogram_atom_idx (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Index of the representative atom (Cβ, or Cα for glycine) of each token. Used by the
            confidence head; the trunk does not need it.
        num_diffusion_samples (`int`, *optional*):
            Number of parallel structure samples to draw; the confidence head re-runs once per sample.
            Defaults to `config.num_diffusion_samples`.
        num_sampling_steps (`int`, *optional*):
            Number of diffusion sampling steps. Defaults to `config.structure_head.inference_num_steps`.
        trunk_kwargs:
            The remaining featurized inputs (and `num_loops`), forwarded verbatim to
            [`EsmFold2Model.forward`], which documents them.
        """
        from .modeling_esmfold2 import EsmFold2Output

        n_samples: int = (
            num_diffusion_samples if num_diffusion_samples is not None else self.config.num_diffusion_samples
        )

        trunk = self(
            token_attention_mask=token_attention_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            **trunk_kwargs,
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
            # From the trunk's featurized copy, as that is the one zeroed at padding.
            atom_to_token=trunk.atom_inputs.atom_to_token,
            atom_attention_mask=trunk.atom_inputs.atom_attention_mask,
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

        Only ``num_sampling_steps`` is an argument; the other sampling hyperparameters come from config.
        """
        denoiser = self.structure_head.diffusion_module
        n_atoms = atom_inputs.atom_to_token.shape[1]
        device = single_inputs.device
        target_batch = single_inputs.shape[0] * num_diffusion_samples

        # Built once, so each denoising step is a plain forward with no caching branch inside it. The
        # coordinates below are at ``target_batch`` and no module takes a mask or sees the sample count.
        conditioning = denoiser.prepare_conditioning(
            atom_inputs=atom_inputs,
            pair_trunk=pair_trunk,
            relative_position_encoding=relative_position_encoding,
            single_inputs=single_inputs,
            token_attention_mask=token_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
        )

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

            # One denoising step: the diffusion module's forward, not the model's.
            x_denoised = denoiser(
                x_noisy=x_noisy,
                t_hat=torch.full((target_batch,), t_hat_val, device=device, dtype=torch.float32),
                conditioning=conditioning,
            )

            # Reverse diffusion alignment (Kabsch); coordinates are fp32 for the whole loop.
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
    def infer_protein_as_pdb(self, seq: str, sample_idx: int | None = None, **forward_kwargs) -> str:
        """Fold ``seq`` and render the prediction as a PDB string.

        ``sample_idx`` picks which diffusion sample to render; by default the best-ranked one.
        """
        from .protein_utils import output_to_pdb, prepare_protein_features

        features = prepare_protein_features(seq)
        features = {k: v.to(self.device) for k, v in features.items()}
        output = self.fold(**features, **forward_kwargs)
        return output_to_pdb(output, features, sample_idx=sample_idx)

    @staticmethod
    def output_to_pdb(output: EsmFold2Output, features: dict[str, Tensor], sample_idx: int | None = None) -> str:
        """Render a PDB string from an [`EsmFold2Output`] and the input ``features`` it was produced from.

        ``sample_idx`` picks which diffusion sample to render; by default the best-ranked one.
        """
        from .protein_utils import output_to_pdb as _output_to_pdb

        return _output_to_pdb(output, features, sample_idx=sample_idx)
