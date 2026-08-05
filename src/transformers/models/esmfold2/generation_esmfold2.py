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
import torch.nn.functional as F
from torch import Tensor


if TYPE_CHECKING:
    from .modeling_esmfold2 import EsmFold2AtomInputs, EsmFold2Output


def _random_rotations(num_rotations: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """``num_rotations`` uniformly random 3x3 rotation matrices, via random unit quaternions."""
    quaternions = torch.randn((num_rotations, 4), dtype=dtype, device=device)
    scale = torch.sqrt((quaternions * quaternions).sum(dim=1))
    signs = torch.where(quaternions[:, 0] < 0, -scale, scale)
    quaternions = quaternions / signs[:, None]
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(dim=-1)
    return torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(num_rotations, 3, 3)


def _center_random_augmentation(
    coords: Tensor, atom_mask: Tensor, second_coords: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    """Algorithm 19: center + random rotation + translation. ``second_coords`` rides along."""
    batch_size = coords.shape[0]
    mask = atom_mask.unsqueeze(-1)  # [B, A, 1]
    denominator = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (coords * mask).sum(dim=1, keepdim=True) / denominator
    coords = coords - mean
    if second_coords is not None:
        second_coords = second_coords - mean

    rotations = _random_rotations(batch_size, coords.dtype, coords.device)
    coords = coords @ rotations
    if second_coords is not None:
        second_coords = second_coords @ rotations

    translations = torch.randn_like(coords[:, 0:1, :])
    coords = coords + translations
    if second_coords is not None:
        second_coords = second_coords + translations
    return coords, second_coords


def _weighted_rigid_align(coords: Tensor, target_coords: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
    """Kabsch alignment: align ``coords`` onto ``target_coords`` with the given weights."""
    weights = (mask * weights).unsqueeze(-1)  # [batch_size, num_atoms, 1]
    denominator = weights.sum(dim=-2, keepdim=True).clamp(min=1e-8)
    centroid = (coords * weights).sum(dim=-2, keepdim=True) / denominator
    target_centroid = (target_coords * weights).sum(dim=-2, keepdim=True) / denominator
    centered_coords = coords - centroid
    centered_target_coords = target_coords - target_centroid
    covariance = (weights * centered_target_coords).transpose(-1, -2) @ centered_coords
    U, _, Vh = torch.linalg.svd(covariance, driver="gesvd" if covariance.is_cuda else None)
    determinant = torch.linalg.det(U @ Vh)
    ones = torch.ones_like(determinant)
    rotation = U @ torch.diag_embed(torch.stack([ones, ones, determinant], dim=-1)) @ Vh
    return centered_coords @ rotation.transpose(-1, -2) + target_centroid


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

        num_samples: int = (
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
            num_diffusion_samples=num_samples,
            num_sampling_steps=num_sampling_steps,
        )

        confidence_output = self.confidence_head(
            single_inputs=trunk.single_inputs.detach(),
            pair_states=trunk.pair_states.detach(),
            predicted_coords=sample_coords.detach(),
            distogram_atom_idx=distogram_atom_idx,
            token_attention_mask=token_attention_mask,
            # From the trunk's featurized copy, as that is the one zeroed at padding.
            atom_to_token=trunk.atom_inputs.atom_to_token,
            atom_attention_mask=trunk.atom_inputs.atom_attention_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=num_samples,
            relative_position_encoding=trunk.relative_position_encoding.detach(),
            token_bonds_encoding=trunk.token_bonds_encoding.detach(),
        )

        return EsmFold2Output(
            distogram_logits=trunk.distogram_logits,
            sample_atom_coords=sample_coords,
            **confidence_output,
        )

    def inference_noise_schedule(self, num_steps: int | None = None, device: torch.device | None = None) -> Tensor:
        """Karras power-law noise schedule."""
        head_config = self.config.structure_head
        steps = head_config.inference_num_steps if num_steps is None else int(num_steps)
        if steps == 1:
            return torch.tensor(
                [head_config.inference_sigma_max_ratio * head_config.diffusion_module.sigma_data, 0.0],
                device=device,
                dtype=torch.float32,
            )
        inverse_exponent = 1.0 / head_config.inference_exponent
        ramp = torch.arange(steps, device=device, dtype=torch.float32)
        base = head_config.inference_sigma_max_ratio**inverse_exponent + (ramp / (steps - 1)) * (
            head_config.inference_sigma_min_ratio**inverse_exponent
            - head_config.inference_sigma_max_ratio**inverse_exponent
        )
        schedule = head_config.diffusion_module.sigma_data * base.pow(head_config.inference_exponent)
        return F.pad(schedule, (0, 1), value=0.0)

    def _build_noise_schedule(self, num_sampling_steps: int | None, device: torch.device) -> tuple[Tensor, Tensor]:
        """Karras σ schedule (Algorithm 18) + per-step γ churn factors, capped at
        ``config.structure_head.inference_sigma_cap``.

        Read from ``config`` at call time, so edits after loading apply.
        """
        head_config = self.config.structure_head
        steps = head_config.inference_num_steps if num_sampling_steps is None else int(num_sampling_steps)
        schedule = self.inference_noise_schedule(steps, device)
        # Truncate the high-σ tail above the cap and re-prepend the cap, so sampling starts from it.
        if head_config.inference_sigma_cap is not None:
            schedule = schedule[schedule <= head_config.inference_sigma_cap]
            schedule = F.pad(schedule, (1, 0), value=head_config.inference_sigma_cap)
        gammas = torch.where(
            schedule > head_config.gamma_min,
            torch.full_like(schedule, head_config.gamma_0),
            torch.zeros_like(schedule),
        )
        return schedule, gammas

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
        denoiser = self.structure_head
        num_atoms = atom_inputs.atom_to_token.shape[1]
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

        schedule, gammas = self._build_noise_schedule(num_sampling_steps, device)

        noise_scale = self.config.structure_head.noise_scale
        step_scale = self.config.structure_head.step_scale

        atom_coords = schedule[0] * torch.randn(target_batch, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = atom_inputs.atom_attention_mask.repeat_interleave(num_diffusion_samples, 0).float()

        prev_denoised_coords: Tensor | None = None

        schedule_steps = list(zip(schedule[:-1], schedule[1:], gammas[1:]))

        for sigma_prev, sigma_next, gamma in schedule_steps:
            atom_coords, prev_denoised_coords = _center_random_augmentation(
                atom_coords, atom_mask, second_coords=prev_denoised_coords
            )

            sigma_prev_value = sigma_prev.item()
            noise_level_value = sigma_prev_value * (1.0 + gamma.item())
            churn_noise_std = noise_scale * max(noise_level_value**2 - sigma_prev_value**2, 0.0) ** 0.5
            noisy_coords = atom_coords + churn_noise_std * torch.randn_like(atom_coords)

            # One denoising step: the diffusion module's forward, not the model's.
            denoised_coords = denoiser(
                noisy_coords=noisy_coords,
                noise_level=torch.full((target_batch,), noise_level_value, device=device, dtype=torch.float32),
                conditioning=conditioning,
            )

            # Reverse diffusion alignment (Kabsch); coordinates are fp32 for the whole loop.
            noisy_coords = _weighted_rigid_align(noisy_coords, denoised_coords, atom_mask, atom_mask)

            # ODE/SDE step
            sigma_next_value = sigma_next.item()
            denoised_direction = (noisy_coords - denoised_coords) / noise_level_value
            atom_coords = noisy_coords + step_scale * (sigma_next_value - noise_level_value) * denoised_direction

            prev_denoised_coords = denoised_coords

        return atom_coords

    @torch.no_grad()
    def infer_protein(self, sequence: str, **forward_kwargs) -> EsmFold2Output:
        """Featurize ``sequence`` and fold it. ``forward_kwargs`` go to [`~EsmFold2Model.fold`]."""
        from .protein_utils import prepare_protein_features

        return self.fold(**prepare_protein_features(sequence, device=self.device), **forward_kwargs)

    @torch.no_grad()
    def infer_protein_as_pdb(self, sequence: str, sample_idx: int | None = None, **forward_kwargs) -> str:
        """Fold ``sequence`` and render the prediction as a PDB string.

        ``sample_idx`` picks which diffusion sample to render; by default the best-ranked one.
        """
        from .protein_utils import output_to_pdb, prepare_protein_features

        features = prepare_protein_features(sequence, device=self.device)
        output = self.fold(**features, **forward_kwargs)
        return output_to_pdb(output, features, sample_idx=sample_idx)
