# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from ..openfold_np import residue_constants
from . import feats
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather, masked_mean, permute_final_dims, tensor_tree_map, tree_map


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()
    log_p = torch.nn.functional.logsigmoid(logits)
    # log_p = torch.log(torch.sigmoid(logits))
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    # log_not_p = torch.log(torch.sigmoid(-logits))
    loss = (-1.0 * labels) * log_p - (1.0 - labels) * log_not_p
    loss = loss.to(dtype=logits_dtype)
    return loss


def torsion_angle_loss(
    a,  # [*, N, 7, 2]
    a_gt,  # [*, N, 7, 2]
    a_alt_gt,  # [*, N, 7, 2]
):
    # [*, N, 7]
    norm = torch.norm(a, dim=-1)

    # [*, N, 7, 2]
    a = a / norm.unsqueeze(-1)

    # [*, N, 7]
    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    diff_norm_alt_gt = torch.norm(a - a_alt_gt, dim=-1)
    min_diff = torch.minimum(diff_norm_gt**2, diff_norm_alt_gt**2)

    # [*]
    l_torsion = torch.mean(min_diff, dim=(-1, -2))
    l_angle_norm = torch.mean(torch.abs(norm - 1), dim=(-1, -2))

    an_weight = 0.02
    return l_torsion + an_weight * l_angle_norm


def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
    Computes FAPE loss.

    Args:
        pred_frames:
            [*, N_frames] Rigid object of predicted frames
        target_frames:
            [*, N_frames] Rigid object of ground truth frames
        frames_mask:
            [*, N_frames] binary mask for the frames
        pred_positions:
            [*, N_pts, 3] predicted atom positions
        target_positions:
            [*, N_pts, 3] ground truth positions
        positions_mask:
            [*, N_pts] positions mask
        length_scale:
            Length scale by which the loss is divided
        l1_clamp_distance:
            Cutoff above which distance errors are disregarded
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps)

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    pred_aff = Rigid.from_tensor_7(traj)
    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        backbone_rigid_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_rigid_mask[None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (1 - use_clamped_fape)

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss


def sidechain_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    renamed_gt_frames = (
        1.0 - alt_naming_is_better[..., None, None, None]
    ) * rigidgroups_gt_frames + alt_naming_is_better[..., None, None, None] * rigidgroups_alt_gt_frames

    # Steamroll the inputs
    sidechain_frames = sidechain_frames[-1]
    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    sidechain_atom_pos = sidechain_atom_pos[-1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(*batch_dims, -1, 3)
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    fape = compute_fape(
        sidechain_frames,
        renamed_gt_frames,
        rigidgroups_gt_exists,
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )

    return fape


def fape_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    bb_loss = backbone_loss(
        traj=out["sm"]["frames"],
        **{**batch, **config.backbone},
    )

    sc_loss = sidechain_loss(
        out["sm"]["sidechain_frames"],
        out["sm"]["positions"],
        **{**batch, **config.sidechain},
    )

    loss = config.backbone.weight * bb_loss + config.sidechain.weight * sc_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    Implements Algorithm 27 (torsionAngleLoss)

    Args:
        angles_sin_cos:
            [*, N, 7, 2] predicted angles
        unnormalized_angles_sin_cos:
            The same angles, but unnormalized
        aatype:
            [*, N] residue indices
        seq_mask:
            [*, N] sequence mask
        chi_mask:
            [*, N, 7] angle mask
        chi_angles_sin_cos:
            [*, N, 7, 2] ground truth angles
        chi_weight:
            Weight for the angle component of the loss
        angle_norm_weight:
            Weight for the normalization component of the loss
    Returns:
        [*] loss tensor
    """
    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )

    true_chi = chi_angles_sin_cos[None]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_angles) ** 2, dim=-1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(*range(len(sq_chi_error.shape))[1:-2], 0, -2, -1)

    sq_chi_loss = masked_mean(chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3))

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(torch.sum(unnormalized_angles_sin_cos**2, dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(*range(len(norm_error.shape))[1:-2], 0, -2, -1)
    angle_norm_loss = masked_mean(seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3))

    loss = loss + angle_norm_weight * angle_norm_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :]) ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )


def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]

    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    score = lddt(all_atom_pred_pos, all_atom_positions, all_atom_mask, cutoff=cutoff, eps=eps)

    score = score.detach()

    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=no_bins)

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (eps + torch.sum(all_atom_mask, dim=-1))

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries**2

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (predicted_aligned_error, max_predicted_aligned_error,) = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    bin_centers = _calculate_bin_centers(boundaries)
    torch.sum(residue_weights)
    n = logits.shape[-2]
    clipped_n = max(n, 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


def tm_loss(
    logits,
    final_affine_tensor,
    backbone_rigid_tensor,
    backbone_rigid_mask,
    resolution,
    max_bin=31,
    no_bins=64,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps=1e-8,
    **kwargs,
):
    pred_affine = Rigid.from_tensor_7(final_affine_tensor)
    backbone_rigid = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum((_points(pred_affine) - _points(backbone_rigid)) ** 2, dim=-1)

    sq_diff = sq_diff.detach()

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)
    boundaries = boundaries**2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(logits, torch.nn.functional.one_hot(true_bins, no_bins))

    square_mask = backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    # Average over the loss dimension
    loss = torch.mean(loss)

    return loss


def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,  # (*, N, 37/14, 3)
    pred_atom_mask: torch.Tensor,  # (*, N, 37/14)
    residue_index: torch.Tensor,  # (*, N)
    aatype: torch.Tensor,  # (*, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide bond between consecutive amino acids.
    This loss corresponds to Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = aatype[..., 1:] == residue_constants.resname_to_idx["PRO"]
    gt_length = (~next_is_proline) * residue_constants.between_res_bond_length_c_n[
        0
    ] + next_is_proline * residue_constants.between_res_bond_length_c_n[1]
    gt_stddev = (~next_is_proline) * residue_constants.between_res_bond_length_stddev_c_n[
        0
    ] + next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    c_n_violation_mask = mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1))
    n_ca_bond_length = torch.sqrt(eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1))

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(eps + (ca_c_n_cos_angle - gt_angle) ** 2)
    ca_c_n_loss_per_residue = torch.nn.functional.relu(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(eps + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = torch.nn.functional.relu(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1)) + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in different peptides coming too close. This
    loss corresponds to the part with different residues of Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_pred_positions[..., :, None, :, None, :] - atom14_pred_positions[..., None, :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (atom14_atom_exists[..., :, None, :, None] * atom14_atom_exists[..., None, :, None, :]).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (residue_index[..., :, None, None, None] < residue_index[..., None, :, None, None])

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(2), num_classes=14)
    c_one_hot = c_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape)
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(residue_index.new_tensor(0), num_classes=14)
    n_one_hot = n_one_hot.reshape(*((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape)
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (residue_index[..., :, None, None, None] + 1) == residue_index[..., None, :, None, None]
    c_n_bonds = neighbour_mask * c_one_hot[..., None, None, :, None] * n_one_hot[..., None, None, None, :]
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = residue_constants.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(*((1,) * len(residue_index.shape[:-1])), 1).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = cys_sg_one_hot[..., None, None, :, None] * cys_sg_one_hot[..., None, None, None, :]
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None] + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(dists_lower_bound - overlap_tolerance_soft - dists)

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(dists_to_low_error, axis=(-3, -1))

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }


def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms in a given peptide. This loss
    corresponds to the part with the same residues of Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
        atom14_pred_positions ([*, N, 14, 3]):
            Predicted positions of atoms in global prediction frame.
        atom14_atom_exists ([*, N, 14]):
            Mask denoting whether atom at positions exists for given amino acid type
        atom14_dists_lower_bound ([*, N, 14]):
            Lower bound on allowed distances.
        atom14_dists_upper_bound ([*, N, 14]):
            Upper bound on allowed distances
        tighten_bounds_for_loss ([*, N]):
            Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum' ([*, N, 14]):
              sum of all clash losses per atom, shape
        * 'per_atom_clash_mask' ([*, N, 14]):
              mask whether atom clashes with any other atom shape
    """
    # Compute the mask for each residue.
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(*((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape)
    dists_masks = atom14_atom_exists[..., :, :, None] * atom14_atom_exists[..., :, None, :] * dists_masks

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_pred_positions[..., :, :, None, :] - atom14_pred_positions[..., :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = torch.nn.functional.relu(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * ((dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound))

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0])

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
    }


def find_structural_violations(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
        aatype=batch["aatype"],
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [residue_constants.van_der_waals_radius[name[0]] for name in residue_constants.atom_types]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = batch["atom14_atom_exists"] * atomtype_radius[batch["residx_atom14_to_atom37"]]

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch["residue_index"],
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_atom_exists = batch["atom14_atom_exists"]
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(restype_atom14_bounds["lower_bound"])[batch["aatype"]]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(restype_atom14_bounds["upper_bound"])[batch["aatype"]]
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(between_residue_clashes["per_atom_clash_mask"], dim=-1)[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # ()
            "angles_ca_c_n_loss_mean": connection_violations["ca_c_n_loss_mean"],  # ()
            "angles_c_n_ca_loss_mean": connection_violations["c_n_ca_loss_mean"],  # ()
            "connections_per_residue_loss_sum": connection_violations["per_residue_loss_sum"],  # (N)
            "connections_per_residue_violation_mask": connection_violations["per_residue_violation_mask"],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes["per_atom_loss_sum"],  # (N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes["per_atom_clash_mask"],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations["per_atom_loss_sum"],  # (N, 14)
            "per_atom_violations": residue_violations["per_atom_violations"],  # (N, 14),
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }


def find_structural_violations_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    config,
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    atom14_pred_positions = to_tensor(atom14_pred_positions)

    out = find_structural_violations(batch, atom14_pred_positions, **config)

    to_np = lambda x: np.array(x)
    np_out = tensor_tree_map(to_np, out)

    return np_out


def extreme_ca_ca_distance_violations(
    pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
    pred_atom_mask: torch.Tensor,  # (N, 37(14))
    residue_index: torch.Tensor,  # (N)
    max_angstrom_tolerance=1.5,
    eps=1e-6,
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are more than 'max_angstrom_tolerance'
    apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.
    Returns:
      Fraction of consecutive CA-CA pairs with violation.
    """
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    ca_ca_distance = torch.sqrt(eps + torch.sum((this_ca_pos - next_ca_pos) ** 2, dim=-1))
    violations = (ca_ca_distance - residue_constants.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    mean = masked_mean(mask, violations, -1)
    return mean


def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    ret = {}
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
    )
    ret["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    ret["violations_between_residue_bond"] = masked_mean(
        batch["seq_mask"],
        violations["between_residues"]["connections_per_residue_violation_mask"],
        dim=-1,
    )
    ret["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    ret["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(violations["within_residues"]["per_atom_violations"], dim=-1)[0],
        dim=-1,
    )
    ret["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return ret


def compute_violation_metrics_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    violations: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    atom14_pred_positions = to_tensor(atom14_pred_positions)
    violations = tree_map(to_tensor, violations, np.ndarray)

    out = compute_violation_metrics(batch, atom14_pred_positions, violations)

    to_np = lambda x: np.array(x)
    return tree_map(to_np, out, torch.Tensor)


def violation_loss(
    violations: Dict[str, torch.Tensor],
    atom14_atom_exists: torch.Tensor,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    num_atoms = torch.sum(atom14_atom_exists)
    l_clash = torch.sum(
        violations["between_residues"]["clashes_per_atom_loss_sum"]
        + violations["within_residues"]["per_atom_loss_sum"]
    )
    l_clash = l_clash / (eps + num_atoms)
    loss = (
        violations["between_residues"]["bonds_c_n_loss_mean"]
        + violations["between_residues"]["angles_ca_c_n_loss_mean"]
        + violations["between_residues"]["angles_c_n_ca_loss_mean"]
        + l_clash
    )

    return loss


def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """
    Find optimal renaming of ground truth based on the predicted positions.

    Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses, such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better. renamed_atom14_gt_positions: Array of
        optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """

    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_pred_positions[..., None, :, None, :] - atom14_pred_positions[..., None, :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"]
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_gt_positions[..., None, :, None, :] - atom14_gt_positions[..., None, :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"]
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_alt_gt_positions[..., None, :, None, :] - atom14_alt_gt_positions[..., None, :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_gt_exists"]
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"]
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        1.0 - alt_naming_is_better[..., None, None]
    ) * atom14_gt_positions + alt_naming_is_better[..., None, None] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (1.0 - alt_naming_is_better[..., None]) * atom14_gt_exists + alt_naming_is_better[
        ..., None
    ] * batch["atom14_alt_gt_exists"]

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }


def experimentally_resolved_loss(
    logits: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    min_resolution: float,
    max_resolution: float,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    errors = sigmoid_cross_entropy(logits, all_atom_mask)
    loss = torch.sum(errors * atom37_atom_exists, dim=-1)
    loss = loss / (eps + torch.sum(atom37_atom_exists, dim=(-1, -2)))
    loss = torch.sum(loss, dim=-1)

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    loss = torch.mean(loss)

    return loss


def masked_msa_loss(logits, true_msa, bert_mask, eps=1e-8, **kwargs):
    """
    Computes BERT-style masked MSA loss. Implements subsection 1.9.9.

    Args:
        logits: [*, N_seq, N_res, 23] predicted residue distribution
        true_msa: [*, N_seq, N_res] true MSA
        bert_mask: [*, N_seq, N_res] MSA mask
    Returns:
        Masked MSA loss
    """
    errors = softmax_cross_entropy(logits, torch.nn.functional.one_hot(true_msa, num_classes=23))

    # FP16-friendly averaging. Equivalent to:
    # loss = (
    #     torch.sum(errors * bert_mask, dim=(-1, -2)) /
    #     (eps + torch.sum(bert_mask, dim=(-1, -2)))
    # )
    loss = errors * bert_mask
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = torch.mean(loss)

    return loss


class AlphaFoldLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super(AlphaFoldLoss, self).__init__()
        self.config = config

    def forward(self, out, batch, _return_breakdown=False):
        if "violation" not in out.keys():
            out["violation"] = find_structural_violations(
                batch,
                out["sm"]["positions"][-1],
                **self.config.violation,
            )

        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    out["sm"]["positions"][-1],
                )
            )

        loss_fns = {
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                **{**batch, **self.config.distogram},
            ),
            "experimentally_resolved": lambda: experimentally_resolved_loss(
                logits=out["experimentally_resolved_logits"],
                **{**batch, **self.config.experimentally_resolved},
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                self.config.fape,
            ),
            "plddt_loss": lambda: lddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.plddt_loss},
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                **{**batch, **self.config.masked_msa},
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                out["sm"]["angles"],
                out["sm"]["unnormalized_angles"],
                **{**batch, **self.config.supervised_chi},
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                **batch,
            ),
        }

        if self.config.tm.enabled:
            loss_fns["tm"] = lambda: tm_loss(
                logits=out["tm_logits"],
                **{**batch, **out, **self.config.tm},
            )

        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                # for k,v in batch.items():
                #    if(torch.any(torch.isnan(v)) or torch.any(torch.isinf(v))):
                #        logging.warning(f"{k}: is nan")
                # logging.warning(f"{loss_name}: {loss}")
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0.0, requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()

        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["aatype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
