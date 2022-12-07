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

from typing import Dict, Optional, Tuple

import torch


def _calculate_bin_centers(boundaries: torch.Tensor) -> torch.Tensor:
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
