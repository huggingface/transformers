# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import torch


def clef_proper_reward(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    option_mask: torch.Tensor,
    is_score: torch.Tensor,
    spherical_weight: float = 0.75,
    rps_weight: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """
    A strictly proper reward of reported distributions over the options of `Q` questions: the log score (floored at
    `log_floor`), plus `spherical_weight` times the spherical score, minus, on score questions, `rps_weight` times the
    ranked probability score divided by the number of options minus one.

    Adapted from `proper_reward` of Laya (https://github.com/NandhaKishorM/laya, commit
    3a38cbe83f6ef556f3125fb0065fdc5a55d05792, Apache-2.0).

    Args:
        probabilities (`torch.Tensor` of shape `(..., Q, K)`): the reported distributions, anything on the padded cells.
        targets (`torch.Tensor` of shape `(Q, K)`): the target distributions, zero on the padded cells.
        option_mask (`torch.BoolTensor` of shape `(Q, K)`): the options that exist.
        is_score (`torch.BoolTensor` of shape `(Q,)`): the score questions, whose options are ordered.

    Returns:
        `torch.Tensor` of shape `(..., Q)`: the reward of each distribution.
    """
    mask = option_mask.float()
    q = probabilities * mask
    log_score = (targets * q.clamp(min=1e-12).log().clamp(min=log_floor)).sum(-1)
    spherical = (targets * q).sum(-1) / q.norm(dim=-1).clamp(min=1e-9)
    ranked = ((q.cumsum(-1) - targets.cumsum(-1)) ** 2 * mask).sum(-1) / (mask.sum(-1).clamp(min=2) - 1)
    return log_score + spherical_weight * spherical - rps_weight * is_score.float() * ranked


def check_clef_label_shapes(labels: torch.Tensor, option_mask: torch.Tensor) -> None:
    """Raises when the shape of `labels` does not give one label per question of `option_mask` (`(Q, K)`)."""
    if labels.is_floating_point():
        if labels.shape != option_mask.shape:
            raise ValueError(
                f"Soft labels must have the shape of the logits {tuple(option_mask.shape)}, got shape "
                f"{tuple(labels.shape)}: give one distribution over the options of each question."
            )
    elif labels.dim() != 1:
        raise ValueError(
            f"Hard labels must have the shape ({option_mask.shape[0]},), one index per question, got shape "
            f"{tuple(labels.shape)}."
        )
    elif labels.shape[0] != option_mask.shape[0]:
        raise ValueError(
            f"Got {labels.shape[0]} labels for {option_mask.shape[0]} questions: give one label per question that has "
            "options, prompt after prompt."
        )


def clef_label_errors(labels: torch.Tensor, option_mask: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Whether each hard label (`(Q,)`) is wrong: neither `ignore_index` nor the index of an option of its question in
    `option_mask` (`(Q, K)`). Computed on the device, without synchronising it; all `False` for soft labels.
    """
    if labels.is_floating_point():
        return torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    labels = labels.to(option_mask.device)
    in_range = (labels >= 0) & (labels < option_mask.shape[-1])
    exists = option_mask.gather(-1, labels.clamp(0, option_mask.shape[-1] - 1)[:, None])[:, 0]
    return (labels != ignore_index) & ~(in_range & exists)


def raise_label_error(
    labels: torch.Tensor, option_mask: torch.Tensor, errors: torch.Tensor, ignore_index: int = -100
) -> None:
    """Raises for the first wrong label of `errors` (see [`clef_label_errors`])."""
    index = int(errors.nonzero()[0, 0])
    count = int(option_mask[index].sum())
    raise ValueError(
        f"The label {int(labels[index])} of question {index} is not one of its {count} options: use an index in "
        f"[0, {count}), or {ignore_index} to ignore the question."
    )


def ClefForDecisionLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    option_mask: torch.Tensor,
    is_score: torch.Tensor,
    label_smoothing: float = 0.0,
    brier_weight: float = 0.0,
    ordinal_weight: float = 0.0,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    rlcd_weight: float = 0.0,
    rlcd_sigma: float = 0.4,
    rlcd_samples: int = 32,
    rlcd_spherical_weight: float = 0.75,
    rlcd_rps_weight: float = 1.0,
    rlcd_normalize_advantage: bool = False,
    generator: torch.Generator | None = None,
    check_labels: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    The loss of the questions of a decision request, with `Q` questions of at most `K` options.

    Args:
        logits (`torch.Tensor` of shape `(Q, K)`): scores of the options, anything on the padded cells.
        labels (`torch.LongTensor` of shape `(Q,)` or `torch.FloatTensor` of shape `(Q, K)`): the index of the right
            option of each question (`ignore_index` to skip the question), or a distribution over its options.
        option_mask (`torch.BoolTensor` of shape `(Q, K)`): the options that exist.
        is_score (`torch.BoolTensor` of shape `(Q,)`): the score questions, whose options are ordered.
        label_smoothing: the weight of the uniform distribution over the options in the target of the cross entropy.
        brier_weight: the weight of the Brier score, the squared distance of the probabilities to the target.
        ordinal_weight: the weight of the expected distance `|i - j|` of the options to the target, on score questions.
        num_items_in_batch: the number of questions the summed cross entropy and Brier terms are divided by, instead
            of the number of questions that are not ignored. It also divides the RLCD term.
        rlcd_weight: the weight of the RLCD policy term (Reinforcement Learning for Calibrated Decisions). Its noisy
            samples of the logits are drawn whenever it is positive: callers skip it out of training.
        rlcd_sigma: the standard deviation of the noise added to the logits by the RLCD term.
        rlcd_samples: the number of noisy samples of the logits of each question.
        rlcd_spherical_weight: the weight of the spherical score in the reward of the RLCD term.
        rlcd_rps_weight: the weight of the ranked probability score in the reward of the RLCD term, on score questions.
        rlcd_normalize_advantage: whether to divide the advantages by their standard deviation over the samples of the
            questions that are not ignored, as Laya's original objective does.
        generator: the generator of the noise of the RLCD term, the global one by default.
        check_labels: whether to check that every hard label is `ignore_index` or an option of its question, which
            synchronises the device once. The shapes are always checked, which costs nothing. `False` for a caller
            that checked the labels already, as [`ClefForDecision`] does.

    Raises a `ValueError` when `labels` or `option_mask` do not match the logits, or a hard label names an option its
    question does not have.

    The RLCD term adapts `rlcd_loss` of Laya (https://github.com/NandhaKishorM/laya, commit
    3a38cbe83f6ef556f3125fb0065fdc5a55d05792, Apache-2.0). Zero mean Gaussian noise over the options of each question
    is added to the detached logits, each sample is rewarded by [`clef_proper_reward`] against the target, and the
    advantage of each sample over the mean reward of the samples of its question weights the Gaussian log density of
    the sample, the only path of its gradient. Its defaults are a revised variant (more samples, advantages not divided
    by their standard deviation); `rlcd_normalize_advantage=True, rlcd_samples=4` gives Laya's original.
    """
    if option_mask.shape != logits.shape:
        raise ValueError(
            f"`option_mask` of shape {tuple(option_mask.shape)} does not match the logits of shape {tuple(logits.shape)}."
        )
    check_clef_label_shapes(labels, option_mask)
    if check_labels:
        errors = clef_label_errors(labels, option_mask, ignore_index)
        if errors.any():
            raise_label_error(labels, option_mask, errors, ignore_index)
    logits = logits.float()
    questions, width = logits.shape
    if labels.is_floating_point():
        target, valid = labels.float(), torch.ones(questions, dtype=torch.bool, device=logits.device)
    else:
        valid = labels != ignore_index
        target = torch.nn.functional.one_hot(labels.clamp(min=0), width).float()
    # Padded cells and ignored questions contribute nothing: they are zeroed in every term
    cells = option_mask & valid[:, None]
    target = target * cells
    options = option_mask.sum(-1, keepdim=True).float()
    log_probs = logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min).log_softmax(-1)
    probs = log_probs.exp() * cells
    log_probs = log_probs * cells

    smoothed = (
        (1 - label_smoothing) * target + label_smoothing * cells / options.clamp(min=1) if label_smoothing else target
    )
    loss = -(smoothed * log_probs).sum()
    if brier_weight:
        loss = loss + brier_weight * ((probs - target) ** 2 * cells).sum()
    normaliser = valid.sum().clamp(min=1) if num_items_in_batch is None else num_items_in_batch
    loss = loss / normaliser

    if ordinal_weight:
        positions = torch.arange(width, device=logits.device)
        distance = (positions[:, None] - positions[None]).abs().float()
        expected = ((probs @ distance) * target).sum(-1) / (options[:, 0] - 1).clamp(min=1)
        scored = is_score & valid
        loss = loss + ordinal_weight * (expected * scored).sum() / scored.sum().clamp(min=1)

    if rlcd_weight < 0:
        # It would push the logits away from the samples with a high reward
        raise ValueError(f"RLCD needs `rlcd_weight >= 0`, got {rlcd_weight}.")
    if rlcd_weight:
        if rlcd_sigma <= 0 or rlcd_samples < 2:
            raise ValueError(
                f"RLCD needs `rlcd_sigma > 0` and `rlcd_samples >= 2`, got {rlcd_sigma} and {rlcd_samples}."
            )
        mask = option_mask.float()
        # One draw of the full shape, whatever is ignored or padded, so that the noise is reproducible from the seed
        device = logits.device if generator is None else generator.device
        noise = torch.randn((rlcd_samples, questions, width), generator=generator, device=device, dtype=torch.float32)
        noise = noise.to(logits.device) * rlcd_sigma * mask
        noise = (noise - noise.sum(-1, keepdim=True) / options.clamp(min=1)) * mask
        sample = logits.detach() + noise
        with torch.no_grad():
            reported = sample.masked_fill(~option_mask, torch.finfo(sample.dtype).min).softmax(-1)
            reward = clef_proper_reward(
                reported, target, option_mask, is_score, rlcd_spherical_weight, rlcd_rps_weight
            )
            advantage = (reward - reward.mean(0, keepdim=True)) * valid
            if rlcd_normalize_advantage and valid.any():
                advantage = advantage / (advantage[:, valid].std() + 1e-6)
        # The gradient reaches the logits through the Gaussian log density only, and never their padded cells
        log_density = -((sample - logits).masked_fill(~option_mask, 0) ** 2).sum(-1) / (2 * rlcd_sigma**2)
        policy = (-(advantage * log_density).mean(0) * valid).sum() / normaliser
        loss = loss + rlcd_weight * policy
    # A batch with nothing to learn from is zero, and still has a gradient
    return loss + 0.0 * logits.masked_fill(~option_mask, 0).sum()
