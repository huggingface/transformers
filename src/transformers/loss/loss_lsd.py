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
"""
Training loss of continuous-latent TTS models whose flow head is trained by Lagrangian self-distillation (LSD,
https://huggingface.co/papers/2505.18825), e.g. Pocket TTS (Kyutai's `training/modules/samplers.py` and
`training/modules/model.py`): a flow loss on the latents plus an end-of-speech loss. The models hold the loss's state
(the learnt weighting of the LSD terms, the generator of the self-distillation draw) in a loss module that calls these
functions.

The flow head is a callable `flow_head(latents, conditions, times=[start_times, end_times])` returning the velocity that
moves `latents` from `start_times` to `end_times`, with the convention `x_t = t * data + (1 - t) * noise`.
"""

from collections.abc import Callable

import torch
from torch import nn


def end_of_speech_loss(eos_logits: torch.Tensor, eos_labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy of the end-of-speech logits, averaged over the positions: every target frame is a negative and
    the position after the last one (whose input is the last target frame) the positive.

    Args:
        eos_logits (`torch.Tensor` of shape `(num_positions,)`): End-of-speech logits.
        eos_labels (`torch.Tensor` of shape `(num_positions,)`): `1` at the end-of-speech position, `0` elsewhere.
    """
    eos_logits = eos_logits.float()
    is_eos = eos_labels.bool()
    loss = torch.where(is_eos, nn.functional.softplus(-eos_logits), nn.functional.softplus(eos_logits))
    return loss.mean()


def _grad_to_latents_only(flow_head: nn.Module, latents, conditions, times):
    """Runs the flow head with its parameters and conditions detached: the gradient only reaches `latents`."""
    parameters = {name: parameter.detach() for name, parameter in flow_head.named_parameters()}
    return torch.func.functional_call(flow_head, parameters, (latents, conditions.detach()), {"times": times})


def lsd_flow_loss(
    flow_head: nn.Module,
    conditions: torch.Tensor,
    targets: torch.Tensor,
    loss_weighting: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    distill_generator: torch.Generator | None = None,
    training: bool = True,
    num_samples: int = 4,
    distill_probability: float = 0.25,
    diagonal_weight: float = 0.75,
    time_logit_mean: float = 0.4,
    time_logit_std: float = 1.0,
) -> torch.Tensor:
    """
    LSD flow loss, averaged over the target frames and their noise draws.

    Each frame is repeated `num_samples` times, each with its own Gaussian noise. The diagonal term regresses the
    instantaneous velocity at a time `t ~ sigmoid(N(time_logit_mean, time_logit_std))` onto `data - noise`. The
    self-distillation term asks the jump from `s` to `t` (`s <= t`, both drawn the same way) to move like the
    instantaneous flow at its endpoint: its time derivative comes from a forward-mode JVP, and the endpoint flow passes
    gradient to the jump only (not to the head's parameters, Kyutai's "minimal" stop-gradient). In training, the
    self-distillation term is computed with probability `distill_probability`, drawn from `distill_generator` (identical
    on every rank) and reweighted to keep the expected loss; outside training it is always computed. With
    `loss_weighting`, each term `L` becomes `L * exp(w) / latent_dim - w`, `w` the learnt uncertainty of its times.

    Args:
        flow_head (`nn.Module`): The flow head, `flow_head(latents, conditions, times=[start, end])`.
        conditions (`torch.Tensor` of shape `(num_frames, hidden_size)`): Backbone hidden states of the target frames.
        targets (`torch.Tensor` of shape `(num_frames, latent_dim)`): The (normalised) target latents.
        loss_weighting (`Callable`, *optional*): The learnt `w(start_times, end_times)`, of shape `(..., 1)`.
        distill_generator (`torch.Generator`, *optional*): CPU generator of the self-distillation draw.
        training (`bool`, *optional*, defaults to `True`): Whether the self-distillation term is drawn.
        num_samples (`int`, *optional*, defaults to 4): Noise draws per frame.
        distill_probability (`float`, *optional*, defaults to 0.25): Probability of computing the self-distillation term.
        diagonal_weight (`float`, *optional*, defaults to 0.75): Weight of the diagonal term; the self-distillation one
            weighs `1 - diagonal_weight`.
        time_logit_mean (`float`, *optional*, defaults to 0.4): Mean of the logit-normal time distribution.
        time_logit_std (`float`, *optional*, defaults to 1.0): Standard deviation of the logit-normal time distribution.
    """
    # Repeated sample-major, as the original flattens `(num_samples, batch, frames)`, so the noise lines up.
    conditions = conditions.repeat(num_samples, 1)
    data = targets.repeat(num_samples, 1)
    noise = torch.randn_like(data)
    latent_dim = data.shape[-1]

    def velocity(start_times, end_times, latents):
        return flow_head(latents, conditions, times=[start_times, end_times])

    # Diagonal term: the instantaneous flow at `t`.
    times = torch.sigmoid(torch.randn_like(data[..., :1]) * time_logit_std + time_logit_mean)
    noisy = times * data + (1 - times) * noise
    flow_diagonal = (velocity(times, times, noisy) - (data - noise)).square().sum(dim=-1)
    if loss_weighting is not None:
        log_variance = loss_weighting(times, times).squeeze(-1)
        flow_diagonal = flow_diagonal * log_variance.exp() / latent_dim - log_variance

    skip_distill = (
        training
        and distill_probability < 1.0
        and float(torch.rand((), generator=distill_generator)) >= distill_probability
    )
    if skip_distill:
        return (diagonal_weight * flow_diagonal).mean()

    # Self-distillation term: the jump from `s` to `t` against the instantaneous flow at its endpoint.
    logits = torch.randn_like(data[..., :2]) * time_logit_std + time_logit_mean
    start_times = torch.sigmoid(logits.min(-1)[0])[..., None]
    end_times = torch.sigmoid(logits.max(-1)[0])[..., None]
    start = start_times * data + (1 - start_times) * noise
    jump_velocity, jump_velocity_dt = torch.func.jvp(
        velocity,
        (start_times, end_times, start),
        (torch.zeros_like(start_times), torch.ones_like(end_times), torch.zeros_like(start)),
    )
    end = start + (end_times - start_times) * jump_velocity
    end_dt = jump_velocity + (end_times - start_times) * jump_velocity_dt
    end_velocity = _grad_to_latents_only(flow_head, end, conditions, [end_times, end_times])
    flow_distill = (end_dt - end_velocity).square().sum(dim=-1)
    if loss_weighting is not None:
        log_variance = loss_weighting(start_times, end_times).squeeze(-1)
        flow_distill = flow_distill * log_variance.exp() / latent_dim - log_variance

    distill_weight = (1 - diagonal_weight) / (distill_probability if training else 1.0)
    return (diagonal_weight * flow_diagonal + distill_weight * flow_distill).mean()
