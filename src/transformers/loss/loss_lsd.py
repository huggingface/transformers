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
"""Lagrangian self-distillation (LSD) loss terms, and the Pocket TTS training loss built on them."""

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn


# Adapted from https://github.com/kyutai-labs/pocket-tts/blob/b94db177bc0f77517d9b124843e25605d6f9aed8/training/modules/utils.py#L100-L124
class _InputOnlyGrad(torch.autograd.Function):
    """`function(inputs)`, whose backward reaches `inputs` only, never the parameters `function` uses."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        with torch.enable_grad():
            outputs = function(inputs)
        ctx.save_for_backward(inputs, outputs)
        return outputs.detach()

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        inputs, outputs = ctx.saved_tensors
        return torch.autograd.grad(outputs, inputs, grad_outputs)[0], None


def lsd_diagonal_loss(
    velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    noise: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """
    Diagonal (flow matching) term of the Lagrangian self-distillation loss from [How to build a consistency model:
    Learning flow maps via self-distillation](https://huggingface.co/papers/2505.18825):
    `|v_{t,t}(x_t) - (data - noise)|^2`, with `x_t = t * data + (1 - t) * noise`.

    Args:
        velocity (`Callable`): `velocity(x, s, t)`, returns `v_{s,t}(x)`.
        data (`torch.Tensor` of shape `(batch_size, latent_dim)`): Target latents.
        noise (`torch.Tensor` of shape `(batch_size, latent_dim)`): Gaussian noise.
        times (`torch.Tensor` of shape `(batch_size, 1)`): Times `t`.

    Returns:
        `torch.Tensor` of shape `(batch_size,)`: the term of each row.
    """
    noisy = times * data + (1 - times) * noise
    return (velocity(noisy, times, times) - (data - noise)).square().sum(dim=-1)


def lsd_distillation_loss(
    velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    noise: torch.Tensor,
    start_times: torch.Tensor,
    end_times: torch.Tensor,
    stop_gradient: Literal["full", "minimal"] = "full",
) -> torch.Tensor:
    """
    Lagrangian term of the Lagrangian self-distillation loss from [How to build a consistency model: Learning flow maps
    via self-distillation](https://huggingface.co/papers/2505.18825):
    `|d/dt X_{s,t}(x_s) - sg(v_{t,t}(X_{s,t}(x_s)))|^2`, with the jump `X_{s,t}(x) = x + (t - s) * v_{s,t}(x)`.

    Args:
        velocity (`Callable`): `velocity(x, s, t)`, returns `v_{s,t}(x)`.
        data (`torch.Tensor` of shape `(batch_size, latent_dim)`): Target latents.
        noise (`torch.Tensor` of shape `(batch_size, latent_dim)`): Gaussian noise.
        start_times (`torch.Tensor` of shape `(batch_size, 1)`): Times `s`.
        end_times (`torch.Tensor` of shape `(batch_size, 1)`): Times `t`, with `s <= t`.
        stop_gradient (`str`, *optional*, defaults to `"full"`): `"full"` gives the target no gradient (the paper),
            `"minimal"` lets it reach the target's input `X_{s,t}(x_s)` but not the model's parameters.

    Returns:
        `torch.Tensor` of shape `(batch_size,)`: the term of each row.
    """
    start = start_times * data + (1 - start_times) * noise
    jump_velocity, jump_velocity_dt = torch.func.jvp(
        velocity,
        (start, start_times, end_times),
        (torch.zeros_like(start), torch.zeros_like(start_times), torch.ones_like(end_times)),
    )
    end = start + (end_times - start_times) * jump_velocity
    end_dt = jump_velocity + (end_times - start_times) * jump_velocity_dt
    if stop_gradient == "minimal":
        target = _InputOnlyGrad.apply(end, lambda latents: velocity(latents, end_times, end_times))
    else:
        with torch.no_grad():
            target = velocity(end, end_times, end_times)
    return (end_dt - target).square().sum(dim=-1)


def end_of_speech_loss(eos_logits: torch.Tensor, eos_labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy of the end-of-speech logits, `eos_labels` being `1` at the end-of-speech position."""
    eos_logits = eos_logits.float()
    is_eos = eos_labels.bool()
    loss = torch.where(is_eos, nn.functional.softplus(-eos_logits), nn.functional.softplus(eos_logits))
    return loss.mean()


def PocketTTSForConditionalGenerationLoss(
    flow_head: nn.Module,
    conditions: torch.Tensor,
    targets: torch.Tensor,
    eos_logits: torch.Tensor,
    eos_labels: torch.Tensor,
    config,
    loss_weighting: nn.Module,
    distill_generator: torch.Generator,
    training: bool,
    **kwargs,
) -> torch.Tensor:
    """
    Kyutai's Pocket TTS training loss: the LSD flow loss of the target frames, with Kyutai's recipe (logit-normal times,
    both terms weighted on every row, the self-distillation term on a fraction of training steps, learnt weighting),
    plus `config.eos_loss_weight` times the end-of-speech loss.

    Args:
        flow_head (`nn.Module`): The flow head, `flow_head(latents, conditions, times=[s, t])`.
        conditions (`torch.Tensor` of shape `(num_frames, hidden_size)`): Backbone hidden states of the target frames.
        targets (`torch.Tensor` of shape `(num_frames, latent_dim)`): Normalised target latents.
        eos_logits (`torch.Tensor` of shape `(num_positions,)`): End-of-speech logits.
        eos_labels (`torch.Tensor` of shape `(num_positions,)`): `1` at the end-of-speech position, `0` elsewhere.
        config ([`PocketTTSConfig`]): Holds the loss hyperparameters.
        loss_weighting (`nn.Module`): Learnt log-variance `w(s, t)` of the LSD terms.
        distill_generator (`torch.Generator`): Generator of the self-distillation draw, identical on every rank.
        training (`bool`): Whether the self-distillation term may be skipped (always computed outside training).
    """
    # `num_samples` noise draws per frame, sample-major as in the original
    conditions = conditions.repeat(config.flow_loss_num_samples, 1)
    targets = targets.repeat(config.flow_loss_num_samples, 1)
    latent_dim = targets.shape[-1]

    def velocity(latents, start_times, end_times):
        return flow_head(latents, conditions, times=[start_times, end_times])

    def time_logits(num_times):
        return torch.randn_like(targets[..., :num_times]) * config.lsd_time_logit_std + config.lsd_time_logit_mean

    # EDM2-style weighting of the paper, with `w` negated and the term averaged over dims (Kyutai)
    def weighted(loss, start_times, end_times):
        log_variance = loss_weighting(start_times, end_times).squeeze(-1)
        return loss * log_variance.exp() / latent_dim - log_variance

    noise = torch.randn_like(targets)
    times = torch.sigmoid(time_logits(1))
    flow_diagonal = weighted(lsd_diagonal_loss(velocity, targets, noise, times), times, times)

    distill_probability = config.lsd_distill_probability if training else 1.0
    if distill_probability < 1.0 and float(torch.rand((), generator=distill_generator)) >= distill_probability:
        flow_loss = (config.lsd_diagonal_weight * flow_diagonal).mean()
    else:
        logits = time_logits(2)
        start_times = torch.sigmoid(logits.min(-1)[0])[..., None]
        end_times = torch.sigmoid(logits.max(-1)[0])[..., None]
        flow_distill = lsd_distillation_loss(velocity, targets, noise, start_times, end_times, stop_gradient="minimal")
        flow_distill = weighted(flow_distill, start_times, end_times)
        distill_weight = (1 - config.lsd_diagonal_weight) / distill_probability
        flow_loss = (config.lsd_diagonal_weight * flow_diagonal + distill_weight * flow_distill).mean()

    return flow_loss + config.eos_loss_weight * end_of_speech_loss(eos_logits, eos_labels)
