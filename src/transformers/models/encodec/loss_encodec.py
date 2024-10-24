import typing as tp
from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F
from torch import autograd


"""
    Balancer code directly copied from: https://github.com/facebookresearch/encodec/blob/main/encodec/balancer.py
"""


class Balancer:
    """Loss balancer.

    The loss balancer combines losses together to compute gradients for the backward.
    A call to the balancer will weight the losses according the specified weight coefficients.
    A call to the backward method of the balancer will compute the gradients, combining all the losses and
    potentially rescaling the gradients, which can help stabilize the training and reasonate
    about multiple losses with varying scales.

    Expected usage:
        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            balancer.backward(losses, x)

    ..Warning:: It is unclear how this will interact with DistributedDataParallel,
        in particular if you have some losses not handled by the balancer. In that case
        you can use `encodec.distrib.sync_grad(model.parameters())` and
        `encodec.distrib.sync_buffwers(model.buffers())` as a safe alternative.

    Args:
        weights (Dict[str, float]): Weight coefficient for each loss. The balancer expect the losses keys
            from the backward method to match the weights keys to assign weight to each of the provided loss.
        rescale_grads (bool): Whether to rescale gradients or not, without. If False, this is just
            a regular weighted sum of losses.
        total_norm (float): Reference norm when rescaling gradients, ignored otherwise.
        emay_decay (float): EMA decay for averaging the norms when `rescale_grads` is True.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not. This only holds
            when rescaling the gradients.
        epsilon (float): Epsilon value for numerical stability.
        monitor (bool): Whether to store additional ratio for each loss key in metrics.
    """

    def __init__(
        self,
        weights: tp.Dict[str, float],
        rescale_grads: bool = True,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        per_batch_item: bool = True,
        epsilon: float = 1e-12,
        monitor: bool = False,
    ):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.averager = averager(ema_decay)
        self.epsilon = epsilon
        self.monitor = monitor
        self.rescale_grads = rescale_grads
        self._metrics: tp.Dict[str, tp.Any] = {}

    @property
    def metrics(self):
        return self._metrics

    def backward(self, losses: tp.Dict[str, torch.Tensor], input: torch.Tensor):
        norms = {}
        grads = {}
        for name, loss in losses.items():
            (grad,) = autograd.grad(loss, [input], retain_graph=True, allow_unused=True)
            if grad is not None:
                if self.per_batch_item:
                    dims = tuple(range(1, grad.dim()))
                    norm = grad.norm(dim=dims).mean()
                else:
                    norm = grad.norm()
                norms[name] = norm
                grads[name] = grad

        count = 1
        if self.per_batch_item:
            count = len(next(iter(grads.values())))
        avg_norms = average_metrics(self.averager(norms), count)
        total = sum(avg_norms.values())

        self._metrics = {}
        if self.monitor:
            for k, v in avg_norms.items():
                self._metrics[f"ratio_{k}"] = v / total

        total_weights = sum([self.weights[k] for k in avg_norms])
        ratios = {k: w / total_weights for k, w in self.weights.items()}

        out_grad: tp.Any = 0
        for name, avg_norm in avg_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (self.epsilon + avg_norm)
                grad = grads[name] * scale
            else:
                grad = self.weights[name] * grads[name]
            out_grad += grad
        input.backward(out_grad)


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_distributed():
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)


def average_metrics(metrics: tp.Dict[str, float], count=1.0):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))


def averager(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: tp.Dict[str, tp.Any], weight: float = 1) -> tp.Dict[str, float]:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}

    return _update


def compute_discriminator_loss(
    real_logits: List[torch.Tensor], fake_logits: List[torch.Tensor], num_discriminators: int
) -> torch.Tensor:
    """
    Compute the discriminator loss based on real and fake logits.

    Args:
        real_logits (List[torch.Tensor]): List of real logits from discriminators.
        fake_logits (List[torch.Tensor]): List of fake logits from discriminators.
        num_discriminators (int): Number of discriminators.

    Returns:
        torch.Tensor: The computed discriminator loss.
    """
    loss = 0.0
    for real_logit, fake_logit in zip(real_logits, fake_logits):
        loss += torch.mean(F.relu(1 - real_logit)) + torch.mean(F.relu(1 + fake_logit))
    return loss / num_discriminators


def compute_generator_adv_loss(fake_logits: List[torch.Tensor], num_discriminators: int) -> torch.Tensor:
    """
    Compute the generator adversarial loss using fake logits.

    Args:
        fake_logits (List[torch.Tensor]): List of fake logits from discriminators.
        num_discriminators (int): Number of discriminators.

    Returns:
        torch.Tensor: The computed generator adversarial loss.
    """
    loss = 0.0
    for fake_logit in fake_logits:
        loss += torch.mean(F.relu(1 - fake_logit))
    return loss / num_discriminators


def compute_feature_matching_loss(
    real_features: List[List[torch.Tensor]], fake_features: List[List[torch.Tensor]], num_discriminators: int
):
    """
    Compute the feature matching loss between real and fake features.

    Args:
        real_features (List[List[torch.Tensor]]): List of lists containing real features from each discriminator.
        fake_features (List[List[torch.Tensor]]): List of lists containing fake features from each discriminator.
        num_discriminators (int): Number of discriminators.

    Returns:
        torch.Tensor: The computed feature matching loss.
    """
    fm_loss = 0
    for k in range(num_discriminators):
        for real_feat, fake_feat in zip(real_features[k], fake_features[k]):
            fm_loss += F.l1_loss(fake_feat, real_feat.detach()) / torch.mean(torch.abs(real_feat.detach()))
    fm_loss /= num_discriminators * len(real_features[0])
    return fm_loss
