# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


class CategoricalMixture:
    def __init__(self, param, bins=50, start=0, end=1):
        # All tensors are of shape ..., bins.
        self.logits = param
        bins = torch.linspace(
            start, end, bins + 1, device=self.logits.device, dtype=self.logits.dtype
        )
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        # Shapes are:
        #     self.probs: ... x bins
        #     true      : ...
        true_index = (
            (
                true.unsqueeze(-1)
                - self.v_bins[
                    [
                        None,
                    ]
                    * true.ndim
                ]
            )
            .abs()
            .argmin(-1)
        )
        nll = self.logits.log_softmax(-1)
        return torch.take_along_dim(nll, true_index.unsqueeze(-1), dim=-1).squeeze(-1)

    def mean(self):
        return (self.logits.softmax(-1) @ self.v_bins.unsqueeze(1)).squeeze(-1)


def categorical_lddt(logits, bins=50):
    # Logits are ..., 37, bins.
    return CategoricalMixture(logits, bins=bins).mean()
