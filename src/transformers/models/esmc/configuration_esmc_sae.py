# Copyright 2026 Biohub. All rights reserved.
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
"""ESMC sparse autoencoder (SAE) configuration."""

from dataclasses import dataclass

from ...configuration_utils import PretrainedConfig  # type: ignore[import]


@dataclass
class ESMCSAEParams:
    """Parameters for one backbone layer's SAE inside :class:`ESMCSAEModel`.

    The SAE itself is an internal ``nn.Module``; this dataclass just bundles
    the handful of fields needed to instantiate one.
    """

    d_model: int = 2560
    codebook_dim: int = 65536
    k: int = 64
    layer: int = 0


class ESMCSAEConfig(PretrainedConfig):
    """
    Configuration class for [`ESMCSAEModel`] — a container that holds one
    SAE per backbone layer for a fixed ``(model, codebook_dim, k)`` group.

    All SAEs in a container share ``d_model``, ``codebook_dim``, and ``k``;
    they differ only in the backbone layer they were trained on.
    ``available_layers`` lists the backbone-layer indices the repo ships;
    each entry ``i`` is stored on disk as ``layer_{i}.safetensors`` (the
    filename index *is* the backbone layer, so a single-layer repo for
    layer 23 stores ``layer_23.safetensors``).

    Args:
        d_model (`int`, *optional*, defaults to 2560):
            Dimensionality of the ESMC hidden states fed into the SAEs.
        codebook_dim (`int`, *optional*, defaults to 65536):
            Number of sparse features in each SAE's codebook.
        k (`int`, *optional*, defaults to 64):
            Top-k sparsity per SAE.
        available_layers (`list[int]`, *optional*, defaults to ``[0]``):
            Which backbone-layer indices the repo ships.
    """

    model_type = "esmc_sae"

    def __init__(
        self,
        d_model: int = 2560,
        codebook_dim: int = 65536,
        k: int = 64,
        available_layers: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.codebook_dim = codebook_dim
        self.k = k
        self.available_layers = (
            list(available_layers) if available_layers is not None else [0]
        )


__all__ = ["ESMCSAEConfig", "ESMCSAEParams"]
