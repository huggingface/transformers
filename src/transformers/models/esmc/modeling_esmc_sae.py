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
"""PyTorch ESMC SAE (Sparse Autoencoder) model.

* :class:`ESMCSAEModel` — the published HF container, one repo per
  ``(backbone, codebook_dim, k)`` group. Each backbone layer ships as a
  ``layer_{i}.safetensors`` shard; ``from_pretrained`` downloads the whole
  snapshot but loads no weights — callers materialize the layers they need
  via :meth:`initialize_layers`. Single-layer repos auto-load so bare
  ``forward(x)`` works.
* :class:`_ESMCSAELayer` — internal ``nn.Module`` that holds the weights for
  one ``(backbone, codebook_dim, k, layer)`` SAE. Not a published HF artifact;
  obtained only via ``model.layers["<idx>"]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ...modeling_outputs import ModelOutput  # type: ignore[import]
from ...modeling_utils import PreTrainedModel  # type: ignore[import]
from ...utils import auto_docstring  # type: ignore[import]
from .configuration_esmc_sae import ESMCSAEConfig, ESMCSAEParams


@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`ESMCSAEModel`].
    """
)
class ESMCSAEOutput(ModelOutput):
    feature_magnitudes: torch.Tensor
    reconstruction_loss: Optional[torch.Tensor] = None

    def to_sparse(self) -> None:
        self.feature_magnitudes = self.feature_magnitudes.to_sparse()


class _ESMCSAELayer(nn.Module):
    """One backbone layer's SAE — internal building block of :class:`ESMCSAEModel`.

    Not exposed via ``AutoModel`` and not loadable on its own. Obtain one
    via ``model.layers["<layer_idx>"]`` after calling ``initialize_layers``.
    """

    def __init__(self, params: ESMCSAEParams):
        super().__init__()
        self.params = params

        self.W_enc = nn.Parameter(torch.empty(params.d_model, params.codebook_dim))
        self.W_dec = nn.Parameter(torch.empty(params.codebook_dim, params.d_model))
        self.b_dec = nn.Parameter(torch.zeros(params.d_model))
        # Per-feature normalization stats. Trained alongside the SAE for some
        # variants; for variants that don't ship them, leaving these as ones
        # makes ``_get_sae_outputs``'s ``features / max * idf`` a no-op.
        self.register_buffer("idf", torch.ones(params.codebook_dim))
        self.register_buffer("max", torch.ones(params.codebook_dim))

    @property
    def layer(self) -> int:
        """Backbone-layer index this SAE is trained against."""
        return self.params.layer

    def forward(self, x: torch.Tensor, **_kwargs: object) -> ESMCSAEOutput:
        del _kwargs
        x = self._zscore_normalize_representation(x)

        x_with_pre_encoder_bias = x - self.b_dec
        preactivations = F.relu(x_with_pre_encoder_bias @ self.W_enc)

        topk = torch.topk(preactivations, self.params.k, dim=-1)
        feature_magnitudes = torch.zeros_like(preactivations).scatter(
            -1, topk.indices, topk.values
        )

        reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

        reconstruction_loss = (reconstructed - x).pow(2).mean(dim=-1)

        return ESMCSAEOutput(
            feature_magnitudes=feature_magnitudes,
            reconstruction_loss=reconstruction_loss,
        )

    def get_sae_output(
        self, layer_states: torch.Tensor, token_mask: torch.Tensor
    ) -> ESMCSAEOutput:
        _, _, v_len = layer_states.shape
        nonpad_states = layer_states[token_mask].view(-1, v_len)
        return self(nonpad_states)

    def _zscore_normalize_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=-1, keepdim=True)
        x = x - x_mean
        x_std = x.std(dim=-1, keepdim=True)
        return x / (x_std + 1e-5)


@auto_docstring
class ESMCSAEPreTrainedModel(PreTrainedModel):
    config_class = ESMCSAEConfig
    base_model_prefix = "esmc_sae"


@auto_docstring(
    custom_intro="""
    HF container holding one SAE per backbone layer, all sharing the same
    ``(d_model, codebook_dim, k)``.

    ``from_pretrained`` downloads the entire repo (every ``layer_{i}.safetensors``)
    into the local HF cache but does **not** load any weights into memory.
    Callers materialize the layers they actually need by calling
    :meth:`initialize_layers`. The full set is available on disk after the
    first call, so subsequent layer switches read from the local cache without
    re-downloading.

    Examples::

        model = ESMCSAEModel.from_pretrained(
            "biohub/esmc-6b-2024-12-sae-k64-codebook16384"
        )
        model.initialize_layers([60])                  # ~2.5 GB into memory
        out = model(layer_states, layer=60)            # forward through layer 60
        model.initialize_layers([45])                  # add layer 45 (cached locally)
        model.release_layer(60)                        # free layer 60
    """
)
class ESMCSAEModel(ESMCSAEPreTrainedModel):
    def __init__(self, config: ESMCSAEConfig):
        super().__init__(config)
        # Layers are populated lazily by ``initialize_layers``; the container
        # starts empty so ``from_pretrained`` doesn't materialize hundreds of
        # GB of unused parameters.
        self.layers = nn.ModuleDict()
        # Zero-element buffer that rides along with ``.to(device/dtype)``.
        # ``initialize_layers`` reads its current device/dtype so SAEs added
        # after ``model.to("cuda")`` land on CUDA without re-passing ``device=``.
        self.register_buffer("_device_marker", torch.empty(0), persistent=False)
        self._snapshot_dir: Optional[str] = None
        self.post_init()

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls, pretrained_model_name_or_path: str | os.PathLike, *model_args, **kwargs
    ) -> "ESMCSAEModel":
        """Download (or reuse cached) the full repo and return the model.

        By default no weights are read into memory and the caller must invoke
        :meth:`initialize_layers` before running :meth:`forward`. The single
        exception is when the repo ships exactly one layer: that layer is
        auto-loaded (honoring ``torch_dtype`` / ``device`` if passed) so the
        bare ``forward(x)`` call just works.

        Honored kwargs: ``revision``, ``cache_dir``, ``token``,
        ``allow_patterns``, ``local_files_only``, ``force_download`` (forwarded
        to ``snapshot_download``); ``torch_dtype`` and ``device`` (used by the
        single-layer auto-load path; otherwise pass them to
        :meth:`initialize_layers`). Behavioral kwargs that imply work we do
        not perform (``device_map``, ``low_cpu_mem_usage``,
        ``quantization_config``, ``attn_implementation``) raise so the user
        isn't silently misled. Other HF housekeeping kwargs (``config``,
        ``trust_remote_code``, ``adapter_kwargs``, …) are accepted and
        ignored — they only matter for the standard loader, which we bypass.
        """
        del model_args
        torch_dtype = kwargs.pop("torch_dtype", None)
        device = kwargs.pop("device", None)
        local_dir = _resolve_snapshot_dir(pretrained_model_name_or_path, kwargs)
        unsupported = {
            "device_map",
            "low_cpu_mem_usage",
            "quantization_config",
            "attn_implementation",
            "max_memory",
            "offload_folder",
            "offload_state_dict",
        } & kwargs.keys()
        if unsupported:
            raise TypeError(
                f"Unsupported kwargs to ESMCSAEModel.from_pretrained: "
                f"{sorted(unsupported)}. The standard HF loader is bypassed —"
                " call initialize_layers(..., device=, dtype=) instead."
            )
        config = ESMCSAEConfig.from_pretrained(local_dir)
        model = cls(config)
        model._snapshot_dir = str(local_dir)
        if device is not None:
            model.to(device)
        if torch_dtype is not None:
            model.to(torch_dtype)
        if len(config.available_layers) == 1:
            model.initialize_layers(list(config.available_layers))
        return model

    def initialize_layers(
        self,
        layers: list[int],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Load the requested layers from the local snapshot into memory.

        Layers already present in :attr:`self.layers` are skipped — calling
        ``initialize_layers([23])`` twice is idempotent. ``device`` / ``dtype``
        default to wherever the model itself lives (via the ``_device_marker``
        buffer that moves with ``.to(...)``), so the common pattern of
        ``model.to("cuda"); model.initialize_layers([7])`` Just Works.
        """
        assert self._snapshot_dir is not None, (
            "ESMCSAEModel has no snapshot directory — call "
            "from_pretrained first, or set _snapshot_dir manually."
        )
        if device is None:
            device = self._device_marker.device
        if dtype is None:
            dtype = self._device_marker.dtype
        snapshot_dir = Path(self._snapshot_dir)
        available = set(self.config.available_layers)
        for layer_idx in layers:
            key = str(layer_idx)
            if key in self.layers:
                continue
            if layer_idx not in available:
                raise KeyError(
                    f"Layer {layer_idx} is not in this repo. "
                    f"available_layers={sorted(available)}"
                )
            shard = snapshot_dir / f"layer_{layer_idx}.safetensors"
            if not shard.exists():
                raise FileNotFoundError(
                    f"Missing layer file {shard} — config lists layer "
                    f"{layer_idx} as available but the shard is not on disk."
                )
            params = ESMCSAEParams(
                d_model=self.config.d_model,
                codebook_dim=self.config.codebook_dim,
                k=self.config.k,
                layer=layer_idx,
            )
            # Build on the meta device so we don't allocate weights that
            # ``load_state_dict`` would immediately overwrite.
            with torch.device("meta"):
                layer = _ESMCSAELayer(params)
            layer.to_empty(device=device)
            layer.load_state_dict(load_file(str(shard)))
            layer.to(dtype=dtype)
            self.layers[key] = layer

    def release_layer(self, layer: int) -> None:
        """Drop the named layer from memory. No-op if not loaded."""
        key = str(layer)
        if key in self.layers:
            del self.layers[key]

    def loaded_layers(self) -> list[int]:
        """Sorted list of layer indices currently materialized in memory."""
        return sorted(int(k) for k in self.layers.keys())

    def forward(
        self, x: torch.Tensor, layer: int | None = None, **kwargs: object
    ) -> ESMCSAEOutput:
        if layer is None:
            if len(self.layers) == 1:
                # Unambiguous: exactly one layer loaded → use it.
                ((_only_key, only_layer),) = self.layers.items()
                return only_layer(x, **kwargs)
            if len(self.layers) == 0:
                raise RuntimeError(
                    "No layers loaded — call "
                    f"initialize_layers([...]) first. "
                    f"available_layers={self.config.available_layers}"
                )
            raise RuntimeError(
                "Multiple layers are loaded — please select one via "
                f"forward(x, layer=<idx>). Loaded layers: {self.loaded_layers()}"
            )
        key = str(layer)
        if key not in self.layers:
            raise KeyError(
                f"Layer {layer} is not loaded. Call "
                f"initialize_layers([{layer}]) first. Loaded layers: "
                f"{self.loaded_layers()}"
            )
        return self.layers[key](x, **kwargs)

    def save_pretrained(  # type: ignore[override]
        self, save_directory: str | os.PathLike, *args, **kwargs
    ) -> None:
        """Write ``config.json`` plus one ``layer_{i}.safetensors`` per loaded layer.

        Only layers currently in :attr:`self.layers` are written.
        ``available_layers`` in the saved config is synced to what's actually
        on disk so a ``release_layer`` + ``save_pretrained`` round-trip never
        advertises a layer whose shard is missing.
        """
        del args, kwargs
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        # Sync available_layers to what we're about to write — never advertise
        # a layer that isn't on disk in this repo.
        self.config.available_layers = self.loaded_layers()
        self.config.save_pretrained(str(save_directory))
        for key, layer in self.layers.items():
            shard = save_directory / f"layer_{key}.safetensors"
            save_file(
                {
                    k: v.detach().cpu().contiguous()
                    for k, v in layer.state_dict().items()
                },
                str(shard),
            )


def _resolve_snapshot_dir(
    pretrained_model_name_or_path: str | os.PathLike, kwargs: dict
) -> str:
    """Local dir → return as-is; hub id → ``snapshot_download`` it.

    A directory only counts as "local" if it actually contains ``config.json``,
    so a stale subdir named like a hub id (``./biohub/esmc-...``)
    doesn't accidentally shadow the hub fetch.

    Pops the standard ``snapshot_download`` keyword args from ``kwargs`` so
    callers can forward them via ``from_pretrained``.
    """
    path = Path(pretrained_model_name_or_path)
    if path.is_dir() and (path / "config.json").exists():
        return str(path)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=str(pretrained_model_name_or_path),
        revision=kwargs.pop("revision", None),
        cache_dir=kwargs.pop("cache_dir", None),
        token=kwargs.pop("token", None),
        allow_patterns=kwargs.pop("allow_patterns", None),
        local_files_only=kwargs.pop("local_files_only", False),
        force_download=kwargs.pop("force_download", False),
    )


__all__ = ["ESMCSAEModel", "ESMCSAEOutput", "ESMCSAEPreTrainedModel"]
