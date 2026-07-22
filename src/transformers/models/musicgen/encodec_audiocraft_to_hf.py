# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""Map AudioCraft EnCodec `best_state` tensors into Hugging Face `EncodecModel` weights (e.g. `facebook/audiogen-medium`)."""

from __future__ import annotations

from typing import Any

import torch

from transformers import EncodecConfig, EncodecModel


def _load_torch_checkpoint(path: str, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _require_omegaconf():
    try:
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError(
            "AudioGen / AudioCraft EnCodec conversion requires `omegaconf`. Install with `pip install omegaconf`."
        ) from e
    return OmegaConf


def build_encodec_config_from_audiocraft_compression_pkg(pkg: dict[str, Any]) -> EncodecConfig:
    """Build `EncodecConfig` for AudioGen-Medium's 16 kHz EnCodec (Hub `compression_state_dict.bin`)."""
    OmegaConf = _require_omegaconf()
    cfg = OmegaConf.create(pkg["xp.cfg"])
    OmegaConf.resolve(cfg)
    seanet = cfg.seanet
    rvq = cfg.rvq
    return EncodecConfig(
        sampling_rate=int(cfg.sample_rate),
        target_bandwidths=(2.2,),
        upsampling_ratios=list(seanet.ratios),
        audio_channels=int(cfg.encodec.channels),
        norm_type="weight_norm",
        num_filters=int(seanet.n_filters),
        hidden_size=int(seanet.dimension),
        num_residual_layers=int(seanet.n_residual_layers),
        kernel_size=int(seanet.kernel_size),
        last_kernel_size=int(seanet.last_kernel_size),
        compress=int(seanet.compress),
        num_lstm_layers=int(seanet.lstm),
        use_causal_conv=bool(seanet.causal),
        pad_mode=str(seanet.pad_mode),
        codebook_size=int(rvq.bins),
        use_conv_shortcut=False,
    )


def _map_non_quant_key(k: str) -> str:
    k = k.replace("encoder.model.", "encoder.layers.")
    k = k.replace("decoder.model.", "decoder.layers.")
    k = k.replace(".convtr.convtr.", ".conv.")
    if ".conv.conv." in k:
        k = k.replace(".conv.conv.", ".conv.")
    if k.endswith(".conv.bias"):
        return k
    if k.endswith(".conv.weight_g"):
        return k.replace(".conv.weight_g", ".conv.parametrizations.weight.original0")
    if k.endswith(".conv.weight_v"):
        return k.replace(".conv.weight_v", ".conv.parametrizations.weight.original1")
    return k


def audiocraft_compression_state_to_hf_state_dict(best_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename AudioCraft `CompressionModel` keys to `transformers.EncodecModel` state dict keys."""
    out: dict[str, torch.Tensor] = {}
    for ak, v in best_state.items():
        if ak.startswith("quantizer"):
            hk = ak.replace("quantizer.vq.layers.", "quantizer.layers.").replace("._codebook.", ".codebook.")
        else:
            hk = _map_non_quant_key(ak)
        out[hk] = v
    return out


def load_encodec_from_audiocraft_compression_pkg(pkg: dict[str, Any]) -> EncodecModel:
    enc_cfg = build_encodec_config_from_audiocraft_compression_pkg(pkg)
    model = EncodecModel(enc_cfg)
    new_sd = audiocraft_compression_state_to_hf_state_dict(pkg["best_state"])
    model.load_state_dict(new_sd, strict=True)
    return model


def load_encodec_from_audiocraft_hub(
    repo_id: str = "facebook/audiogen-medium", filename: str = "compression_state_dict.bin"
) -> EncodecModel:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename)
    pkg = _load_torch_checkpoint(path)
    return load_encodec_from_audiocraft_compression_pkg(pkg)
