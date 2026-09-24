# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Convert AudioGen (AudioCraft) Hub checkpoints to `MusicgenForConditionalGeneration`.

Example Hub layout: `facebook/audiogen-medium` ships `state_dict.bin` (LM) and `compression_state_dict.bin` (EnCodec).

Requires: `pip install omegaconf` (for `xp.cfg` resolution). Does not require the `audiocraft` package.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from transformers import (
    AutoTokenizer,
    EncodecFeatureExtractor,
    MusicgenDecoderConfig,
    MusicgenForConditionalGeneration,
    MusicgenProcessor,
    T5EncoderModel,
)
from transformers.models.musicgen.convert_musicgen_transformers import (
    EXPECTED_MISSING_KEYS,
    rename_state_dict,
)
from transformers.models.musicgen.encodec_audiocraft_to_hf import load_encodec_from_audiocraft_compression_pkg
from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def _require_omegaconf():
    try:
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError(
            "AudioGen conversion requires `omegaconf`. Install with `pip install omegaconf`."
        ) from e
    return OmegaConf


def _load_torch_checkpoint(path: str, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def decoder_config_from_lm_xp_cfg(cfg) -> MusicgenDecoderConfig:
    """Build `MusicgenDecoderConfig` from AudioCraft `xp.cfg` embedded in `state_dict.bin`."""
    OmegaConf = _require_omegaconf()
    OmegaConf.resolve(cfg)
    t = cfg.transformer_lm
    return MusicgenDecoderConfig(
        hidden_size=int(t.dim),
        num_hidden_layers=int(t.num_layers),
        num_attention_heads=int(t.num_heads),
        ffn_dim=int(t.dim * t.hidden_scale),
        num_codebooks=int(t.n_q),
        audio_channels=1,
        vocab_size=int(t.card),
        max_position_embeddings=int(t.card),
        pad_token_id=int(t.card),
        bos_token_id=int(t.card),
    )


def decoder_config_from_json(path: str) -> MusicgenDecoderConfig:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    return MusicgenDecoderConfig(**raw)


@torch.no_grad()
def convert_audiogen_checkpoint(
    repo_id: str = "facebook/audiogen-medium",
    lm_filename: str = "state_dict.bin",
    compression_filename: str = "compression_state_dict.bin",
    pytorch_dump_folder: str | None = None,
    push_to_hub: str | None = None,
    device: str = "cpu",
    decoder_config_override: str | None = None,
    text_encoder_id: str = "google-t5/t5-base",
    skip_forward_test: bool = False,
) -> MusicgenForConditionalGeneration:
    from huggingface_hub import hf_hub_download

    OmegaConf = _require_omegaconf()

    lm_path = hf_hub_download(repo_id=repo_id, filename=lm_filename)
    comp_path = hf_hub_download(repo_id=repo_id, filename=compression_filename)

    lm_pkg = _load_torch_checkpoint(lm_path, map_location=device)
    comp_pkg = _load_torch_checkpoint(comp_path, map_location=device)

    cfg = OmegaConf.create(lm_pkg["xp.cfg"])
    if decoder_config_override:
        decoder_config = decoder_config_from_json(decoder_config_override)
    else:
        decoder_config = decoder_config_from_lm_xp_cfg(cfg)

    raw_lm_state = lm_pkg["best_state"]
    if (
        isinstance(raw_lm_state, dict)
        and "model" in raw_lm_state
        and isinstance(raw_lm_state["model"], dict)
        and not any(str(k).startswith(("emb", "transformer", "linears")) for k in raw_lm_state)
    ):
        raw_lm_state = raw_lm_state["model"]
    decoder_state_dict: OrderedDict = OrderedDict(raw_lm_state)
    decoder_state_dict, enc_dec_proj_state_dict = rename_state_dict(decoder_state_dict, decoder_config.hidden_size)

    text_encoder = T5EncoderModel.from_pretrained(text_encoder_id)
    audio_encoder = load_encodec_from_audiocraft_compression_pkg(comp_pkg)
    decoder = MusicgenForCausalLM(decoder_config).eval().to(device)

    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)
    for key in missing_keys.copy():
        if key.startswith(("text_encoder", "audio_encoder")) or key in EXPECTED_MISSING_KEYS:
            missing_keys.remove(key)
    if len(missing_keys) > 0:
        raise ValueError(f"Missing key(s) in state_dict: {missing_keys}")
    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected key(s) in state_dict: {unexpected_keys}")

    model = MusicgenForConditionalGeneration(
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        decoder=decoder,
    )
    model.enc_to_dec_proj.load_state_dict(enc_dec_proj_state_dict)

    if not skip_forward_test:
        input_ids = torch.arange(0, 2 * decoder_config.num_codebooks, dtype=torch.long, device=device).reshape(2, -1)
        decoder_input_ids = input_ids.reshape(2 * decoder_config.num_codebooks, -1)
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits
        if logits.shape != (2 * decoder_config.num_codebooks, 1, decoder_config.vocab_size):
            raise ValueError(f"Unexpected logits shape {logits.shape}; check decoder config / checkpoint pairing.")

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_id)
    feature_extractor = EncodecFeatureExtractor(
        feature_size=decoder_config.audio_channels,
        sampling_rate=audio_encoder.config.sampling_rate,
        padding_side="left",
    )
    processor = MusicgenProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    model.generation_config.decoder_start_token_id = decoder_config.pad_token_id
    model.generation_config.pad_token_id = decoder_config.pad_token_id
    model.generation_config.max_length = int(30 * audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True
    model.generation_config.guidance_scale = 3.0

    if pytorch_dump_folder is not None:
        Path(pytorch_dump_folder).mkdir(parents=True, exist_ok=True)
        logger.info("Saving converted AudioGen model to %s", pytorch_dump_folder)
        model.save_pretrained(pytorch_dump_folder)
        processor.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        logger.info("Pushing to Hugging Face Hub: %s", push_to_hub)
        model.push_to_hub(push_to_hub)
        processor.push_to_hub(push_to_hub)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="facebook/audiogen-medium")
    parser.add_argument("--pytorch_dump_folder", type=str, required=True)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--decoder_config_json",
        type=str,
        default=None,
        help="Optional JSON override for `MusicgenDecoderConfig`.",
    )
    parser.add_argument("--skip_forward_test", action="store_true")
    args = parser.parse_args()
    convert_audiogen_checkpoint(
        repo_id=args.repo_id,
        pytorch_dump_folder=args.pytorch_dump_folder,
        push_to_hub=args.push_to_hub,
        device=args.device,
        decoder_config_override=args.decoder_config_json,
        skip_forward_test=args.skip_forward_test,
    )
