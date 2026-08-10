# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""Convert original-author WavTokenizer checkpoints to Transformers format.

The transformers WavTokenizer implementation is an inference-only port. The original checkpoints are full Lightning
training checkpoints that additionally contain GAN discriminators, loss modules and the original SEANet decoder
(which is bypassed at inference in favor of the Vocos backbone + ISTFT head). All of these training-only components
are dropped during conversion; only the inference path (encoder, quantizer, decoder backbone, ISTFT head) is kept.

The converter supports the currently available 24 kHz releases and infers their 40- or 75-token architecture from
checkpoint tensor shapes:

    novateur/WavTokenizer:
        WavTokenizer_small_600_24k_4096.ckpt
        WavTokenizer_small_320_24k_4096.ckpt
    novateur/WavTokenizer-medium-speech-75token:
        wavtokenizer_medium_speech_320_24k.ckpt
        wavtokenizer_medium_speech_320_24k_v2.ckpt
    novateur/WavTokenizer-medium-music-audio-75token:
        wavtokenizer_medium_music_audio_320_24k.ckpt
        wavtokenizer_medium_music_audio_320_24k_v2.ckpt
    novateur/WavTokenizer-large-unify-40token:
        wavtokenizer_large_unify_600_24k.ckpt
    novateur/WavTokenizer-large-speech-75token:
        wavtokenizer_large_speech_320_v2.ckpt

Example:
    python src/transformers/models/wavtokenizer/convert_wavtokenizer_checkpoint.py \
        --checkpoint_path wavtokenizer_large_unify_600_24k.ckpt \
        --output_dir ./wavtokenizer-large-unify-40token-hf
"""

import argparse
import math
import re

import torch

from transformers import WavTokenizerConfig, WavTokenizerFeatureExtractor, WavTokenizerModel, logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# Training-only / unused branches of the original Lightning checkpoint.
IGNORE_PREFIXES = (
    "multiperioddisc.",
    "multiresddisc.",
    "dac.",
    "dacdiscriminator.",
    "melspec_loss.",
    # The original SEANet *decoder* is bypassed at inference (the Vocos backbone + ISTFT head replace it).
    "feature_extractor.encodec.decoder.",
)

# Buffers that transformers re-creates (non-persistent) instead of loading.
IGNORE_KEYS = ("head.istft.window",)


_DOWNSAMPLING_WEIGHT_RE = re.compile(r"^feature_extractor\.encodec\.encoder\.model\.(\d+)\.conv\.conv\.weight_v$")
_LSTM_WEIGHT_RE = re.compile(r"^feature_extractor\.encodec\.encoder\.model\.\d+\.lstm\.weight_ih_l(\d+)$")
_CONVNEXT_WEIGHT_RE = re.compile(r"^backbone\.convnext\.(\d+)\.dwconv\.weight$")


def _require_tensor(state_dict: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state_dict:
        raise ValueError(f"Cannot infer WavTokenizer architecture: checkpoint is missing `{key}`.")
    value = state_dict[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Cannot infer WavTokenizer architecture: `{key}` is not a tensor.")
    return value


def infer_wavtokenizer_config(state_dict: dict[str, torch.Tensor]) -> WavTokenizerConfig:
    """Infer the inference architecture of a released original WavTokenizer checkpoint.

    The original releases ship as bare ``.ckpt`` files without a config and differ in architecture (40-token models:
    hop length 600, ratios ``[6, 5, 5, 4]``; 75-token models: hop length 320, ratios ``[8, 5, 4, 2]``), so the config
    is inferred from checkpoint tensor shapes. Cross-checks between the shapes validate that the checkpoint is a
    supported WavTokenizer release and raise otherwise.
    """
    downsampling_layers = []
    for key, value in state_dict.items():
        match = _DOWNSAMPLING_WEIGHT_RE.match(key)
        if match is None or not isinstance(value, torch.Tensor) or value.ndim != 3:
            continue
        # The initial/final encoder convolutions do not double the channel dimension. Each downsampling
        # convolution does, and its kernel size is twice its stride/downsampling ratio.
        if value.shape[0] == 2 * value.shape[1]:
            kernel_size = value.shape[-1]
            if kernel_size % 2:
                raise ValueError(
                    "Cannot infer WavTokenizer architecture: encoder downsampling kernel "
                    f"`{key}` has odd size {kernel_size}."
                )
            downsampling_layers.append((int(match.group(1)), kernel_size // 2))

    downsampling_layers.sort()
    if len(downsampling_layers) != 4:
        raise ValueError(
            "Cannot infer WavTokenizer architecture: expected 4 channel-doubling encoder downsampling "
            f"convolutions, found {len(downsampling_layers)}."
        )
    upsampling_ratios = [ratio for _, ratio in reversed(downsampling_layers)]

    initial_conv = _require_tensor(state_dict, "feature_extractor.encodec.encoder.model.0.conv.conv.weight_v")
    codebook = _require_tensor(state_dict, "feature_extractor.encodec.quantizer.vq.layers.0._codebook.embed")
    backbone_embed = _require_tensor(state_dict, "backbone.embed.weight")
    intermediate = _require_tensor(state_dict, "backbone.convnext.0.pwconv1.weight")
    adanorm_scale = _require_tensor(state_dict, "backbone.norm.scale.weight")
    head = _require_tensor(state_dict, "head.out.weight")

    if initial_conv.ndim != 3 or codebook.ndim != 2 or backbone_embed.ndim != 3 or head.ndim != 2:
        raise ValueError("Cannot infer WavTokenizer architecture: checkpoint contains invalid inference tensor ranks.")

    lstm_layers = sorted(
        {int(match.group(1)) for key in state_dict if (match := _LSTM_WEIGHT_RE.match(key)) is not None}
    )
    if lstm_layers != list(range(len(lstm_layers))) or not lstm_layers:
        raise ValueError("Cannot infer WavTokenizer architecture: encoder LSTM layers are missing or non-contiguous.")

    convnext_layers = sorted(
        {int(match.group(1)) for key in state_dict if (match := _CONVNEXT_WEIGHT_RE.match(key)) is not None}
    )
    if convnext_layers != list(range(len(convnext_layers))) or not convnext_layers:
        raise ValueError(
            "Cannot infer WavTokenizer architecture: decoder ConvNeXt layers are missing or non-contiguous."
        )

    codebook_size, codebook_dim = codebook.shape
    decoder_hidden_size, hidden_size = backbone_embed.shape[:2]
    if codebook_dim != hidden_size:
        raise ValueError(
            "Cannot infer WavTokenizer architecture: codebook dimension "
            f"({codebook_dim}) differs from decoder input dimension ({hidden_size})."
        )
    if head.shape[1] != decoder_hidden_size:
        raise ValueError(
            "Cannot infer WavTokenizer architecture: ISTFT head input dimension "
            f"({head.shape[1]}) differs from decoder hidden size ({decoder_hidden_size})."
        )

    hop_length = math.prod(upsampling_ratios)
    checkpoint_n_fft = head.shape[0] - 2
    if checkpoint_n_fft != 4 * hop_length:
        raise ValueError(
            "Cannot infer WavTokenizer architecture: ISTFT head implies "
            f"n_fft={checkpoint_n_fft}, but encoder ratios {upsampling_ratios} imply "
            f"n_fft={4 * hop_length}."
        )

    return WavTokenizerConfig(
        num_filters=initial_conv.shape[0],
        upsampling_ratios=upsampling_ratios,
        num_lstm_layers=len(lstm_layers),
        hidden_size=hidden_size,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        decoder_hidden_size=decoder_hidden_size,
        decoder_intermediate_size=intermediate.shape[0],
        decoder_num_layers=len(convnext_layers),
        adanorm_num_embeddings=adanorm_scale.shape[0],
    )


def remap_key(key: str) -> str | None:
    """Map an original state-dict key to the Transformers WavTokenizerModel naming; None to drop."""
    if key in IGNORE_KEYS or any(key.startswith(prefix) for prefix in IGNORE_PREFIXES):
        return None

    # Encoder: feature_extractor.encodec.encoder.model.{i}... -> encoder_model.encoder.layers.{i}...
    key = key.replace("feature_extractor.encodec.encoder.model.", "encoder_model.encoder.layers.")
    # Quantizer: ...quantizer.vq.layers.0._codebook.* -> encoder_model.quantizer.codebook.*
    key = key.replace(
        "feature_extractor.encodec.quantizer.vq.layers.0._codebook.", "encoder_model.quantizer.codebook."
    )
    # ISTFT head: head.out.* -> head.linear.*
    key = key.replace("head.out.", "head.linear.")

    # Original SConv1d nests the conv as `.conv.conv` with legacy weight-norm keys `weight_g`/`weight_v`;
    # transformers EncodecConv1d-style modules use `.conv` with torch parametrized weight norm.
    key = re.sub(r"\.conv\.conv\.weight_g$", ".conv.parametrizations.weight.original0", key)
    key = re.sub(r"\.conv\.conv\.weight_v$", ".conv.parametrizations.weight.original1", key)
    key = re.sub(r"\.conv\.conv\.bias$", ".conv.bias", key)

    return key


def convert_checkpoint(checkpoint_path: str, output_dir: str, push_to_hub: str | None = None):
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = original.get("state_dict", original)

    converted = {}
    for key, value in state_dict.items():
        new_key = remap_key(key)
        if new_key is not None:
            converted[new_key] = value

    config = infer_wavtokenizer_config(state_dict)
    logger.info(
        "Inferred WavTokenizer architecture: upsampling_ratios=%s, hop_length=%s, frame_rate=%s",
        config.upsampling_ratios,
        config.hop_length,
        config.frame_rate,
    )
    model = WavTokenizerModel(config)
    model.load_state_dict(converted, strict=True)
    model.eval()

    feature_extractor = WavTokenizerFeatureExtractor(sampling_rate=config.sampling_rate, hop_length=config.hop_length)

    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    logger.info(f"Saved converted model + feature extractor to {output_dir}")

    if push_to_hub:
        model.push_to_hub(push_to_hub)
        feature_extractor.push_to_hub(push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to the original .ckpt file")
    parser.add_argument("--output_dir", required=True, help="Where to save the converted model")
    parser.add_argument("--push_to_hub", default=None, help="Optional Hub repo id to push to")
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint_path, args.output_dir, args.push_to_hub)
