# Copyright 2026 The Emu team, BAAI, The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""Convert the original EMU3.5 Vision Tokenizer to the Apertus 1.5 `Apertus1p5VisionTokenizerModel` format.

`Apertus1p5VisionTokenizerModel` is an encode-only port of BAAI's EMU3.5 Vision Tokenizer
(`BAAI/Emu3.5-VisionTokenizer`, Apache-2.0): Apertus 1.5 consumes discrete image codes but never generates
images, so the decoder branch is dropped. The remaining tensors keep their original names (there is no
renaming), and `load_state_dict(strict=True)` is the correctness gate for both the tensor set and the
geometry derived from the source configuration.

The source may be a local directory or a Hub `repo_id[@revision]`, of which only two files are read: a single
float32 `model.safetensors` and `config.json`. A full snapshot download would also pull the redundant 1.8 GB
`model.ckpt`, remote-code modules that this converter deliberately never executes, and a research-style
`config.yaml` whose nested `ddconfig` block uses different field names and is not the mapping source.

The configuration is derived from `config.json` rather than assumed, using the field mapping in
`ORIGINAL_CONFIG_FIELDS` below plus `dropout`. The decoder-only `out_ch` and `double_z` are not mapped; the
released tokenizer sets `double_z` false, and a source setting it true would widen the kept
`encoder.conv_out`, which the strict load rejects. Every mapped field except `dropout` is key- or
shape-bearing, so a wrong mapping fails that load rather than passing silently. In particular, `resolution`
and `attn_resolutions` decide which encoder stages carry attention blocks, and therefore change the key set
itself.

The output directory is the `--vision_tokenizer_checkpoint` input of `convert_apertus1p5_weights_to_hf.py`,
which assembles the full Apertus 1.5 composite checkpoint.

Example:
    python src/transformers/models/apertus1p5/convert_apertus1p5_vision_tokenizer_to_hf.py \
        --checkpoint_path BAAI/Emu3.5-VisionTokenizer \
        --output_dir /path/to/apertus1p5-vision-tokenizer-hf \
        --verify
"""

import argparse
import json
import os
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from transformers import Apertus1p5VisionTokenizerConfig, Apertus1p5VisionTokenizerModel, logging
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_NAME
from transformers.utils.hub import cached_file


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# The port implements the encoder, the quantizer and `quant_conv` only; the original decoder branch and the
# projection feeding it have no counterpart and are dropped.
DROPPED_PREFIXES = ("decoder.", "post_quant_conv.")

# `Apertus1p5VisionTokenizerConfig` field <- original EMU3.5 `config.json` field. `out_ch` and `double_z`
# belong to the dropped decoder. `dropout` is handled separately: it is the only mapped field that carries no
# parameters, so a source omitting it stays unambiguous.
ORIGINAL_CONFIG_FIELDS = {
    "codebook_size": "codebook_size",
    "embed_dim": "embed_dim",
    "latent_channels": "z_channels",
    "in_channels": "in_channels",
    "base_channels": "ch",
    "channel_multiplier": "ch_mult",
    "num_res_blocks": "num_res_blocks",
    "attn_resolutions": "attn_resolutions",
    "resolution": "resolution",
}

ORIGINAL_MODEL_TYPE = "Emu3p5VisionVQ"

# `--verify` inputs are intentionally small
VERIFY_IMAGE_SIZES = ((64, 64), (64, 96), (96, 64), (49, 45))


def _resolve_source_file(checkpoint_path: str, filename: str) -> str:
    """Resolve one file of a local directory or a Hub `repo_id[@revision]`, downloading it if needed."""
    if os.path.isdir(checkpoint_path):
        resolved = cached_file(checkpoint_path, filename)
    else:
        repo_id, _, revision = checkpoint_path.partition("@")
        resolved = cached_file(repo_id, filename, revision=revision or None)
    if resolved is None:
        # `cached_file` returns None instead of raising for a missing `config.json`
        raise OSError(f"{checkpoint_path!r} does not contain `{filename}`.")
    return resolved


def read_original_config(checkpoint_path: str) -> dict:
    """Read the original EMU3.5 `config.json`.

    The repo also ships a research-style `config.yaml` with a nested `ddconfig` block and different field
    names; only `config.json` is read.
    """
    with open(_resolve_source_file(checkpoint_path, CONFIG_NAME)) as config_file:
        return json.load(config_file)


def convert_config(original_config: dict) -> Apertus1p5VisionTokenizerConfig:
    """Build the port's configuration from the original EMU3.5 configuration."""
    model_type = original_config.get("model_type")
    if model_type != ORIGINAL_MODEL_TYPE:
        # the likeliest mistake is pointing at the EMU3 (v1) tokenizer, whose fields map cleanly but whose
        # architecture differs, turning this into a wall of strict-load key errors further down
        logger.warning(
            f"Expected an EMU3.5 vision tokenizer (`model_type` {ORIGINAL_MODEL_TYPE!r}), got {model_type!r}."
        )

    missing = sorted({source for source in ORIGINAL_CONFIG_FIELDS.values() if source not in original_config})
    if missing:
        raise ValueError(
            f"The source configuration is missing {', '.join(f'`{key}`' for key in missing)}; expected the "
            "`config.json` of an EMU3.5 vision tokenizer such as `BAAI/Emu3.5-VisionTokenizer`."
        )

    return Apertus1p5VisionTokenizerConfig(
        dropout=original_config.get("dropout", 0.0),
        **{target: original_config[source] for target, source in ORIGINAL_CONFIG_FIELDS.items()},
    )


def convert_state_dict(original_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop the original decoder branch; the encode path keeps its original key names one to one."""
    return {key: value for key, value in original_state_dict.items() if not key.startswith(DROPPED_PREFIXES)}


def _check_fp32_source(state_dict: dict[str, torch.Tensor]) -> None:
    """Reject half-precision sources.

    `load_state_dict` copies into the model's float32 parameters, so a half-precision source would produce a
    float32-looking checkpoint whose codes are already degraded, and no later check could tell the difference,
    because code assignment is a precision-sensitive argmax.
    """
    for key, value in state_dict.items():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(
                f"The source stores `{key}` in {value.dtype}; vision tokenizer weights must be stored in "
                "float32 (half-precision weights flip discrete codes)."
            )


def convert_checkpoint(checkpoint_path: str, output_dir: str) -> None:
    """Convert an original EMU3.5 vision tokenizer into `Apertus1p5VisionTokenizerModel` format."""
    if os.path.isdir(checkpoint_path) and os.path.realpath(checkpoint_path) == os.path.realpath(output_dir):
        # the conversion overwrites `config.json` and `model.safetensors`, which would destroy the source
        # (and corrupt the Hub cache in place if `--checkpoint_path` pointed at a cached snapshot)
        raise ValueError(
            f"`output_dir` resolves to the same directory as the source checkpoint {checkpoint_path!r}; "
            "choose a separate output directory."
        )

    config = convert_config(read_original_config(checkpoint_path))
    # filtering the freshly loaded dict in one expression releases the dropped decoder tensors immediately
    state_dict = convert_state_dict(load_file(_resolve_source_file(checkpoint_path, SAFE_WEIGHTS_NAME)))
    _check_fp32_source(state_dict)
    logger.info(f"keeping {len(state_dict)} encoder tensors (dropped {', '.join(DROPPED_PREFIXES)})")

    model = Apertus1p5VisionTokenizerModel(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    model.save_pretrained(output_dir)
    logger.info(f"Saved converted model to {output_dir}")


def _example_pixel_values(height: int, width: int, seed: int = 0) -> torch.Tensor:
    """Deterministic pixel values in the encoder's input contract: RGB in [-1, 1], as the image processor emits.

    Dense random values are the most sensitive input for the bit-exactness checks below: structured images
    produce long runs of identical codes that would mask a difference.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, height, width, generator=generator) * 2 - 1


def _comparable(value):
    """Normalize a config value for comparison; sequence fields round-trip through JSON as lists."""
    return list(value) if isinstance(value, (list, tuple)) else value


def verify(output_dir: str, checkpoint_path: str) -> None:
    """Reload a converted checkpoint and check its stored precision, the configuration derived from the source,
    the code-grid geometry, batched encoding, the save/reload round trip, and the float32 keep under a
    bfloat16 load. Raises `RuntimeError` listing every failed check."""
    failed_checks = []

    model, loading_info = Apertus1p5VisionTokenizerModel.from_pretrained(output_dir, output_loading_info=True)
    loading_problems = {kind: keys for kind, keys in loading_info.items() if keys}
    if loading_problems:
        raise RuntimeError(f"The converted checkpoint did not load cleanly: {loading_problems}")
    model = model.eval()
    config = model.config

    # --- stored precision: the weights must be saved in float32 ---------------------------------------------
    # The dtypes are read from the written file, not from the loaded model. `_keep_in_fp32_modules_strict`
    # covers every submodule, so `from_pretrained` upcasts whatever the checkpoint holds and the live
    # parameters are float32 either way. The tensor set itself is already covered: the clean-load check above
    # rejects any missing or unexpected key.
    with safe_open(os.path.join(output_dir, SAFE_WEIGHTS_NAME), framework="pt") as written:
        dtypes = {written.get_slice(key).get_dtype() for key in written.keys()}
    dtypes_ok = dtypes == {"F32"}
    if not dtypes_ok:
        failed_checks.append("stored dtype")
    print(f"[{'PASS' if dtypes_ok else 'FAIL'}] stored dtype: weights saved as {sorted(dtypes)}, expected F32")

    # --- configuration: the saved config must match what the current mapping derives from the source --------
    expected_config = convert_config(read_original_config(checkpoint_path))
    mismatched = {
        field: (getattr(config, field), getattr(expected_config, field))
        for field in (*ORIGINAL_CONFIG_FIELDS, "dropout")
        if _comparable(getattr(config, field)) != _comparable(getattr(expected_config, field))
    }
    architectures_ok = config.architectures == ["Apertus1p5VisionTokenizerModel"]
    config_ok = not mismatched and architectures_ok
    if not config_ok:
        failed_checks.append("config")
    print(
        f"[{'PASS' if config_ok else 'FAIL'}] config: channel_multiplier {list(config.channel_multiplier)}, "
        f"spatial factor {config.spatial_scale_factor}, codebook {config.codebook_size}, "
        f"architectures {config.architectures}" + (f", mismatched {mismatched}" if mismatched else "")
    )

    # --- code-grid geometry, including a size that is not a multiple of the spatial factor ------------------
    factor = config.spatial_scale_factor
    baselines = {}
    geometry_problems = []
    for height, width in VERIFY_IMAGE_SIZES:
        with torch.no_grad():
            codes = model.encode(_example_pixel_values(height, width))
        baselines[height, width] = codes
        if tuple(codes.shape) != (1, height // factor, width // factor):
            geometry_problems.append(f"{height}x{width} -> {tuple(codes.shape)}")
        elif not (int(codes.min()) >= 0 and int(codes.max()) < config.codebook_size):
            geometry_problems.append(f"{height}x{width} codes out of range")
    if geometry_problems:
        failed_checks.append("geometry")
    print(
        f"[{'PASS' if not geometry_problems else 'FAIL'}] geometry: "
        + ", ".join(f"{h}x{w}->{h // factor}x{w // factor}" for h, w in VERIFY_IMAGE_SIZES)
        + (f"; problems {geometry_problems}" if geometry_problems else "")
    )

    # --- batching: the encoder has global attention, so batched and per-image codes must still agree --------
    first, second = _example_pixel_values(64, 64, seed=1), _example_pixel_values(64, 64, seed=2)
    with torch.no_grad():
        batched = model.encode(torch.cat([first, second], dim=0))
        singles = torch.cat([model.encode(first), model.encode(second)], dim=0)
    batch_ok = torch.equal(batched, singles)
    if not batch_ok:
        failed_checks.append("batch")
    print(f"[{'PASS' if batch_ok else 'FAIL'}] batch: same-size batched encode == per-image encode")

    # --- save/reload round trip ----------------------------------------------------------------------------
    reference_size = VERIFY_IMAGE_SIZES[0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        reloaded = Apertus1p5VisionTokenizerModel.from_pretrained(tmp_dir).eval()
        with torch.no_grad():
            codes = reloaded.encode(_example_pixel_values(*reference_size))
        reload_ok = set(reloaded.state_dict()) == set(model.state_dict()) and torch.equal(
            codes, baselines[reference_size]
        )
        del reloaded
    if not reload_ok:
        failed_checks.append("reload")
    print(f"[{'PASS' if reload_ok else 'FAIL'}] reload: save_pretrained/from_pretrained round trip is bit-exact")

    # --- the float32 guard on a half-precision load ---------------------------------------------------------
    guarded = Apertus1p5VisionTokenizerModel.from_pretrained(output_dir, dtype=torch.bfloat16).eval()
    weights_fp32 = guarded.encoder.conv_in.weight.dtype == torch.float32
    with torch.no_grad():
        codes = guarded.encode(_example_pixel_values(*reference_size))
    dtype_ok = weights_fp32 and torch.equal(codes, baselines[reference_size])
    del guarded
    if not dtype_ok:
        failed_checks.append("dtype")
    print(
        f"[{'PASS' if dtype_ok else 'FAIL'}] dtype: a bfloat16 load keeps the weights float32 "
        f"({weights_fp32}) and the codes bit-exact"
    )

    if failed_checks:
        raise RuntimeError(f"Vision tokenizer verification failed: {failed_checks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Original EMU3.5 vision tokenizer: local directory or Hub `repo_id[@revision]` "
        "(e.g. BAAI/Emu3.5-VisionTokenizer)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to save the converted model (the `--vision_tokenizer_checkpoint` input of "
        "convert_apertus1p5_weights_to_hf.py)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Reload the converted checkpoint and check the stored precision, the derived configuration, the "
        "code-grid geometry, batched encoding, the save/reload round trip, and the float32 keep under a "
        "bfloat16 load",
    )
    parser.add_argument(
        "--push_to_hub",
        default=None,
        help="Optional Hub repo id to push the converted model to, after `--verify` if that is set",
    )
    args = parser.parse_args()

    convert_checkpoint(args.checkpoint_path, args.output_dir)
    if args.verify:
        verify(args.output_dir, args.checkpoint_path)
    if args.push_to_hub:
        # pushed last so that `--verify` gates publication instead of following it
        Apertus1p5VisionTokenizerModel.from_pretrained(args.output_dir).push_to_hub(args.push_to_hub)
