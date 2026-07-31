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
"""End-to-end parity checks between Transformers and the original WavTokenizer implementation.

For a named checkpoint, the script resolves its original-author Hub repository and filename, downloads the raw
checkpoint, and converts it by calling `convert_wavtokenizer_checkpoint.convert_checkpoint`. It can also use a custom
local checkpoint or process the complete released checkpoint matrix. Pass `--skip_convert` to reuse an existing
converted output directory.

Unless `--skip_convert` is used, each run verifies strict conversion. Every run checks bit-exact encoding, decoded
waveform parity, forward roundtrips, batching, feature extraction, Auto classes, and save/reload behavior.

Examples:
    python scripts/check_wavtokenizer_parity.py --original_repo /path/to/WavTokenizer --checkpoint large-unify-40 --output_dir /tmp/wavtokenizer-hf

    python scripts/check_wavtokenizer_parity.py --original_repo /path/to/WavTokenizer --all-checkpoints --output_dir /tmp/wavtokenizer-matrix
"""

import argparse
import gc
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


DEFAULT_CHECKPOINT = "large-unify-40"
ORIGINAL_CONFIGS = {
    600: "configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    320: "configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
}
PASS, FAIL, INFO = "PASS", "FAIL", "INFO"


@dataclass(frozen=True)
class CheckpointSpec:
    repo_id: str
    filename: str


CHECKPOINTS = {
    "small-speech-40": CheckpointSpec("novateur/WavTokenizer", "WavTokenizer_small_600_24k_4096.ckpt"),
    "small-speech-75": CheckpointSpec("novateur/WavTokenizer", "WavTokenizer_small_320_24k_4096.ckpt"),
    "medium-speech-75": CheckpointSpec(
        "novateur/WavTokenizer-medium-speech-75token", "wavtokenizer_medium_speech_320_24k.ckpt"
    ),
    "medium-speech-75-v2": CheckpointSpec(
        "novateur/WavTokenizer-medium-speech-75token", "wavtokenizer_medium_speech_320_24k_v2.ckpt"
    ),
    "medium-music-audio-75": CheckpointSpec(
        "novateur/WavTokenizer-medium-music-audio-75token", "wavtokenizer_medium_music_audio_320_24k.ckpt"
    ),
    "medium-music-audio-75-v2": CheckpointSpec(
        "novateur/WavTokenizer-medium-music-audio-75token", "wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
    ),
    "large-unify-40": CheckpointSpec(
        "novateur/WavTokenizer-large-unify-40token", "wavtokenizer_large_unify_600_24k.ckpt"
    ),
    "large-speech-75-v2": CheckpointSpec(
        "novateur/WavTokenizer-large-speech-75token", "wavtokenizer_large_speech_320_v2.ckpt"
    ),
}


def make_test_signals(hop_length: int, sampling_rate: int) -> dict[str, np.ndarray]:
    """Create deterministic mono float32 signals, including hop-boundary and odd-length inputs."""
    rng = np.random.default_rng(seed=0)
    time = lambda seconds: np.arange(int(seconds * sampling_rate)) / sampling_rate  # noqa: E731
    speech_time = time(3.0)
    speech_like = (
        (
            0.3 * np.sin(2 * np.pi * 120 * speech_time)
            + 0.2 * np.sin(2 * np.pi * 240 * speech_time)
            + 0.1 * np.sin(2 * np.pi * 800 * speech_time)
        )
        * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * speech_time))
        * (np.sin(2 * np.pi * 0.4 * speech_time) > -0.6)
    )
    long_mixed = np.concatenate(
        [
            0.4 * np.sin(2 * np.pi * (50 + 4000 * time(20.0)) * time(20.0)),
            0.2 * rng.standard_normal(int(15.0 * sampling_rate)),
            0.5 * np.sin(2 * np.pi * 440.0 * time(15.0)),
            np.zeros(int(10.0 * sampling_rate)),
        ]
    )
    signals = {
        "sine_440hz_1s": 0.5 * np.sin(2 * np.pi * 440.0 * time(1.0)),
        "chirp_1.7s": 0.4 * np.sin(2 * np.pi * (100 + 2000 * time(1.7)) * time(1.7)),
        "noise_0.35s": 0.2 * rng.standard_normal(int(0.35 * sampling_rate)),
        "silence_0.5s": np.zeros(int(0.5 * sampling_rate)),
        "impulse_short": np.eye(1, 2 * hop_length + 1, hop_length)[0],
        "len_hop_multiple": 0.3 * rng.standard_normal(20 * hop_length),
        "len_hop_plus_one": 0.3 * rng.standard_normal(20 * hop_length + 1),
        "speech_like_3s": speech_like,
        "long_mixed_60s": long_mixed,
    }
    return {name: signal.astype(np.float32) for name, signal in signals.items()}


def load_original_model(original_repo: str, config_path: str, checkpoint_path: str):
    repo = Path(original_repo).resolve()
    if not (repo / "decoder" / "pretrained.py").exists():
        raise FileNotFoundError(f"{repo} does not look like a WavTokenizer clone (no decoder/pretrained.py)")
    sys.path.insert(0, str(repo))
    from decoder.pretrained import WavTokenizer as OriginalWavTokenizer  # noqa: PLC0415

    return OriginalWavTokenizer.from_pretrained0802(config_path, checkpoint_path).eval()


def original_encode(original, waveform: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        _, codes = original.encode_infer(waveform, bandwidth_id=torch.tensor([0]))
    return codes.squeeze(0) if codes.dim() == 3 else codes


def original_decode(original, codes: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        features = original.codes_to_features(codes.unsqueeze(0))
        return original.decode(features, bandwidth_id=torch.tensor([0]))


def run_checkpoint(name, checkpoint_path, output_dir, config_path, args):
    results: list[tuple[str, str, str]] = []

    def record(status: str, test_name: str, detail: str = ""):
        results.append((status, test_name, detail))
        print(f"[{name}] [{status}] {test_name}" + (f" - {detail}" if detail else ""))

    if not args.skip_convert:
        from transformers.models.wavtokenizer.convert_wavtokenizer_checkpoint import convert_checkpoint

        try:
            convert_checkpoint(checkpoint_path, output_dir)
            record(PASS, "convert: strict state-dict load + save_pretrained")
        except Exception as exception:  # noqa: BLE001
            record(FAIL, "convert", f"{type(exception).__name__}: {exception}")
            return results

    from transformers import (
        AutoFeatureExtractor,
        AutoModel,
        AutoModelForAudioTokenization,
        WavTokenizerFeatureExtractor,
        WavTokenizerModel,
    )

    model = WavTokenizerModel.from_pretrained(output_dir).eval()
    feature_extractor = WavTokenizerFeatureExtractor.from_pretrained(output_dir)
    hop_length = model.config.hop_length
    sampling_rate = model.config.sampling_rate
    if feature_extractor.hop_length != hop_length or feature_extractor.sampling_rate != sampling_rate:
        record(
            FAIL,
            "config",
            "model and feature extractor disagree on sampling rate or hop length",
        )
        return results

    if config_path is None:
        if hop_length not in ORIGINAL_CONFIGS:
            record(FAIL, "config", f"no original reference YAML registered for hop_length={hop_length}")
            return results
        config_path = str(Path(args.original_repo).resolve() / ORIGINAL_CONFIGS[hop_length])
    original = load_original_model(args.original_repo, config_path, checkpoint_path)
    signals = make_test_signals(hop_length, sampling_rate)
    our_codes: dict[str, torch.Tensor] = {}

    for signal_name, signal in signals.items():
        waveform = torch.from_numpy(signal)[None, :]
        with torch.no_grad():
            ours = model.encode(waveform[:, None, :]).audio_codes[:, 0, :]
        theirs = original_encode(original, waveform)
        our_codes[signal_name] = ours
        expected_codes = math.ceil(signal.shape[-1] / hop_length)
        if ours.shape != theirs.shape:
            record(FAIL, f"encode[{signal_name}]", f"shape ours {tuple(ours.shape)} vs original {tuple(theirs.shape)}")
        elif not torch.equal(ours.long(), theirs.long()):
            different = (ours.long() != theirs.long()).sum().item()
            record(FAIL, f"encode[{signal_name}]", f"{different}/{ours.numel()} codes differ")
        elif ours.shape[-1] != expected_codes:
            record(FAIL, f"encode[{signal_name}]", f"{ours.shape[-1]} codes, expected {expected_codes}")
        else:
            record(PASS, f"encode[{signal_name}]", f"{ours.shape[-1]} codes bit-exact")

    for signal_name, codes in our_codes.items():
        with torch.no_grad():
            ours = model.decode(codes[:, None, :]).audio_values
        theirs = original_decode(original, codes)
        expected_length = codes.shape[-1] * hop_length
        expected_ours_shape = (codes.shape[0], 1, expected_length)
        expected_theirs_shape = (codes.shape[0], expected_length)
        if ours.shape != expected_ours_shape or theirs.shape != expected_theirs_shape:
            record(
                FAIL,
                f"decode[{signal_name}]",
                f"shape ours {tuple(ours.shape)} vs expected {expected_ours_shape}; "
                f"original {tuple(theirs.shape)} vs expected {expected_theirs_shape}",
            )
            continue
        max_difference = (ours[:, 0, :] - theirs).abs().max().item()
        status = PASS if max_difference <= args.decode_atol else FAIL
        record(status, f"decode[{signal_name}]", f"max|delta|={max_difference:.2e} (atol={args.decode_atol})")

    del original
    gc.collect()

    signal = signals["chirp_1.7s"]
    waveform = torch.from_numpy(signal)[None, None, :]
    with torch.no_grad():
        forward_output = model(waveform)
        encoded = model.encode(waveform).audio_codes
        decoded = model.decode(encoded).audio_values[..., : waveform.shape[-1]]
    consistent = torch.equal(forward_output.audio_codes, encoded) and torch.equal(forward_output.audio_values, decoded)
    record(PASS if consistent else FAIL, "roundtrip: forward == encode + decode")

    batch = torch.from_numpy(np.stack([signals["len_hop_multiple"], signals["len_hop_multiple"][::-1].copy()]))
    with torch.no_grad():
        batched = model.encode(batch[:, None, :]).audio_codes
        singles = torch.cat([model.encode(batch[index : index + 1, None, :]).audio_codes for index in range(2)])
    record(PASS if torch.equal(batched, singles) else FAIL, "batch: batched encode == per-sample encode")

    signal = signals["len_hop_multiple"]
    inputs = feature_extractor(signal, sampling_rate=sampling_rate, return_tensors="pt")
    with torch.no_grad():
        fe_codes = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"]).audio_codes
        raw_codes = model.encode(torch.from_numpy(signal)[None, None, :]).audio_codes
    record(PASS if torch.equal(fe_codes, raw_codes) else FAIL, "feature extractor == raw tensor path")

    auto_model = AutoModel.from_pretrained(output_dir).eval()
    auto_model_ok = type(auto_model).__name__ == "WavTokenizerModel"
    if auto_model_ok:
        with torch.no_grad():
            auto_model_ok = torch.equal(auto_model.encode(waveform).audio_codes, our_codes["chirp_1.7s"][:, None, :])
    del auto_model
    gc.collect()

    auto_tokenizer = AutoModelForAudioTokenization.from_pretrained(output_dir).eval()
    auto_feature_extractor = AutoFeatureExtractor.from_pretrained(output_dir)
    auto_inputs = auto_feature_extractor(signal, sampling_rate=sampling_rate, return_tensors="pt")
    auto_tokenizer_ok = (
        type(auto_tokenizer).__name__ == "WavTokenizerModel"
        and type(auto_feature_extractor).__name__ == "WavTokenizerFeatureExtractor"
    )
    if auto_tokenizer_ok:
        with torch.no_grad():
            auto_tokenizer_codes = auto_tokenizer.encode(
                auto_inputs["input_values"], padding_mask=auto_inputs["padding_mask"]
            ).audio_codes
        auto_tokenizer_ok = torch.equal(auto_tokenizer_codes, raw_codes)
    record(PASS if auto_model_ok and auto_tokenizer_ok else FAIL, "auto classes")
    del auto_tokenizer, auto_feature_extractor, auto_inputs
    gc.collect()

    import tempfile

    with tempfile.TemporaryDirectory() as temporary_directory:
        model.save_pretrained(temporary_directory)
        reloaded = WavTokenizerModel.from_pretrained(temporary_directory).eval()
        with torch.no_grad():
            reload_ok = torch.equal(reloaded.encode(waveform).audio_codes, our_codes["chirp_1.7s"][:, None, :])
        del reloaded
    record(PASS if reload_ok else FAIL, "save_pretrained/from_pretrained roundtrip")
    gc.collect()

    try:
        bf16_model = WavTokenizerModel.from_pretrained(output_dir, dtype=torch.bfloat16).eval()
        with torch.no_grad():
            bf16_codes = bf16_model.encode(waveform.to(torch.bfloat16)).audio_codes
        agreement = (bf16_codes == our_codes["chirp_1.7s"][:, None, :]).float().mean().item()
        record(INFO, "BF16 encode agreement vs FP32", f"{agreement:.1%}")
        del bf16_model
    except Exception as exception:  # noqa: BLE001
        record(INFO, "BF16 encode", f"not supported: {type(exception).__name__}: {exception}")

    del model, feature_extractor
    gc.collect()
    return results


def summarize(results, title="PARITY SUMMARY"):
    passed = sum(status == PASS for status, _, _ in results)
    failed = sum(status == FAIL for status, _, _ in results)
    print("\n" + "=" * 80)
    print(f"{title}: {passed} passed, {failed} failed, {len(results) - passed - failed} informational")
    for status, name, detail in results:
        if status == FAIL:
            print(f"  FAILED: {name} - {detail}")
    print("=" * 80)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original_repo", required=True, help="Path to a clone of jishengpeng/WavTokenizer")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint", choices=sorted(CHECKPOINTS), help=f"Named checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    source.add_argument("--checkpoint_path", help="Custom original .ckpt path")
    parser.add_argument("--all-checkpoints", action="store_true", help="Run the complete released checkpoint matrix")
    parser.add_argument("--config_path", help="Custom original config YAML for a single checkpoint")
    parser.add_argument("--output_dir", required=True, help="Converted model directory, or matrix base directory")
    parser.add_argument("--skip_convert", action="store_true", help="Reuse already converted output directories")
    parser.add_argument("--decode_atol", type=float, default=1e-4, help="Absolute decoded-audio tolerance")
    args = parser.parse_args(argv)
    if args.all_checkpoints and (args.checkpoint is not None or args.checkpoint_path or args.config_path):
        parser.error("--all-checkpoints cannot be combined with --checkpoint, --checkpoint_path, or --config_path")
    return args


def main():
    args = parse_args()
    if args.all_checkpoints:
        from huggingface_hub import hf_hub_download

        aggregate = []
        for name, spec in CHECKPOINTS.items():
            try:
                checkpoint_path = hf_hub_download(spec.repo_id, spec.filename)
            except Exception as exception:  # noqa: BLE001
                results = [(FAIL, "download", f"{type(exception).__name__}: {exception}")]
            else:
                try:
                    results = run_checkpoint(
                        name,
                        checkpoint_path,
                        str(Path(args.output_dir) / name),
                        None,
                        args,
                    )
                except Exception as exception:  # noqa: BLE001
                    results = [(FAIL, "unexpected parity execution", f"{type(exception).__name__}: {exception}")]
            summarize(results, title=f"PARITY SUMMARY [{name}]")
            aggregate.extend((status, f"{name}: {test_name}", detail) for status, test_name, detail in results)
        summarize(aggregate, title="AGGREGATE PARITY SUMMARY")
        sys.exit(1 if any(status == FAIL for status, _, _ in aggregate) else 0)

    if args.checkpoint_path:
        name = "custom"
        checkpoint_path = args.checkpoint_path
    else:
        from huggingface_hub import hf_hub_download

        name = args.checkpoint or DEFAULT_CHECKPOINT
        spec = CHECKPOINTS[name]
        checkpoint_path = hf_hub_download(spec.repo_id, spec.filename)

    results = run_checkpoint(name, checkpoint_path, args.output_dir, args.config_path, args)
    summarize(results, title=f"PARITY SUMMARY [{name}]")
    sys.exit(1 if any(status == FAIL for status, _, _ in results) else 0)


if __name__ == "__main__":
    main()
