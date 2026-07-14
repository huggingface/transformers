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
"""
End-to-end parity check between the transformers `WavTokenizerModel` port and the ORIGINAL WavTokenizer
implementation (https://github.com/jishengpeng/WavTokenizer), covering every intended use case in transformers.

What it verifies (per test signal: sines, chirp, noise, silence, impulse, hop-multiple and odd lengths):
  1. convert    - the conversion script converts the original .ckpt with a strict state-dict load
  2. encode     - our codes are BIT-EXACT equal to the original `encode_infer` codes
  3. decode     - our decoded waveform matches the original decode (allclose, reports max abs diff)
  4. roundtrip  - forward() == encode()+decode() internally consistent
  5. batch      - equal-length batched encode is bit-exact vs per-sample encode
  6. fe         - the WavTokenizerFeatureExtractor path yields identical codes to the raw-tensor path
  7. auto       - AutoModel / AutoModelForAudioTokenization / AutoFeatureExtractor resolve on the converted
                  dir and reproduce identical codes
  8. reload     - save_pretrained/from_pretrained round trip is bit-exact
  9. dtype      - (informational) bf16 encode agreement rate vs fp32

Requirements: a clone of the original repo (jishengpeng/WavTokenizer or the swiss-ai fork) for the reference
implementation, and the original checkpoint (auto-downloaded from `novateur/WavTokenizer-large-unify-40token`
unless --checkpoint_path is given).

Example:
    python scripts/check_wavtokenizer_parity.py \
        --original_repo /path/to/WavTokenizer \
        --output_dir /tmp/wavtokenizer-hf \
        [--checkpoint_path /path/to/wavtokenizer_large_unify_600_24k.ckpt] \
        [--config_path /path/to/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


HOP = 600
SAMPLING_RATE = 24000
DEFAULT_CKPT_REPO = "novateur/WavTokenizer-large-unify-40token"
DEFAULT_CKPT_FILE = "wavtokenizer_large_unify_600_24k.ckpt"
DEFAULT_CONFIG_RELPATH = "configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"

PASS, FAIL, INFO = "PASS", "FAIL", "INFO"


def make_test_signals() -> dict[str, np.ndarray]:
    """Deterministic signals covering the intended input space (mono, 24 kHz, float32)."""
    rng = np.random.default_rng(seed=0)
    t = lambda seconds: np.arange(int(seconds * SAMPLING_RATE)) / SAMPLING_RATE  # noqa: E731
    # speech-like: amplitude-modulated harmonic stack in the speech band with pauses
    t_sp = t(3.0)
    speech_like = (
        (
            0.3 * np.sin(2 * np.pi * 120 * t_sp)
            + 0.2 * np.sin(2 * np.pi * 240 * t_sp)
            + 0.1 * np.sin(2 * np.pi * 800 * t_sp)
        )
        * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t_sp))  # ~3 Hz syllable-rate envelope
        * (np.sin(2 * np.pi * 0.4 * t_sp) > -0.6)  # intermittent pauses
    )
    # long mixed content: chirp + noise + sine + silence, 60 s total (2400 codes)
    long_mixed = np.concatenate(
        [
            0.4 * np.sin(2 * np.pi * (50 + 4000 * t(20.0)) * t(20.0)),
            0.2 * rng.standard_normal(int(15.0 * SAMPLING_RATE)),
            0.5 * np.sin(2 * np.pi * 440.0 * t(15.0)),
            np.zeros(int(10.0 * SAMPLING_RATE)),
        ]
    )
    signals = {
        "sine_440hz_1s": 0.5 * np.sin(2 * np.pi * 440.0 * t(1.0)),
        "chirp_1.7s": 0.4 * np.sin(2 * np.pi * (100 + 2000 * t(1.7)) * t(1.7)),
        "noise_0.35s": 0.2 * rng.standard_normal(int(0.35 * SAMPLING_RATE)),
        "silence_0.5s": np.zeros(int(0.5 * SAMPLING_RATE)),
        "impulse_short": np.eye(1, 2 * HOP + 1, HOP)[0],  # single 1.0 in the middle, odd length
        "len_hop_multiple": 0.3 * rng.standard_normal(20 * HOP),
        "len_hop_plus_one": 0.3 * rng.standard_normal(20 * HOP + 1),
        "len_sub_hop": 0.3 * rng.standard_normal(HOP - 1),
        "speech_like_3s": speech_like,
        "long_mixed_60s": long_mixed,
    }
    return {name: sig.astype(np.float32) for name, sig in signals.items()}


def load_original_model(original_repo: str, config_path: str | None, checkpoint_path: str):
    """Load the reference model through the ORIGINAL repo code (decoder.pretrained.WavTokenizer)."""
    repo = Path(original_repo).resolve()
    if not (repo / "decoder" / "pretrained.py").exists():
        raise FileNotFoundError(f"{repo} does not look like a WavTokenizer clone (no decoder/pretrained.py)")
    sys.path.insert(0, str(repo))
    from decoder.pretrained import WavTokenizer as OriginalWavTokenizer  # noqa: PLC0415

    config_path = config_path or str(repo / DEFAULT_CONFIG_RELPATH)
    model = OriginalWavTokenizer.from_pretrained0802(config_path, checkpoint_path)
    return model.eval()


def original_encode(original, waveform: torch.Tensor) -> torch.Tensor:
    """Original codes for a (B, T) waveform -> (B, N) int64."""
    bandwidth_id = torch.tensor([0])
    with torch.no_grad():
        _, codes = original.encode_infer(waveform, bandwidth_id=bandwidth_id)
    return codes.squeeze(0) if codes.dim() == 3 else codes  # (n_q=1, B, N) -> (B, N)


def original_decode(original, codes: torch.Tensor) -> torch.Tensor:
    """Original waveform for (B, N) codes -> (B, T)."""
    bandwidth_id = torch.tensor([0])
    with torch.no_grad():
        features = original.codes_to_features(codes.unsqueeze(0))  # (1, B, N)
        audio = original.decode(features, bandwidth_id=bandwidth_id)
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original_repo", required=True, help="Path to a clone of jishengpeng/WavTokenizer")
    parser.add_argument("--config_path", default=None, help="Original config yaml (defaults to the 40-token one)")
    parser.add_argument("--checkpoint_path", default=None, help="Original .ckpt (downloaded from the Hub if unset)")
    parser.add_argument("--output_dir", required=True, help="Where the converted HF model is written/reused")
    parser.add_argument("--skip_convert", action="store_true", help="Reuse an existing converted --output_dir")
    parser.add_argument("--decode_atol", type=float, default=1e-4, help="Absolute tolerance for decoded audio")
    args = parser.parse_args()

    results: list[tuple[str, str, str]] = []

    def record(status: str, name: str, detail: str = ""):
        results.append((status, name, detail))
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download

        print(f"Downloading original checkpoint {DEFAULT_CKPT_REPO}/{DEFAULT_CKPT_FILE} ...")
        checkpoint_path = hf_hub_download(repo_id=DEFAULT_CKPT_REPO, filename=DEFAULT_CKPT_FILE)

    # ---- 1. convert -------------------------------------------------------------------------------------
    if not args.skip_convert:
        from transformers.models.wavtokenizer.convert_wavtokenizer_checkpoint import convert_checkpoint

        try:
            convert_checkpoint(checkpoint_path, args.output_dir)
            record(PASS, "convert: strict state-dict load + save_pretrained")
        except Exception as exc:  # noqa: BLE001
            record(FAIL, "convert", f"{type(exc).__name__}: {exc}")
            print("Conversion failed — aborting.")
            _summarize(results)
            sys.exit(1)

    from transformers import (
        AutoFeatureExtractor,
        AutoModel,
        AutoModelForAudioTokenization,
        WavTokenizerFeatureExtractor,
        WavTokenizerModel,
    )

    model = WavTokenizerModel.from_pretrained(args.output_dir).eval()
    feature_extractor = WavTokenizerFeatureExtractor.from_pretrained(args.output_dir)
    original = load_original_model(args.original_repo, args.config_path, checkpoint_path)

    signals = make_test_signals()
    our_codes: dict[str, torch.Tensor] = {}

    # ---- 2. encode parity (bit-exact) -------------------------------------------------------------------
    for name, signal in signals.items():
        waveform = torch.from_numpy(signal)[None, :]  # (1, T)
        with torch.no_grad():
            ours = model.encode(waveform[:, None, :]).audio_codes[:, 0, :]  # (1, N)
        theirs = original_encode(original, waveform)
        our_codes[name] = ours
        expected_n = math.ceil(signal.shape[-1] / HOP)
        if ours.shape != theirs.shape:
            record(FAIL, f"encode[{name}]", f"shape ours {tuple(ours.shape)} vs original {tuple(theirs.shape)}")
        elif not torch.equal(ours.long(), theirs.long()):
            n_diff = (ours.long() != theirs.long()).sum().item()
            record(FAIL, f"encode[{name}]", f"{n_diff}/{ours.numel()} codes differ")
        elif ours.shape[-1] != expected_n:
            record(FAIL, f"encode[{name}]", f"{ours.shape[-1]} codes, expected ceil(T/hop)={expected_n}")
        else:
            record(PASS, f"encode[{name}]", f"{ours.shape[-1]} codes bit-exact")

    # ---- 3. decode parity --------------------------------------------------------------------------------
    for name in signals:
        codes = our_codes[name]
        if codes.shape[-1] < 2:
            # single-code decode is unsupported by the architecture (GroupNorm over one time step);
            # our port raises a clear ValueError by design — verify the guard fires
            try:
                with torch.no_grad():
                    model.decode(codes[:, None, :])
                record(FAIL, f"decode[{name}]", "expected ValueError for single-code decode, got none")
            except ValueError:
                record(PASS, f"decode[{name}]", "single-code guard raised as designed")
            continue
        with torch.no_grad():
            ours = model.decode(codes[:, None, :]).audio_values[:, 0, :]
        theirs = original_decode(original, codes)
        theirs = theirs.reshape(ours.shape[0], -1)[:, : ours.shape[-1]]
        max_diff = (ours[..., : theirs.shape[-1]] - theirs).abs().max().item()
        status = PASS if max_diff <= args.decode_atol else FAIL
        record(status, f"decode[{name}]", f"max|Δ|={max_diff:.2e} (atol={args.decode_atol})")

    # ---- 4. roundtrip forward ----------------------------------------------------------------------------
    signal = signals["chirp_1.7s"]
    waveform = torch.from_numpy(signal)[None, None, :]
    with torch.no_grad():
        fwd = model(waveform)
        enc = model.encode(waveform).audio_codes
        dec = model.decode(enc).audio_values[..., : waveform.shape[-1]]
    ok = torch.equal(fwd.audio_codes, enc) and torch.equal(fwd.audio_values, dec)
    record(PASS if ok else FAIL, "roundtrip: forward == encode + decode")

    # ---- 5. batched vs single ----------------------------------------------------------------------------
    batch = torch.from_numpy(np.stack([signals["len_hop_multiple"], signals["len_hop_multiple"][::-1].copy()]))
    with torch.no_grad():
        batched = model.encode(batch[:, None, :]).audio_codes
        singles = torch.cat([model.encode(batch[i : i + 1, None, :]).audio_codes for i in range(2)], dim=0)
    record(PASS if torch.equal(batched, singles) else FAIL, "batch: batched encode == per-sample encode")

    # ---- 6. feature extractor path -----------------------------------------------------------------------
    signal = signals["len_hop_multiple"]  # hop multiple: FE pads nothing, must match the raw path exactly
    inputs = feature_extractor(signal, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    with torch.no_grad():
        fe_codes = model.encode(inputs["input_values"], padding_mask=inputs["padding_mask"]).audio_codes
        raw_codes = model.encode(torch.from_numpy(signal)[None, None, :]).audio_codes
    record(PASS if torch.equal(fe_codes, raw_codes) else FAIL, "fe: feature-extractor path == raw-tensor path")

    # ---- 7. auto classes ----------------------------------------------------------------------------------
    auto_model = AutoModel.from_pretrained(args.output_dir).eval()
    auto_tok = AutoModelForAudioTokenization.from_pretrained(args.output_dir).eval()
    auto_fe = AutoFeatureExtractor.from_pretrained(args.output_dir)
    ok = (
        type(auto_model).__name__ == "WavTokenizerModel"
        and type(auto_tok).__name__ == "WavTokenizerModel"
        and type(auto_fe).__name__ == "WavTokenizerFeatureExtractor"
    )
    if ok:
        with torch.no_grad():
            ok = torch.equal(
                auto_model.encode(waveform).audio_codes,
                our_codes["chirp_1.7s"][:, None, :],
            )
    record(PASS if ok else FAIL, "auto: AutoModel/AutoModelForAudioTokenization/AutoFeatureExtractor")

    # ---- 8. save/load round trip --------------------------------------------------------------------------
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded = WavTokenizerModel.from_pretrained(tmp).eval()
        with torch.no_grad():
            ok = torch.equal(reloaded.encode(waveform).audio_codes, our_codes["chirp_1.7s"][:, None, :])
    record(PASS if ok else FAIL, "reload: save_pretrained/from_pretrained round trip")

    # ---- 9. dtype (informational) -------------------------------------------------------------------------
    try:
        bf16 = WavTokenizerModel.from_pretrained(args.output_dir, dtype=torch.bfloat16).eval()
        with torch.no_grad():
            bf16_codes = bf16.encode(waveform.to(torch.bfloat16)).audio_codes
        agreement = (bf16_codes == our_codes["chirp_1.7s"][:, None, :]).float().mean().item()
        record(INFO, "dtype: bf16 encode agreement vs fp32", f"{agreement:.1%}")
    except Exception as exc:  # noqa: BLE001
        record(INFO, "dtype: bf16 encode", f"not supported: {type(exc).__name__}: {exc}")

    _summarize(results)
    sys.exit(1 if any(status == FAIL for status, _, _ in results) else 0)


def _summarize(results):
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print("\n" + "=" * 80)
    print(f"PARITY SUMMARY: {n_pass} passed, {n_fail} failed, {len(results) - n_pass - n_fail} informational")
    for status, name, detail in results:
        if status == FAIL:
            print(f"  FAILED: {name} — {detail}")
    print("=" * 80)


if __name__ == "__main__":
    main()
