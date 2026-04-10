# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Convert PhoneticXeus (ESPnet/Lightning) checkpoint to HuggingFace format.

Usage:
    python convert_phoneticxeus_checkpoint.py \
        --checkpoint_path changelinglab/PhoneticXeus \
        --output_dir ./phoneticxeus_hf
"""

import argparse
import re
import sys

import torch

from transformers import PhoneticXeusConfig, PhoneticXeusForCTC, PhoneticXeusTokenizer, Wav2Vec2FeatureExtractor


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load ESPnet/Lightning checkpoint, handling pickled module references."""
    import types

    # Stub modules that may be pickled in the checkpoint
    for mod_name in ["src", "lightning"]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            m.__path__ = []
            m.__file__ = ""
            sys.modules[mod_name] = m

    # Recursively stub submodules
    class _StubFinder:
        def find_module(self, fullname, path=None):
            if fullname.startswith(("src.", "lightning.")):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            m = types.ModuleType(fullname)
            m.__path__ = []
            m.__file__ = ""
            m.__loader__ = self
            sys.modules[fullname] = m
            return m

    sys.meta_path.insert(0, _StubFinder())

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state_dict from Lightning checkpoint
    if "state_dict" in state:
        sd = state["state_dict"]
        # Strip "net." prefix from Lightning module wrapping
        sd = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    else:
        sd = state

    return sd


_PREFIX = PhoneticXeusForCTC.base_model_prefix


def convert_key(key: str) -> str | None:
    """Map a single ESPnet state_dict key to the HuggingFace equivalent."""

    if key.startswith("frontend.layers."):
        return key.replace("frontend.layers.", f"{_PREFIX}.feature_extractor.conv_layers.")

    if key.startswith("preencoder.linear_out."):
        return key.replace("preencoder.linear_out.", f"{_PREFIX}.feature_projection.projection.")

    if key.startswith("encoder.embed.0.convs.0."):
        return key.replace("encoder.embed.0.convs.0.", f"{_PREFIX}.encoder.pos_conv_embed.conv.")

    if key.startswith("encoder.after_norm."):
        return key.replace("encoder.after_norm.", f"{_PREFIX}.encoder.layer_norm.")

    if key.startswith("encoder.conditioning_layer."):
        return key.replace("encoder.conditioning_layer.", f"{_PREFIX}.encoder.conditioning_layer.")

    m = re.match(r"encoder\.encoders\.(\d+)\.(.*)", key)
    if m:
        layer_idx, rest = m.group(1), m.group(2)
        if rest.startswith("attn."):
            rest = rest.replace("attn.", "self_attn.", 1)
        return f"{_PREFIX}.encoder.layers.{layer_idx}.{rest}"

    if key.startswith("ctc.ctc_lo."):
        return key.replace("ctc.ctc_lo.", "lm_head.")

    return None


def convert_state_dict(sd: dict) -> dict:
    """Convert full ESPnet state dict to HuggingFace format."""
    new_sd = {}
    skipped = []
    for key, value in sd.items():
        new_key = convert_key(key)
        if new_key is not None:
            new_sd[new_key] = value
        else:
            skipped.append(key)

    if skipped:
        print(f"Skipped {len(skipped)} keys: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    return new_sd


def main():
    parser = argparse.ArgumentParser(description="Convert PhoneticXeus checkpoint to HuggingFace format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to .ckpt file or HuggingFace repo (e.g., 'changelinglab/PhoneticXeus')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for HuggingFace model files",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default=None,
        help="Path to ipa_vocab.json. If not provided, downloads from HF repo.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="If set, push model to this HuggingFace Hub repo",
    )
    args = parser.parse_args()

    # Resolve checkpoint path
    ckpt_path = args.checkpoint_path
    if not ckpt_path.endswith(".ckpt") and not ckpt_path.endswith(".pth"):
        # Assume HuggingFace repo
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(ckpt_path, "checkpoint-22000.ckpt")
        print(f"Downloaded checkpoint to: {ckpt_path}")

    # Load and convert
    print("Loading checkpoint...")
    sd = load_checkpoint(ckpt_path)
    print(f"Loaded {len(sd)} keys from checkpoint")

    print("Converting state dict...")
    hf_sd = convert_state_dict(sd)
    print(f"Converted to {len(hf_sd)} HuggingFace keys")

    # Create config
    config = PhoneticXeusConfig()
    print(
        f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}, vocab={config.vocab_size}"
    )

    # Create model and load weights
    print("Creating HuggingFace model...")
    model = PhoneticXeusForCTC(config)
    load_info = model.load_state_dict(hf_sd, strict=False)
    print(f"Missing keys: {load_info.missing_keys}")
    print(f"Unexpected keys: {load_info.unexpected_keys}")

    # Verify with dummy forward pass
    print("Running verification forward pass...")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 16000)  # 1 second of audio
        output = model(dummy_input)
        logits = output.logits
        print(f"Output shape: {logits.shape}")  # Should be (1, T, 428)
        assert logits.shape[-1] == config.vocab_size, (
            f"Expected vocab_size={config.vocab_size}, got {logits.shape[-1]}"
        )
        print("Verification passed!")

    # Save
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    # Save feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    feature_extractor.save_pretrained(args.output_dir)

    # Save tokenizer + processor if vocab provided
    if args.vocab_file:
        import shutil

        from transformers import PhoneticXeusProcessor

        shutil.copy(args.vocab_file, f"{args.output_dir}/vocab.json")
        tokenizer = PhoneticXeusTokenizer(f"{args.output_dir}/vocab.json")
        tokenizer.save_pretrained(args.output_dir)
        processor = PhoneticXeusProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(args.output_dir)
        print(f"Saved tokenizer + processor (vocab_size={tokenizer.vocab_size})")

    if args.push_to_hub:
        print(f"Pushing to hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        config.push_to_hub(args.push_to_hub)
        feature_extractor.push_to_hub(args.push_to_hub)

    print("Done!")


if __name__ == "__main__":
    main()
