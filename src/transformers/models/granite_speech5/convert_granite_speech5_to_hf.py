# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Convert an original Granite Speech 5.0 Turbo CTC checkpoint to the native HuggingFace
`GraniteSpeech5ForCTC` format.

The upstream checkpoint ships as a remote-code model (its `config.json` carries an `auto_map` pointing at
`CtcConformerForCTC`) with the conformer encoder stored under an `encoder.` prefix. The native model follows
the `ParakeetEncoder` layout instead, so the conformer blocks are renamed:
  * the sub-modules' pre-norms are hoisted onto the block (`ff1.pre_norm` -> `norm_feed_forward1`,
    `attn.pre_norm` -> `norm_self_att`, `conv.norm` -> `norm_conv`, `post_norm` -> `norm_out`),
  * the feed-forward and attention projections take their canonical names (`ff1.up_proj` ->
    `feed_forward1.linear1`, `attn.to_q` -> `self_attn.q_proj`, ...),
  * the CTC head `encoder.out` (shared with the encoder's mid-layer self-conditioning) is additionally tied
    to the top-level `ctc_head`, so only the encoder copy is stored.

The original front-end constants (from `config.json` / `preprocessor_config.json`) map onto
`GraniteSpeech5FeatureExtractor`, and the bundled fast `tokenizer.json` (blank token `<|blank|>` at id 0)
onto `ParakeetTokenizer`, whose CTC-collapsing decode this model shares.

Example:
    python src/transformers/models/granite_speech5/convert_granite_speech5_to_hf.py \
        --input_path path/or/repo/of/the/original/checkpoint \
        --output_path converted_granite_speech5 \
        --run_sanity_check
"""

import argparse
import copy
import json
import re

import torch
from safetensors.torch import load_file

from transformers import (
    GraniteSpeech5CTCConfig,
    GraniteSpeech5EncoderConfig,
    GraniteSpeech5FeatureExtractor,
    GraniteSpeech5ForCTC,
    GraniteSpeech5Processor,
    ParakeetTokenizer,
)
from transformers.utils.hub import cached_file


# Ordered regex renamings from the original checkpoint namespace to the native HF layout. Order matters: the
# sub-modules' pre-norms must be hoisted onto the block before the generic sub-module renames run, and
# `conv.batch_norm` -> `conv.norm` must run after `conv.norm` -> `norm_conv`.
ORIGINAL_TO_HF_WEIGHT_MAPPING = {
    # pre-norms are hoisted out of the sub-modules onto the conformer block
    r"\.ff1\.pre_norm": ".norm_feed_forward1",
    r"\.ff2\.pre_norm": ".norm_feed_forward2",
    r"\.attn\.pre_norm": ".norm_self_att",
    r"\.conv\.norm": ".norm_conv",
    r"\.post_norm": ".norm_out",
    # feed-forward projections
    r"\.ff1\.up_proj": ".feed_forward1.linear1",
    r"\.ff1\.down_proj": ".feed_forward1.linear2",
    r"\.ff2\.up_proj": ".feed_forward2.linear1",
    r"\.ff2\.down_proj": ".feed_forward2.linear2",
    # attention projections
    r"\.attn\.to_q": ".self_attn.q_proj",
    r"\.attn\.to_k": ".self_attn.k_proj",
    r"\.attn\.to_v": ".self_attn.v_proj",
    r"\.attn\.to_out": ".self_attn.o_proj",
    r"\.attn\.rel_pos_emb": ".self_attn.rel_pos_emb",
    # convolution module: the depthwise convolution loses its padding wrapper (the kernel size is odd, so
    # the wrapper's padding is symmetric and folds into the convolution itself)
    r"\.conv\.depth_conv\.conv": ".conv.depthwise_conv",
    r"\.conv\.batch_norm": ".conv.norm",
}


def convert_key(key, mapping):
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def convert_state_dict(state_dict):
    """Rename every key from the original namespace onto the native `GraniteSpeech5ForCTC` layout.

    Dtypes are preserved as the original stores them, i.e. bfloat16 weights next to float32 batch-norm
    running stats: bfloat16 is the precision the model was trained and evaluated in (the original's
    `transcribe()` runs its forward under bfloat16 autocast), so upcasting would only double the
    checkpoint and make float32 the default `from_pretrained` dtype.
    """
    return {convert_key(key, ORIGINAL_TO_HF_WEIGHT_MAPPING): value for key, value in state_dict.items()}


def check_frontend_is_supported(original_config: dict):
    """The delta concatenation and the pair-wise frame stacking are hardcoded in the native front-end, so
    the encoder's input width follows from `num_mel_bins` alone. Refuse a checkpoint that departs from it."""
    delta_expansion = GraniteSpeech5FeatureExtractor.delta_expansion
    frame_stacking = GraniteSpeech5FeatureExtractor.frame_stacking
    if not original_config["deltas"] or original_config["stack_factor"] != frame_stacking:
        raise ValueError(
            f"This converter only supports the front-end `GraniteSpeech5FeatureExtractor` implements "
            f"(deltas=True, stack_factor={frame_stacking}), got deltas={original_config['deltas']}, "
            f"stack_factor={original_config['stack_factor']}."
        )
    input_dim = original_config["n_mels"] * delta_expansion * frame_stacking
    if input_dim != original_config["input_dim"]:
        raise ValueError(
            f"Inconsistent original config: n_mels={original_config['n_mels']} with deltas and "
            f"stack_factor={frame_stacking} gives an input width of {input_dim}, but input_dim is "
            f"{original_config['input_dim']}."
        )


def convert_config(original_config: dict) -> GraniteSpeech5CTCConfig:
    encoder_config = GraniteSpeech5EncoderConfig(
        num_mel_bins=original_config["n_mels"],
        num_hidden_layers=original_config["num_layers"],
        hidden_size=original_config["hidden_dim"],
        intermediate_size=original_config["hidden_dim"] * original_config["feedforward_mult"],
        num_attention_heads=original_config["num_heads"],
        head_dim=original_config["dim_head"],
        vocab_size=original_config["output_dim"],
        context_size=original_config["context_size"],
        max_position_embeddings=original_config["max_pos_emb"],
        attention_dropout=original_config["dropout"],
        activation_dropout=original_config["dropout"],
        conv_kernel_size=original_config["conv_kernel_size"],
        conv_expansion_factor=original_config["conv_expansion_factor"],
        subsample_layers=list(original_config["subsample_layers"]),
    )
    # the CTC blank shares id 0 with the tokenizer's `<|blank|>` token
    return GraniteSpeech5CTCConfig(
        encoder_config=encoder_config, vocab_size=original_config["output_dim"], pad_token_id=0
    )


def convert_feature_extractor(original_config: dict) -> GraniteSpeech5FeatureExtractor:
    return GraniteSpeech5FeatureExtractor(
        num_mel_bins=original_config["n_mels"],
        sampling_rate=original_config["sample_rate"],
        n_fft=original_config["n_fft"],
        win_length=original_config["win_length"],
        hop_length=original_config["hop_length"],
        delta_win_length=original_config["delta_win_length"],
        logmel_floor_db=original_config["logmel_floor_db"],
    )


def convert_and_write_model(input_path: str, output_path: str, run_sanity_check: bool = False):
    with open(cached_file(input_path, "config.json")) as f:
        original_config = json.load(f)
    check_frontend_is_supported(original_config)
    config = convert_config(original_config)

    original_state_dict = load_file(cached_file(input_path, "model.safetensors"))
    state_dict = convert_state_dict(original_state_dict)

    with torch.device("meta"):
        model = GraniteSpeech5ForCTC(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
    # `ctc_head` is tied to `encoder.out`
    if set(missing_keys) != {"ctc_head.weight", "ctc_head.bias"} or unexpected_keys:
        raise ValueError(f"Unexpected conversion mismatch: missing={missing_keys}, unexpected={unexpected_keys}")
    model.tie_weights()
    # materialize the (non-persistent, hence still on the meta device) relative distances buffer
    model.encoder.attention_dists = torch.nn.Buffer(model.encoder.compute_attention_dists(), persistent=False)

    # the CTC blank is id 0 of the vocabulary, whatever the checkpoint happens to call it
    tokenizer_file = cached_file(input_path, "tokenizer.json")
    with open(tokenizer_file) as f:
        vocabulary = json.load(f)["model"]["vocab"]
    blank_token = next(token for token, index in vocabulary.items() if index == 0)
    tokenizer = ParakeetTokenizer(tokenizer_file=tokenizer_file, pad_token=blank_token, padding=True)
    feature_extractor = convert_feature_extractor(original_config)
    processor = GraniteSpeech5Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    if run_sanity_check:
        # the checkpoint mixes bfloat16 weights with float32 batch-norm stats, which batch-norm rejects;
        # `from_pretrained` casts every float tensor to the load dtype, so sanity-check that dtype directly
        run_sanity_checks(copy.deepcopy(model).to(torch.bfloat16), processor)

    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print(f"Converted model and processor saved to {output_path}")


def run_sanity_checks(model, processor):
    from datasets import Audio, load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
    samples = [x["array"] for x in ds.sort("id")[:2]["audio"]]

    model.eval()
    inputs = processor(samples, sampling_rate=processor.feature_extractor.sampling_rate)
    with torch.no_grad():
        predicted_ids = model.generate(**inputs)
    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print("Sanity-check transcriptions:")
    for transcription in transcriptions:
        print(f"  {transcription!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Local directory or Hub repo id of the original (remote-code) checkpoint.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the converted model.")
    parser.add_argument(
        "--run_sanity_check",
        action="store_true",
        help="Transcribe two librispeech samples with the converted model before saving.",
    )
    args = parser.parse_args()
    convert_and_write_model(args.input_path, args.output_path, run_sanity_check=args.run_sanity_check)


if __name__ == "__main__":
    main()
