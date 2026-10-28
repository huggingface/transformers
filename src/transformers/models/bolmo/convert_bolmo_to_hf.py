# Copyright 2026 the HuggingFace Team. All rights reserved.
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
Converts a remote-code Bolmo checkpoint (e.g. `allenai/Bolmo-7B`) to the native format. Weights are already named
right, only the config and tokenizer change:

    python src/transformers/models/bolmo/convert_bolmo_to_hf.py --input_dir allenai/Bolmo-7B --output_dir Bolmo-7B-hf
"""

import argparse
import json
import os
import tempfile

from huggingface_hub import hf_hub_download

from transformers import BolmoConfig, BolmoForCausalLM, BolmoTokenizer


def write_vocab(subword_tokenizer: str, save_dir: str) -> str:
    """Subword vocab with the eos/pad tokens renamed to Bolmo's, so the expansion maps them to Bolmo's special bytes."""
    with open(hf_hub_download(subword_tokenizer, "tokenizer.json"), encoding="utf-8") as f:
        tokenizer = json.load(f)
    vocab = dict(tokenizer["model"]["vocab"])
    vocab.update({token["content"]: token["id"] for token in tokenizer["added_tokens"]})
    rename = {"<|endoftext|>": "<bos>", "<|pad|>": "<pad>"}
    vocab_file = os.path.join(save_dir, "vocab.json")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump({rename.get(token, token): i for token, i in vocab.items()}, f, ensure_ascii=False)
    return vocab_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Remote-code Bolmo repo or folder")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subword_tokenizer", default="allenai/dolma2-tokenizer")
    args = parser.parse_args()

    with open(hf_hub_download(args.input_dir, "config.json"), encoding="utf-8") as f:
        original = json.load(f)
    for key in (
        "auto_map",
        "architectures",
        "tokenizer_config",
        "block_type",
        "norm_type",
        "use_qk_norm",
        "use_head_qk_norm",
        "boundary_predictor_lookahead",
        "boundary_threshold",
        "add_expanded_embeddings",
    ):
        original.pop(key, None)
    config = BolmoConfig(**original)

    model = BolmoForCausalLM.from_pretrained(args.input_dir, config=config)
    model.save_pretrained(args.output_dir)

    with tempfile.TemporaryDirectory() as tmp:
        BolmoTokenizer(write_vocab(args.subword_tokenizer, tmp)).save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
