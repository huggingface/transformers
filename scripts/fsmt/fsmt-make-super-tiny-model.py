#!/usr/bin/env python
# coding: utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# This script creates a super tiny model that is useful inside tests, when we just want to test that
# the machinery works, without needing to the check the quality of the outcomes.
#
# This version creates a tiny vocab first, and then a tiny model - so the outcome is truly tiny -
# all files ~60KB. As compared to taking a full-size model, reducing to the minimum its layers and
# emb dimensions, but keeping the full vocab + merges files, leading to ~3MB in total for all files.
# The latter is done by `fsmt-make-super-tiny-model.py`.
#
# It will be used then as "stas/tiny-wmt19-en-ru"

from pathlib import Path
import json
import tempfile

from transformers import FSMTTokenizer, FSMTConfig, FSMTForConditionalGeneration
from transformers.models.fsmt.tokenization_fsmt import VOCAB_FILES_NAMES

mname_tiny = "tiny-wmt19-en-ru"

# Build

# borrowed from a test 
vocab = [ "l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "w</w>", "r</w>", "t</w>", "lo", "low", "er</w>", "low</w>", "lowest</w>", "newer</w>", "wider</w>", "<unk>", ]
vocab_tokens = dict(zip(vocab, range(len(vocab))))
merges = ["l o 123", "lo w 1456", "e r</w> 1789", ""]

with tempfile.TemporaryDirectory() as tmpdirname:
    build_dir = Path(tmpdirname)
    src_vocab_file = build_dir / VOCAB_FILES_NAMES["src_vocab_file"]
    tgt_vocab_file = build_dir / VOCAB_FILES_NAMES["tgt_vocab_file"]
    merges_file = build_dir / VOCAB_FILES_NAMES["merges_file"]
    with open(src_vocab_file, "w") as fp: fp.write(json.dumps(vocab_tokens))
    with open(tgt_vocab_file, "w") as fp: fp.write(json.dumps(vocab_tokens))
    with open(merges_file, "w") as fp   : fp.write("\n".join(merges))

    tokenizer = FSMTTokenizer(
        langs=["en", "ru"],
        src_vocab_size = len(vocab),
        tgt_vocab_size = len(vocab),
        src_vocab_file=src_vocab_file,
        tgt_vocab_file=tgt_vocab_file,
        merges_file=merges_file,
    )
    
config = FSMTConfig(
    langs=['ru', 'en'],
    src_vocab_size=1000, tgt_vocab_size=1000,
    d_model=4,
    encoder_layers=1, decoder_layers=1,
    encoder_ffn_dim=4, decoder_ffn_dim=4,
    encoder_attention_heads=1, decoder_attention_heads=1,
)

tiny_model = FSMTForConditionalGeneration(config)
print(f"num of params {tiny_model.num_parameters()}")

# Test
batch = tokenizer(["Making tiny model"], return_tensors="pt")
outputs = tiny_model(**batch)

print("test output:", len(outputs.logits[0]))

# Save
tiny_model.half() # makes it smaller
tiny_model.save_pretrained(mname_tiny)
tokenizer.save_pretrained(mname_tiny)

print(f"Generated {mname_tiny}")

# Upload
# transformers-cli upload tiny-wmt19-en-ru
