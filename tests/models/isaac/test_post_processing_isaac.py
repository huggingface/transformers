# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for Isaac processor post-processing helpers."""

import pytest

from transformers import PythonBackend
from transformers.models.isaac.image_processing_isaac_fast import IsaacImageProcessorFast
from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import require_torch


class SimpleIsaacTokenizer(PythonBackend):
    vocab_files_names = {}
    model_input_names = ["input_ids"]

    def __init__(self):
        self._vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "<image>": 4,
        }
        self._ids_to_tokens = {idx: tok for tok, idx in self._vocab.items()}
        super().__init__(
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            extra_special_tokens=["<image>"],
            model_max_length=512,
        )

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        clean = text.replace("\n", " ").strip()
        if not clean:
            return []
        return [token for token in clean.split(" ") if token]

    def _convert_token_to_id(self, token):
        if token not in self._vocab:
            next_id = len(self._vocab)
            self._vocab[token] = next_id
            self._ids_to_tokens[next_id] = token
        return self._vocab[token]

    def _convert_id_to_token(self, index):
        return self._ids_to_tokens.get(index, self.unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1
        return [self.bos_token_id] + list(token_ids_0) + [self.eos_token_id]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


def _make_processor():
    return IsaacProcessor(image_processor=IsaacImageProcessorFast(), tokenizer=SimpleIsaacTokenizer())


@require_torch
def test_post_process_generation_extracts_boxes_and_cleans_text():
    processor = _make_processor()

    generated_text = (
        "No, it is not safe to cross the street. "
        '<point_box mention="traffic light" t="0.5">(808, 247), (863, 386)</point_box>'
    )

    clean_text, annotations = processor.post_process_generation(generated_text)

    assert clean_text == "No, it is not safe to cross the street."
    assert len(annotations) == 1
    box = annotations[0]
    assert box.mention == "traffic light"
    assert box.t == pytest.approx(0.5)
    assert box.top_left.x == 808
    assert box.top_left.y == 247
    assert box.bottom_right.x == 863
    assert box.bottom_right.y == 386
