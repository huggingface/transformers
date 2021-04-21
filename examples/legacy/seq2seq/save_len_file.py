#!/usr/bin/env python
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

import fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer
from utils import Seq2SeqDataset, pickle_save


def save_len_file(
    tokenizer_name, data_dir, max_source_length=1024, max_target_length=1024, consider_target=False, **kwargs
):
    """Save max(src_len, tgt_len) for each example to allow dynamic batching."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    train_ds = Seq2SeqDataset(tok, data_dir, max_source_length, max_target_length, type_path="train", **kwargs)
    pad = tok.pad_token_id

    def get_lens(ds):
        dl = tqdm(
            DataLoader(ds, batch_size=512, num_workers=8, shuffle=False, collate_fn=ds.collate_fn),
            desc=str(ds.len_file),
        )
        max_lens = []
        for batch in dl:
            src_lens = batch["input_ids"].ne(pad).sum(1).tolist()
            tgt_lens = batch["labels"].ne(pad).sum(1).tolist()
            if consider_target:
                for src, tgt in zip(src_lens, tgt_lens):
                    max_lens.append(max(src, tgt))
            else:
                max_lens.extend(src_lens)
        return max_lens

    train_lens = get_lens(train_ds)
    val_ds = Seq2SeqDataset(tok, data_dir, max_source_length, max_target_length, type_path="val", **kwargs)
    val_lens = get_lens(val_ds)
    pickle_save(train_lens, train_ds.len_file)
    pickle_save(val_lens, val_ds.len_file)


if __name__ == "__main__":
    fire.Fire(save_len_file)
