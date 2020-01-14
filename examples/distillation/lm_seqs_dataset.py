# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
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
""" Dataset to distilled models
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import logger


class LmSeqsDataset(Dataset):
    """Custom Dataset wrapping language modeling sequences.

    Each sample will be retrieved by indexing the list of token_ids and their corresponding lengths.

    Input:
    ------
        params: `NameSpace` parameters
        data: `List[np.array[int]]
    """

    def __init__(self, params, data):
        self.params = params

        self.token_ids = np.array(data)
        self.lengths = np.array([len(t) for t in data])

        self.check()
        self.remove_long_sequences()
        self.remove_empty_sequences()
        self.remove_unknown_sequences()
        self.check()
        self.print_statistics()

    def __getitem__(self, index):
        return (self.token_ids[index], self.lengths[index])

    def __len__(self):
        return len(self.lengths)

    def check(self):
        """
        Some sanity checks
        """
        assert len(self.token_ids) == len(self.lengths)
        assert all(self.lengths[i] == len(self.token_ids[i]) for i in range(len(self.lengths)))

    def remove_long_sequences(self):
        """
        Sequences that are too long are splitted by chunk of max_model_input_size.
        """
        max_len = self.params.max_model_input_size
        indices = self.lengths > max_len
        logger.info(f"Splitting {sum(indices)} too long sequences.")

        def divide_chunks(l, n):
            return [l[i : i + n] for i in range(0, len(l), n)]

        new_tok_ids = []
        new_lengths = []
        if self.params.mlm:
            cls_id, sep_id = self.params.special_tok_ids["cls_token"], self.params.special_tok_ids["sep_token"]
        else:
            cls_id, sep_id = self.params.special_tok_ids["bos_token"], self.params.special_tok_ids["eos_token"]

        for seq_, len_ in zip(self.token_ids, self.lengths):
            assert (seq_[0] == cls_id) and (seq_[-1] == sep_id), seq_
            if len_ <= max_len:
                new_tok_ids.append(seq_)
                new_lengths.append(len_)
            else:
                sub_seqs = []
                for sub_s in divide_chunks(seq_, max_len - 2):
                    if sub_s[0] != cls_id:
                        sub_s = np.insert(sub_s, 0, cls_id)
                    if sub_s[-1] != sep_id:
                        sub_s = np.insert(sub_s, len(sub_s), sep_id)
                    assert len(sub_s) <= max_len
                    assert (sub_s[0] == cls_id) and (sub_s[-1] == sep_id), sub_s
                    sub_seqs.append(sub_s)

                new_tok_ids.extend(sub_seqs)
                new_lengths.extend([len(l) for l in sub_seqs])

        self.token_ids = np.array(new_tok_ids)
        self.lengths = np.array(new_lengths)

    def remove_empty_sequences(self):
        """
        Too short sequences are simply removed. This could be tunedd.
        """
        init_size = len(self)
        indices = self.lengths > 11
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)
        logger.info(f"Remove {init_size - new_size} too short (<=11 tokens) sequences.")

    def remove_unknown_sequences(self):
        """
        Remove sequences with a (too) high level of unknown tokens.
        """
        if "unk_token" not in self.params.special_tok_ids:
            return
        else:
            unk_token_id = self.params.special_tok_ids["unk_token"]
        init_size = len(self)
        unk_occs = np.array([np.count_nonzero(a == unk_token_id) for a in self.token_ids])
        indices = (unk_occs / self.lengths) < 0.5
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)
        logger.info(f"Remove {init_size - new_size} sequences with a high level of unknown tokens (50%).")

    def print_statistics(self):
        """
        Print some statistics on the corpus. Only the master process.
        """
        if not self.params.is_master:
            return
        logger.info(f"{len(self)} sequences")
        # data_len = sum(self.lengths)
        # nb_unique_tokens = len(Counter(list(chain(*self.token_ids))))
        # logger.info(f'{data_len} tokens ({nb_unique_tokens} unique)')

        # unk_idx = self.params.special_tok_ids['unk_token']
        # nb_unkown = sum([(t==unk_idx).sum() for t in self.token_ids])
        # logger.info(f'{nb_unkown} unknown tokens (covering {100*nb_unkown/data_len:.2f}% of the data)')

    def batch_sequences(self, batch):
        """
        Do the padding and transform into torch.tensor.
        """
        token_ids = [t[0] for t in batch]
        lengths = [t[1] for t in batch]
        assert len(token_ids) == len(lengths)

        # Max for paddings
        max_seq_len_ = max(lengths)

        # Pad token ids
        if self.params.mlm:
            pad_idx = self.params.special_tok_ids["pad_token"]
        else:
            pad_idx = self.params.special_tok_ids["unk_token"]
        tk_ = [list(t.astype(int)) + [pad_idx] * (max_seq_len_ - len(t)) for t in token_ids]
        assert len(tk_) == len(token_ids)
        assert all(len(t) == max_seq_len_ for t in tk_)

        tk_t = torch.tensor(tk_)  # (bs, max_seq_len_)
        lg_t = torch.tensor(lengths)  # (bs)
        return tk_t, lg_t
