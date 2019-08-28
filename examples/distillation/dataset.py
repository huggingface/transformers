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
""" Dataloaders to train DistilBERT
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
from typing import List
import math
from itertools import chain
from collections import Counter
import numpy as np
import torch

from utils import logger

class Dataset:
    def __init__(self,
                 params,
                 data):
        self.params = params
        self.tokens_per_batch = params.tokens_per_batch
        self.batch_size = params.batch_size
        self.shuffle = params.shuffle
        self.group_by_size = params.group_by_size

        self.token_ids = np.array(data)
        self.lengths = np.uint16([len(t) for t in data])

        self.check()
        self.remove_long_sequences()
        self.remove_empty_sequences()
        self.check()
        self.print_statistics()

    def __len__(self):
        return len(self.lengths)

    def check(self):
        """
        Some sanity checks
        """
        assert len(self.token_ids) == len(self.lengths)

    def remove_long_sequences(self):
        """
        Sequences that are too long are splitted by chunk of max_position_embeddings.
        """
        indices = self.lengths >= self.params.max_position_embeddings
        logger.info(f'Splitting {sum(indices)} too long sequences.')

        def divide_chunks(l, n):
            return [l[i:i + n] for i in range(0, len(l), n)]

        new_tok_ids = []
        new_lengths = []
        cls_id, sep_id = self.params.special_tok_ids['cls_token'], self.params.special_tok_ids['sep_token']
        max_len = self.params.max_position_embeddings

        for seq_, len_ in zip(self.token_ids, self.lengths):
            if len_ <= max_len:
                new_tok_ids.append(seq_)
                new_lengths.append(len_)
            else:
                sub_seqs = []
                for sub_s in divide_chunks(seq_, max_len-2):
                    if sub_s[0] != cls_id:
                        sub_s = np.insert(sub_s, 0, cls_id)
                    if sub_s[-1] != sep_id:
                        sub_s = np.insert(sub_s, len(sub_s), cls_id)
                    assert len(sub_s) <= max_len
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
        indices = self.lengths > 5
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)
        logger.info(f'Remove {init_size - new_size} too short (<=5 tokens) sequences.')

    def print_statistics(self):
        """
        Print some statistics on the corpus. Only the master process.
        """
        if not self.params.is_master:
            return
        logger.info(f'{len(self)} sequences')
        # data_len = sum(self.lengths)
        # nb_unique_tokens = len(Counter(list(chain(*self.token_ids))))
        # logger.info(f'{data_len} tokens ({nb_unique_tokens} unique)')

        # unk_idx = self.params.special_tok_ids['unk_token']
        # nb_unkown = sum([(t==unk_idx).sum() for t in self.token_ids])
        # logger.info(f'{nb_unkown} unknown tokens (covering {100*nb_unkown/data_len:.2f}% of the data)')

    def select_data(self, a: int, b: int):
        """
        Select a subportion of the data.
        """
        n_sequences = len(self)
        assert 0 <= a < b <= n_sequences, ValueError(f'`0 <= a < b <= n_sequences` is not met with a={a} and b={b}')

        logger.info(f'Selecting sequences from {a} to {b} (excluded).')
        self.token_ids = self.token_ids[a:b]
        self.lengths = self.lengths[a:b]

        self.check()

    def split(self):
        """
        Distributed training: split the data accross the processes.
        """
        assert self.params.n_gpu > 1
        logger.info('Splitting the data accross the processuses.')
        n_seq = len(self)
        n_seq_per_procesus = n_seq // self.params.world_size
        a = n_seq_per_procesus * self.params.global_rank
        b = a + n_seq_per_procesus
        self.select_data(a=a, b=b)

    def batch_sequences(self,
                        token_ids: List[List[int]],
                        lengths: List[int]):
        """
        Do the padding and transform into torch.tensor.
        """
        assert len(token_ids) == len(lengths)

        # Max for paddings
        max_seq_len_ = max(lengths)

        # Pad token ids
        pad_idx = self.params.special_tok_ids['pad_token']
        tk_ = [list(t.astype(int)) + [pad_idx]*(max_seq_len_-len(t)) for t in token_ids]
        assert len(tk_) == len(token_ids)
        assert all(len(t) == max_seq_len_ for t in tk_)

        tk_t = torch.tensor(tk_)                  # (bs, max_seq_len_)
        lg_t = torch.tensor(lengths.astype(int))  # (bs)
        return tk_t, lg_t

    def get_batches_iterator(self,
                             batches):
        """
        Return an iterator over batches.
        """
        for sequences_ids in batches:
            token_ids, lengths = self.batch_sequences(self.token_ids[sequences_ids],
                                                    self.lengths[sequences_ids])
            yield (token_ids, lengths)

    def get_iterator(self,
                     seed: int = None):
        """
        Return a data iterator.
        """
        rng = np.random.RandomState(seed)

        n_sequences = len(self)
        indices = np.arange(n_sequences)

        if self.group_by_size:
            indices = indices[np.argsort(self.lengths[indices], kind='mergesort')]

        if self.tokens_per_batch == -1:
            batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        else:
            assert self.tokens_per_batch > 0
            batch_ids = np.cumsum(self.lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [indices[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        if self.shuffle:
            rng.shuffle(batches)

        assert n_sequences == sum([len(x) for x in batches])
        assert self.lengths[indices].sum() == sum([self.lengths[x].sum() for x in batches])

        return self.get_batches_iterator(batches=batches)
