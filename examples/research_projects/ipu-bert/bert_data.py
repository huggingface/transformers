# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import random
import multiprocessing
import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset
from transformers import BertTokenizerFast
from tfrecord.reader import tfrecord_loader


TFRECORD_KEYS = (           # Torch Model Keys
    'input_ids',            # input_ids                  : tokens after masking
    'input_mask',           # attention_mask             : 1 if padding token, 0 otherwise
    'segment_ids',          # token_type_ids             : sentence 0 or 1
    'masked_lm_positions',  # masked_lm_positions        : position of masked tokens in input_ids
    'masked_lm_ids',        # masked_lm_labels=None      : label of masked tokens with padding as 0.
    'next_sentence_labels'  # next_sentence_label=None   : 1 if next sentence, 0 otherwise
)


def expand_glob_files(files):
    result = []
    for filepath in files:
        expanded = glob.glob(filepath)
        if len(expanded) < 1:
            raise FileNotFoundError(f"Could not find file: {filepath}")
        result += expanded
    return result


class TFRecordPretrainingDataset(IterableDataset):
    def __init__(self,
                 input_files,
                 file_buffer_size=100,
                 shuffle=True,
                 seed=42):
        self.files = expand_glob_files(input_files)
        self.file_buffer_size = file_buffer_size
        self.shuffle = shuffle
        if self.shuffle:
            self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.file_index = 0
        self.data_index = 0

    def samples_per_file(self, filename):
        reader = tfrecord_loader(filename,
                                 None,
                                 list(TFRECORD_KEYS))
        count = 0
        for _ in reader:
            count += 1
        return count

    def __len__(self):
        if getattr(self, "_len", None) is None:
            pool = multiprocessing.Pool(
                min(multiprocessing.cpu_count(), len(self.files)))
            num_samples = pool.map(self.samples_per_file, self.files)
            pool.close()
            pool.join()
            self._len = sum(num_samples)
        return self._len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            self.shard = None

        self.reset()
        if self.shuffle:
            self._rng.shuffle(self.files)
        self.load_data()
        return self

    def __next__(self):
        if self.data_index >= len(self.data):
            self.load_data()
        data = self.data[self.data_index]
        self.data_index += 1
        return data

    def load_data(self):
        # This drops the remainder
        if self.file_index >= len(self.files):
            raise StopIteration
        self.data = []
        # Load multiple files into the data buffer at a time
        for _ in range(self.file_buffer_size):
            self.data += self.load_file()
            self.file_index += 1
            if self.file_index >= len(self.files):
                break
        if self.shuffle:
            self._rng.shuffle(self.data)
        self.data_index = 0

    def load_file(self):
        reader = tfrecord_loader(self.files[self.file_index],
                                 self.files[self.file_index].replace(".tfrecord", ".index"),
                                 list(TFRECORD_KEYS),
                                 self.shard)
        data = []
        for datum in reader:
            data.append([datum[key] for key in TFRECORD_KEYS])
        return data


class GeneratedPretrainingDataset(Dataset):
    def __init__(self, vocab_size, sequence_length, mask_tokens, length=1, seed=42):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.mask_tokens = mask_tokens
        self.length = length
        self.seed = seed
        self.data = self.generate_data()

    def generate_data(self):
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            tokens = torch.randint(0, self.vocab_size,
                                   [self.sequence_length],
                                   dtype=torch.long)
            mask = torch.ones_like(tokens)
            types = torch.zeros_like(tokens)
            masked_lm_positions = torch.randint(0, self.sequence_length,
                                                [self.mask_tokens],
                                                dtype=torch.long)
            masked_lm_label = torch.randint(0, self.vocab_size,
                                            [self.mask_tokens],
                                            dtype=torch.long)
            next_sentence_label = torch.randint(0, 2, [1], dtype=torch.long)
        return tokens, mask, types, masked_lm_positions, masked_lm_label, next_sentence_label

    def __len__(self):
        return self.length

    def __getitem__(self, __):
        return self.data


def get_generated_datum(config):
    result = []
    dataset = GeneratedPretrainingDataset(config.vocab_size,
                                          config.sequence_length,
                                          config.mask_tokens)
    data = (dataset[i] for i in range(config.samples_per_step))
    for batches in zip(*data):
        result.append(torch.stack(batches))
    return result


def get_dataset(config):
    if config.dataset == 'generated':
        dataset = GeneratedPretrainingDataset(config.vocab_size,
                                              config.sequence_length,
                                              config.mask_tokens,
                                              config.samples_per_step,
                                              config.random_seed)
    elif config.dataset == 'pretraining':
        dataset = TFRecordPretrainingDataset(config.input_files,
                                             file_buffer_size=config.file_buffer_size,
                                             seed=config.random_seed)
    else:
        raise RuntimeError(f"Unknown dataset '{config.dataset}', aborting.")
    return dataset


if __name__ == "__main__":

    print("\nYou are executing bert_data directly.")
    print("Let's read the first input from sample dataset.")

    dataset = TFRecordPretrainingDataset(["data/sample_text.tfrecord"])
    print("dataset length: ", len(dataset), "\n")
    first = next(iter(dataset))
    named_datum = zip(['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids', 'next_sentence_labels'], first)
    for (name, value) in iter(named_datum):
        print(name, value.shape, value.dtype, type(value), value, "\n\n")

    print("And now, we are going to decode the tokens.\n")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased",
                                                  do_lower_case=True)
    print("\n\n", tokenizer.decode(first[0]), "\n\n")
