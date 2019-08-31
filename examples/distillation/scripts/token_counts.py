# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
Preprocessing script before training DistilBERT.
"""
from collections import Counter
import argparse
import pickle

from examples.distillation.utils import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Token Counts for smoothing the masking probabilities in MLM (cf XLM/word2vec)")
    parser.add_argument("--data_file", type=str, default="data/dump.bert-base-uncased.pickle",
                        help="The binarized dataset.")
    parser.add_argument("--token_counts_dump", type=str, default="data/token_counts.bert-base-uncased.pickle",
                        help="The dump file.")
    parser.add_argument("--vocab_size", default=30522, type=int)
    args = parser.parse_args()

    logger.info(f'Loading data from {args.data_file}')
    with open(args.data_file, 'rb') as fp:
        data = pickle.load(fp)

    logger.info('Counting occurences for MLM.')
    counter = Counter()
    for tk_ids in data:
        counter.update(tk_ids)
    counts = [0]*args.vocab_size
    for k, v in counter.items():
        counts[k] = v

    logger.info(f'Dump to {args.token_counts_dump}')
    with open(args.token_counts_dump, 'wb') as handle:
        pickle.dump(counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
