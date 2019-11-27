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
Preprocessing script before distillation.
"""
import argparse
import logging
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer
# from tqdm import tqdm

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)

def main():
    parser = argparse.ArgumentParser(description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids).")
    parser.add_argument('--file_path', type=str, default='data/dump.txt',
                        help='The path to the data.')
    parser.add_argument('--tokenizer_type', type=str, default='bert', choices=['bert', 'roberta', 'gpt2'])
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased',
                        help="The tokenizer to use.")
    parser.add_argument('--dump_file', type=str, default='data/dump',
                        help='The dump file prefix.')
    args = parser.parse_args()


    logger.info(f'Loading Tokenizer ({args.tokenizer_name})')
    if args.tokenizer_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['cls_token'] # `[CLS]`
        sep = tokenizer.special_tokens_map['sep_token'] # `[SEP]`
    elif args.tokenizer_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['cls_token'] # `<s>`
        sep = tokenizer.special_tokens_map['sep_token'] # `</s>`
    elif args.tokenizer_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['bos_token'] # `<|endoftext|>`
        sep = tokenizer.special_tokens_map['eos_token'] # `<|endoftext|>`    

    logger.info(f'Loading text from {args.file_path}')
    data = []
    with open(args.file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(f'{bos} {line.strip()} {sep}')

    logger.info(f'Start encoding')
    logger.info(f'{len(data)} examples to process.')

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        rslt = executor.map(encode, repeat(tokenizer), data)
        # rslt = list(tqdm(executor.map(encode, repeat(tokenizer), data), total=len(data), desc='Encoding examples...'))

        logger.info('Finished binarization')
        logger.info(f'{len(data)} examples processed.')

        rslt_ = executor.map(np.uint16, rslt)
        # rslt_ = list(tqdm(executor.map(np.uint16, rslt), total=len(rslt), desc='Converting examples...'))
        
        random.shuffle(rslt_)
        
        dp_file = f'{args.dump_file}.{args.tokenizer_name}.pickle'
        logger.info(f'Dump to {dp_file}')
        pickle.dump(rslt_, open(dp_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
