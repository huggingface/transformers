# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset $CORPUS_DIR/train.txt
          --eval_dataset $CORPUS_DIR/eval.txt
          --output_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                     AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME,
                                     WarmupLinearSchedule)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def load_lm_dataset(dataset_path, newline_delimiter=' '):
    """ Output a list of strings """
    output = []
    with open(dataset_path, encoding='utf_8') as f:
        for line in tqdm(f):
            line = line.strip()
            if len(line) > 0:
                output.append(line)
    output = newline_delimiter.join(output)
    return output

def chunk_indexable(indexable, batch_len):
    N_splits = len(indexable) // batch_len
    for n in range(N_splits):
        start = batch_len * n
        end = start + batch_len
        chunk = indexable[start:end]
        yield chunk

def pre_process_datasets(encoded_datasets, input_len):
    """ Pre-process datasets, each represented as a string

        To a tensor of shape (n_batch, input_len)
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset) // input_len
        input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
        lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)
        for i, chunk in enumerate(chunk_indexable(dataset, input_len)):
            chunk_len = len(chunk)
            input_ids[i, :chunk_len] = chunk
        all_inputs = (input_ids,)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs)) # pylint: disable=not-callable
    return tensor_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training \
                        steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before\
                        performing a backward/update pass.")
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pylint: disable=no-member
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    special_tokens = []
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens) # pylint: disable=unused-variable
    model = OpenAIGPTLMHeadModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))
    model.to(device)

    # Load and encode the datasets
    if not args.train_dataset and not args.eval_dataset:
        assert(False) # need at least one of training or eval dataset

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)

    logger.info("Encoding dataset...")
    train_tensor_dataset, eval_tensor_dataset = None, None
    if args.cache_path is not None:
        if os.path.exists(args.cache_path):
            train_tensor_dataset, eval_tensor_dataset = torch.load(args.cache_path)

    if train_tensor_dataset is None:
        train_dataset = load_lm_dataset(args.train_dataset)
        eval_dataset = load_lm_dataset(args.eval_dataset)
        datasets = (train_dataset, eval_dataset)
        encoded_datasets = tokenize_and_encode(datasets)

        # Compute the max input length for the Transformer
        max_length = model.config.n_positions // 2 - 2

        # Prepare inputs tensors and dataloaders
        tensor_datasets = pre_process_datasets(encoded_datasets, max_length)
        train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]

        if args.cache_path is not None and not os.path.exists(args.cache_path):
            torch.save((train_tensor_dataset, eval_tensor_dataset), args.cache_path)

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Prepare optimizer
    if args.do_train:
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps //\
                (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader)\
                // args.gradient_accumulation_steps * args.num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            nb_tr_examples = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for _, batch in enumerate(tqdm_bar):
                input_ids, = tuple(t.to(device) for t in batch)
                losses = model(input_ids, labels=input_ids)
                loss = losses[0].mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                nb_tr_steps += 1
                nb_tr_examples += input_ids.size(0)
                avg_tr_loss = tr_loss / nb_tr_steps
                tqdm_bar.desc = "T loss: {:.2e} T ppl: {:.2e} lr: {:.2e}".format(avg_tr_loss, np.exp(avg_tr_loss), scheduler.get_lr()[0])

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = OpenAIGPTLMHeadModel.from_pretrained(args.output_dir)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.output_dir)
        model.to(device)

    if args.do_eval:
        model.eval()
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, = batch
            with torch.no_grad():
               lm_loss = model(input_ids, labels=input_ids)[0]

            eval_loss += lm_loss.mean().item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_ppl = np.exp(eval_loss)
        train_loss = tr_loss/nb_tr_steps if args.do_train else None
        train_ppl = np.exp(train_loss) if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_ppl': eval_ppl,
                  'train_loss': train_loss,
                  'train_ppl': train_ppl}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == '__main__':
    main()
