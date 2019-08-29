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
Training DistilBERT.
"""
import os
import argparse
import pickle
import json
import shutil
import numpy as np
import torch

from pytorch_transformers import BertTokenizer, BertForMaskedLM
from pytorch_transformers import DistilBertForMaskedLM, DistilBertConfig

from distiller import Distiller
from utils import git_log, logger, init_gpu_params, set_seed
from dataset import Dataset


def main():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--dump_path", type=str, required=True,
                        help="The output directory (log, checkpoints, parameters, etc.)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="The binarized file (tokenized + tokens_to_ids) and grouped by sequence.")
    parser.add_argument("--token_counts", type=str, required=True,
                        help="The token counts in the data_file for MLM.")
    parser.add_argument("--force", action='store_true',
                        help="Overwrite dump_path if it already exists.")

    parser.add_argument("--vocab_size", default=30522, type=int,
                        help="The vocabulary size.")
    parser.add_argument("--max_position_embeddings", default=512, type=int,
                        help="Maximum sequence length we can model (including [CLS] and [SEP]).")
    parser.add_argument("--sinusoidal_pos_embds", action='store_false',
                        help="If true, the position embeddings are simply fixed with sinusoidal embeddings.")
    parser.add_argument("--n_layers", default=6, type=int,
                        help="Number of Transformer blocks.")
    parser.add_argument("--n_heads", default=12, type=int,
                        help="Number of heads in the self-attention module.")
    parser.add_argument("--dim", default=768, type=int,
                        help="Dimension through the network. Must be divisible by n_heads")
    parser.add_argument("--hidden_dim", default=3072, type=int,
                        help="Intermediate dimension in the FFN.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Dropout.")
    parser.add_argument("--attention_dropout", default=0.1, type=float,
                        help="Dropout in self-attention.")
    parser.add_argument("--activation", default='gelu', type=str,
                        help="Activation to use in self-attention")
    parser.add_argument("--tie_weights_", action='store_false',
                        help="If true, we tie the embeddings matrix with the projection over the vocabulary matrix. Default is true.")

    parser.add_argument("--from_pretrained_weights", default=None, type=str,
                        help="Load student initialization checkpoint.")
    parser.add_argument("--from_pretrained_config", default=None, type=str,
                        help="Load student initialization architecture config.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="The teacher BERT model.")

    parser.add_argument("--temperature", default=2., type=float,
                        help="Temperature for the softmax temperature.")
    parser.add_argument("--alpha_ce", default=0.5, type=float,
                        help="Linear weight for the distillation loss. Must be >=0.")
    parser.add_argument("--alpha_mlm", default=0.5, type=float,
                        help="Linear weight for the MLM loss. Must be >=0.")
    parser.add_argument("--alpha_mse", default=0.0, type=float,
                        help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument("--mlm_mask_prop", default=0.15, type=float,
                        help="Proportion of tokens for which we need to make a prediction.")
    parser.add_argument("--word_mask", default=0.8, type=float,
                        help="Proportion of tokens to mask out.")
    parser.add_argument("--word_keep", default=0.1, type=float,
                        help="Proportion of tokens to keep.")
    parser.add_argument("--word_rand", default=0.1, type=float,
                        help="Proportion of tokens to randomly replace.")
    parser.add_argument("--mlm_smoothing", default=0.7, type=float,
                        help="Smoothing parameter to emphasize more rare tokens (see XLM, similar to word2vec).")
    parser.add_argument("--restrict_ce_to_mask", action='store_true',
                        help="If true, compute the distilation loss only the [MLM] prediction distribution.")

    parser.add_argument("--n_epoch", type=int, default=3,
                        help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size (for each process).")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="If specified, modify the batches so that they have approximately this number of tokens.")
    parser.add_argument("--shuffle", action='store_false',
                        help="If true, shuffle the sequence order. Default is true.")
    parser.add_argument("--group_by_size", action='store_false',
                        help="If true, group sequences that have similar length into the same batch. Default is true.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=50,
                        help="Gradient accumulation for larger training batches.")
    parser.add_argument("--warmup_prop", default=0.05, type=float,
                        help="Linear warmup proportion.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--initializer_range", default=0.02, type=float,
                        help="Random initialization range.")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--n_gpu", type=int, default=1,
                        help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56,
                        help="Random seed")

    parser.add_argument("--log_interval", type=int, default=500,
                        help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=4000,
                        help="Checkpoint interval.")
    args = parser.parse_args()


    ## ARGS ##
    init_gpu_params(args)
    set_seed(args)
    if args.is_master:
        if os.path.exists(args.dump_path):
            if not args.force:
                raise ValueError(f'Serialization dir {args.dump_path} already exists, but you have not precised wheter to overwrite it'
                                   'Use `--force` if you want to overwrite it')
            else:
                shutil.rmtree(args.dump_path)

        if not os.path.exists(args.dump_path):
            os.makedirs(args.dump_path)
        logger.info(f'Experiment will be dumped and logged in {args.dump_path}')


        ### SAVE PARAMS ###
        logger.info(f'Param: {args}')
        with open(os.path.join(args.dump_path, 'parameters.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        git_log(args.dump_path)
    assert (args.from_pretrained_weights is None and args.from_pretrained_config is None) or \
           (args.from_pretrained_weights is not None and args.from_pretrained_config is not None)


    ### TOKENIZER ###
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    special_tok_ids = {}
    for tok_name, tok_symbol in bert_tokenizer.special_tokens_map.items():
        idx = bert_tokenizer.all_special_tokens.index(tok_symbol)
        special_tok_ids[tok_name] = bert_tokenizer.all_special_ids[idx]
    logger.info(f'Special tokens {special_tok_ids}')
    args.special_tok_ids = special_tok_ids


    ## DATA LOADER ##
    logger.info(f'Loading data from {args.data_file}')
    with open(args.data_file, 'rb') as fp:
        data = pickle.load(fp)


    assert os.path.isfile(args.token_counts)
    logger.info(f'Loading token counts from {args.token_counts} (already pre-computed)')
    with open(args.token_counts, 'rb') as fp:
        counts = pickle.load(fp)
        assert len(counts) == args.vocab_size
    token_probs = np.maximum(counts, 1) ** -args.mlm_smoothing
    for idx in special_tok_ids.values():
        token_probs[idx] = 0.  # do not predict special tokens
    token_probs = torch.from_numpy(token_probs)


    train_dataloader = Dataset(params=args, data=data)
    logger.info(f'Data loader created.')


    ## STUDENT ##
    if args.from_pretrained_weights is not None:
        assert os.path.isfile(os.path.join(args.from_pretrained_weights))
        assert os.path.isfile(os.path.join(args.from_pretrained_config))
        logger.info(f'Loading pretrained weights from {args.from_pretrained_weights}')
        logger.info(f'Loading pretrained config from {args.from_pretrained_config}')
        stu_architecture_config = DistilBertConfig.from_json_file(args.from_pretrained_config)
        student = DistilBertForMaskedLM.from_pretrained(args.from_pretrained_weights,
                                                     config=stu_architecture_config)
    else:
        args.vocab_size_or_config_json_file = args.vocab_size
        stu_architecture_config = DistilBertConfig(**vars(args))
        student = DistilBertForMaskedLM(stu_architecture_config)


    if args.n_gpu > 0:
        student.to(f'cuda:{args.local_rank}')
    logger.info(f'Student loaded.')


    ## TEACHER ##
    teacher = BertForMaskedLM.from_pretrained(args.bert_model)
    if args.n_gpu > 0:
        teacher.to(f'cuda:{args.local_rank}')
    logger.info(f'Teacher loaded from {args.bert_model}.')

    ## DISTILLER ##
    torch.cuda.empty_cache()
    distiller = Distiller(params=args,
                          dataloader=train_dataloader,
                          token_probs=token_probs,
                          student=student,
                          teacher=teacher)
    distiller.train()
    logger.info("Let's go get some drinks.")


if __name__ == "__main__":
    main()
