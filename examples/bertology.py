#!/usr/bin/env python3

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        args.device, n_gpu, bool(args.local_rank != -1), args.fp16))

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    

if __name__ == '__main__':
    run_model()


