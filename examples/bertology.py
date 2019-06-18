#!/usr/bin/env python3

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import BertModel, BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-uncased',
                                                help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertModel.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    

if __name__ == '__main__':
    run_model()


