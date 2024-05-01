#!/usr/bin/env python
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
""" PyTorch Transformer XL model evaluation script.
    Adapted from https://github.com/kimiyoung/transformer-xl.
    In particular https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/eval.py

    This script with default values evaluates a pretrained Transformer-XL on WikiText 103
"""


import argparse
import logging
import math
import time

import torch

from transformers import TransfoXLCorpus, TransfoXLLMHeadModel


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Language Model")
    parser.add_argument("--model_name", type=str, default="transfo-xl/transfo-xl-wt103", help="pretrained model name")
    parser.add_argument(
        "--split", type=str, default="test", choices=["all", "valid", "test"], help="which split to evaluate"
    )
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--tgt_len", type=int, default=128, help="number of tokens to predict")
    parser.add_argument("--ext_len", type=int, default=0, help="length of the extended context")
    parser.add_argument("--mem_len", type=int, default=1600, help="length of the retained previous heads")
    parser.add_argument("--clamp_len", type=int, default=1000, help="max positional embedding index")
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA even though CUA is available")
    parser.add_argument("--work_dir", type=str, required=True, help="path to the work_dir")
    parser.add_argument("--no_log", action="store_true", help="do not log the eval result")
    parser.add_argument("--same_length", action="store_true", help="set same length attention with masking")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    args = parser.parse_args()
    assert args.ext_len >= 0, "extended context length must be non-negative"

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    # Load a pre-processed dataset
    # You can also build the corpus yourself using TransfoXLCorpus methods
    # The pre-processing involve computing word frequencies to prepare the Adaptive input and SoftMax
    # and tokenizing the dataset
    # The pre-processed corpus is a convertion (using the conversion script )
    corpus = TransfoXLCorpus.from_pretrained(args.model_name)

    va_iter = corpus.get_iterator("valid", args.batch_size, args.tgt_len, device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator("test", args.batch_size, args.tgt_len, device=device, ext_len=args.ext_len)

    # Load a pre-trained model
    model = TransfoXLLMHeadModel.from_pretrained(args.model_name)
    model.to(device)

    logger.info(
        "Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}".format(
            args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len
        )
    )

    model.reset_memory_length(args.mem_len)
    if args.clamp_len > 0:
        model.clamp_len = args.clamp_len
    if args.same_length:
        model.same_length = True

    ###############################################################################
    # Evaluation code
    ###############################################################################
    def evaluate(eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_len, total_loss = 0, 0.0
        start_time = time.time()
        with torch.no_grad():
            mems = None
            for idx, (data, target, seq_len) in enumerate(eval_iter):
                ret = model(data, lm_labels=target, mems=mems)
                loss, _, mems = ret
                loss = loss.mean()
                total_loss += seq_len * loss.item()
                total_len += seq_len
            total_time = time.time() - start_time
        logger.info("Time : {:.2f}s, {:.2f}ms/segment".format(total_time, 1000 * total_time / (idx + 1)))
        return total_loss / total_len

    # Run on test data.
    if args.split == "all":
        test_loss = evaluate(te_iter)
        valid_loss = evaluate(va_iter)
    elif args.split == "valid":
        valid_loss = evaluate(va_iter)
        test_loss = None
    elif args.split == "test":
        test_loss = evaluate(te_iter)
        valid_loss = None

    def format_log(loss, split):
        log_str = "| {0} loss {1:5.2f} | {0} ppl {2:9.3f} ".format(split, loss, math.exp(loss))
        return log_str

    log_str = ""
    if valid_loss is not None:
        log_str += format_log(valid_loss, "valid")
    if test_loss is not None:
        log_str += format_log(test_loss, "test")

    logger.info("=" * 100)
    logger.info(log_str)
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
