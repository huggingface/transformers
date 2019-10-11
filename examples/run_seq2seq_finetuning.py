# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018 Microsoft and The HuggingFace Inc.  All rights reserved.
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
""" Finetuning seq2seq models for sequence generation.

We use the procedure described in [1] to finetune models for sequence
generation. Let S1 and S2 be the source and target sequence respectively; we
pack them using the start of sequence [SOS] and end of sequence [EOS] token:

    [SOS] S1 [EOS] S2 [EOS]

We then mask a fixed percentage of token from S2 at random and learn to predict
the masked words. [EOS] can be masked during finetuning so the model learns to
terminate the generation process.

[1] Dong Li, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng
Gao, Ming Zhou, and Hsiao-Wuen Hon.  “Unified Language Model Pre-Training for
Natural Language Understanding and Generation.” (May 2019) ArXiv:1905.03197
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Fine-tune the pretrained model on the corpus. """
    # Data sampler
    # Data loader
    # Training
    raise NotImplementedError


def evaluate(args, model, tokenizer, prefix=""):
    raise NotImplementedError


def main():
    raise NotImplementedError


def __main__():
    main()
