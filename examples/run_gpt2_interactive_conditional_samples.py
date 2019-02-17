#!/usr/bin/env python3

import argparse
import logging

import torch
import numpy as np

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda'):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device)
    prev = context
    output = context
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            prev = torch.multinomial(logits, 1)
            output = torch.cat((output, prev), dim=1)
    return output

def interact_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    args = parser.parse_args()
    print(args)

    if args.batch_size is None:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    while True:
        raw_text = input("Model prompt >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Model prompt >>> ")
        context_tokens = enc.encode(raw_text)
        generated = 0
        for _ in range(args.nsamples // args.batch_size):
            out = sample_sequence(
                model=model, length=args.length,
                context=context_tokens,
                batch_size=args.batch_size,
                temperature=args.temperature, top_k=args.top_k, device=device
            )
            out = out[:, len(context_tokens):]
            for i in range(args.batch_size):
                generated += 1
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
        print("=" * 80)

if __name__ == '__main__':
    interact_model()

