import argparse
import functools
import json
import math
import numpy as np
import os
import pickle
import torch

from random import choice
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from time import time
from tqdm import tqdm, trange

from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer

from transformers import AdamW, get_linear_schedule_with_warmup


# prepare question for generation, returns x, y prefix
def prepare_question(question, document, tokenizer, args, answer='', n_start=0, device='cuda:0'):
    if '--T--' not in question:
        question = question.strip() + ' --T--'
    if 'bart-' in args.model_type or 't5-' in args.model_type:
        if args.seq2seq_type == 'qd-a':
            in_st = "{} {} {}".format(
                question.lower().strip(),
                tokenizer.sep_token,
                document.lower().strip(),
            )
            out_st = "{} {}".format(tokenizer.bos_token, answer)
        elif args.seq2seq_type == 'd-qa':
            in_st = document.lower().strip()
            out_st = "{} {} {} {}".format(
                tokenizer.bos_token,
                question.lower().strip(),
                tokenizer.sep_token,
                answer,
            )
    else:
        raise NotImplementedError
    x, _, y, _ = make_batch([(in_st, out_st)], tokenizer, args, device)
    if 'bart-' in args.model_type or 't5-' in args.model_type:
        cur_len = list(y[0]).index(tokenizer.pad_token_id)
        y = y[:,:1 + n_start]
    return x, y


# self-contained answer question from our examples
def answer_example(qda_dct, model, tokenizer, args, n_start=0, n_beams=4, n_samples=4, min_length=32, max_length=128, rep_pen=2., verbose=True):
    if verbose:
        print('QUESTION-{}:'.format(qda_dct['id']), qda_dct['question'][:256])
        print('DOCUMENT: \t', qda_dct['document'][:256], '.....')
    answer = qda_dct['answer'] if n_start > 0 else ''
    x, y = prepare_question(qda_dct['question'], qda_dct['document'], tokenizer, args, answer, n_start)
    answers = {}
    if n_beams > 0:
        if verbose:
            print('\n\n-----beam search')
        pred = model.generate(
            input_ids=x,
            prefix_ids=y,
            num_beams=n_beams,
            min_length=min_length + y.shape[-1],
            max_length=max_length + y.shape[-1],  # +2 from original because we start at step=1 and stop before max_length
            early_stopping=True,
            repetition_penalty=rep_pen,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
        assert(pred.shape[-1] >= min_length)
        # ans = tokenizer.decode(list(pred[y.shape[-1]:]), skip_special_tokens=True).replace('</A>', '').strip()
        ans = tokenizer.decode(list(pred), skip_special_tokens=True).replace('</A>', '').strip()
        answers['beam'] = ans
        if verbose:
            print('BEAM ANS:  ', ans)
    if n_samples > 0:
        if verbose:
            print('\n\n-----sampling')
        preds = model.generate(
            input_ids=x,
            decoder_input_ids=y,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=10,
            top_p=0.98,
            min_length=min_length + y.shape[-1],
            max_length=max_length + y.shape[-1],  # +2 from original because we start at step=1 and stop before max_length
            early_stopping=True,
            repetition_penalty=rep_pen,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        ans_ls = [
            # tokenizer.decode(list(pred[y.shape[-1]:]), skip_special_tokens=True).replace('</A>', '').strip()
            tokenizer.decode(list(pred), skip_special_tokens=True).replace('</A>', '').strip()
            for pred in preds
        ]
        answers['samples'] = ans_ls
        if verbose:
            for i, ans in enumerate(ans_ls):
                    print('SAMPLE-{}:  '.format(i+1), ans + '\n')
    if verbose:
        print('\n\n-----original answer')
        print('GOLD ANS:  ', qda_dct['answer'])
    return answers


class ELI5Dataset(Dataset):

    def __init__(self, example_dicts, tokenizer, args):
        self.data = example_dicts
        self.model_type = args.model_type
        self.s2s_type = args.seq2seq_type
        self.tokenizer = tokenizer
        # sampling for multi-task
        sample_tasks = [
            ((['question', 'document'], ['answer']), 0.3),
            ((['document'], ['question', 'answer']), 0.3),
            ((['question'], ['answer']), 0.2),
            ((['question', 'answer'], ['document']), 0.1),
            ((['question'], ['answer', 'document']), 0.1),
        ]
        self.tasks = [t for t, p in sample_tasks]
        self.probs = [p for t, p in sample_tasks]
        self.probs = [p / sum(self.probs) for p in self.probs]

    def __len__(self):
        return len(self.data)

    def make_example(self, qda_dict):
        if 'bart-' in self.model_type:
            if self.s2s_type == 'qd-a':
                in_st = "{} {} {}".format(
                    qda_dict['question'].lower().replace(' --t--', '').strip(),
                    self.tokenizer.sep_token,
                    qda_dict['document'].lower().strip(),
                )
                out_st = "{} {} {}".format(
                    self.tokenizer.bos_token,
                    qda_dict['answer'].lower().strip(),
                    self.tokenizer.eos_token,
                )
            elif self.s2s_type == 'd-qa':
                in_st = "{}".format(
                    qda_dict['document'].lower().strip(),
                )
                out_st = "{} {} answer: {} {}".format(
                    self.tokenizer.bos_token,
                    qda_dict['question'].lower().replace(' --t--', '').strip(),
                    qda_dict['answer'].lower().strip(),
                    self.tokenizer.eos_token,
                )
        elif 't5-' in self.model_type:
            if self.s2s_type == 'qd-a':
                in_st = "long form question: {} context: {}".format(
                    qda_dict['question'].lower().replace(' --t--', '').strip(),
                    qda_dict['document'].lower().strip(),
                )
                out_st = "{} {} {}".format(
                    self.tokenizer.bos_token,
                    qda_dict['answer'].lower().strip(),
                    self.tokenizer.eos_token,
                )
            elif self.s2s_type == 'd-qa':
                in_st = "context: {}".format(
                    qda_dict['document'].lower().strip(),
                )
                out_st = "{} long form question: {} answer: {} {}".format(
                    self.tokenizer.bos_token,
                    qda_dict['question'].replace(' --t--', '').lower().strip(),
                    qda_dict['answer'].lower().strip(),
                    self.tokenizer.eos_token,
                )
        else:
            raise NotImplementedError
        return (in_st, out_st)

    def __getitem__(self, idx):
        return self.make_example(self.data[idx])


def make_batch(sent_pairs, tokenizer, args, device='cpu'):
    tok_batch_x = tokenizer.batch_encode_plus(
        [x for x, y in sent_pairs],
        max_length=args.max_length,
        return_tensors="pt",
        add_special_tokens=False,
        pad_to_max_length=True
    )
    tok_batch_y = tokenizer.batch_encode_plus(
        [y for x, y in sent_pairs],
        max_length=args.max_length, # TODO: change to max_length
        return_tensors="pt",
        add_special_tokens=False,
        pad_to_max_length=True
    )
    x, x_mask = (tok_batch_x['input_ids'].to(device), tok_batch_x['attention_mask'].to(device))
    y, y_mask = (tok_batch_y['input_ids'].to(device), tok_batch_y['attention_mask'].to(device))
    return x, x_mask, y, y_mask


def train(model, dataset, tokenizer, optimizer, scheduler, args, print_freq=10):
    # make data sampler
    train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(
        make_batch,
        tokenizer=tokenizer,
        args=args,
        device='cuda:0'
    )
    train_dataloader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=model_collate_fn
    )
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
    pad_id = tokenizer.pad_token_id
    # training loop
    for e in range(1):
        st_time = time()
        loc_steps = 0
        loc_loss = 0.0
        tot_steps = 0
        tot_loss = 0.0
        st_time = time()
        for step, batch in enumerate(epoch_iterator):
            x, x_mask, y, y_mask = batch
            y_ids = y[:,:-1].contiguous()
            y_labels = y[:,1:].clone()
            y_labels[y[:,1:] == pad_id] = -100
            m_out = model(
                input_ids=x,
                attention_mask=x_mask,
                decoder_input_ids=y_ids,
                lm_labels=y_labels,
            )
            loss = m_out[0]
            loss.backward()
            if step % args.batches_per_step == 0:
                if args.clip_norm > 0.:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            # book-keeping
            loc_loss += loss.item()
            loc_steps += 1
            tot_loss += loss.item()
            tot_steps += 1
            if step % (print_freq * args.batches_per_step) == 0:
                print(
                    "{:2d} - {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                        e,
                        step // args.batches_per_step,
                        len(dataset) // (args.batch_size * args.batches_per_step),
                        loc_loss / loc_steps,
                        time() - st_time,
                    )
                )
                loc_loss = 0
                loc_steps = 0
    return (tot_loss / tot_steps)


def evaluate(model, dataset, tokenizer, args, print_freq=10):
    with torch.no_grad():
        # make data sampler
        eval_sampler = SequentialSampler(dataset)
        model_collate_fn = functools.partial(
            make_batch,
            tokenizer=tokenizer,
            args=args,
            device='cuda:0'
        )
        eval_dataloader = DataLoader(
            dataset,
            sampler=eval_sampler,
            batch_size=args.batch_size,
            collate_fn=model_collate_fn
        )
        epoch_iterator = tqdm(eval_dataloader, desc="Iteration", disable=True)
        pad_id = tokenizer.pad_token_id
        for e in range(1):
            st_time = time()
            loc_steps = 0
            loc_loss = 0.0
            tot_steps = 0
            tot_loss = 0.0
            st_time = time()
            for step, batch in enumerate(epoch_iterator):
                x, x_mask, y, y_mask = batch
                y_ids = y[:,:-1].contiguous()
                y_labels = y[:,1:].clone()
                y_labels[y[:,1:] == pad_id] = -100
                m_out = model(
                    input_ids=x,
                    attention_mask=x_mask,
                    decoder_input_ids=y_ids,
                    lm_labels=y_labels,
                )
                loss = m_out[0]
                # book-keeping
                loc_loss += loss.item()
                loc_steps += 1
                tot_loss += loss.item()
                tot_steps += 1
                if step % print_freq == 0:
                    print(
                        "{:2d} - {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                            e,
                            step // args.batches_per_step,
                            len(dataset) // (args.batch_size * args.batches_per_step),
                            loc_loss / loc_steps,
                            time() - st_time,
                        )
                    )
                    loc_loss = 0
                    loc_steps = 0
    return (tot_loss / tot_steps)


def make_tokenizer(args, data):
    if 'bart-' in args.model_type:
        tokenizer = BartTokenizer.from_pretrained(args.model_type)
        tokenizer.sep_token = tokenizer.bos_token
    elif 't5-' in args.model_type:
        tokenizer = T5Tokenizer.from_pretrained(args.model_type)
        tokenizer.bos_token = tokenizer.unk_token
        tokenizer.sep_token = tokenizer.unk_token
    else:
        sep_tokens = ['<Q>', '</Q>', '<QC>', '</QC>', '<A>', '</A>', '<p>']
        tok_dir = "charbpe-lower-eli5-30k"
        tok_voc = os.path.join(tok_dir, "eli5-30k-bpe-lower-vocab.json")
        tok_mrg = os.path.join(tok_dir, "eli5-30k-bpe-lower-merges.txt")
        if os.path.isdir(tok_dir) and os.path.isfile(tok_voc) and os.path.isfile(tok_mrg):
            tokenizer = CharBPETokenizer(tok_voc, tok_mrg)
        else:
            print("start training BPE tokenizer")
            if not os.path.isdir(tok_dir):
                os.mkdir(tok_dir)
            print("-- start writing text file for tokenizer")
            fl = open('train_qda_lines_lower.txt', 'w')
            for dct in train:
                q = dct['question'].replace(' --T--', '')
                _ = fl.write(q.strip().lower() + '\n')
                _ = fl.write(dct['answer'].lower().strip() + '\n')
                d = dct['document'].replace('<P> ', '').strip()
                _ = fl.write(d.lower() + '\n')
            fl.close()
            print("-- done writing text file for tokenizer")
            tokenizer = CharBPETokenizer()
            tokenizer.train(["train_qda_lines_lower.txt"], vocab_size=30000)
            tokenizer.save("charbpe-lower-eli5-30k", "eli5-30k-bpe-lower")
            print("done training BPE tokenizer")
        tokenizer.add_tokens(sep_tokens)
        tokenizer.add_special_tokens(['[PAD]', '[BOS]', '[EOS]'])
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[PAD]'))
        tokenizer.enable_truncation(args.max_length)
        tokenizer.pad_token_id = tokenizer.token_to_id('[PAD]')
        tokenizer.bos_token_id = tokenizer.token_to_id('[BOS]')
        tokenizer.eos_token_id = tokenizer.token_to_id('[EOS]')
        tokenizer.sep_token_id = tokenizer.token_to_id('<A>')
    return tokenizer


def make_model(tokenizer, args):
    if 'bart-' in args.model_type:
        model = BartForConditionalGeneration.from_pretrained(
            args.model_type,
            output_past=False
        )
    elif 't5-' in args.model_type:
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_type,
            output_past=False
        )
    else:
        raise NotImplementedError
    return model


def load_saved(args_file, model_file):
    args = pickle.load(open(args_file, 'rb'))
    tokenizer = make_tokenizer(args, data=None) # fails if tokenizer needs to be learned
    model = make_model(tokenizer, args)
    model.load_state_dict(torch.load(model_file))
    return (model, tokenizer, args)


def main():
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument(
        "-mt",
        "--model_type",
        default="encoder-decoder",
        type=str,
        help="pretrained model in [t5-small, t5-base, bart-large]"
    )
    parser.add_argument(
        "-s2st",
        "--seq2seq_type",
        default="qd-a",
        type=str,
        help="sequence-to-sequence task in [qd-a, d-qa]"
    )
    parser.add_argument(
        "--init_std",
        default=2e-2,
        type=float,
        help="standard deviation initialization"
    )
    # optimization arguments
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=8,
        type=int,
        help="Number of examples per batch (on GPU)",
    )
    parser.add_argument(
        "-bps",
        "--batches_per_step",
        default=1,
        type=int,
        help="Number of batches backward steps before applying gradients",
    )
    parser.add_argument(
        "--clip_norm",
        default=0.0,
        type=float,
        help="Optional norm clipping (if > 0.)"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=2e-4,
        type=float,
        help="Adam learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.,
        type=float,
        help="Adam weight decay"
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Adam epsilon value"
    )
    parser.add_argument(
        "--warmup_steps",
        default=1000,
        type=int,
        help="Number of steps for linear warmup",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=15,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "-es",
        "--epoch_size",
        default=-1,
        type=int,
        help="Max number of examples per epoch",
    )
    # data arguments
    parser.add_argument(
        "--max_length",
        default=1024,
        type=int,
        help="Maximum sequence length for both input and output",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        default="eli5_model",
        type=str,
        help="name of saved model",
    )
    # misc arguments
    parser.add_argument(
        "--print_freq",
        default=100,
        type=int,
        help="Printing frequencey",
    )
    args = parser.parse_args()
    pickle.dump(args, open("{}_args.pk".format(args.model_name), "wb"))
    print('----- loading data')
    train_data = json.load(open('eli5wiki_train_expanded.json', encoding='utf-8'))
    # train_data = json.load(open('explainlikeimfive_train.json', encoding='utf-8'))
    # valid_data = json.load(open('explainlikeimfive_valid.json', encoding='utf-8'))
    # test_data = json.load(open('explainlikeimfive_test.json', encoding='utf-8'))
    print('----- loaded data, loading tokenizer')
    tokenizer = make_tokenizer(args, train_data)
    print('----- making model')
    model = make_model(tokenizer, args)
    _ = model.to('cuda:0')
    print('----- making optimizer')
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(args.epochs + 1) * (len(train_data) // (args.batch_size * args.batches_per_step))
    )
    print('----- training')
    if args.epoch_size <= 0 or args.epoch_size >= len(train_data):
        eli5_dset = ELI5Dataset(train_data, tokenizer, args)
    for e in range(args.epochs):
        print('---------- starting epoch', e+1)
        if args.epoch_size > 0 and args.epoch_size < len(train_data):
            nep = math.ceil(len(train_data) / args.epoch_size)
            ep = e % nep
            eli5_dset = ELI5Dataset(
                train_data[ep * args.epoch_size:(ep + 1) * args.epoch_size],
                tokenizer,
                args
            )
        train(model, eli5_dset, tokenizer, optimizer, scheduler, args, print_freq=args.print_freq)
        torch.save(model.state_dict(), "{}_{}.pth".format(args.model_name, e+1))


if __name__ == "__main__":
    main()

