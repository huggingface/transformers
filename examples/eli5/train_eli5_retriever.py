import argparse
import functools
import json
import math
import pickle
import torch

from random import choice, randint
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from time import time
from tqdm import tqdm, trange

from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup


def make_batch(qda_dct_ls, tokenizer, max_len=64, max_start_idx=32, device='cpu'):
    q_ls = [dct['question'] for dct in qda_dct_ls]
    a_ls = [dct['answer']  for dct in qda_dct_ls]
    a_ls = [' '.join(a.split()[randint(0, max(0, len(a.split()) - max_start_idx)):]) for a in a_ls]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=64, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks['input_ids']).to(device),
        torch.LongTensor(q_toks['attention_mask']).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=64, pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']).to(device),
        torch.LongTensor(a_toks['attention_mask']).to(device),
    )
    return (q_ids, q_mask, a_ids, a_mask)


class SimpleELI5Dataset(Dataset):
    def __init__(self, examples_list):
        self.examples = examples_list

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RetrievalQAEmbedder(torch.nn.Module):
    def __init__(self, sent_encoder, dim):
        super(RetrievalQAEmbedder, self).__init__()
        self.sent_encoder = sent_encoder
        self.project_q = torch.nn.Linear(dim, 128, bias=False)
        self.project_a = torch.nn.Linear(dim, 128, bias=False)

    def embed_questions(self, q_ids, q_mask):
        _, q_reps = self.sent_encoder(q_ids, attention_mask=q_mask)
        return self.project_q(q_reps)

    def embed_answerss(self, a_ids, a_mask):
        _, a_reps = self.sent_encoder(a_ids, attention_mask=a_mask)
        return self.project_a(a_reps)

    def forward(self, q_ids, q_mask, a_ids, a_mask):
        q_reps = self.embed_questions(q_ids, q_mask)
        a_reps = self.embed_answerss(a_ids, a_mask)
        compare_scores = torch.mm(q_reps, a_reps.t())
        return compare_scores


def train_epoch(model, dataset, tokenizer, optimizer, scheduler, args, e=0):
    # make iterator
    train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(
        make_batch,
        tokenizer=tokenizer,
        max_len=args.max_length,
        max_start_idx=args.max_length // 2,
        device='cuda:0'
    )
    train_dataloader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=model_collate_fn
    )
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
    # define loss
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch in enumerate(epoch_iterator):
        q_ids, q_mask, a_ids, a_mask = batch
        scores = model(q_ids, q_mask, a_ids, a_mask)
        loss_qa = ce_loss(scores, torch.arange(scores.shape[1]).to('cuda:0'))
        loss_aq = ce_loss(scores.t(), torch.arange(scores.shape[0]).to('cuda:0'))
        loss = (loss_qa + loss_aq) / 2
        # optimizer
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        # some printing within the epoch
        loc_loss += loss.item()
        loc_steps += 1
        if step % args.print_freq == 0:
            print(
                "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                    e, step,
                    len(dataset) // args.batch_size,
                    loc_loss / loc_steps,
                    time() - st_time,
                )
            )
            loc_loss = 0
            loc_steps = 0


def main():
    parser = argparse.ArgumentParser()
    # optimization arguments
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=256,
        type=int,
        help="Number of examples per batch (on GPU)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=2e-5,
        type=float,
        help="Adam learning rate"
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Adam epsilon value"
    )
    parser.add_argument(
        "--warmup_steps",
        default=100,
        type=int,
        help="Number of steps for linear warmup",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5,
        type=int,
        help="Number of training epochs",
    )
    # data arguments
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help="Maximum sequence length for both input and output",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        default="eli5_qa_embedder",
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
    # make dataset
    wiki_train_exp = json.load(open('eli5wiki_train_expanded.json'))
    train_dset = SimpleELI5Dataset(wiki_train_exp)
    # make model
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-8_H-512_A-8")
    bert_model = AutoModel.from_pretrained("google/bert_uncased_L-8_H-512_A-8").to('cuda:0')
    qa_embedder = RetrievalQAEmbedder(bert_model, 512).to('cuda:0')
    # make optimizer
    params = [p for p in qa_embedder.parameters()]
    optimizer = AdamW(params, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=10 * len(train_dset)
    )
    # train
    for e in range(args.epochs):
        train_epoch(qa_embedder, train_dset, tokenizer, optimizer, scheduler, args, e)
        torch.save(
            {
                'model': qa_embedder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            '{}_{}.pth'.format(args.model_name, e))


if __name__ == "__main__":
    main()
