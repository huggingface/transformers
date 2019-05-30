import argparse
import math
import time
import torch
import random
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from apex import amp
from apex.optimizers import FusedAdam
from apex.parallel import DistributedDataParallel as DDP

from pytorch_pretrained_bert.datasets import LazyDataset, JSONDataset, BertDataset, PreprocessedBertDataset
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    model_group = parser.add_argument_group('model', 'model configuration')

    model_group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='dropout probability for attention weights')
    model_group.add_argument('--num-attention-heads', type=int, default=16,
                       help='num of transformer attention heads')
    model_group.add_argument('--hidden-size', type=int, default=1024,
                       help='transformer hidden size')
    model_group.add_argument('--intermediate-size', type=int, default=4096,
                       help='transformer embedding dimension for FFN')
    model_group.add_argument('--num-layers', type=int, default=24,
                       help='number of layers')
    model_group.add_argument('--layernorm-epsilon', type=float, default=1e-12,
                       help='layer norm epsilon')
    model_group.add_argument('--hidden-dropout', type=float, default=0.0,
                       help='dropout probability for hidden state transformer')
    model_group.add_argument('--max-position-embeddings', type=int, default=512,
                       help='maximum number of position embeddings to use')

    opt_group = parser.add_argument_group('train', 'optimization configuration')
    opt_group.add_argument('--seed', type=int, default=1,
                           help='Random seed of experiment..')
    opt_group.add_argument('--batch-size', type=int, default=4,
                       help='Data Loader batch size')
    opt_group.add_argument('--num-workers', type=int, default=1,
                           help='Number of workers in DataLoader')
    opt_group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    opt_group.add_argument('--train-iters', type=int, default=100,
                           help='Number of training iterations')
    opt_group.add_argument('--use-fp16', action='store_true',
                           help='Training with FP16?')

    return parser.parse_args()


def train_epoch(model, iterator, optimizer):
    model.train()
    total_loss  = 0.0
    n_iters = 0

    for batch in tqdm(iterator):
        n_iters += 1

        # move to batch to gpu
        for k, v in batch.items():
            batch[k] = batch[k].cuda()

        # forward pass
        loss = model.forward(batch['input_tokens'],
                             batch['segment_ids'],
                             batch['attention_mask'],
                             batch['lm_labels'],
                             batch['is_random_next'])
        total_loss += loss.item()

        if args.use_fp16:
            # backward pass
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # optimizer step
        optimizer.step()
        optimizer.zero_grad()
    return total_loss/n_iters

def eval_epoch(model, iterator):
    model.eval()

    total_loss = 0.0
    n_iters = 0

    with torch.no_grad():
        for batch in tqdm(iterator):
            n_iters += 1

            # move to batch to gpu
            for k, v in batch.items():
                batch[k] = batch[k].cuda()

            # forward pass
            loss = model.forward(batch['input_tokens'],
                                 batch['segment_ids'],
                                 batch['attention_mask'],
                                 batch['lm_labels'],
                                 batch['is_random_next'])
            total_loss += loss.item()
    return total_loss/n_iters

if __name__ == '__main__':
    args = get_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    rank = 0
    local_rank = 0
    world_size = 1

    distributed = ('WORLD_SIZE' in os.environ) and (int(os.environ['WORLD_SIZE']) > 1)
    if 'WORLD_SIZE' in os.environ:
        print('Initializing torch.distributed')
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        rank = int(os.environ['RANK'])

    print('Loading tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir='./data')
    vocab_size = len(tokenizer.vocab)

    print('Building model...')
    model_config = BertConfig(vocab_size,
                              hidden_size=args.hidden_size,
                              num_hidden_layers=args.num_layers,
                              num_attention_heads=args.num_attention_heads,
                              intermediate_size=args.intermediate_size)
    model = BertForPreTraining(model_config)
    print(' > number of parameters: {}'.format(
        sum([p.nelement() for p in model.parameters()])))
    model.cuda()

    # optimizer = FusedAdam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    opt_level = 'O0' # FP32
    if args.use_fp16:
        opt_level = 'O2'

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=opt_level,
                                      loss_scale='dynamic',
                                      keep_batchnorm_fp32=False)
    if distributed:
        model = DDP(model)

    print('Loading data and creating iterators...')
    dataset = LazyDataset('/home/hdvries/data/preprocessed_test.txt', use_mmap=True)
    dataset = JSONDataset(dataset)
    dataset = PreprocessedBertDataset(dataset, tokenizer)

    num_train_examples = math.ceil(len(dataset)*0.9)
    train_set, valid_set = random_split(dataset, [num_train_examples, len(dataset) - num_train_examples])
    # train_set = Subset(dataset, range(12))
    # valid_set = Subset(dataset, [i+12 for i in range(12)])

    if distributed:
        train_sampler = DistributedSampler(train_set)
        valid_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = RandomSampler(train_set)
        valid_sampler = RandomSampler(valid_set)

    train_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_iterator = DataLoader(train_set, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)

    valid_sampler = BatchSampler(valid_sampler, args.batch_size, drop_last=True)
    valid_iterator = DataLoader(valid_set, batch_sampler=valid_sampler, num_workers=args.num_workers, pin_memory=True)

    t = time.time()
    n_iters = args.train_iters
    it = 0

    with open('/home/hdvries/logs/log.%s.txt' % str(rank), 'w', encoding='utf-8') as f:
        for epoch in range(200):
            train_loss = train_epoch(model, train_iterator, optimizer)
            valid_loss = eval_epoch(model, valid_iterator)

            log_str  = ' epoch{:2d} |'.format(epoch)
            log_str += ' train_loss: {:.3E} |'.format(train_loss)
            log_str += ' valid_loss: {:.3E} |'.format(valid_loss)
            f.write(log_str)










