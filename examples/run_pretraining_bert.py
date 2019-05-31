import argparse
import logging
import math
import time
import torch
import random
import os

from apex import amp
from apex.optimizers import FusedAdam
from apex.parallel import DistributedDataParallel as DDP
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

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


def calc_loss(scores, labels):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    flat_scores = scores.view(-1, scores.shape[-1])
    flat_labels = labels.view(-1)
    _, ind = flat_scores.max(1)
    accuracy = float(sum(flat_labels==ind))/float((sum(flat_labels != -1)))
    loss = loss_fn(flat_scores, flat_labels)
    return loss, accuracy



def train_epoch(model, iterator, optimizer):
    model.train()
    total_lm_loss = 0.0
    total_nsp_loss = 0.0
    total_lm_acc = 0.0
    total_nsp_acc = 0.0
    n_iters = 0

    for batch in tqdm(iterator):
        n_iters += 1

        # move to batch to gpu
        for k, v in batch.items():
            batch[k] = batch[k].cuda()

        # forward pass
        lm_scores, nsp_scores = model.forward(batch['input_tokens'],
                                             batch['segment_ids'],
                                             batch['attention_mask'])
        lm_loss, lm_acc = calc_loss(lm_scores.view(-1, lm_scores.shape[-1]), batch['lm_labels'].view(-1))
        nsp_loss, nsp_acc = calc_loss(nsp_scores.view(-1, 2), batch['is_random_next'].view(-1))
        loss = lm_loss + nsp_loss
        total_nsp_acc += nsp_acc
        total_lm_acc += lm_acc
        total_nsp_loss += nsp_loss.item()
        total_lm_loss += lm_loss.item()

        if args.use_fp16:
            # backward pass
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # optimizer step
        optimizer.step()
        optimizer.zero_grad()
    return total_lm_loss/n_iters, total_nsp_loss/n_iters, total_lm_acc/n_iters, total_nsp_acc/n_iters

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


def create_logger(save_path):
    logger = logging.getLogger()
    # Debug = write everything
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(save_path, 'a', 1000000, 1)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


if __name__ == '__main__':
    args = get_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    rank = 0
    local_rank = 0
    world_size = 1

    distributed = ('WORLD_SIZE' in os.environ) and (int(os.environ['WORLD_SIZE']) > 1)
    if 'WORLD_SIZE' in os.environ:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        rank = int(os.environ['RANK'])

    logger = create_logger('/home/hdvries/logs/log.%s.txt' % rank)
    logger.info(args)

    logger.info('Loading tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir='./data')
    vocab_size = len(tokenizer.vocab)

    logger.info('Building model...')
    model_config = BertConfig(vocab_size,
                              hidden_size=args.hidden_size,
                              num_hidden_layers=args.num_layers,
                              num_attention_heads=args.num_attention_heads,
                              intermediate_size=args.intermediate_size)
    model = BertForPreTraining(model_config)
    logger.info(' > number of parameters: {}'.format(
        sum([p.nelement() for p in model.parameters()])))
    model.cuda()

    # optimizer = FusedAdam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


    opt_level = 'O0' # FP32
    if args.use_fp16:
        opt_level = 'O2'

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=opt_level,
                                      loss_scale='dynamic',
                                      keep_batchnorm_fp32=False)

    if distributed:
        model = DDP(model)

    logger.info('Loading data and creating iterators...')
    dataset = LazyDataset('/home/hdvries/data/preprocessed_test.txt', use_mmap=True)
    dataset = JSONDataset(dataset)
    dataset = PreprocessedBertDataset(dataset, tokenizer)

    num_train_examples = math.ceil(len(dataset)*0.9)
    # train_set, valid_set = random_split(dataset, [num_train_examples, len(dataset) - num_train_examples])
    train_set = Subset(dataset, range(12))
    valid_set = Subset(dataset, [i+12 for i in range(12)])

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

    # with open('/home/hdvries/logs/log.%s.txt' % str(rank), 'w', encoding='utf-8') as f:
    for epoch in range(200):
        train_lm_loss, train_nsp_loss, train_lm_acc, train_nsp_acc = train_epoch(model, train_iterator, optimizer)
        valid_loss = eval_epoch(model, valid_iterator)

        log_str  = ' epoch{:2d} |'.format(epoch)
        log_str += ' train_lm_loss: {:.3E} |'.format(train_lm_loss)
        log_str += ' train_nsp_loss: {:.3E} |'.format(train_nsp_loss)
        log_str += ' train_lm_acc: {:.3E} |'.format(train_lm_acc)
        log_str += ' train_nsp_acc: {:.3E} |'.format(train_nsp_acc)
        log_str += ' valid_loss: {:.3E} |'.format(valid_loss)
        logger.info(log_str)










