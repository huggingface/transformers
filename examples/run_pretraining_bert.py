import argparse
import time
import torch
import os

from torch.utils.data import DataLoader
from apex import amp
from apex.optimizers import FusedAdam
from apex.parallel import DistributedDataParallel as DDP

from pytorch_pretrained_bert.datasets import LazyDataset, JSONDataset, BertDataset
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
    opt_group.add_argument('--batch-size', type=int, default=4,
                       help='Data Loader batch size')
    opt_group.add_argument('--num-workers', type=int, default=1,
                           help='Number of workers in DataLoader')
    opt_group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    opt_group.add_argument('--train-iters', type=int, default=100,
                           help='Number of training iterations')
    opt_group.add_argument('--use-fp16', type=bool, default=True,
                           help='Training with FP16?')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    local_rank = 0
    world_size = 1


    distributed = ('WORLD_SIZE' in os.environ) and (int(os.environ['WORLD_SIZE']) > 1)
    if 'WORLD_SIZE' in os.environ:
        print('Initializing torch.distributed')
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

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

    optimizer = FusedAdam(model.parameters())

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
    wiki_dataset = LazyDataset('/home/nathan/data/wiki/enwiki.txt', use_mmap=False)
    wiki_dataset = JSONDataset(wiki_dataset, key='text')

    #bookcorpus_dataset = LazyDataset('')

    dataset = BertDataset(wiki_dataset, tokenizer)

    iterator = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    t = time.time()
    n_iters = args.train_iters
    for i, batch in enumerate(iterator):
        if i == n_iters:
            break

        # move to batch to gpu
        for k, v in batch.items():
            batch[k] = batch[k].cuda()

        # forward pass
        loss = model.forward(batch['input_tokens'],
                             batch['segment_ids'],
                             batch['attention_mask'],
                             batch['lm_labels'],
                             batch['is_random_next'])
        # backward pass
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # optimizer step
        optimizer.step()
        optimizer.zero_grad()

    print(n_iters/(time.time() - t))


