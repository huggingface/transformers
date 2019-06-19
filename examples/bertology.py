#!/usr/bin/env python3
import os
import argparse
import logging
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_pretrained_bert import BertForSequenceClassification, BertTokenizer

from run_classifier_dataset_utils import processors, output_modes, convert_examples_to_features, compute_metrics


logger = logging.getLogger(__name__)


def entropy(p):
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)

def print_1d_tensor(tensor, prefix=""):
    if tensor.dtype != torch.long:
        logger.info(prefix + "\t".join(f"{x:.5f}" for x in tensor.cpu().data))
    else:
        logger.info(prefix + "\t".join(f"{x:d}" for x in tensor.cpu().data))

def print_2d_tensor(tensor):
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        print_1d_tensor(tensor[row], prefix=f"layer {row + 1}:\t")

def compute_heads_importance(args, model, eval_dataloader):
    """ Example on how to use model outputs to compute:
        - head attention entropy (activated by setting output_attentions=True when we created the model
        - head importance scores according to http://arxiv.org/abs/1905.10650
            (activated by setting keep_multihead_output=True when we created the model)
    """
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # Do a forward pass
        all_attentions, logits = model(input_ids, segment_ids, input_mask)

        # Update head attention entropy
        for layer, attn in enumerate(all_attentions):
            masked_entropy = entropy(attn.detach()) * input_mask.float().unsqueeze(1)
            attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

        # Update head importance scores with regards to our loss
        # First backpropagate to populate the gradients
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
        loss.backward()
        # Second compute importance scores according to http://arxiv.org/abs/1905.10650
        multihead_outputs = model.bert.get_multihead_outputs()
        for layer, mh_layer_output in enumerate(multihead_outputs):
            dot = torch.einsum("bhli,bhli->bhl", [mh_layer_output.grad, mh_layer_output])
            head_importance[layer] += dot.abs().sum(-1).sum(0).detach()

        tot_tokens += input_mask.float().detach().sum().data

    # Normalize
    attn_entropy /= tot_tokens
    head_importance /= tot_tokens
    if args.normalize_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    return attn_entropy, head_importance

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='bert-base-cased-finetuned-mrpc', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--task_name", type=str, default='mrpc', help="The name of the task to train.")
    parser.add_argument("--data_dir", type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances.")
    parser.add_argument("--overwrite_output_dir", action='store_true', help="Whether to overwrite data in output directory")

    parser.add_argument("--normalize_importance", action='store_true', help="Whether to normalize importance score between 0 and 1")

    parser.add_argument("--try_pruning", action='store_true', help="Whether to try to prune head until a threshold of accuracy.")
    parser.add_argument("--pruning_threshold", default=0.9, type=float, help="Pruning threshold of accuracy.")

    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    args = parser.parse_args()

    # Setup devices and distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(args.device, n_gpu, bool(args.local_rank != -1)))

    # Set seeds
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.seed)

    # Prepare GLUE task
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Prepare output directory
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Load model & tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only one distributed process download model & vocab
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # Load a model with all BERTology options on:
    #   output_attentions => will output attention weights
    #   keep_multihead_output => will store gradient of attention head outputs for head importance computation
    #       see: http://arxiv.org/abs/1905.10650
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                          num_labels=num_labels,
                                                          output_attentions=True,
                                                          keep_multihead_output=True)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only one distributed process download model & vocab
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model.eval()

    # Prepare dataset for the GLUE task
    eval_examples = processor.get_dev_examples(args.data_dir)
    cached_eval_features_file = os.path.join(args.data_dir, 'dev_{0}_{1}_{2}'.format(
        list(filter(None, args.model_name_or_path.split('/'))).pop(), str(args.max_seq_length), str(task_name)))
    try:
        eval_features = torch.load(cached_eval_features_file)
    except:
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving eval features to cache file %s", cached_eval_features_file)
            torch.save(eval_features, cached_eval_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long if output_mode == "classification" else torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if args.data_subset > 0:
        eval_data = Subset(eval_data, list(range(args.data_subset)))

    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    # Print/save training arguments
    print(args)
    torch.save(args, os.path.join(args.output_dir, 'run_args.bin'))

    # To showcase some BERTology methods, we will compute:
    #   - the average entropy of each head over the dev set
    #   - the importance score of each head over the dev set as explained in http://arxiv.org/abs/1905.10650
    n_layers, n_heads = model.bert.config.num_hidden_layers, model.bert.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)
    tot_tokens = 0.0

    # Compute head entropy and importance score
    attn_entropy, head_importance = compute_heads_importance(args, model, eval_dataloader)

    # Print/save matrices
    np.save(os.path.join(args.output_dir, 'attn_entropy.npy'), attn_entropy)
    np.save(os.path.join(args.output_dir, 'head_importance.npy'), head_importance)

    logger.info("Attention entropies")
    print_2d_tensor(attn_entropy)
    logger.info("Head importance scores")
    print_2d_tensor(head_importance)
    logger.info("Head ranked by importance scores")
    head_ranks = torch.zeros(n_layers * n_heads, dtype=torch.long, device=args.device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(head_importance.numel())
    print_2d_tensor(head_ranks.view_as(head_importance))

    # Do pruning if we want to
    if args.try_pruning and args.pruning_threshold > 0.0 and args.pruning_threshold < 1.0:
        
        

if __name__ == '__main__':
    run_model()
