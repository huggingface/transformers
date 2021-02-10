#!/usr/bin/env python3
""" This script is adapted from the Bertology pruning code (https://github.com/huggingface/transformers/blob/783d7d2629e97c5f0c5f9ef01b8c66410275c204/examples/research_projects/bertology/run_bertology.py)
to prune GPT-like models. The author is @altsoph.
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm

from transformers import GPT2LMHeadModel


logger = logging.getLogger(__name__)


def save_model(model, dirpath):
    # save results
    if os.path.exists(dirpath):
        if os.path.exists(os.path.join(dirpath, "config.json")) and os.path.isfile(
            os.path.join(dirpath, "config.json")
        ):
            os.remove(os.path.join(dirpath, "config.json"))
        if os.path.exists(os.path.join(dirpath, "pytorch_model.bin")) and os.path.isfile(
            os.path.join(dirpath, "pytorch_model.bin")
        ):
            os.remove(os.path.join(dirpath, "pytorch_model.bin"))
    else:
        os.makedirs(dirpath)
    model.save_pretrained(dirpath)


def entropy(p, unlogit=False):
    """ Compute the entropy of a probability distribution """
    exponent = 2
    if unlogit:
        p = torch.pow(p, exponent)
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def print_2d_tensor(tensor):
    """ Print a 2D tensor """
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(
    args, model, eval_dataloader, compute_entropy=True, compute_importance=True, head_mask=None, actually_pruned=False
):
    """This method shows how to compute:
    - head attention entropy
    - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(args.device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(args.device)

    head_mask.requires_grad_(requires_grad=True)
    # If actually pruned attention multi-head, set head mask to None to avoid shape mismatch
    if actually_pruned:
        head_mask = None

    tot_tokens = 0.0
    total_loss = 0.0
    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        inputs = tuple(t.to(args.device) for t in inputs)
        (input_ids,) = inputs

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(input_ids, labels=input_ids, head_mask=head_mask)
        #  (loss), lm_logits, presents, (all hidden_states), (attentions)
        loss, _, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        loss.backward()  # Backpropagate to populate the gradients in the head mask
        total_loss += loss.detach().cpu().numpy()
        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach(), True)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).sum(0).detach()

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()
        tot_tokens += torch.ones_like(input_ids).float().detach().sum().data

    # Normalize
    attn_entropy /= tot_tokens
    head_importance /= tot_tokens
    # Layerwise importance normalization
    if not args.dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not args.dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    # Print matrices
    if compute_entropy:
        logger.info("Attention entropies")
        print_2d_tensor(attn_entropy)
    if compute_importance:
        logger.info("Head importance scores")
        print_2d_tensor(head_importance)
    logger.info("Head ranked by importance scores")
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=args.device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel(), device=args.device
    )
    head_ranks = head_ranks.view_as(head_importance)
    print_2d_tensor(head_ranks)
    return attn_entropy, head_importance, total_loss


def mask_heads(args, model, eval_dataloader):
    """This method shows how to mask head (set some heads to zero), to test the effect on the network,
    based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    _, head_importance, loss = compute_heads_importance(args, model, eval_dataloader, compute_entropy=False)
    original_score = 1 / loss  # instead of downsteam score use the LM loss
    logger.info("Pruning: original score: %f, threshold: %f", original_score, original_score * args.masking_threshold)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * args.masking_amount))

    current_score = original_score
    while current_score >= original_score * args.masking_threshold:
        head_mask = new_head_mask.clone().detach()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            print("BREAK BY num_to_mask")
            break

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        logger.info("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        new_head_mask = new_head_mask.clone().detach()
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, loss = compute_heads_importance(
            args, model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
        )
        current_score = 1 / loss
        logger.info(
            "Masking: current score: %f, remaining heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )

    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(os.path.join(args.output_dir, "head_mask.npy"), head_mask.detach().cpu().numpy())

    return head_mask


def prune_heads(args, model, eval_dataloader, head_mask):
    """This method shows how to prune head (remove heads weights) based on
    the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, loss = compute_heads_importance(
        args, model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=head_mask
    )
    score_masking = 1 / loss
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = dict(
        (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist()) for layer in range(len(head_mask))
    )

    for k, v in heads_to_prune.items():
        if isinstance(v, int):
            heads_to_prune[k] = [
                v,
            ]

    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
    model.prune_heads(heads_to_prune)
    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, _, loss = compute_heads_importance(
        args,
        model,
        eval_dataloader,
        compute_entropy=False,
        compute_importance=False,
        head_mask=None,
        actually_pruned=True,
    )

    score_pruning = 1 / loss
    new_time = datetime.now() - before_time

    logger.info(
        "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
        original_num_params,
        pruned_num_params,
        pruned_num_params / original_num_params * 100,
    )
    logger.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    logger.info("Pruning: speed ratio (original timing / new timing): %f percents", original_time / new_time * 100)
    save_model(model, args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances."
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Whether to overwrite data in output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )

    parser.add_argument(
        "--try_masking", action="store_true", help="Whether to try to mask head until a threshold of accuracy."
    )
    parser.add_argument(
        "--masking_threshold",
        default=0.9,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument("--metric_name", default="acc", type=str, help="Metric to use for head masking.")

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, sequences shorter padded.",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup devices and distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(args.device, args.n_gpu, bool(args.local_rank != -1)))

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)

    # Distributed and parallel training
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Print/save training arguments
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(args, os.path.join(args.output_dir, "run_args.bin"))
    logger.info("Training/evaluation parameters %s", args)

    # Prepare dataset
    numpy_data = np.concatenate(
        [
            np.loadtxt(args.data_dir, dtype=np.int64),
        ]
    )
    train_tensor_dataset = (torch.from_numpy(numpy_data),)
    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    eval_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # Compute head entropy and importance score
    compute_heads_importance(args, model, eval_dataloader)

    # Try head masking (set heads to zero until the score goes under a threshole)
    # and head pruning (remove masked heads and see the effect on the network)
    if args.try_masking and args.masking_threshold > 0.0 and args.masking_threshold < 1.0:
        head_mask = mask_heads(args, model, eval_dataloader)
        prune_heads(args, model, eval_dataloader, head_mask)


if __name__ == "__main__":
    main()
