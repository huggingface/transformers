import argparse
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import fire

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


logger = getLogger(__name__)

try:
    from .utils import (
        Seq2SeqDataset,
        calculate_bleu,
        calculate_rouge,
        parse_numeric_cl_kwargs,
        save_json,
        use_task_specific_params,
    )
except ImportError:
    from utils import (
        Seq2SeqDataset,
        calculate_bleu,
        calculate_rouge,
        parse_numeric_cl_kwargs,
        save_json,
        use_task_specific_params,
    )

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

def eval_data_dir(
    data_dir,
    save_path: str,
    model_name: str,
    bs: int = 8,
    max_source_length: int = 1024,
    device: str = DEFAULT_DEVICE,
    type_path='val',
    n_obs=None,
    fp16=False,
    num_beams: int = 4,
    task="summarization",
    local_rank=None,
    **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    model_name = str(model_name)
    torch.distributed.init_process_group(backend="nccl", world_size=dist.get_world_size(), rank=local_rank)
    world_size = dist.get_world_size()
    if local_rank is None:
        local_rank = torch.distributed.get_rank()
    torch.distributed.init_process_group(backend="nccl", world_size=dist.get_world_size(), rank=local_rank)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(f'setting device ={device}')
    torch.cuda.set_device(device)
    save_dir, basename = Path(save_path).parent, Path(save_path).name
    save_path = save_dir.joinpath(f'rank_{local_rank}_{basename}')
    print(f'rank: {local_rank} saving to {save_path}')
    # assume multi-gpu
    model = DistributedDataParallel(model, device_ids=[local_rank], world_size=world_size)


    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    use_task_specific_params(model, task)
    ds = Seq2SeqDataset(
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length=1024,
        type_path=type_path,
        n_obs=n_obs,
        prefix=model.config.prefix,
    )
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=local_rank) #ds.make_sortish_sampler(bs, distributed=gpus > 1)

    data_loader = DataLoader(ds, sampler=sampler, batch_size=bs, collate_fn=ds.collate_fn)

    #start_time = time.time()
    # update config with task specific params
    i = 0
    results = []
    for batch in tqdm(data_loader):
        i += 1

        summaries = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            num_beams=num_beams,
            **generate_kwargs,
        )
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for pred, label in zip(dec, labels):
            results.append(dict(pred=pred, label=label))
        save_json(results, save_path)
    return results

def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--save_path", type=str, help="where to save", default='multigpu_generations.json')
    parser.add_argument("--prefix", type=str, default='test', help="where to save summaries")

    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument("--local_rank", type=int, default=-1, required=False, help="should be passed by distributed.launch")

    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed = parse_numeric_cl_kwargs(rest)
    if parsed:
        print(f"parsed the following generate kwargs: {parsed}")
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.reference_path is None and Path(args.score_path).exists():
        warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")
    eval_data_dir(
        args.input_path,
        args.save_path,
        args.model_name,
        prefix=args.prefix,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        local_rank=args.local_rank,
        **parsed,
    )


if __name__ == "__main__":
    # Usage for MT:
    run_generate()
