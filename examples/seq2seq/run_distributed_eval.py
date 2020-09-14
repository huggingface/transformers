import argparse
from logging import getLogger
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


logger = getLogger(__name__)

try:
    from .utils import Seq2SeqDataset, parse_numeric_cl_kwargs, save_json, use_task_specific_params
except ImportError:
    from utils import Seq2SeqDataset, parse_numeric_cl_kwargs, save_json, use_task_specific_params


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval_data_dir(
    data_dir,
    save_dir: str,
    model_name: str,
    bs: int = 8,
    max_source_length: int = 1024,
    type_path="val",
    n_obs=None,
    fp16=False,
    save_source=False,
    num_beams: int = 4,
    task="summarization",
    local_rank=None,
    **generate_kwargs,
) -> Dict:
    """Run evaluation on part of the data for one gpu and save to {save_dir}/rank_{rank}_output.json"""
    model_name = str(model_name)
    assert local_rank is not None
    torch.distributed.init_process_group(backend="nccl", rank=local_rank)

    save_dir = Path(save_dir)
    save_path = save_dir.joinpath(f"rank_{local_rank}_output.json")
    torch.cuda.set_device(local_rank)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    use_task_specific_params(model, task)  # update config with task specific params
    if max_source_length is None:
        max_source_length = tokenizer.model_max_length
    ds = Seq2SeqDataset(
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length=1024,
        type_path=type_path,
        n_obs=n_obs,
        prefix=model.config.prefix,
    )
    sampler = ds.make_sortish_sampler(bs, distributed=True)
    data_loader = DataLoader(ds, sampler=sampler, batch_size=bs, collate_fn=ds.collate_fn)
    dec_kwargs = dict(skip_special_tokens=True, clean_up_tokenization_spaces=False)  # tokenizer.decode
    results = []
    for batch in tqdm(data_loader):
        summaries = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            num_beams=num_beams,
            **generate_kwargs,
        )
        preds = tokenizer.batch_decode(summaries, **dec_kwargs)
        labels = tokenizer.batch_decode(batch["labels"], **dec_kwargs)
        if save_source:
            docs = tokenizer.batch_decode(batch["input_ids"], **dec_kwargs)
        for i in range(len(labels)):
            label, pred = labels[i], preds[i]
            if save_source:
                results.append(dict(pred=pred, label=label, source=docs[i]))
            else:
                results.append(dict(pred=pred, label=label))
    save_json(results, save_path)
    return results


def run_generate():
    parser = argparse.ArgumentParser(
        epilog="Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate"
    )
    parser.add_argument("--input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument(
        "--model_name",
        type=str,
        help="like facebook/bart-large-cnn,t5-base, etc.",
        default="sshleifer/distilbart-xsum-12-3",
    )
    parser.add_argument("--save_dir", type=str, help="where to save", default="tmp_gen")
    parser.add_argument("--max_source_length", type=int, default=None)
    parser.add_argument(
        "--type_path", type=str, default="test", help="which subset to evaluate typically train/val/test"
    )
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
    parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--local_rank", type=int, default=-1, required=False, help="should be passed by distributed.launch"
    )

    parser.add_argument(
        "--n_obs", type=int, default=None, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_source", action="store_true")

    args, rest = parser.parse_known_args()
    generate_kwargs = parse_numeric_cl_kwargs(rest)
    if generate_kwargs:
        print(f"parsed the following generate kwargs: {generate_kwargs}")
    Path(args.save_dir).mkdir(exist_ok=True)
    eval_data_dir(
        args.input_path,
        args.save_dir,
        args.model_name,
        type_path=args.type_path,
        batch_size=args.bs,
        fp16=args.fp16,
        task=args.task,
        local_rank=args.local_rank,
        n_obs=args.n_obs,
        save_source=args.save_source,
        max_source_length=args.max_source_length,
        **generate_kwargs,
    )


if __name__ == "__main__":
    # Usage for MT:
    run_generate()
