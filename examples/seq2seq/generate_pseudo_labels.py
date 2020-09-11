import argparse
import json
import time
import warnings
from logging import getLogger
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader


logger = getLogger(__name__)

try:
    from .utils import calculate_bleu, calculate_rouge, parse_numeric_cl_kwargs, use_task_specific_params, Seq2SeqDataset, save_json
except ImportError:
    from utils import calculate_bleu, calculate_rouge, parse_numeric_cl_kwargs, use_task_specific_params, Seq2SeqDataset, save_json

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries_or_translations(
    data_dir,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    max_source_length: int=1024,
    device: str = DEFAULT_DEVICE,
    n_obs=None,
    fp16=False,
    num_return_sequences:int=10,
    num_beams: int=10,
    task="summarization",
    **generate_kwargs,
) -> Dict:
    """Save model.generate results to <out_file>, and return how long it took."""
    #fout = Path(out_file).open("w", encoding="utf-8")
    Path(out_file).parent.mkdir(exist_ok=True)
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Inferred tokenizer type: {tokenizer.__class__}")  # if this is wrong, check config.model_type.
    use_task_specific_params(model, task)
    ds = Seq2SeqDataset(
        tokenizer,
        data_dir,
        max_source_length, max_target_length=1024,
        type_path='train',
        n_obs=n_obs,
        prefix=model.config.prefix,
    )
    sampler = ds.make_sortish_sampler(batch_size)

    data_loader = DataLoader(
        ds,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=ds.collate_fn

    )

    start_time = time.time()
    # update config with task specific params
    i = 0
    results = []
    for batch in tqdm(data_loader):
        i+=1

        summaries = model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            **generate_kwargs,
        )
        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        chunked_preds = chunks(dec, num_return_sequences)

        for i,label in enumerate(labels):
            best_pred, best_score = '', -1
            for j in range(num_return_sequences):
                pred = chunked_preds[i][j]
                score = calculate_rouge([pred], [label])['rouge2']
                if score > best_score:
                    best_score = score
                    best_pred = pred
            results.append(dict(label=label, best_pred=best_pred, best_score=best_score))
        save_json(results, out_file)

    runtime = int(time.time() - start_time)  # seconds
    return dict(n_obs=n_obs, runtime=runtime, seconds_per_sample=round(runtime / n_obs, 4))
import fire

if __name__ == '__main__':
    fire.Fire(generate_summaries_or_translations)

# def run_generate():
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
#     # parser.add_argument("input_path", type=str, help="like cnn_dm/test.source")
#     # parser.add_argument("save_path", type=str, help="where to save summaries")
#     # parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test.target")
#     # parser.add_argument("--score_path", type=str, required=False, default="metrics.json", help="where to save metrics")
#     # parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
#     # parser.add_argument("--task", type=str, default="summarization", help="used for task_specific_params + metrics")
#     # parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
#     # parser.add_argument(
#     #     "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
#     # )
#     # parser.add_argument("--fp16", action="store_true")
#     # # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
#     # args, rest = parser.parse_known_args()
#     # parsed = parse_numeric_cl_kwargs(rest)
#     # if parsed:
#     #     print(f"parsed the following generate kwargs: {parsed}")
#     # Path(args.save_path).parent.mkdir(exist_ok=True)
#     # if args.reference_path is None and Path(args.score_path).exists():
#     #     warnings.warn(f"score_path {args.score_path} will be overwritten unless you type ctrl-c.")
#     generate_summaries_or_translations(
#         args.input_path,
#         args.save_path,
#         args.model_name,
#         batch_size=args.bs,
#         device=args.device,
#         fp16=args.fp16,
#         task=args.task,
#         **parsed,
#     )
#
#
# if __name__ == "__main__":
#     # Usage for MT:
#     run_generate()
