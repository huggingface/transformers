import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


try:
    from .finetune import calculate_rouge, use_task_specific_params
except ImportError:
    from finetune import calculate_rouge, use_task_specific_params

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(
    examples: list, out_file: str, model_name: str, batch_size: int = 8, device: str = DEFAULT_DEVICE, fp16=False,
) -> None:
    fout = Path(out_file).open("w", encoding="utf-8")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # update config with summarization specific params
    use_task_specific_params(model, "summarization")

    for batch in tqdm(list(chunks(examples, batch_size))):
        if "t5" in model_name:
            batch = [model.config.prefix + text for text in batch]
        dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True).to(
            device
        )
        summaries = model.generate(**dct)

        dec = tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("output_path", type=str, help="where to save summaries")
    parser.add_argument("model_name", type=str, help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument("--score_path", type=str, required=False, help="where to save the rouge score in json format")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]

    generate_summaries(
        examples, args.output_path, args.model_name, batch_size=args.bs, device=args.device, fp16=args.fp16
    )
    if args.score_path is not None:
        output_lns = [x.rstrip() for x in open(args.output_path).readlines()]
        reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]

        rouge: dict = calculate_rouge(output_lns, reference_lns)

        json.dump(rouge, open("score_path", "w+"))


if __name__ == "__main__":
    run_generate()
