import argparse
from pathlib import Path

import torch
from sacrebleu import corpus_bleu
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_translations(lns, output_file_path, model_size, batch_size, device):
    model = T5ForConditionalGeneration.from_pretrained(model_size)
    model.to(device)

    tokenizer = T5Tokenizer.from_pretrained(model_size)

    # update config with summarization specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("translation_en_to_de", {}))

    with Path(output_file_path).open("w") as output_file:
        for batch in tqdm(list(chunks(lns, batch_size))):
            batch = [model.config.prefix + text for text in batch]

            dct = tokenizer.batch_encode_plus(batch, max_length=512, return_tensors="pt", pad_to_max_length=True)

            input_ids = dct["input_ids"].to(device)
            attention_mask = dct["attention_mask"].to(device)

            translations = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            dec = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in translations
            ]

            for hypothesis in dec:
                output_file.write(hypothesis + "\n")


def calculate_bleu_score(output_lns, refs_lns, score_path):
    bleu = corpus_bleu(output_lns, [refs_lns])
    result = "BLEU score: {}".format(bleu.score)
    with Path(score_path).open("w") as score_file:
        score_file.write(result)


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_size",
        type=str,
        help="T5 model size, either 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'. Defaults to 't5-base'.",
        default="t5-base",
    )
    parser.add_argument(
        "input_path", type=str, help="like wmt/newstest2014.en",
    )
    parser.add_argument(
        "output_path", type=str, help="where to save translation",
    )
    parser.add_argument(
        "reference_path", type=str, help="like wmt/newstest2014.de",
    )
    parser.add_argument(
        "score_path", type=str, help="where to save the bleu score",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, required=False, help="batch size: how many to summarize at a time",
    )
    parser.add_argument(
        "--no_cuda", default=False, type=bool, help="Whether to force the execution on CPU.",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    dash_pattern = (" ##AT##-##AT## ", "-")

    # Read input lines into python
    with open(args.input_path, "r") as input_file:
        input_lns = [x.strip().replace(dash_pattern[0], dash_pattern[1]) for x in input_file.readlines()]

    generate_translations(input_lns, args.output_path, args.model_size, args.batch_size, args.device)

    # Read generated lines into python
    with open(args.output_path, "r") as output_file:
        output_lns = [x.strip() for x in output_file.readlines()]

    # Read reference lines into python
    with open(args.reference_path, "r") as reference_file:
        refs_lns = [x.strip().replace(dash_pattern[0], dash_pattern[1]) for x in reference_file.readlines()]

    calculate_bleu_score(output_lns, refs_lns, args.score_path)


if __name__ == "__main__":
    run_generate()
