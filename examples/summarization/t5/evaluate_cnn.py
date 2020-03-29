import argparse
from pathlib import Path

import torch
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(lns, output_file_path, model_size, batch_size, device):
    output_file = Path(output_file_path).open("w")

    model = T5ForConditionalGeneration.from_pretrained(model_size)
    model.to(device)

    tokenizer = T5Tokenizer.from_pretrained(model_size)

    # update config with summarization specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("summarization", {}))

    for batch in tqdm(list(chunks(lns, batch_size))):
        batch = [model.config.prefix + text for text in batch]

        dct = tokenizer.batch_encode_plus(batch, max_length=512, return_tensors="pt", pad_to_max_length=True)
        input_ids = dct["input_ids"].to(device)
        attention_mask = dct["attention_mask"].to(device)

        summaries = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

        for hypothesis in dec:
            output_file.write(hypothesis + "\n")
            output_file.flush()


def calculate_rouge(output_lns, reference_lns, score_path):
    score_file = Path(score_path).open("w")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    score_file.write(
        "ROUGE_1: \n{} \n\n ROUGE_2: \n{} \n\n ROUGE_L: \n{} \n\n".format(
            result["rouge1"], result["rouge2"], result["rougeL"]
        )
    )


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_size",
        type=str,
        help="T5 model size, either 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'. Defaults to 't5-base'.",
        default="t5-base",
    )
    parser.add_argument(
        "input_path", type=str, help="like cnn_dm/test_articles_input.txt",
    )
    parser.add_argument(
        "output_path", type=str, help="where to save summaries",
    )
    parser.add_argument("reference_path", type=str, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument(
        "score_path", type=str, help="where to save the rouge score",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, required=False, help="batch size: how many to summarize at a time",
    )
    parser.add_argument(
        "--no_cuda", default=False, type=bool, help="Whether to force the execution on CPU.",
    )

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    source_lns = [x.rstrip() for x in open(args.input_path).readlines()]

    generate_summaries(source_lns, args.output_path, args.model_size, args.batch_size, args.device)

    output_lns = [x.rstrip() for x in open(args.output_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]

    calculate_rouge(output_lns, reference_lns, args.score_path)


if __name__ == "__main__":
    run_generate()
