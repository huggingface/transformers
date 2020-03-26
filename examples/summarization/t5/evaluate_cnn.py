import argparse
from pathlib import Path

from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(lns, out_file, batch_size):
    fout = Path(out_file).open("w")

    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    tokenizer = T5Tokenizer.from_pretrained("t5-large")

    # update config with summarization specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("summarization", {}))

    for batch in tqdm(list(chunks(lns, batch_size))):
        batch = [model.config.prefix + text for text in batch]

        dct = tokenizer.batch_encode_plus(batch, max_length=512, return_tensors="tf", pad_to_max_length=True)
        summaries = model.generate(input_ids=dct["input_ids"], attention_mask=dct["attention_mask"],)
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


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


def _run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_path", type=str, help="like cnn_dm/test_articles_input.txt",
    )
    parser.add_argument(
        "output_path", type=str, help="where to save summaries",
    )
    parser.add_argument("reference_path", type=str, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument(
        "score_path", type=str, help="where to save the rouge score",
    )
    parser.add_argument(
        "--bs", type=int, default=8, required=False, help="batch size: how many to summarize at a time",
    )
    args = parser.parse_args()
    #    source_lns = [x.rstrip() for x in open(args.source_path).readlines()]
    #
    #    generate_summaries(source_lns, args.output_path, args.bs)

    output_lns = [x.rstrip() for x in open(args.output_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]

    calculate_rouge(output_lns, reference_lns, args.score_path)


if __name__ == "__main__":
    _run_generate()
