import argparse
from pathlib import Path

from sacrebleu import corpus_bleu
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_translations(lns, out_file, batch_size):
    fout = Path(out_file).open("w")

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # update config with summarization specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("translation_en_to_de", {}))

    for batch in tqdm(list(chunks(lns, batch_size))):
        batch = [model.config.prefix + text for text in batch]

        dct = tokenizer.batch_encode_plus(batch, max_length=512, return_tensors="pt", pad_to_max_length=True)

        translations = model.generate(
            input_ids=dct["input_ids"],
            attention_mask=dct["attention_mask"],
        )
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in translations]

        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


def calculate_bleu_score(output_lns, refs_lns, score_path):
    bleu = corpus_bleu(output_lns, [refs_lns])
    result = "BLEU score: {}".format(bleu.score)
    score_file = Path(score_path).open("w")
    score_file.write(result)


def _run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_path", type=str, help="like wmt/newstest2013.en",
    )
    parser.add_argument(
        "output_path", type=str, help="where to save translation",
    )
    parser.add_argument(
        "reference_path", type=str, help="like wmt/newstest2013.de",
    )
    parser.add_argument(
        "score_path", type=str, help="where to save the bleu score",
    )
    parser.add_argument(
        "--bs", type=int, default=16, required=False, help="batch size: how many to summarize at a time",
    )
    args = parser.parse_args()
    dash_pattern = (" ##AT##-##AT## ", "-")

    input_lns = [x.strip().replace(dash_pattern[0], dash_pattern[1]) for x in open(args.source_path).readlines()]

    generate_translations(input_lns, args.output_path, args.bs)

    output_lns = [x.strip() for x in open(args.output_path).readlines()]
    refs_lns = [x.strip().replace(dash_pattern[0], dash_pattern[1]) for x in open(args.reference_path).readlines()]

    calculate_bleu_score(output_lns, refs_lns, args.score_path)


if __name__ == "__main__":
    _run_generate()
