import argparse
from pathlib import Path

from tqdm import tqdm

from transformers import T5_PREFIX_PATTERNS, T5Tokenizer, TFT5ForConditionalGeneration


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(lns, out_file, batch_size):
    fout = Path(out_file).open("w")

    model = TFT5ForConditionalGeneration.from_pretrained("t5-large")
    tokenizer = T5Tokenizer.from_pretrained("t5-large")

    max_length = 140
    min_length = 55

    for batch in tqdm(list(chunks(lns, batch_size))):
        batch = [T5_PREFIX_PATTERNS["summarization"] + text for text in batch]

        dct = tokenizer.batch_encode_plus(batch, max_length=512, return_tensors="tf", pad_to_max_length=True)
        summaries = model.generate(
            input_ids=dct["input_ids"],
            attention_mask=dct["attention_mask"],
            num_beams=4,
            length_penalty=2.0,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
            bos_token_id=tokenizer.pad_token_id,
        )
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]

        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


def _run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_path", type=str, help="like cnn_dm/test.source",
    )
    parser.add_argument(
        "output_path", type=str, help="where to save summaries",
    )
    parser.add_argument(
        "--bs", type=int, default=8, required=False, help="batch size: how many to summarize at a time",
    )
    args = parser.parse_args()
    lns = [x.rstrip() for x in open(args.source_path).readlines()]
    generate_summaries(lns, args.output_path, args.bs)


if __name__ == "__main__":
    _run_generate()
