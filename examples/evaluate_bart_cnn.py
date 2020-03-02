import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import BartForMaskedLM, BartTokenizer


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main(source_path, out_file, batch_size=8, device=DEFAULT_DEVICE):
    lns = [" " + x.rstrip() for x in open(source_path).readlines()]
    fout = Path(out_file).open("w")
    model = BartForMaskedLM.from_pretrained("bart-large-cnn", output_past=True,)
    tok = BartTokenizer.from_pretrained("bart-large")
    for batch in tqdm(list(chunks(lns, batch_size))):
        dct = tok.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)
        hypotheses_hf = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_beams=4,
            length_penalty=2.0,
            max_length=140,
            min_len=55,
            no_repeat_ngram_size=3,
        )
        dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in hypotheses_hf]
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-source_path", type=str, required=True, help="like cnn_dm/test.source",
    )
    parser.add_argument(
        "-outfile", type=str, required=True, help="where to save summaries",
    )
    parser.add_argument(
        "--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, required=False, help="How many to summarize at a time",
    )

    main()
