import fire

from utils import calculate_rouge


def calculate_rouge_path(pred_path, tgt_path, cleaned_up_tokenization_spaces=False, use_stemmer=True):
    output_lns = [x.rstrip() for x in open(pred_path).readlines()]
    reference_lns = [x.rstrip() for x in open(tgt_path).readlines()][: len(output_lns)]
    return calculate_rouge(
        output_lns,
        reference_lns,
        cleaned_up_tokenization_spaces=cleaned_up_tokenization_spaces,
        use_stemmer=use_stemmer,
    )


if __name__ == "__main__":
    fire.Fire(calculate_rouge_path)
