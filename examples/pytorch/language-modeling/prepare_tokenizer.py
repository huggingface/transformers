"""Train a ByteLevelBPETokenizer based on a given dataset. The dataset must be on the HF Hub.
This script is adaptated from the Transformers example in https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling
"""
from os import PathLike
from pathlib import Path
from typing import Union

from datasets import load_dataset

from transformers import AutoTokenizer


def train_tokenizer(
    dataset_name: str = "oscar",
    dataset_config_name: str = "unshuffled_deduplicated_nl",
    dataset_split: str = "train",
    dataset_textcol: str = "text",
    vocab_size: int = 50265,
    base_tokenizer_name: str = "facebook/bart-base",
    dout: Union[str, PathLike] = ".",
):
    # load dataset
    dataset = load_dataset(dataset_name, dataset_config_name, split=dataset_split)
    # Instantiate tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

    def batch_iterator(batch_size=1024):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size][dataset_textcol]

    # Customized training
    tokenizer = old_tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=vocab_size)

    # Save to disk
    pdout = Path(dout).resolve()
    pdout.mkdir(exist_ok=True, parents=True)
    tokenizer.save_model(str(pdout))


def main():
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cparser.add_argument("dataset_name", help="Name of dataset to use for tokenizer training")
    cparser.add_argument(
        "--dataset_config_name", default=None, help="Name of the config to use for tokenizer training"
    )
    cparser.add_argument(
        "--dataset_split", default=None, help="Name of the split to use for tokenizer training (typically 'train')"
    )
    cparser.add_argument(
        "--dataset_textcol", default="text", help="Name of the text column to use for tokenizer training"
    )
    cparser.add_argument("--vocab_size", type=int, default=50265, help="Vocabulary size")
    cparser.add_argument(
        "--base_tokenizer_name",
        default="facebook/bart-base",
        help=(
            "Base tokenizer to start from. We'll reuse the same special tokens and tokenization"
            "algorithm from this tokenizer"
        ),
    )
    cparser.add_argument("--dout", default=".", help="Path to directory to save tokenizer.json file")

    train_tokenizer(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
