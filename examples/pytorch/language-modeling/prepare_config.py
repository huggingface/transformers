"""Prepare a config file based on a pretrained model.
This script is adaptated from the Transformers example in https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling
"""
from os import PathLike
from pathlib import Path
from typing import Union

from transformers import BartConfig


def prepare_config(
    pretrained_model_name: str = "facebook/bart-base", vocab_size: int = 50265, dout: Union[str, PathLike] = "."
):
    config = BartConfig.from_pretrained(pretrained_model_name, vocab_size=vocab_size)

    # Save to disk
    pdout = Path(dout).resolve()
    pdout.mkdir(exist_ok=True, parents=True)
    config.save_pretrained(str(pdout))


def main():
    import argparse

    cparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cparser.add_argument(
        "--pretrained_model_name",
        default="facebook/bart-base",
        help="Name of the config to use for tokenizer training",
    )
    cparser.add_argument("--vocab_size", type=int, default=50265, help="Vocabulary size")
    cparser.add_argument("--dout", default=".", help="Path to directory to save tokenizer.json file")

    prepare_config(**vars(cparser.parse_args()))


if __name__ == "__main__":
    main()
