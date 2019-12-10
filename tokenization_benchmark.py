import datetime
from typing import Dict

from transformers import BertTokenizer
import argparse
import itertools
import re

import sys
import unicodedata
from collections import defaultdict

UNICODE_CATEGORIES: Dict[str, list] = defaultdict(list)
for c in map(chr, range(sys.maxunicode + 1)):
    UNICODE_CATEGORIES[unicodedata.category(c)].append(c)

CONTROL_CAT_KEYS = [cat_key for cat_key in UNICODE_CATEGORIES.keys() if cat_key.startswith("C")]

def _og_is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _og_is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _og_clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _og_is_control(char):
            continue
        if _og_is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def build_features(tokenizer, file_path, block_size=512):
    examples = []
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    print(len(text))

    start = datetime.datetime.now()
    tokens = tokenizer.tokenize(text)
    end = datetime.datetime.now()
    print(f"Tokenization took: {end - start}")

    start = datetime.datetime.now()
    tokenized_text = tokenizer.convert_tokens_to_ids(tokens)
    end = datetime.datetime.now()
    print(f"Conversion to ids took: {end - start}")

    start = datetime.datetime.now()
    for i in range(
        0, len(tokenized_text) - block_size + 1, block_size
    ):  # Truncate in block of block_size
        examples.append(
            tokenizer.build_inputs_with_special_tokens(
                tokenized_text[i : i + block_size]
            )
        )
    end = datetime.datetime.now()
    print(f"Examples building took: {end - start}")
    print("Done.")


our_whitespace_chars = set([" ", "\t", "\n", "\r"] + UNICODE_CATEGORIES['Zs'])


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    return char in our_whitespace_chars


our_control_chars = controls = set(itertools.chain.from_iterable([UNICODE_CATEGORIES[cat_key] for cat_key in CONTROL_CAT_KEYS])).difference({'\t', '\n', '\r'})
our_control_chars.add('\uFFFD')


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    return char in our_control_chars


def _regex_clean_text(text):
    new_text = re.sub(r'\s+', ' ', text).strip()
    return "".join(ch for ch in new_text if not _is_control(ch))

def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""

    output = []
    for char in text:
        if _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def write_text(file, text: str):
    with open(file, "wb") as f:
        f.write(text.encode("utf-8"))



def clean_text_benchmark(tokenizer: BertTokenizer, file_path):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    start = datetime.datetime.now()
    text1 = _regex_clean_text(text)
    text2 = _og_clean_text(text)
    write_text('text1.txt', text1)
    write_text('text2.txt', text2)
    print(text1 == text2)
    end = datetime.datetime.now()
    print(f"Text cleaning took: {end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file", type=str, help="Location to read txt.",
    )

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    build_features(tokenizer, args.file)
    # clean_text_benchmark(tokenizer, args.file)
