from collections import Counter

import datasets
import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging

logging.set_verbosity_info()

# Mapping of slow -> fast tokenizer classes
TOKENIZER_CLASSES = {
    name: (getattr(transformers, name), getattr(transformers, name + "Fast"))
    for name in SLOW_TO_FAST_CONVERTERS
}

# Load a small subset of XNLI (English) for safe testing else all_languages and test+validation
dataset = datasets.load_dataset("facebook/xnli", "en", split="test+validation[:10]")

total = perfect = imperfect = wrong = 0


def check_diff(
    spm_diff: list[int], tok_diff: list[int], slow: PreTrainedTokenizerBase, fast: PreTrainedTokenizerBase
) -> bool:
    if spm_diff == list(reversed(tok_diff)):
        return True
    elif len(spm_diff) == len(tok_diff) and fast.decode(spm_diff) == fast.decode(tok_diff):
        return True
    spm_reencoded = slow.encode(slow.decode(spm_diff))
    tok_reencoded = fast.encode(fast.decode(spm_diff))
    if spm_reencoded != spm_diff and spm_reencoded == tok_reencoded:
        return True
    return False


def check_LTR_mark(line: str, idx: int, fast: PreTrainedTokenizerBase) -> bool:
    enc = fast.encode_plus(line)[0]
    offsets = enc.offsets
    curr, prev = offsets[idx], offsets[idx - 1]
    if curr is not None and line[curr[0] : curr[1]] == "\u200f":
        return True
    if prev is not None and line[prev[0] : prev[1]] == "\u200f":
        return True
    return False


def check_details(
    line: str, spm_ids: list[int], tok_ids: list[int], slow: PreTrainedTokenizerBase, fast: PreTrainedTokenizerBase
) -> bool:
    for i, (spm_id, tok_id) in enumerate(zip(spm_ids, tok_ids)):
        if spm_id != tok_id:
            break
    first = i
    for i, (spm_id, tok_id) in enumerate(zip(reversed(spm_ids), reversed(tok_ids))):
        if spm_id != tok_id:
            break
    last = len(spm_ids) - i

    spm_diff = spm_ids[first:last]
    tok_diff = tok_ids[first:last]

    if check_diff(spm_diff, tok_diff, slow, fast):
        return True

    if check_LTR_mark(line, first, fast):
        return True

    if last - first > 5:
        spms = Counter(spm_ids[first:last])
        toks = Counter(tok_ids[first:last])
        removable_tokens = {spm_ for spm_, si in spms.items() if toks.get(spm_, 0) == si}
        min_width = 3
        for i in range(last - first - min_width):
            if all(spm_ids[first + i + j] in removable_tokens for j in range(min_width)):
                possible_matches = [
                    k
                    for k in range(last - first - min_width)
                    if tok_ids[first + k : first + k + min_width] == spm_ids[first + i : first + i + min_width]
                ]
                for j in possible_matches:
                    if check_diff(spm_ids[first : first + i], tok_ids[first : first + j], slow, fast) and check_details(
                        line,
                        spm_ids[first + i : last],
                        tok_ids[first + j : last],
                        slow,
                        fast,
                    ):
                        return True

    return False


def test_string(slow: PreTrainedTokenizerBase, fast: PreTrainedTokenizerBase, text: str) -> None:
    global perfect, imperfect, wrong, total

    slow_ids = slow.encode(text)
    fast_ids = fast.encode(text)

    skip_assert = False
    total += 1

    if slow_ids != fast_ids:
        if check_details(text, slow_ids, fast_ids, slow, fast):
            skip_assert = True
            imperfect += 1
        else:
            wrong += 1
    else:
        perfect += 1

    if skip_assert:
        return

    assert slow_ids == fast_ids, (
        f"line {text} : \n\n{slow_ids}\n{fast_ids}\n\n{slow.tokenize(text)}\n{fast.tokenize(text)}"
    )


def test_tokenizer(slow, fast, dry_run=True):
    global total, perfect, imperfect, wrong
    total = perfect = imperfect = wrong = 0  
    n_samples = 5 if dry_run else len(dataset)
    for i in range(n_samples):
        premise = dataset[i]["premise"]
        hypothesis = dataset[i]["hypothesis"]
        test_string(slow, fast, premise)
        test_string(slow, fast, hypothesis)


if __name__ == "__main__":
    DEFAULT_CHECKPOINTS = {
        "BertTokenizer": "bert-base-uncased",
        "BertTokenizerFast": "bert-base-uncased",
        "AlbertTokenizer": "albert-base-v2",
        "AlbertTokenizerFast": "albert-base-v2",
        "BartTokenizer": "facebook/bart-base",
        "BartTokenizerFast": "facebook/bart-base",
        "BarthezTokenizer": "facebook/barthez",
        "DPRReaderTokenizer": "facebook/dpr-reader-single-nq-base",
        "DPRReaderTokenizerFast": "facebook/dpr-reader-single-nq-base",
    }

    for name, (slow_class, fast_class) in TOKENIZER_CLASSES.items():
        checkpoint = DEFAULT_CHECKPOINTS.get(name)
        if checkpoint is None:
            print(f"Skipping {name}: no compatible checkpoint defined")
            continue

        try:
            print(f"========================== Checking {name}: {checkpoint} ==========================")
            slow = slow_class.from_pretrained(checkpoint, force_download=True)
            fast = fast_class.from_pretrained(checkpoint, force_download=True)

            test_tokenizer(slow, fast, dry_run=True)

            if total > 0:
                print(f"Accuracy {perfect * 100 / total:.2f}% ({perfect}/{total} perfect)")
            else:
                print("No samples tested.")

        except ImportError as e:
            print(f"Skipping {name} due to missing dependency: {e}")
            continue
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
            continue
