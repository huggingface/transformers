from collections import Counter
import datasets
import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from transformers.utils import logging

import logging as py_logging
import traceback

py_logging.basicConfig(level=py_logging.INFO)

TOKENIZER_CLASSES = {
    name: (getattr(transformers, name), getattr(transformers, name + "Fast")) for name in SLOW_TO_FAST_CONVERTERS
}

dataset = datasets.load_dataset("facebook/xnli", split="test+validation")  # no-script

total = 0
perfect = 0
imperfect = 0
wrong = 0


def check_diff(spm_diff, tok_diff, slow, fast):
    try:
        if spm_diff == list(reversed(tok_diff)):
            return True
        elif len(spm_diff) == len(tok_diff) and fast.decode(spm_diff) == fast.decode(tok_diff):
            return True
        spm_reencoded = slow.encode(slow.decode(spm_diff))
        tok_reencoded = fast.encode(fast.decode(spm_diff))
        if spm_reencoded != spm_diff and spm_reencoded == tok_reencoded:
            return True
        return False
    except Exception as e:
        py_logging.error(f"Error in check_diff function: {e}")
        return False


def check_LTR_mark(line, idx, fast):
    try:
        enc = fast.encode_plus(line)[0]
        offsets = enc.offsets
        curr, prev = offsets[idx], offsets[idx - 1]
        if curr is not None and line[curr[0] : curr[1]] == "\u200f":
            return True
        if prev is not None and line[prev[0] : prev[1]] == "\u200f":
            return True
        return False
    except Exception as e:
        py_logging.error(f"Error in check_LTR_mark function: {e}")
        return False


def check_details(line, spm_ids, tok_ids, slow, fast):
    try:
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

            removable_tokens = {spm_ for (spm_, si) in spms.items() if toks.get(spm_, 0) == si}
            min_width = 3
            for i in range(last - first - min_width):
                if all(spm_ids[first + i + j] in removable_tokens for j in range(min_width)):
                    possible_matches = [
                        k
                        for k in range(last - first - min_width)
                        if tok_ids[first + k : first + k + min_width] == spm_ids[first + i : first + i + min_width]
                    ]
                    for j in possible_matches:
                        if check_diff(spm_ids[first : first + i], tok_ids[first : first + j], sp, tok) and check_details(
                            line,
                            spm_ids[first + i : last],
                            tok_ids[first + j : last],
                            slow,
                            fast,
                        ):
                            return True

        py_logging.info(f"Spm: {[fast.decode([spm_ids[i]]) for i in range(first, last)]}")
        try:
            py_logging.info(f"Tok: {[fast.decode([tok_ids[i]]) for i in range(first, last)]}")
        except Exception:
            pass

        fast.decode(spm_ids[:first])
        fast.decode(spm_ids[last:])
        wrong = fast.decode(spm_ids[first:last])
        py_logging.info(f"\n\n{wrong}\n\n")
        return False
    except Exception as e:
        py_logging.error(f"Error in check_details function: {e}")
        return False


def test_string(slow, fast, text):
    global perfect, imperfect, wrong, total
    try:
        slow_ids = slow.encode(text)
        fast_ids = fast.encode(text)

        total += 1

        if slow_ids != fast_ids:
            if check_details(text, slow_ids, fast_ids, slow, fast):
                imperfect += 1
            else:
                wrong += 1
        else:
            perfect += 1

        if total % 10000 == 0:
            py_logging.info(f"({perfect} / {imperfect} / {wrong} ----- {perfect + imperfect + wrong})")

        assert (
            slow_ids == fast_ids
        ), f"line {text} : \n\n{slow_ids}\n{fast_ids}\n\n{slow.tokenize(text)}\n{fast.tokenize(text)}"

    except Exception as e:
        py_logging.error(f"Error in test_string function: {e}")


def test_tokenizer(slow, fast):
    global batch_total
    try:
        for i in range(len(dataset)):
            for text in dataset[i]["premise"].values():
                test_string(slow, fast, text)

            for text in dataset[i]["hypothesis"]["translation"]:
                test_string(slow, fast, text)

    except Exception as e:
        py_logging.error(f"Error in test_tokenizer function: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        for name, (slow_class, fast_class) in TOKENIZER_CLASSES.items():
            checkpoint_names = list(slow_class.max_model_input_sizes.keys())
            for checkpoint in checkpoint_names:
                perfect = 0
                imperfect = 0
                wrong = 0
                total = 0

                py_logging.info(f"========================== Checking {name}: {checkpoint} ==========================")
                slow = slow_class.from_pretrained(checkpoint, force_download=True)
                fast = fast_class.from_pretrained(checkpoint, force_download=True)
                test_tokenizer(slow, fast)
                py_logging.info(f"Accuracy {perfect * 100 / total:.2f}")

    except Exception as e:
        py_logging.error(f"Error in main script execution: {e}")
        traceback.print_exc()
