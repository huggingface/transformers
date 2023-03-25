import os

import jsonlines
import numpy as np
from tqdm import tqdm


DOC_STRIDE = 2048
MAX_LENGTH = 4096
SEED = 42
PROCESS_TRAIN = os.environ.pop("PROCESS_TRAIN", "false")
CATEGORY_MAPPING = {"null": 0, "short": 1, "long": 2, "yes": 3, "no": 4}


def _get_single_answer(example):
    def choose_first(answer, is_long_answer=False):
        assert isinstance(answer, list)
        if len(answer) == 1:
            answer = answer[0]
            return {k: [answer[k]] for k in answer} if is_long_answer else answer
        for a in answer:
            if is_long_answer:
                a = {k: [a[k]] for k in a}
            if len(a["start_token"]) > 0:
                break
        return a

    answer = {"id": example["id"]}
    annotation = example["annotations"]
    yes_no_answer = annotation["yes_no_answer"]
    if 0 in yes_no_answer or 1 in yes_no_answer:
        answer["category"] = ["yes"] if 1 in yes_no_answer else ["no"]
        answer["start_token"] = answer["end_token"] = []
        answer["start_byte"] = answer["end_byte"] = []
        answer["text"] = ["<cls>"]
    else:
        answer["category"] = ["short"]
        out = choose_first(annotation["short_answers"])
        if len(out["start_token"]) == 0:
            # answer will be long if short is not available
            answer["category"] = ["long"]
            out = choose_first(annotation["long_answer"], is_long_answer=True)
            out["text"] = []
        answer.update(out)

    # disregard some samples
    if len(answer["start_token"]) > 1 or answer["start_token"] == answer["end_token"]:
        answer["remove_it"] = True
    else:
        answer["remove_it"] = False

    cols = ["start_token", "end_token", "start_byte", "end_byte", "text"]
    if not all([isinstance(answer[k], list) for k in cols]):
        raise ValueError("Issue in ID", example["id"])

    return answer


def get_context_and_ans(example, assertion=False):
    """Gives new context after removing <html> & new answer tokens as per new context"""
    answer = _get_single_answer(example)
    # bytes are of no use
    del answer["start_byte"]
    del answer["end_byte"]

    # handle yes_no answers explicitly
    if answer["category"][0] in ["yes", "no"]:  # category is list with one element
        doc = example["document"]["tokens"]
        context = []
        for i in range(len(doc["token"])):
            if not doc["is_html"][i]:
                context.append(doc["token"][i])
        return {
            "context": " ".join(context),
            "answer": {
                "start_token": -100,  # ignore index in cross-entropy
                "end_token": -100,  # ignore index in cross-entropy
                "category": answer["category"],
                "span": answer["category"],  # extra
            },
        }

    # later, help in removing all no answers
    if answer["start_token"] == [-1]:
        return {
            "context": "None",
            "answer": {
                "start_token": -1,
                "end_token": -1,
                "category": "null",
                "span": "None",  # extra
            },
        }

    # handling normal samples

    cols = ["start_token", "end_token"]
    answer.update({k: answer[k][0] if len(answer[k]) > 0 else answer[k] for k in cols})  # e.g. [10] == 10

    doc = example["document"]["tokens"]
    start_token = answer["start_token"]
    end_token = answer["end_token"]

    context = []
    for i in range(len(doc["token"])):
        if not doc["is_html"][i]:
            context.append(doc["token"][i])
        else:
            if answer["start_token"] > i:
                start_token -= 1
            if answer["end_token"] > i:
                end_token -= 1
    new = " ".join(context[start_token:end_token])

    # checking above code
    if assertion:
        """checking if above code is working as expected for all the samples"""
        is_html = doc["is_html"][answer["start_token"] : answer["end_token"]]
        old = doc["token"][answer["start_token"] : answer["end_token"]]
        old = " ".join([old[i] for i in range(len(old)) if not is_html[i]])
        if new != old:
            print("ID:", example["id"])
            print("New:", new, end="\n")
            print("Old:", old, end="\n\n")

    return {
        "context": " ".join(context),
        "answer": {
            "start_token": start_token,
            "end_token": end_token - 1,  # this makes it inclusive
            "category": answer["category"],  # either long or short
            "span": new,  # extra
        },
    }


def get_strided_contexts_and_ans(example, tokenizer, doc_stride=2048, max_length=4096, assertion=True):
    # overlap will be of doc_stride - q_len

    out = get_context_and_ans(example, assertion=assertion)
    answer = out["answer"]

    # later, removing these samples
    if answer["start_token"] == -1:
        return {
            "example_id": example["id"],
            "input_ids": [[-1]],
            "labels": {
                "start_token": [-1],
                "end_token": [-1],
                "category": ["null"],
            },
        }

    input_ids = tokenizer(example["question"]["text"], out["context"]).input_ids
    q_len = input_ids.index(tokenizer.sep_token_id) + 1

    # return yes/no
    if answer["category"][0] in ["yes", "no"]:  # category is list with one element
        inputs = []
        category = []
        q_indices = input_ids[:q_len]
        doc_start_indices = range(q_len, len(input_ids), max_length - doc_stride)
        for i in doc_start_indices:
            end_index = i + max_length - q_len
            slice = input_ids[i:end_index]
            inputs.append(q_indices + slice)
            category.append(answer["category"][0])
            if slice[-1] == tokenizer.sep_token_id:
                break

        return {
            "example_id": example["id"],
            "input_ids": inputs,
            "labels": {
                "start_token": [-100] * len(category),
                "end_token": [-100] * len(category),
                "category": category,
            },
        }

    splitted_context = out["context"].split()
    complete_end_token = splitted_context[answer["end_token"]]
    answer["start_token"] = len(
        tokenizer(
            " ".join(splitted_context[: answer["start_token"]]),
            add_special_tokens=False,
        ).input_ids
    )
    answer["end_token"] = len(
        tokenizer(" ".join(splitted_context[: answer["end_token"]]), add_special_tokens=False).input_ids
    )

    answer["start_token"] += q_len
    answer["end_token"] += q_len

    # fixing end token
    num_sub_tokens = len(tokenizer(complete_end_token, add_special_tokens=False).input_ids)
    if num_sub_tokens > 1:
        answer["end_token"] += num_sub_tokens - 1

    old = input_ids[answer["start_token"] : answer["end_token"] + 1]  # right & left are inclusive
    start_token = answer["start_token"]
    end_token = answer["end_token"]

    if assertion:
        """This won't match exactly because of extra gaps => visaully inspect everything"""
        new = tokenizer.decode(old)
        if answer["span"] != new:
            print("ISSUE IN TOKENIZATION")
            print("OLD:", answer["span"])
            print("NEW:", new, end="\n\n")

    if len(input_ids) <= max_length:
        return {
            "example_id": example["id"],
            "input_ids": [input_ids],
            "labels": {
                "start_token": [answer["start_token"]],
                "end_token": [answer["end_token"]],
                "category": answer["category"],
            },
        }

    q_indices = input_ids[:q_len]
    doc_start_indices = range(q_len, len(input_ids), max_length - doc_stride)

    inputs = []
    answers_start_token = []
    answers_end_token = []
    answers_category = []  # null, yes, no, long, short
    for i in doc_start_indices:
        end_index = i + max_length - q_len
        slice = input_ids[i:end_index]
        inputs.append(q_indices + slice)
        assert len(inputs[-1]) <= max_length, "Issue in truncating length"

        if start_token >= i and end_token <= end_index - 1:
            start_token = start_token - i + q_len
            end_token = end_token - i + q_len
            answers_category.append(answer["category"][0])  # ["short"] -> "short"
        else:
            start_token = -100
            end_token = -100
            answers_category.append("null")
        new = inputs[-1][start_token : end_token + 1]

        answers_start_token.append(start_token)
        answers_end_token.append(end_token)
        if assertion:
            """checking if above code is working as expected for all the samples"""
            if new != old and new != [tokenizer.cls_token_id]:
                print("ISSUE in strided for ID:", example["id"])
                print("New:", tokenizer.decode(new))
                print("Old:", tokenizer.decode(old), end="\n\n")
        if slice[-1] == tokenizer.sep_token_id:
            break

    return {
        "example_id": example["id"],
        "input_ids": inputs,
        "labels": {
            "start_token": answers_start_token,
            "end_token": answers_end_token,
            "category": answers_category,
        },
    }


def prepare_inputs(example, tokenizer, doc_stride=2048, max_length=4096, assertion=False):
    example = get_strided_contexts_and_ans(
        example,
        tokenizer,
        doc_stride=doc_stride,
        max_length=max_length,
        assertion=assertion,
    )

    return example


def save_to_disk(hf_data, file_name):
    with jsonlines.open(file_name, "a") as writer:
        for example in tqdm(hf_data, total=len(hf_data), desc="Saving samples ... "):
            labels = example["labels"]
            for ids, start, end, cat in zip(
                example["input_ids"],
                labels["start_token"],
                labels["end_token"],
                labels["category"],
            ):
                if start == -1 and end == -1:
                    continue  # leave waste samples with no answer
                if cat == "null" and np.random.rand() < 0.6:
                    continue  # removing 50 % samples
                writer.write(
                    {
                        "input_ids": ids,
                        "start_token": start,
                        "end_token": end,
                        "category": CATEGORY_MAPPING[cat],
                    }
                )


if __name__ == "__main__":
    """Running area"""
    from datasets import load_dataset

    from transformers import BigBirdTokenizer

    data = load_dataset("natural_questions")
    tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

    data = data["train" if PROCESS_TRAIN == "true" else "validation"]

    fn_kwargs = {
        "tokenizer": tokenizer,
        "doc_stride": DOC_STRIDE,
        "max_length": MAX_LENGTH,
        "assertion": False,
    }
    data = data.map(prepare_inputs, fn_kwargs=fn_kwargs)
    data = data.remove_columns(["annotations", "document", "id", "question"])
    print(data)

    np.random.seed(SEED)
    cache_file_name = "nq-training.jsonl" if PROCESS_TRAIN == "true" else "nq-validation.jsonl"
    save_to_disk(data, file_name=cache_file_name)
