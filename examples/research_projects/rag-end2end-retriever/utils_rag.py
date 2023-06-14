import itertools
import json
import linecache
import os
import pickle
import re
import socket
import string
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import git
import torch
from torch.utils.data import Dataset

from transformers import BartTokenizer, RagTokenizer, T5Tokenizer


def encode_line(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        # Need to add eos token manually for T5
        if isinstance(self.tokenizer, T5Tokenizer):
            source_line += self.tokenizer.eos_token
            tgt_line += self.tokenizer.eos_token

        # Pad source and target to the right
        source_tokenizer = (
            self.tokenizer.question_encoder if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        )
        target_tokenizer = self.tokenizer.generator if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer

        source_inputs = encode_line(source_tokenizer, source_line, self.max_source_length, "right")
        target_inputs = encode_line(target_tokenizer, tgt_line, self.max_target_length, "right")

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        tgt_pad_token_id = (
            self.tokenizer.generator.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        src_pad_token_id = (
            self.tokenizer.question_encoder.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        y = trim_batch(target_ids, tgt_pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, src_pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
        }
        return batch


logger = getLogger(__name__)


def flatten_list(summary_ids: List[List]):
    return list(itertools.chain.from_iterable(summary_ids))


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
        "hostname": str(socket.gethostname()),
    }
    return repo_infos


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def calculate_exact_match(output_lns: List[str], reference_lns: List[str]) -> Dict:
    assert len(output_lns) == len(reference_lns)
    em = 0
    for hypo, pred in zip(output_lns, reference_lns):
        em += exact_match_score(hypo, pred)
    if len(output_lns) > 0:
        em /= len(output_lns)
    return {"em": em}


def is_rag_model(model_prefix):
    return model_prefix.startswith("rag")


def set_extra_model_params(extra_params, hparams, config):
    equivalent_param = {p: p for p in extra_params}
    # T5 models don't have `dropout` param, they have `dropout_rate` instead
    equivalent_param["dropout"] = "dropout_rate"
    for p in extra_params:
        if getattr(hparams, p, None):
            if not hasattr(config, p) and not hasattr(config, equivalent_param[p]):
                logger.info("config doesn't have a `{}` attribute".format(p))
                delattr(hparams, p)
                continue
            set_p = p if hasattr(config, p) else equivalent_param[p]
            setattr(config, set_p, getattr(hparams, p))
            delattr(hparams, p)
    return hparams, config
