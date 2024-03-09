import os
from typing import List, Tuple, Union

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, cached_file
from transformers.utils.hub import get_checkpoint_shard_files


def check_list_type(list_of_list: List[List[Union[int, float]]], error_message: str) -> None:
    if list_of_list is None:
        return

    assert isinstance(list_of_list, list), error_message
    assert isinstance(list_of_list[0], list), error_message


def flatten_and_convert_to_tensors(x: List[int], device: torch.device) -> torch.Tensor:
    y = []
    for sequence in x:
        y.extend(sequence)

    return torch.tensor(y, device=device)


def download_repo(repo_name_or_path: str) -> Tuple[AutoConfig, AutoTokenizer, str]:
    config = _download_config(repo_name_or_path)
    tokenizer = _download_tokenizer(repo_name_or_path)
    model_path = None

    if os.path.isdir(repo_name_or_path):
        model_path = repo_name_or_path
    else:
        # try downloading model weights
        try:
            model_path = cached_file(repo_name_or_path, SAFE_WEIGHTS_NAME)
            model_path = os.path.dirname(model_path)
        except:
            # try downloading model weights if they are sharded
            try:
                sharded_filename = cached_file(repo_name_or_path, SAFE_WEIGHTS_INDEX_NAME)
                get_checkpoint_shard_files(repo_name_or_path, sharded_filename)
                model_path = os.path.dirname(sharded_filename)
            except:
                pass

    return config, tokenizer, model_path


def _download_config(repo_name_or_path: str) -> AutoConfig:
    try:
        config = AutoConfig.from_pretrained(repo_name_or_path)
    except:
        config = None

    return config


def _download_tokenizer(repo_name_or_path: str) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_name_or_path)
    except:
        tokenizer = None

    return tokenizer
