#!/usr/bin/env python

import os
from pathlib import Path
from typing import Dict, List

import fire
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils.logging import get_logger


logger = get_logger(__name__)


def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


def sanitize(sd):
    return {remove_prefix(k, "model."): v for k, v in sd.items()}


def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]):
    new_sd = {}
    for k in state_dicts[0].keys():
        tensors = [sd[k] for sd in state_dicts]
        new_t = sum(tensors) / len(tensors)
        assert isinstance(new_t, torch.Tensor)
        new_sd[k] = new_t
    return new_sd


def convert_pl_to_hf(pl_ckpt_path: str, hf_src_model_dir: str, save_path: str) -> None:
    """Cleanup a pytorch-lightning .ckpt file or experiment dir and save a huggingface model with that state dict.
    Silently allows extra pl keys (like teacher.) Puts all ckpt models into CPU RAM at once!

    Args:
        pl_ckpt_path (:obj:`str`): Path to a .ckpt file saved by pytorch_lightning or dir containing ckpt files.
            If a directory is passed, all .ckpt files inside it will be averaged!
        hf_src_model_dir (:obj:`str`): Path to a directory containing a correctly shaped checkpoint
        save_path (:obj:`str`): Directory to save the new model

    """
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_src_model_dir)
    if os.path.isfile(pl_ckpt_path):
        ckpt_files = [pl_ckpt_path]
    else:
        assert os.path.isdir(pl_ckpt_path)
        ckpt_files = list(Path(pl_ckpt_path).glob("*.ckpt"))
        assert ckpt_files, f"could not find any ckpt files inside the {pl_ckpt_path} directory"

    if len(ckpt_files) > 1:
        logger.info(f"averaging the weights of {ckpt_files}")

    state_dicts = [sanitize(torch.load(x, map_location="cpu")["state_dict"]) for x in ckpt_files]
    state_dict = average_state_dicts(state_dicts)

    missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
    assert not missing, f"missing keys: {missing}"
    hf_model.save_pretrained(save_path)
    try:
        tok = AutoTokenizer.from_pretrained(hf_src_model_dir)
        tok.save_pretrained(save_path)
    except Exception:
        pass
        # dont copy tokenizer if cant


if __name__ == "__main__":
    fire.Fire(convert_pl_to_hf)
