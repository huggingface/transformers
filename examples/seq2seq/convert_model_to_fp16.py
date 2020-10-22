#!/usr/bin/env python

from typing import Union

import fire
import torch
from tqdm import tqdm


def convert(src_path: str, map_location: str = "cpu", save_path: Union[str, None] = None) -> None:
    """Convert a pytorch_model.bin or model.pt file to torch.float16 for faster downloads, less disk space."""
    state_dict = torch.load(src_path, map_location=map_location)
    for k, v in tqdm(state_dict.items()):
        if not isinstance(v, torch.Tensor):
            raise TypeError("FP16 conversion only works on paths that are saved state dics, like pytorch_model.bin")
        state_dict[k] = v.half()
    if save_path is None:  # overwrite src_path
        save_path = src_path
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    fire.Fire(convert)
