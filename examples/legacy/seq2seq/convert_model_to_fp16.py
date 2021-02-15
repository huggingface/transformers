#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import fire
import torch
from tqdm import tqdm


def convert(src_path: str, map_location: str = "cpu", save_path: Union[str, None] = None) -> None:
    """Convert a pytorch_model.bin or model.pt file to torch.float16 for faster downloads, less disk space."""
    state_dict = torch.load(src_path, map_location=map_location)
    for k, v in tqdm(state_dict.items()):
        if not isinstance(v, torch.Tensor):
            raise TypeError("FP16 conversion only works on paths that are saved state dicts, like pytorch_model.bin")
        state_dict[k] = v.half()
    if save_path is None:  # overwrite src_path
        save_path = src_path
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    fire.Fire(convert)
