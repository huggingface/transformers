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

import fire

from utils import calculate_rouge, save_json


def calculate_rouge_path(pred_path, tgt_path, save_path=None, **kwargs):
    """Kwargs will be passed to calculate_rouge"""
    pred_lns = [x.strip() for x in open(pred_path).readlines()]
    tgt_lns = [x.strip() for x in open(tgt_path).readlines()][: len(pred_lns)]
    metrics = calculate_rouge(pred_lns, tgt_lns, **kwargs)
    if save_path is not None:
        save_json(metrics, save_path, indent=None)
    return metrics  # these print nicely


if __name__ == "__main__":
    fire.Fire(calculate_rouge_path)
