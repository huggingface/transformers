# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from types import SimpleNamespace

import torch

from transformers.pipelines.any_to_any import AnyToAnyPipeline
from transformers.pipelines.any_to_any import ReturnType as AnyReturnType
from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline
from transformers.pipelines.image_text_to_text import ReturnType as ImageReturnType


def test_multiple_candidates():
    vocab = {10: "P", 11: "Q", 21: " x", 22: " y", 23: " z", 24: " w"}

    def decode(batch, **kwargs):
        return ["".join(vocab[int(t)] for t in seq) for seq in batch]

    for cls, returns in ((ImageTextToTextPipeline, ImageReturnType), (AnyToAnyPipeline, AnyReturnType)):
        for n in (1, 2):
            sequences = [[10, 21], [11, 23]] if n == 1 else [[10, 21], [10, 22], [11, 23], [11, 24]]
            pipe = SimpleNamespace(
                model=SimpleNamespace(generate=lambda **kw: torch.tensor(sequences)),
                processor=SimpleNamespace(
                    post_process_image_text_to_text=decode, post_process_multimodal_output=decode
                ),
                tokenizer=SimpleNamespace(response_template=None),
                _prepare_generate_kwargs=lambda kw, **other: kw,
            )
            output = cls._forward(pipe, {"text": ["A", "B"], "input_ids": torch.tensor([[10], [11]])})
            assert tuple(output["generated_sequence"].shape) == ((2, n, 2) if n == 2 else (2, 2))
            prompts = ["A"] * n + ["B"] * n

            for mode in returns:
                records = []
                for i in range(2):
                    item = {
                        "prompt_text": prompts[i * n],
                        "input_ids": output["input_ids"][i : i + 1],
                        "generated_sequence": output["generated_sequence"][i : i + 1],
                    }
                    kw = {"generation_mode": "text"} if cls is AnyToAnyPipeline else {}
                    records += cls.postprocess(pipe, item, return_type=mode, **kw)

                assert [r["input_text"] for r in records] == prompts
                actual = (
                    [r["generated_token_ids"].tolist() for r in records]
                    if mode == returns.TENSORS
                    else [r["generated_text"] for r in records]
                )
                texts = [vocab[s[-1]] for s in sequences]
                expected = (
                    sequences
                    if mode == returns.TENSORS
                    else texts
                    if mode == returns.NEW_TEXT
                    else [p + t for p, t in zip(prompts, texts)]
                )
                assert actual == expected
