# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch

from transformers.feature_extraction_utils import BatchFeature
from transformers.models.voxtral.processing_voxtral import VoxtralProcessor
from transformers.pipelines.any_to_any import AnyToAnyPipeline
from transformers.pipelines.any_to_any import Chat as AnyChat
from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline
from transformers.utils.chat_template_utils import Chat as ImageChat


class MistralCommonBackend:
    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": [[1, 2, 3]]}


@pytest.mark.parametrize("kind", ["processor", "any", "image"])
@pytest.mark.parametrize("prefill", [False, True])
def test_mistral_chat_controls(kind, prefill):
    messages = [{"role": "user", "content": "Hi"}]
    if prefill:
        messages.append({"role": "assistant", "content": "Hello"})

    tokenizer = MistralCommonBackend()

    if kind == "processor":
        processor = object.__new__(VoxtralProcessor)
        processor.tokenizer = tokenizer
        processor._merge_kwargs = lambda **_: {
            "text_kwargs": dict(VoxtralProcessor.text_kwargs),
            "audio_kwargs": {},
        }

        result = processor.apply_chat_template(messages, continue_final_message=prefill)
        assert tokenizer.kwargs.get("continue_final_message", False) is prefill

    else:
        seen = {}

        def apply(_messages, **kwargs):
            seen.update(kwargs)
            return BatchFeature({"input_ids": torch.tensor([[1, 2, 3]])})

        processor = SimpleNamespace(tokenizer=tokenizer, apply_chat_template=apply)
        pipe = AnyToAnyPipeline if kind == "any" else ImageTextToTextPipeline
        chat = AnyChat if kind == "any" else ImageChat

        result = pipe.preprocess(
            SimpleNamespace(processor=processor, dtype=torch.float32),
            chat(messages),
            continue_final_message=prefill,
            padding=True,
            processor_kwargs={"unused": 1},
            unrelated_kwarg=7,
        )

        assert seen == {
            "continue_final_message": prefill,
            "add_generation_prompt": not prefill,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
        }

    assert result["input_ids"].tolist() == [[1, 2, 3]]
