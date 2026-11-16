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

from transformers import VoxtralProcessor
from transformers.testing_utils import require_mistral_common, require_torch, slow


@slow
@require_mistral_common
@require_torch
def test_voxtral_assistant_prefill_matches_mistral_tokenizer():
    processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
    assert processor.tokenizer.__class__.__name__ == "MistralCommonBackend"

    conversation = [
        {"role": "user", "content": "What is the answer?"},
        {"role": "assistant", "content": "The answer is"},
    ]

    expected = processor.tokenizer.apply_chat_template(
        conversation,
        continue_final_message=True,
        tokenize=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )
    actual = processor.apply_chat_template(
        conversation,
        continue_final_message=True,
    )

    assert actual["input_ids"].tolist() == expected["input_ids"].tolist()
