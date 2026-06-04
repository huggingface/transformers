# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from ..gemma4_assistant.modeling_gemma4_assistant import Gemma4AssistantForCausalLM


class Gemma4UnifiedAssistantForCausalLM(Gemma4AssistantForCausalLM):
    def forward(**super_kwargs):
        r"""
        shared_kv_states (`dict[str, torch.Tensor` of shape `(batch_size, 1, q_len, kv_len)`, *optional*):
            A dictionary containing the computed KV values for the last layer of each `layer_type` in this model.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma4UnifiedAssistantForCausalLM, Gemma4UnifiedForConditionalGeneration

        >>> model = Gemma4UnifiedForConditionalGeneration.from_pretrained("google/gemma-4-e2b-it")
        >>> assistant_model = Gemma4UnifiedAssistantForCausalLM.from_pretrained("google/gemma-4-e2b-it-assistant")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, assistant_model=assistant_model, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        "What is your favorite condiment?"
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "Gemma4UnifiedAssistantPreTrainedModel",  # noqa: F822
    "Gemma4UnifiedAssistantForCausalLM",
]
