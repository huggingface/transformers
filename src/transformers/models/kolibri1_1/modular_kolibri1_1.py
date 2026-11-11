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

from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..kolibri1.configuration_kolibri1 import Kolibri1Config
from ..kolibri1.modeling_kolibri1 import (
    Kolibri1ForCausalLM,
    Kolibri1Model,
    Kolibri1PreTrainedModel,
)


@auto_docstring(checkpoint="Aleph-Alpha/Kolibri-1.1")
@strict
class Kolibri1_1Config(Kolibri1Config):
    r"""
    ```python
    >>> from transformers import Kolibri1_1Model, Kolibri1_1Config

    >>> configuration = Kolibri1_1Config()
    >>> model = Kolibri1_1Model(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "kolibri1_1"


class Kolibri1_1PreTrainedModel(Kolibri1PreTrainedModel):
    pass


class Kolibri1_1Model(Kolibri1Model):
    pass


class Kolibri1_1ForCausalLM(Kolibri1ForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Kolibri1_1ForCausalLM

        >>> model = Kolibri1_1ForCausalLM.from_pretrained("Aleph-Alpha/Kolibri-1.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("Aleph-Alpha/Kolibri-1.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        return super().forward(**super_kwargs)


__all__ = [
    "Kolibri1_1Config",
    "Kolibri1_1ForCausalLM",
    "Kolibri1_1Model",
    "Kolibri1_1PreTrainedModel",
]
