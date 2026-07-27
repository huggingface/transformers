# Copyright 2022 The HuggingFace Inc. team.
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
"""
Processor class for Donut.
"""

import re

from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


class DonutProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "return_mm_token_type_ids": False,
            "return_text_replacement_offsets": False,
        },
    }


logger = logging.get_logger(__name__)


@auto_docstring
class DonutProcessor(ProcessorMixin):
    valid_processor_kwargs = DonutProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[DonutProcessorKwargs],
    ):
        if images is not None and text is not None:
            kwargs.setdefault("add_special_tokens", False)

        model_inputs = super().__call__(images=images, text=text, **kwargs)
        if text is not None and images is not None:
            model_inputs["labels"] = model_inputs["input_ids"]
        return model_inputs

    @property
    def model_input_names(self):
        return super().model_input_names + ["labels"]

    def token2json(self, tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = self.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            # We want r"<s_(.*?)>" but without ReDOS risk, so do it manually in two parts
            potential_start = re.search(r"<s_", tokens, re.IGNORECASE)
            if potential_start is None:
                break
            start_token = tokens[potential_start.start() :]
            if ">" not in start_token:
                break
            start_token = start_token[: start_token.index(">") + 1]
            key = start_token[len("<s_") : -len(">")]
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = self.token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if output:
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


__all__ = ["DonutProcessor"]
