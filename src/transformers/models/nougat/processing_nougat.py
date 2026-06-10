# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for Nougat.
"""

from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class NougatProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class NougatProcessor(ProcessorMixin):
    valid_processor_kwargs = NougatProcessorKwargs

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(self, images=None, text=None, **kwargs):
        result = super().__call__(images=images, text=text, **kwargs)
        if text is not None and images is not None and "input_ids" in result:
            result["labels"] = result.pop("input_ids")
        return result

    def post_process_generation(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.post_process_generation`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.post_process_generation(*args, **kwargs)


__all__ = ["NougatProcessor"]
