# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class Ovis2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
    }


@auto_docstring
class Ovis2Processor(ProcessorMixin):
    valid_processor_kwargs = Ovis2ProcessorKwargs
    unused_input_names = ["grids"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<image>",
        image_seq_length=256,
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        image_seq_length (`int`, *optional*, defaults to 256):
            The number of image tokens to be used for each image in the input.
        """
        self.image_seq_length = image_seq_length
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        grid = image_inputs["grids"][image_idx].tolist()
        row, col = grid[0], grid[1]
        placeholder = f"<IMG_START>{'<IMG_ATOM>' * self.image_seq_length}<IMG_GRID>"
        if row * col > 1:
            for r in range(row):
                for c in range(col):
                    placeholder += f"{'<IMG_ATOM>' * self.image_seq_length}"
                    if c < col - 1:
                        placeholder += "<IMG_COL>"
                if r < row - 1:
                    placeholder += "<IMG_ROW>"
        placeholder += "<IMG_END>"
        return placeholder


__all__ = ["Ovis2Processor"]
