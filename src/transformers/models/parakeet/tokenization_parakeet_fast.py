# coding=utf-8
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

import itertools
from typing import Optional, Union

from ...tokenization_utils_fast import PreTrainedTokenizerFast


class ParakeetTokenizerFast(PreTrainedTokenizerFast):
    """
    Inherits all methods from [`PreTrainedTokenizerFast`]. Users should refer to this superclass for more information regarding those methods,
    except for `_decode` which is overridden to adapt it to CTC decoding:
    1. Group consecutive tokens
    2. Filter out the blank token
    """

    def _decode(
        self,
        token_ids: Union[int, list[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        group_tokens: bool = True,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if group_tokens:
            token_ids = [token_group[0] for token_group in itertools.groupby(token_ids)]

        # for CTC we filter out the blank token, which is the pad token
        token_ids = [token for token in token_ids if token != self.pad_token_id]

        return super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["ParakeetTokenizerFast"]
