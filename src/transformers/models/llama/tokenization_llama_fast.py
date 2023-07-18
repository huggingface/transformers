# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
import os
from shutil import copyfile
from typing import TYPE_CHECKING, Optional, Tuple

from tokenizers import processors

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
from ...utils.versions import require_version


if TYPE_CHECKING:
    from transformers.pipelines.conversational import Conversation

require_version("tokenizers>=0.13.3")

if is_sentencepiece_available():
    from .tokenization_llama import LlamaTokenizer
else:
    LlamaTokenizer = None

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your\
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not\
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class LlamaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no normalization.

    ```
    from transformers import LlamaTokenizerFast

    tokenizer = LlaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.encode("Hello this is a test")
    >>> [1, 15043, 445, 338, 263, 1243]
    ```

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.


    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.

        clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`):
            Wether to cleanup spaces after decoding, cleanup consists in removing potential artifacts like extra
            spaces.

        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = LlamaTokenizer
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token=True,
        add_eos_token=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()

        self.vocab_file = vocab_file
        self.can_save_slow_tokenizer = False if not self.vocab_file else True

    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id

        eos = self.eos_token
        eos_token_id = self.eos_token_id

        single = f"{(bos+':0 ') * self.add_bos_token}$A:0{(' '+eos+':0') * self.add_eos_token}"
        pair = f"{single}{(' '+bos+':1') * self.add_bos_token} $B:1{(' '+eos+':1') * self.add_eos_token}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    def _build_conversation_input_ids(self, conversation: "Conversation"):
        """Builds the input ids for a conversation.
        This is the format used in the provided examples. System prompts should be manually added at the beginning of
        the conversation. If no system prompt is given, the `DEFAULT_SYSTEM_PROMPT` will be used.
        ```
        <bos>[INST] B_SYS SytemPrompt E_SYS Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST]
        ```

        If you want to use your own system prompt, make sure to use both `B_SYS` and `E_SYS` use the following:
        ```python
        >>> from transformers import Conversation

        >>> Conversation(
        ...     "<<SYS>>\n Only answer with emojis, and charades\n<</SYS>>\n\nHow can I build a house in 10 septs?"
        ... )
        ```
        Args:
            conversation (`Conversation`):
                Conversation to build input ids for.
        Returns:
            `List[int]`:
                Input ids for the conversation.
        """
        dialogue = list(conversation.iter_texts())
        if not all([is_user for is_user, msg in dialogue[::2]]) or not all(
            [not is_user for is_user, msg in dialogue[1::2]]
        ):
            raise ValueError(
                "The model only supports 'user' and 'assistant' roles, starting with user and alternating (u/a/u/a/u...)"
            )

        dialog_tokens = []
        if len(conversation.past_user_inputs) > 0:
            if not conversation.past_user_inputs[0].startswith(B_SYS) or E_SYS not in conversation.past_user_inputs[0]:
                conversation.past_user_inputs[0] = (
                    B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + conversation.past_user_inputs[0]
                )
        elif not dialogue[0][1].startswith(B_SYS) or E_SYS not in dialogue[0][1]:
            dialogue[0] = (dialogue[0][0], B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + dialogue[0][1])

        dialog_tokens += sum(
            [
                [self.bos_token_id]
                + self.encode(
                    f"{B_INST} {(prompt[1]).strip()} {E_INST} {(answer[1]).strip()} ", add_special_tokens=False
                )
                + [self.eos_token_id]
                for prompt, answer in zip(dialogue[::2], dialogue[1::2])
            ],
            [],
        )
        if not (dialogue[-1][0]):
            raise ValueError(f"Last message must be from user, got {dialogue[-1]['role']}")
        dialog_tokens += [self.bos_token_id] + self.encode(
            f"{B_INST} {(dialogue[-1][1]).strip()} {E_INST}", add_special_tokens=False
        )
        return dialog_tokens
