# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

"""Tokenization classes for LLaMA."""
import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "hf-internal-testing/llama-tokenizer": 2048,
}
SPIECE_UNDERLINE = "▁"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class LlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        legacy (`bool`, *optional*):
            Whether or not the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of #24622
            and #25224 which includes fixes to properly handle tokens that appear after special tokens. A simple
            example:

            - `legacy=True`:
            ```python
            >>> from transformers import T5Tokenizer

            >>> tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=True)
            >>> tokenizer.encode("Hello <extra_id_0>.")
            [8774, 32099, 3, 5, 1]
            ```
            - `legacy=False`:
            ```python
            >>> from transformers import T5Tokenizer

            >>> tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
            >>> tokenizer.encode("Hello <extra_id_0>.")  # the extra space `[3]` is no longer here
            [8774, 32099, 5, 1]
            ```
            Checkout the [pull request](https://github.com/huggingface/transformers/pull/24565) for more details.

    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=True,
        spaces_between_special_tokens=False,
        legacy=None,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. If you see this, DO NOT PANIC! This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thouroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565"
            )
            legacy = True

        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self.sp_model = self.get_spm_processor()

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy,
            **kwargs,
        )

    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(self):
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        if self.legacy:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer

        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(self, text: "TextInput", add_special_tokens=False, **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        tokens = self.sp_model.encode(text, out_type=str)
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. Encode string + prefix ex: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. Remove self.unk_token from ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # since we manually add the prefix space, we have to remove it when decoding
        if tokens[0].startswith(SPIECE_UNDERLINE):
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0 and self.legacy:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

    @property
    def default_chat_template(self):
        """
        LLaMA uses [INST] and [/INST] to indicate user messages, and <<SYS>> and <</SYS>> to indicate system messages.
        Assistant messages do not have special tokens, because LLaMA chat models are generally trained with strict
        user/assistant/user/assistant message ordering, and so assistant messages can be identified from the ordering
        rather than needing special tokens. The system message is partly 'embedded' in the first user message, which
        results in an unusual token ordering when it is present. This template should definitely be changed if you wish
        to fine-tune a model with more flexible role ordering!

        The output should look something like:

        <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST] Answer <eos> <bos>[INST] Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST]
        """

        template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
            "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
        template = template.replace("USE_DEFAULT_PROMPT", "true" if self.use_default_system_prompt else "false")
        default_message = DEFAULT_SYSTEM_PROMPT.replace("\n", "\\n").replace("'", "\\'")
        template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)

        return template
