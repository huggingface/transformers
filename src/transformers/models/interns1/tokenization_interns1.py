# coding=utf-8
# Copyright 2025 The Intern team and Shanghai AI Lab team. All rights reserved.
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
"""Tokenization classes for InternS1."""

import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import lru_cache
from typing import Optional, Union

import regex as re
import sentencepiece as spm

from ...models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from ...tokenization_utils_base import AddedToken, TextInput
from ...utils import logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning_once(
        "If tokenization with SMILES formula is of necessity, please 'pip install RDKit' for better tokenization quality."
    )
    RDKIT_AVAILABLE = False

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "sp_model_SMILES": "tokenizer_SMILES.model",
    "sp_model_IUPAC": "tokenizer_IUPAC.model",
    "sp_model_FASTA": "tokenizer_FASTA.model",
}

PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class InternS1CheckModuleMixin(ABC):
    """
    Basic auto-detection module.

    Note that short strings are ignored by this module.
    """

    def __init__(self, *, min_length: int):
        self.min_length = min_length
        self.REGEX = self._build_regex()
        self.auto_detect_token = []
        self.truncation = False

    @abstractmethod
    def _build_regex(self):
        pass

    @abstractmethod
    def check_legitimacy(self, candidate: str) -> bool:
        pass

    def re_split(self, texts: Union[str, list[str]]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]

        total_results = []

        for text in texts:
            results = []
            current_pos = 0
            for match in self.REGEX.finditer(text):
                candidate = match.group(1)

                if len(candidate) >= self.min_length:
                    match_start, match_end = match.span(1)

                    if not self.check_legitimacy(candidate):
                        continue

                    if not self.truncation:
                        if match_start > 0 and text[match_start - 1].encode("UTF-8").isalpha():
                            continue
                        if match_end < len(text) and text[match_end].encode("UTF-8").isalpha():
                            continue

                    if match_start > current_pos:
                        non_candidate_part = text[current_pos:match_start]
                        results.append(non_candidate_part)
                else:
                    continue

                results.extend([self.auto_detect_token[0], candidate, self.auto_detect_token[1]])
                current_pos = match_end

            if current_pos < len(text):
                remaining_part = text[current_pos:]
                results.append(remaining_part)

            total_results.extend(results)

        return total_results


class FastaCheckModule(InternS1CheckModuleMixin):
    """
    Protein sequence auto-detection module.

    Automatically detects protein sequence using regex patterns.
    """

    def __init__(self, *, min_length: int = 27):
        super().__init__(min_length=min_length)
        self.auto_detect_token = ["<FASTA_AUTO_DETECT>", "</FASTA_AUTO_DETECT>"]
        self.truncation = True

    def _build_regex(self):
        return re.compile(r"([A-Z]{" + str(self.min_length) + r",})")

    def check_legitimacy(self, candidate: str):
        return True


# fmt: off
bonds = ["-", "=", "#", ":", "/", "\\", ".", "$"]
organic_symbols = ["B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]
other_allows = bonds + ["[", "]", "(", ")", ";"]
aromatic_symbols = ["b", "c", "n", "o", "s", "p"]
elements = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]
# fmt: on


class SmilesCheckModule(InternS1CheckModuleMixin):
    """
    SMILES molecular sequence auto-detection module.

    Automatically detects and validates SMILES strings in text using regex patterns
    or chemical syntax rules. Uses RDKit for precise validation when available,
    otherwise falls back to rule-based validation.
    """

    def __init__(self, *, min_length: int = 10):
        super().__init__(min_length=min_length)
        self.auto_detect_token = ["<SMILES_AUTO_DETECT>", "</SMILES_AUTO_DETECT>"]
        self._SQ_BRACKET_BAN_1 = re.compile(r"(?:[A-GI-Z]|[a-z]){3,}")
        self._SQ_BRACKET_BAN_2 = re.compile(r"\d{4,}")

    def _build_regex(self):
        # fmt: off
        _two_letter_elements = [
            'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'Ba', 'Be', 'Bh', 'Bi', 'Bk', 'Br', 'Ca', 'Cd',
            'Ce', 'Cf', 'Cl', 'Cm', 'Cn', 'Co', 'Cr', 'Cs', 'Cu', 'Db', 'Ds', 'Dy', 'Er', 'Es', 'Eu', 'Fe',
            'Fl', 'Fm', 'Fr', 'Ga', 'Gd', 'Ge', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'In', 'Ir', 'Kr', 'La', 'Li',
            'Lr', 'Lu', 'Lv', 'Mc', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'Na', 'Nb', 'Nd', 'Ne', 'Nh', 'Ni', 'No',
            'Np', 'Og', 'Os', 'Pa', 'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rg',
            'Rh', 'Rn', 'Ru', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te', 'Th',
            'Ti', 'Tl', 'Tm', 'Ts', 'Xe', 'Yb', 'Zn', 'Zr'
        ]
        _single_letter_elements = [
            "B", "C", "F", "H", "I", "K", "N", "O", "P", "S", "U", "V", "W", "Y", 'b', 'c', 'n', 'o', 'p', 's'
        ]
        # fmt: on
        all_elements_sorted = sorted(_two_letter_elements + _single_letter_elements, key=lambda x: (-len(x), x))
        elements_pattern_str = "|".join(all_elements_sorted)

        bracket_atom_pattern_str = r"\[[^\]]+\]"
        other_single_chars_pattern_str = r"[\(\)\.=\-#@\d\$\%\*:\+\-\/\\]"
        smiles_unit_pattern = (
            r"(?:"
            + bracket_atom_pattern_str
            + r"|"
            + elements_pattern_str
            + r"|"
            + other_single_chars_pattern_str
            + r")"
        )
        core_sequence_pattern = rf"(?>{smiles_unit_pattern}){{10,}}"
        constrained_core_sequence_pattern = rf"(?![:.=]){core_sequence_pattern}(?<![:.=])"

        final_regex_str = rf"({constrained_core_sequence_pattern})"

        COMPILED_REGEX = re.compile(final_regex_str)
        return COMPILED_REGEX

    def check_legitimacy_slow(self, candidate: str) -> bool:
        """Check legitimacy with RDKit"""
        if sum(1 for char in candidate if char.encode("UTF-8").isalpha()) < 5:
            return False

        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            return False
        else:
            return True

    def check_legitimacy_fast(self, candidate: str) -> bool:
        """Check legitimacy with hard rules"""
        if sum(1 for char in candidate if char.encode("UTF-8").isalpha()) < 5:
            return False

        if not self.check_rings_and_brackets(candidate):
            return False
        else:
            return True

    def check_legitimacy(self, candidate: str) -> bool:
        if RDKIT_AVAILABLE:
            return self.check_legitimacy_slow(candidate)
        else:
            return self.check_legitimacy_fast(candidate)

    def check_brackets(self, text):
        matches = re.findall(r"\[([^\[\]]*)\]", text)
        for part in matches:
            if "(" in part or ")" in part:
                return False
            if len(part) == 0:
                return False
            if part[0] in elements or part[0] in aromatic_symbols or part[:2] in elements:
                return True
        return True

    def check_rings_and_brackets(self, text):
        rings = {}
        left_sq_bracket, right_sq_bracket = 0, 0
        left_pt_bracket, right_pt_bracket = 0, 0
        all_lower = True
        digits_cnt = 0
        pos = 0
        while pos < len(text):
            step = 0
            c = text[pos]
            if ord(c) >= 65 and ord(c) <= 90:
                all_lower = False
            if (pos == len(text) - 1 or pos == 0) and c in bonds:
                return False
            if pos > 0 and text[pos - 1] in bonds and text[pos] in bonds:
                return False
            if c == "[":
                step = 1
                left_sq_bracket += 1
                if left_sq_bracket > right_sq_bracket + 1:
                    return False
                if pos == len(text) - 1:
                    return False
                if "]" not in text[pos + 1 :]:
                    return False
                bracket_span = text[pos + 1 : text.find("]")]

                if self._SQ_BRACKET_BAN_1.search(bracket_span) or self._SQ_BRACKET_BAN_2.search(bracket_span):
                    return False

                matches = re.findall(r"\d+", bracket_span)
                if len(matches) > 2:
                    return False
            if c == "]":
                step = 1
                right_sq_bracket += 1
                if right_sq_bracket > left_sq_bracket:
                    return False

            if c == "(":
                step = 1
                left_pt_bracket += 1
            if c == ")":
                step = 1
                right_pt_bracket += 1
                if right_pt_bracket > left_pt_bracket:
                    return False

            if left_sq_bracket == right_sq_bracket:
                if c.isdigit():
                    digits_cnt += 1
                    step = 1
                    if (
                        pos == 0
                        or (pos == 1 and text[pos - 1] != "%")
                        or (pos > 1 and text[pos - 1] != "%" and text[pos - 2] != "%")
                    ):
                        if c in rings:
                            if rings[c] == "unclosed":
                                rings[c] = "closed"
                            else:
                                rings[c] = "unclosed"
                        else:
                            rings[c] = "unclosed"
                if c == "%":
                    if pos >= len(text) - 2 or not text[pos + 1].isdigit() or not text[pos + 2].isdigit():
                        return False
                    step = 3
                    digits_cnt += 1
                    num = text[pos + 1 : pos + 3]
                    if num in rings:
                        if rings[num] == "unclosed":
                            rings[num] = "closed"
                        else:
                            rings[num] = "unclosed"
                    else:
                        rings[num] = "unclosed"
                if step == 0:
                    if (
                        pos < len(text) - 1
                        and text[pos : pos + 2] in organic_symbols + aromatic_symbols + other_allows
                    ):
                        step = 2
                    elif c in organic_symbols + aromatic_symbols + other_allows:
                        step = 1
                    else:
                        return False

            if step == 0:
                step = 1
            pos += step

        if left_sq_bracket != right_sq_bracket or any(v == "unclosed" for v in rings.values()):
            return False
        if all_lower and digits_cnt < 2:
            return False
        return self.check_brackets(text)


@requires(backends=("sentencepiece",))
class InternS1Tokenizer(Qwen2Tokenizer):
    """
    Construct an InternS1 tokenizer. Based on byte-level Byte-Pair-Encoding.

    Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("InternS1Tokenizer", trust_remote_code=True)
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21927, 1879]
    ```
    This is expected.

    Include custom extension to support better domain-specific text tokenization, leveraging a separately trained tokenizer model.
    Users should refer to this superclass [`PreTrainedTokenizer`] for more information regarding those overloaded methods

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not the model should cleanup the spaces that were added when splitting the input text during the
            tokenization process. Not applicable to this tokenizer, since tokenization does not add spaces.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the tokenization process. The default behavior is
            to not split special tokens. This means that if `<|endoftext|>` is the `eos_token`, then `tokenizer.tokenize("<|endoftext|>") =
            ['<|endoftext|>`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("<|endoftext|>")` will be give `['<',
            '|', 'endo', 'ft', 'ext', '|', '>']`. This argument is only supported for `slow` tokenizers for the moment.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        self.extra_tokenizer_start_mapping = {}
        self.extra_tokenizer_end_mapping = {}
        self._extra_special_tokens = []

        self._extra_tokenizer_list = [
            {
                "tokenizer_name": "tokenizer_SMILES",
                "tokenizer_path": os.path.join(os.path.dirname(vocab_file), "tokenizer_SMILES.model"),
                "begin_sp_tokens": ["<SMILES>", "<SELFIES>"],
                "end_sp_tokens": ["</SMILES>", "</SELFIES>"],
                "auto_begin_sp_tokens": ["<SMILES_AUTO_DETECT>"],
                "auto_end_sp_tokens": ["</SMILES_AUTO_DETECT>"],
            },
            {
                "tokenizer_name": "tokenizer_IUPAC",
                "tokenizer_path": os.path.join(os.path.dirname(vocab_file), "tokenizer_IUPAC.model"),
                "begin_sp_tokens": ["<IUPAC>"],
                "end_sp_tokens": ["</IUPAC>"],
                "auto_begin_sp_tokens": [],
                "auto_end_sp_tokens": [],
            },
            {
                "tokenizer_name": "tokenizer_FASTA",
                "tokenizer_path": os.path.join(os.path.dirname(vocab_file), "tokenizer_FASTA.model"),
                "begin_sp_tokens": [],
                "end_sp_tokens": [],
                "auto_begin_sp_tokens": ["<FASTA_AUTO_DETECT>"],
                "auto_end_sp_tokens": ["</FASTA_AUTO_DETECT>"],
            },
        ]
        # Content wrapped in these sp tokens won't be tokenized
        self.protect_begin_sp_tokens = ["<MOLFORMULA>"]
        self.protect_end_sp_tokens = ["</MOLFORMULA>"]

        self.auto_begin_sp_tokens = []
        self.auto_end_sp_tokens = []

        self._unk_token = "<unk>"  # Fall-back

        self.new_sp_token_offset = [26]  # The length of sp token before the start of extra vocab
        self.tokenizer_mapping = OrderedDict()

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

        # keep order for python < 3.7
        self.tokenizer_mapping = OrderedDict([("tokenizer_original", self.encoder)])

        if self._extra_tokenizer_list is not None:
            for tokenizer_config in self._extra_tokenizer_list:
                self._build_extra_tokenizer(tokenizer_config)
                self._update_special_tokens(tokenizer_config)
                self._update_logical_special_tokens(tokenizer_config)
                self.decoder.update(self._build_extra_decoder(tokenizer_config))

        for token in self.protect_begin_sp_tokens:
            self.tokens_trie.add(token)

        for token in self.protect_end_sp_tokens:
            self.tokens_trie.add(token)

        self.new_sp_token_offset.append(
            len(self._added_tokens_decoder) - sum(self.new_sp_token_offset) + len(self._extra_special_tokens)
        )
        self.check_module_list = [SmilesCheckModule(), FastaCheckModule()]

    @property
    def vocab_size(self) -> int:
        """Returns vocab size including extra tokenizer"""
        total_vocab_size = len(self.encoder)
        for tokenizer in self.tokenizer_mapping.values():
            if isinstance(tokenizer, dict):
                continue
            else:
                total_vocab_size += tokenizer.get_piece_size()
        return total_vocab_size + sum(self.new_sp_token_offset)

    def __len__(self) -> int:
        """Overload method"""
        return self.vocab_size

    @property
    def logical_auto_tokens(self):
        """Tokens that won't be decoded and only for switching tokenizer"""
        return self.auto_begin_sp_tokens + self.auto_end_sp_tokens

    @property
    def extra_tokenizer_bos_keys(self):
        return self.extra_tokenizer_start_mapping.keys()

    @property
    def extra_tokenizer_eos_keys(self):
        return self.extra_tokenizer_end_mapping.keys()

    @property
    def protect_sp_tokens(self):
        """Content wrapped by these sp tokens won't apply extra tokenizer"""
        return self.protect_begin_sp_tokens + self.protect_end_sp_tokens

    def _build_extra_tokenizer(self, tokenizer_config: dict) -> None:
        """
        Build domain-specific tokenizers
        and register them in tokenizer_mapping
        """
        _sp_model = spm.SentencePieceProcessor()
        _sp_model.Load(tokenizer_config["tokenizer_path"])
        self.tokenizer_mapping.update({tokenizer_config["tokenizer_name"]: _sp_model})

        for begin_sp_token, end_sp_token in zip(
            tokenizer_config["begin_sp_tokens"], tokenizer_config["end_sp_tokens"]
        ):
            self.extra_tokenizer_start_mapping.update({begin_sp_token: tokenizer_config["tokenizer_name"]})
            self.extra_tokenizer_end_mapping.update({end_sp_token: tokenizer_config["tokenizer_name"]})

        for begin_sp_token, end_sp_token in zip(
            tokenizer_config["auto_begin_sp_tokens"], tokenizer_config["auto_end_sp_tokens"]
        ):
            self.extra_tokenizer_start_mapping.update({begin_sp_token: tokenizer_config["tokenizer_name"]})
            self.extra_tokenizer_end_mapping.update({end_sp_token: tokenizer_config["tokenizer_name"]})

    def _build_extra_decoder(self, tokenizer_config: dict) -> dict[int, str]:
        """Build domain-specific tokenizers' decoder"""
        extra_decoder = {}
        sp_model = self.tokenizer_mapping[tokenizer_config["tokenizer_name"]]
        start_pos = self.vocab_size - sp_model.get_piece_size() - self.new_sp_token_offset[-1]
        extra_decoder.update(
            {i: sp_model.id_to_piece(i - start_pos) for i in range(start_pos, start_pos + sp_model.get_piece_size())}
        )
        return extra_decoder

    def _update_logical_special_tokens(self, tokenizer_config: dict) -> None:
        """Update logical special tokens which serve as special token and won't be mapped to a specific token id"""
        for begin_sp_token, end_sp_token in zip(
            tokenizer_config["auto_begin_sp_tokens"], tokenizer_config["auto_end_sp_tokens"]
        ):
            self.auto_begin_sp_tokens.append(begin_sp_token)
            self.auto_end_sp_tokens.append(end_sp_token)

            self.tokens_trie.add(begin_sp_token)
            self.tokens_trie.add(end_sp_token)

    def _update_special_tokens(self, tokenizer_config: dict):
        """Update special tokens for each modality"""
        offset = sum(self.new_sp_token_offset[1:]) + len(self.logical_auto_tokens)
        new_offset = 0
        for start_key, end_key in zip(
            list(self.extra_tokenizer_bos_keys)[offset // 2 :], list(self.extra_tokenizer_eos_keys)[offset // 2 :]
        ):
            self.tokens_trie.add(start_key)

            if start_key not in tokenizer_config["auto_begin_sp_tokens"]:
                self._added_tokens_encoder.update({start_key: self.vocab_size + new_offset})
                self._added_tokens_decoder.update(
                    {
                        self.vocab_size + new_offset: AddedToken(
                            content=start_key,
                            lstrip=False,
                            normalized=False,
                            rstrip=False,
                            single_word=False,
                            special=True,
                        )
                    }
                )
                self.tokens_trie.add(start_key)
                new_offset += 1

            if end_key not in tokenizer_config["auto_end_sp_tokens"]:
                self._added_tokens_encoder.update({end_key: self.vocab_size + new_offset})
                self._added_tokens_decoder.update(
                    {
                        self.vocab_size + new_offset: AddedToken(
                            content=end_key,
                            lstrip=False,
                            normalized=False,
                            rstrip=False,
                            single_word=False,
                            special=True,
                        )
                    }
                )
                self.tokens_trie.add(end_key)
                new_offset += 1
        self.new_sp_token_offset.append(new_offset)

    @lru_cache(maxsize=None)  # May cause memory leak
    def _extra_tokenizer_offset(self, tokenizer_key) -> int:
        offset = 0
        for index, (tokenizer_name, tokenizer) in enumerate(self.tokenizer_mapping.items()):
            if tokenizer_name == tokenizer_key:
                break
            else:
                offset += len(tokenizer) + self.new_sp_token_offset[index]
        return offset

    def _pop_logical_sp_token(self, extra_tokenizer_stack: list, mapping_name: str) -> None:
        """Switch tokenizer when it comes to an end sp token"""
        extra_tokenizer_end_mapping = extra_tokenizer_stack.pop()
        if extra_tokenizer_end_mapping != self.extra_tokenizer_end_mapping[mapping_name]:
            logger.warning_once(
                f"Encounter incorrect nesting of extra tokenizer: {self.extra_tokenizer_end_mapping[mapping_name]} and {extra_tokenizer_end_mapping}"
            )
            logger.warning_once("This may lead to unexpected behaviour of the tokenizer, please check your input.")

    def tokenize(self, text: TextInput, **kwargs) -> list[str]:
        """
        Converts a string into a sequence of tokens, using the tokenizer.

        It will switch to domain-specific tokenizer once encountering extra/logical sp tokens.

        Args:
            text: TextInput
        """
        split_special_tokens = kwargs.pop("split_special_tokens", self.split_special_tokens)

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase. Might be super slow as well?
            escaped_special_toks = [re.escape(s_tok) for s_tok in (self.all_special_tokens)]
            escaped_special_toks += [
                re.escape(s_tok.content)
                for s_tok in (self._added_tokens_decoder.values())
                if not s_tok.special and s_tok.normalized
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        if split_special_tokens:
            no_split_token = []
            tokens = [text]
        else:
            no_split_token = self._added_tokens_encoder.keys()  # don't split on any of the added tokens
            # "This is something<special_token_1>  else"
            tokens = self.tokens_trie.split(text)

        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = self._added_tokens_decoder.get(self._added_tokens_encoder[token], None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                    if tok_extended.single_word and left and left[-1] != " ":
                        tokens[i - 1] += token
                        tokens[i] = ""
                    elif tok_extended.single_word and right and right[0] != " ":
                        tokens[i + 1] = token + tokens[i + 1]
                        tokens[i] = ""
                else:
                    raise ValueError(
                        f"{tok_extended} cannot be tokenized because it was not properly added"
                        f" to the tokenizer. This means that it is not an `AddedToken` but a {type(tok_extended)}"
                    )

        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []

        # Code for Auto Detect
        if self._extra_tokenizer_list is not None:
            new_tokens = []
            not_split_flag = 0
            for token in tokens:
                if not token:
                    continue
                if token in no_split_token or token in self.protect_sp_tokens:
                    new_tokens.append(token)
                    if token in self.extra_tokenizer_bos_keys or token in self.protect_begin_sp_tokens:
                        not_split_flag += 1  # In case nested sp tokens
                    elif token in self.extra_tokenizer_eos_keys or token in self.protect_end_sp_tokens:
                        not_split_flag = max(0, not_split_flag - 1)
                else:
                    if not_split_flag:
                        new_tokens.append(token)
                    else:
                        for check_module in self.check_module_list:
                            token = check_module.re_split(token)

                        new_tokens.extend(token)
            tokens = new_tokens

        extra_tokenizer_stack = []  # This should be a stack to handle nested extra tokenizer

        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in self.protect_sp_tokens:
                tokenized_text.extend(self._tokenize(token))
            elif token in no_split_token:
                tokenized_text.append(token)
                if token in self.extra_tokenizer_bos_keys:
                    extra_tokenizer_stack.append(self.extra_tokenizer_start_mapping[token])
                elif token in self.extra_tokenizer_eos_keys:
                    if extra_tokenizer_stack:
                        self._pop_logical_sp_token(extra_tokenizer_stack, token)
            elif token in self.auto_begin_sp_tokens:
                tokenized_text.append(token)
                extra_tokenizer_stack.append(self.extra_tokenizer_start_mapping[token])
            elif token in self.auto_end_sp_tokens:
                tokenized_text.append(token)
                if extra_tokenizer_stack:
                    self._pop_logical_sp_token(extra_tokenizer_stack, token)
            else:
                tokenized_text.extend(self._tokenize(token, extra_tokenizer_stack=extra_tokenizer_stack))

        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _add_tokens(self, new_tokens: Union[list[str], list[AddedToken]], special_tokens: bool = False) -> int:
        """
        Modified from `transformers.tokenization_utils._add_tokens`.

        This adaptation supports dynamic tokenizer length due to supplementary tokenizers (e.g., domain-specific or scientific text tokenizers).
        """
        added_tokens = 0
        if new_tokens is None:
            return added_tokens
        # TODO this is fairly slow to improve!
        current_vocab = self.get_vocab().copy()
        new_idx = max(current_vocab.values()) + 1

        for token in new_tokens:
            if not isinstance(token, (str, AddedToken)):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if str(token) == "":
                continue
            if isinstance(token, str):
                if token in self._added_tokens_encoder:
                    continue
                else:
                    # very important for fast and slow equivalence!
                    is_special = token in self.all_special_tokens or special_tokens
                    token = AddedToken(
                        token, rstrip=False, lstrip=False, normalized=not is_special, special=is_special
                    )
            elif special_tokens:
                # doing token.special=True changes the normalization! will fix in rust
                # this is important and the only reason why the AddedTokens in each class are normalized by default
                token.__setstate__({"special": True, "normalized": token.normalized})
            if token in self._added_tokens_decoder:
                continue
            if not token.special and token.normalized and getattr(self, "do_lower_case", False):
                # Normalize if requested
                token.content = token.content.lower()
            if token.content not in current_vocab:
                token_index = new_idx + added_tokens
                current_vocab[token.content] = token_index
                added_tokens += 1
                self._extra_special_tokens.append(token)
            else:
                token_index = current_vocab[token.content]
            if token.special and str(token) not in self.all_special_tokens:
                self._special_tokens_map["additional_special_tokens"].append(token)
            # the setter automatically updates the reverse map
            self._added_tokens_decoder[token_index] = token
            self._added_tokens_encoder[token.content] = token_index
            if self.verbose:
                logger.info(f"Adding {token} to the vocabulary")
        self._update_trie()
        self._update_total_vocab_size()

        if added_tokens and self.tokenizer_mapping:
            self.new_sp_token_offset.append(added_tokens)

        return added_tokens

    def _tokenize(self, text, **kwargs):
        """
        Modified from `transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._tokenize`.

        This adaptation supports domain-specific tokenizers.
        """
        extra_tokenizer_stack = kwargs.pop("extra_tokenizer_stack", False)
        if extra_tokenizer_stack:
            tokenized_text = self.tokenizer_mapping[extra_tokenizer_stack[-1]].encode(text, out_type=str)
            tokenized_id = self.tokenizer_mapping[extra_tokenizer_stack[-1]].encode(text, out_type=int)
            final_tokenized_text = []
            for text_piece, id_piece in zip(tokenized_text, tokenized_id):
                if id_piece == 0:
                    final_tokenized_text.extend(self._bpe_tokenize(text_piece))
                else:
                    final_tokenized_text.append(text_piece)
            return final_tokenized_text
        else:
            return self._bpe_tokenize(text)

    def _bpe_tokenize(self, text, **kwargs):
        text = text.replace(
            "▁", " "
        )  # This discrepancy stems from differing whitespace treatment in SentencePiece versus BPE tokenization.
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]) -> Union[int, list[int]]:
        """
        Modified from `transformers.tokenization_utils.PreTrainedTokenzier.convert_tokens_to_ids`.

        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        This adaptation supports domain-specific tokenizers.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        extra_tokenizer_stack = []

        for token in tokens:
            if token not in self.logical_auto_tokens:
                ids.append(
                    self._convert_token_to_id_with_added_voc(token, extra_tokenizer_stack=extra_tokenizer_stack)
                )
            if token in self.extra_tokenizer_bos_keys:
                extra_tokenizer_stack.append(self.extra_tokenizer_start_mapping[token])
            elif token in self.extra_tokenizer_eos_keys:
                if extra_tokenizer_stack:
                    self._pop_logical_sp_token(extra_tokenizer_stack, token)
        return ids

    def _convert_token_to_id_with_added_voc(self, token, **kwargs):
        """
        Modified from `transformers.tokenization_utils.PreTrainedTokenzier._convert_token_to_id_with_added_voc`.

        This adaptation supports domain-specific tokenizers.
        """
        if token is None:
            return None

        if token in self._added_tokens_encoder:
            return self._added_tokens_encoder[token]
        return self._convert_token_to_id(token, **kwargs)

    def _convert_token_to_id(self, token, **kwargs):
        """
        Modified from `transformers.tokenization_utils.PreTrainedTokenzier._convert_token_to_id`.

        Converts a token (str) in an id using the vocab.

        Fall back to original tokenizer once OOV.
        """
        extra_tokenizer_stack = kwargs.pop("extra_tokenizer_stack", False)
        if extra_tokenizer_stack:
            token_id = self.tokenizer_mapping[extra_tokenizer_stack[-1]].piece_to_id(token)
            if token_id == self.tokenizer_mapping[extra_tokenizer_stack[-1]].unk_id():
                return self.encoder.get(token, self.encoder.get(self._unk_token))
            else:
                return token_id + self._extra_tokenizer_offset(extra_tokenizer_stack[-1])
        else:
            return self.encoder.get(token, self.encoder.get(self._unk_token))

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = text.replace(
            "▁", "Ġ"
        )  # This discrepancy stems from differing whitespace treatment in SentencePiece versus BPE tokenization.
        text = text.replace("\n", "Ċ")
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = False,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        # `spaces_between_special_tokens` defaults to True for _decode in slow tokenizers
        # and cannot be configured elsewhere, but it should default to False for Qwen2Tokenizer
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        """
        Modified from `transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.save_vocabulary` to support saving custom extension.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        sp_model_smiles = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["sp_model_SMILES"]
        )
        sp_model_iupac = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["sp_model_IUPAC"]
        )
        sp_model_fasta = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["sp_model_FASTA"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        with open(sp_model_smiles, "wb") as f:
            f.write(self.tokenizer_mapping["tokenizer_SMILES"].serialized_model_proto())

        with open(sp_model_iupac, "wb") as f:
            f.write(self.tokenizer_mapping["tokenizer_IUPAC"].serialized_model_proto())

        with open(sp_model_fasta, "wb") as f:
            f.write(self.tokenizer_mapping["tokenizer_FASTA"].serialized_model_proto())

        return vocab_file, merge_file


__all__ = ["InternS1Tokenizer"]
