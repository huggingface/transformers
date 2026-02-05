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
"""
Tokenization classes for python tokenizers. For fast tokenizers (provided by HuggingFace's tokenizers library) see
tokenization_utils_tokenizers.py
"""

import bisect
import unicodedata
from collections import OrderedDict
from typing import Any, overload

from .tokenization_utils_base import (
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
)
from .utils import PaddingStrategy, TensorType, add_end_docstrings, logging


logger = logging.get_logger(__name__)

# Slow tokenizers are saved in a vocabulary plus three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self, *args):
        self.data = {}
        self._tokens = set()
        self._termination_char = ""
        self.update(*args)

    def update(self, *args):
        """
        Updates the Trie with new tokens provided as arguments.

        Args:
            *args: Variable number of words to be added to the Trie.
        """
        for token in tuple(*args):
            self.add(token)

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` in `self._termination_char` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.add("Hello 友達")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}

        >>> trie.add("Hello")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        ```
        """
        if not word:
            # Prevent empty string
            return

        self._tokens.add(word)
        ref = self.data
        for char in word:
            ref[char] = ref.setdefault(char, {})
            ref = ref[char]
        ref[self._termination_char] = 1

    def split(self, text: str) -> list[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string split along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS] This is a extra_id_100"]

        >>> trie.add("[CLS]")
        >>> trie.add("extra_id_1")
        >>> trie.add("extra_id_100")
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS]", " This is a ", "extra_id_100"]
        ```
        """
        # indexes are counted left of the chars index.
        # "hello", index 0, is left of h, index 1 is between h and e.
        # index 5 is right of the "o".

        # States are going to capture every possible start (indexes as above)
        # as keys, and have as values, a pointer to the position in the trie
        # where we're at. This is a partial match for now.
        # This enables to keep track of multiple matches while we're iterating
        # the string
        # If the trie contains, "blowing", and "lower" and we encounter the
        # string "blower", we need to split into ["b", "lower"].
        # This is where we need to keep track of multiple possible starts.
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = 0
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it"
                    " anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


class ExtensionsTrie(Trie):
    def __init__(self, *args):
        super().__init__(*args)

    def extensions(self, prefix: str):
        """
        Generates all extensions of a given prefix token in the Trie.

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.add("apple")
        >>> trie.add("app")
        >>> trie.add("application")
        >>> trie.extensions("app")
        ['app', 'apple', 'application']
        ```
        """
        prefix_node = self._get_node(prefix)
        ret = self._collect_tokens(prefix_node)
        return [prefix + token for token in ret]

    def _get_node(self, token: str) -> dict:
        """
        Retrieves the node corresponding to the given token in the Trie.

        Args:
            token (str): The token for which the corresponding node needs to be retrieved.

        Returns:
            dict: The node in the Trie corresponding to the given token.
        """
        node = self.data
        for char in token:
            if char not in node:
                break

            node = node[char]
        return node

    def _collect_tokens(self, node: dict) -> list:
        """
        Generates all tokens in the Trie starting from a given node.

        Args:
            node (dict): The node in the Trie from which tokens need to be generated.

        Returns:
            list: List of tokens generated from the given node.
        """
        tokens = [self._termination_char] if self._termination_char in node else []
        for token, subtrie_head in node.items():
            if token != self._termination_char:
                subtokens = self._collect_tokens(subtrie_head)
                tokens.extend([token + subtoken for subtoken in subtokens])
        return tokens


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def _insert_one_token_to_ordered_list(token_list: list[str], new_token: str):
    """
    Inserts one token to an ordered list if it does not already exist. Note: token_list must be sorted.
    """
    insertion_idx = bisect.bisect_left(token_list, new_token)
    # Checks if new_token is already in the ordered token_list
    if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
        # new_token is in token_list, don't add
        return
    else:
        token_list.insert(insertion_idx, new_token)


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PythonBackend(PreTrainedTokenizerBase):
    """
    Base class for all slow tokenizers.

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
    pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    def __init__(self, **kwargs):
        # 1. Init the parent class

        self.tokens_trie = Trie()

        # Initialize total_vocab_size early to avoid issues if get_vocab() is called early (custom tokenizers)
        self.total_vocab_size = 0

        # 2. init `_added_tokens_decoder` if child class did not
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder: dict[int, AddedToken] = {}

        # 3. if a `added_tokens_decoder` is passed, we are loading from a saved tokenizer, we overwrite
        self._added_tokens_decoder.update(kwargs.pop("added_tokens_decoder", {}))
        self._added_tokens_encoder: dict[str, int] = {k.content: v for v, k in self._added_tokens_decoder.items()}

        # 4. Token type ID configuration for dynamic mask building
        # These can be overridden by subclasses to avoid overriding create_token_type_ids_from_sequences
        self.token_type_ids_pattern = kwargs.pop("token_type_ids_pattern", "bert_style")  # "all_zeros" or "bert_style"
        self.token_type_ids_include_special_tokens = kwargs.pop("token_type_ids_include_special_tokens", True)

        # 5. Special tokens mask configuration
        # Patterns: "none", "cls_sep", "eos", "bos", "bos_eos", "cls_double_sep", "prefix_suffix"
        self.special_tokens_pattern = kwargs.pop("special_tokens_pattern", None)

        # 6. Set backend to "custom" if not already set (for direct PreTrainedTokenizer subclasses)
        if "backend" not in kwargs:
            kwargs["backend"] = "custom"

        # 7. init the parent class
        super().__init__(**kwargs)

        # 4. If some of the special tokens are not part of the vocab, we add them, at the end.
        # V5: the order of addition follows self.SPECIAL_TOKENS_ATTRIBUTES, then extra special tokens
        # Note: _add_tokens will automatically skip tokens that are already in the base vocab
        self._add_tokens(
            [token for token in self.all_special_tokens if token not in self._added_tokens_encoder],
            special_tokens=True,
        )

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def added_tokens_encoder(self) -> dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        return {k.content: v for v, k in sorted(self._added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `dict[str, int]`: The added tokens.
        """
        return dict(sorted(self._added_tokens_decoder.items(), key=lambda item: item[0]))

    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: dict[int, AddedToken | str]) -> dict[int, AddedToken]:
        # Always raise an error if string because users should define the behavior
        for index, token in value.items():
            if not isinstance(token, (str, AddedToken)) or not isinstance(index, int):
                raise TypeError(
                    f"The provided `added_tokens_decoder` has an element of type {index.__class__, token.__class__}, should be a dict of {int, AddedToken | str}"
                )

            self._added_tokens_decoder[index] = AddedToken(token) if isinstance(token, str) else token
            self._added_tokens_encoder[str(token)] = index
        self._update_total_vocab_size()

    def get_added_vocab(self) -> dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `dict[str, int]`: The added tokens.
        """
        return self._added_tokens_encoder

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens.
        """
        # Lazy evaluation: compute if not already set (e.g., during initialization)
        if self.total_vocab_size == 0:
            self._update_total_vocab_size()
        return self.total_vocab_size

    def _update_total_vocab_size(self):
        """
        Update the size of the full vocabulary with the added tokens. Counts the `keys` and not the `values` because
        otherwise if there is a hole in the vocab, we will add tokenizers at a wrong index. This operation is slow and
        is only updated when adding tokens.
        """
        self.total_vocab_size = len(self.get_vocab())

    def _add_tokens(self, new_tokens: list[str] | list[AddedToken], special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary. Special tokens are sometimes already in the
        vocab which is why they have to be handled specifically.

        Args:
            new_tokens (`list[str]`or `list[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is counted as added if it's not already in the vocabulary
                (tested by checking if the tokenizer assign the index of the `unk_token` to them). If a token is part
                of the vocabulary then we simply mark this token as an `AddedToken` which allows to control the
                stripping and normalization of this token. This is NOT possible in `tokenizers`.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            `int`: The number of tokens actually added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        added_tokens = 0
        if new_tokens is None:
            return added_tokens
        # TODO this is fairly slow to improve!
        current_vocab = self.get_vocab().copy()
        new_idx = len(current_vocab)  # only call this once, len gives the last index + 1
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
            else:
                token_index = current_vocab[token.content]

            if token.special and str(token) not in self.all_special_tokens:
                self._extra_special_tokens.append(token)
            # the setter automatically updates the reverse map
            self._added_tokens_decoder[token_index] = token
            self._added_tokens_encoder[token.content] = token_index
            if self.verbose:
                logger.info(f"Adding {token} to the vocabulary")

        self._update_trie()
        self._update_total_vocab_size()
        return added_tokens

    def _update_trie(self, unique_no_split_tokens: list[str] | None = None):
        for token in self._added_tokens_decoder.values():
            if token.content not in self.tokens_trie._tokens:
                self.tokens_trie.add(token.content)
        for token in unique_no_split_tokens or []:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def tokenize(self, text: TextInput, **kwargs) -> list[str]:
        """
        Converts a string into a sequence of tokens, using the tokenizer.

        Args:
            text: The sequence to be encoded.
            **kwargs: Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            The list of tokens.
        """
        split_special_tokens = kwargs.pop("split_special_tokens", self.split_special_tokens)
        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if split_special_tokens:
            # Don't split on any tokens - just tokenize directly
            return self._tokenize(text)

        # Split on added tokens
        tokens = self.tokens_trie.split(text)
        no_split_token = self._added_tokens_encoder.keys()

        # Handle added token properties (lstrip, rstrip, single_word)
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = self._added_tokens_decoder.get(self._added_tokens_encoder[token])
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None

                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        tokens[i + 1] = right.lstrip()
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()
                    if tok_extended.single_word:
                        if left and left[-1] != " ":
                            tokens[i - 1] += token
                            tokens[i] = ""
                        elif right and right[0] != " ":
                            tokens[i + 1] = token + tokens[i + 1]
                            tokens[i] = ""

        # Tokenize non-added tokens
        result = []
        all_special_tokens_set = set(self.all_special_tokens)
        for token in tokens:
            if not token:
                continue
            if token in no_split_token or token in all_special_tokens_set:
                result.append(token)
            else:
                result.extend(self._tokenize(token))

        return result

    def _tokenize(self, text, **kwargs):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _convert_token_to_id_with_added_voc(self, token):
        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def _encode_plus(
        self,
        text: TextInput | PreTokenizedInput | EncodedInput,
        text_pair: TextInput | PreTokenizedInput | EncodedInput | None = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: int | None = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # Detect batched inputs (list of sequences)
        is_batched = isinstance(text, (list, tuple)) and (
            (not text and not is_split_into_words)
            or (text and is_split_into_words and isinstance(text[0], (list, tuple)))
            or (text and not is_split_into_words and isinstance(text[0], (str, list, tuple)))
        )

        if is_batched:
            if text_pair is not None:
                if not isinstance(text_pair, (list, tuple)) or len(text_pair) != len(text):
                    raise ValueError("If `text` is a batch, `text_pair` must also be a batch of the same length.")
            pairs = text_pair if text_pair is not None else [None] * len(text)

            batch_outputs = {}
            for current_text, current_pair in zip(text, pairs):
                # Handle tuples/lists as sequence pairs like ("text1", "text2")
                # For is_split_into_words=True: only unpack if it's a tuple of exactly 2 sequences (pair)
                # Otherwise, treat the list as a single pretokenized sequence
                if (
                    isinstance(current_text, (list, tuple))
                    and current_text
                    and not isinstance(current_text[0], int)
                    and current_pair is None
                ):
                    # Check if this looks like a pair: tuple/list of length 2 where elements are strings or lists/tuples
                    is_pair = (
                        len(current_text) == 2
                        and (isinstance(current_text[0], str) or isinstance(current_text[0], (list, tuple)))
                        and (isinstance(current_text[1], str) or isinstance(current_text[1], (list, tuple)))
                    )
                    if is_pair:
                        current_text, current_pair = current_text
                    elif len(current_text) == 1:
                        current_text = current_text[0]
                    elif not is_split_into_words:
                        # Only raise error for non-pretokenized input
                        raise ValueError(f"Expected a pair of sequences, got {len(current_text)} sequences.")

                current_output = self._encode_plus(
                    text=current_text,
                    text_pair=current_pair,
                    add_special_tokens=add_special_tokens,
                    padding_strategy=PaddingStrategy.DO_NOT_PAD,  # we pad in batch afterward
                    truncation_strategy=truncation_strategy,
                    max_length=max_length,
                    stride=stride,
                    is_split_into_words=is_split_into_words,
                    pad_to_multiple_of=None,  # we pad in batch afterward
                    padding_side=None,  # we pad in batch afterward
                    return_tensors=None,  # We convert the whole batch to tensors at the end
                    return_token_type_ids=return_token_type_ids,
                    return_attention_mask=False,  # we pad in batch afterward
                    return_overflowing_tokens=return_overflowing_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                    return_length=return_length,
                    verbose=verbose,
                    **kwargs,
                )
                for key, value in current_output.items():
                    batch_outputs.setdefault(key, []).append(value)

            # Remove overflow-related keys before tensor conversion if return_tensors is set
            # Slow tokenizers don't support returning these as tensors
            if return_tensors and return_overflowing_tokens:
                batch_outputs.pop("overflowing_tokens", None)
                batch_outputs.pop("num_truncated_tokens", None)

            batch_outputs = self.pad(
                batch_outputs,
                padding=padding_strategy.value,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )

            return BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # Single sequence handling
        def get_input_ids(text):
            if isinstance(text, str):
                # Normal case: tokenize string
                return self.convert_tokens_to_ids(self.tokenize(text, **kwargs))
            if isinstance(text, (list, tuple)) and text:
                if isinstance(text[0], int):
                    return text
                # Pre-tokenized strings
                if isinstance(text[0], str):
                    if is_split_into_words:
                        return self.convert_tokens_to_ids(
                            [tok for word in text for tok in self.tokenize(word, **kwargs)]
                        )
                    return self.convert_tokens_to_ids(text)
            raise ValueError(f"Input must be a string, list of strings, or list of ints, got: {type(text)}")

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            padding_side=padding_side,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> tuple[str, dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            kwargs (`dict[str, Any]`, *optional*):
                Keyword arguments to use for the tokenization.

        Returns:
            `tuple[str, dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        return (text, kwargs)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequences by adding special tokens.

        This method dynamically builds inputs based on the tokenizer's `special_tokens_pattern`:
        - `"none"`: No special tokens
        - `"cls_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
        - `"eos"`: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
        - `"bos"`: [BOS] seq0 or [BOS] seq0 [BOS] seq1
        - `"bos_eos"`: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
        - `"cls_double_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] [SEP] seq1 [SEP]
        - `"prefix_suffix"`: `<prefix_tokens> seq0 [seq1] <suffix_tokens>` (custom prefix/suffix stored on the tokenizer)

        Args:
            token_ids_0 (`list[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of input IDs with the appropriate special tokens.
        """
        if self.special_tokens_pattern == "cls_sep":
            # [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
            if self.cls_token_id is None and self.sep_token_id is None:
                raise ValueError(
                    "Cannot add special tokens following 'cls_sep' pattern because one or several special tokens "
                    f"are not defined (cls_token_id={self.cls_token_id}; sep_token_id={self.sep_token_id})"
                    "Set the required special tokens in tokenizer or update `tokenizer.special_tokens_pattern`"
                )
            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

        elif self.special_tokens_pattern == "eos":
            # seq0 [EOS] or seq0 [EOS] seq1 [EOS]
            if self.eos_token_id is None:
                raise ValueError(
                    "Cannot add special tokens following 'eos' pattern because eos token is not defined "
                    f"(eos_token_id={self.eos_token_id})."
                    "Set the required special tokens in tokenizer or update `tokenizer.special_tokens_pattern`"
                )
            if token_ids_1 is None:
                return token_ids_0 + [self.eos_token_id]
            return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

        elif self.special_tokens_pattern == "bos":
            # [BOS] seq0 or [BOS] seq0 [BOS] seq1
            if self.bos_token_id is None:
                raise ValueError(
                    "Cannot add special tokens following 'bos' pattern because bos token is not defined "
                    f"(bos_token_id={self.bos_token_id})."
                    "Set the required special tokens in tokenizer or update `tokenizer.special_tokens_pattern`"
                )
            if token_ids_1 is None:
                return [self.bos_token_id] + token_ids_0
            return [self.bos_token_id] + token_ids_0 + [self.bos_token_id] + token_ids_1

        elif self.special_tokens_pattern == "bos_eos":
            # [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
            if self.bos_token_id is None and self.eos_token_id is None:
                raise ValueError(
                    "Cannot add special tokens following 'bos_eos' pattern because one or several special tokens "
                    f"are not defined (bos_token_id={self.bos_token_id}; eos_token_id={self.eos_token_id})"
                    "Set the required special tokens in tokenizer or update `tokenizer.special_tokens_pattern`"
                )
                return token_ids_0 if token_ids_1 is None else token_ids_0 + token_ids_1

            if token_ids_1 is None:
                return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

        elif self.special_tokens_pattern == "cls_double_sep":
            # [CLS] seq0 [SEP] or [CLS] seq0 [SEP] [SEP] seq1 [SEP]
            if self.cls_token_id is None and self.sep_token_id is None:
                raise ValueError(
                    "Cannot add special tokens following 'cls_double_sep' pattern because one or several special tokens "
                    f"are not defined (cls_token_id={self.cls_token_id}; sep_token_id={self.sep_token_id})"
                    "Set the required special tokens in tokenizer or update `tokenizer.special_tokens_pattern`"
                )
            if token_ids_1 is None:
                return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            return (
                [self.cls_token_id]
                + token_ids_0
                + [self.sep_token_id, self.sep_token_id]
                + token_ids_1
                + [self.sep_token_id]
            )

        elif self.special_tokens_pattern == "prefix_suffix":
            prefix_tokens = getattr(self, "prefix_tokens", [])
            suffix_tokens = getattr(self, "suffix_tokens", [])
            if token_ids_1 is None:
                return prefix_tokens + token_ids_0 + suffix_tokens
            return prefix_tokens + token_ids_0 + token_ids_1 + suffix_tokens

        else:  # "none" or any other value
            # No special tokens
            if token_ids_1 is None:
                return token_ids_0
            return token_ids_0 + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: list, token_ids_1: list | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        This method dynamically builds the special tokens mask based on the tokenizer's `special_tokens_pattern`:
        - `"none"`: No special tokens (default, returns all 0s)
        - `"cls_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
        - `"eos"`: seq0 [EOS] or seq0 [EOS] seq1 [EOS]
        - `"bos"`: [BOS] seq0 or [BOS] seq0 [BOS] seq1
        - `"bos_eos"`: [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
        - `"cls_double_sep"`: [CLS] seq0 [SEP] or [CLS] seq0 [SEP] [SEP] seq1 [SEP]
        - `"prefix_suffix"`: `<prefix_tokens> seq0 [seq1] <suffix_tokens>`

        Args:
            token_ids_0 (`list[int]`):
                List of ids of the first sequence.
            token_ids_1 (`list[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if self.special_tokens_pattern == "cls_sep":
            # [CLS] seq0 [SEP] or [CLS] seq0 [SEP] seq1 [SEP]
            if token_ids_1 is None:
                return [1] + ([0] * len(token_ids_0)) + [1]
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

        elif self.special_tokens_pattern == "eos":
            # seq0 [EOS] or seq0 [EOS] seq1 [EOS]
            if token_ids_1 is None:
                return ([0] * len(token_ids_0)) + [1]
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

        elif self.special_tokens_pattern == "bos":
            # [BOS] seq0 or [BOS] seq0 [BOS] seq1
            if token_ids_1 is None:
                return [1] + ([0] * len(token_ids_0))
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

        elif self.special_tokens_pattern == "bos_eos":
            # [BOS] seq0 [EOS] or [BOS] seq0 [EOS] seq1 [EOS]
            if token_ids_1 is None:
                return [1] + ([0] * len(token_ids_0)) + [1]
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

        elif self.special_tokens_pattern == "cls_double_sep":
            # [CLS] seq0 [SEP] or [CLS] seq0 [SEP] [SEP] seq1 [SEP]
            if token_ids_1 is None:
                return [1] + ([0] * len(token_ids_0)) + [1]
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

        elif self.special_tokens_pattern == "prefix_suffix":
            prefix_len = len(getattr(self, "prefix_tokens", []))
            suffix_len = len(getattr(self, "suffix_tokens", []))
            mask = [1] * prefix_len + ([0] * len(token_ids_0))
            if token_ids_1 is not None:
                mask += [0] * len(token_ids_1)
            mask += [1] * suffix_len
            return mask

        else:
            return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: list[int], skip_special_tokens: bool = False) -> list[str]: ...

    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `list[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `list[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            return (
                self._added_tokens_decoder[ids].content
                if ids in self._added_tokens_decoder
                else self._convert_id_to_token(ids)
            )

        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(
                self._added_tokens_decoder[index].content
                if index in self._added_tokens_decoder
                else self._convert_id_to_token(index)
            )
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> str:
        """Decode token ids to string."""
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        if isinstance(filtered_tokens, str):
            filtered_tokens = [filtered_tokens]

        text = self.convert_tokens_to_string(filtered_tokens)

        # Apply tokenizer-specific cleanup if available and requested
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        return text

    def prepare_for_model(
        self,
        ids: list[int],
        pair_ids: list[int] | None = None,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = False,
        max_length: int | None = None,
        stride: int = 0,
        pad_to_multiple_of: int | None = None,
        padding_side: str | None = None,
        return_tensors: str | TensorType | None = None,
        return_token_type_ids: bool | None = None,
        return_attention_mask: bool | None = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input ids so it can be used by the model. Adds special tokens, truncates, and pads.

        Args:
            ids: Tokenized input ids of the first sequence.
            pair_ids: Tokenized input ids of the second sequence (optional).
        """
        # Get padding/truncation strategies
        padding_strategy, truncation_strategy, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # Validation
        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        # Truncation
        pair = pair_ids is not None
        num_special = self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0
        total_len = len(ids) + len(pair_ids or []) + num_special

        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + (pair_ids if pair_ids else [])
            token_type_ids = [0] * len(sequence)

        # Build output
        encoded_inputs = {"input_ids": sequence}
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = (
                self.get_special_tokens_mask(ids, pair_ids) if add_special_tokens else [0] * len(sequence)
            )
        if return_overflowing_tokens and not return_tensors and overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length if max_length else 0

        # Check sequence length and warn if needed
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Pad
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        return BatchEncoding(encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis)

    def truncate_sequences(
        self,
        ids: list[int],
        pair_ids: list[int] | None = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: str | TruncationStrategy = "longest_first",
        stride: int = 0,
    ) -> tuple[list[int], list[int], list[int]]:
        """Truncates sequences according to the specified strategy."""
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []

        # ONLY_FIRST or LONGEST_FIRST with single sequence
        if truncation_strategy == TruncationStrategy.ONLY_FIRST or (
            truncation_strategy == TruncationStrategy.LONGEST_FIRST and pair_ids is None
        ):
            window_len = min(len(ids), stride + num_tokens_to_remove)
            if self.truncation_side == "left":
                overflowing_tokens = ids[:window_len]
                ids = ids[num_tokens_to_remove:]
            else:
                overflowing_tokens = ids[-window_len:]
                ids = ids[:-num_tokens_to_remove]

        # LONGEST_FIRST with pair
        elif truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            logger.warning(
                "Be aware, overflowing tokens are not returned for the setting you have chosen,"
                f" i.e. sequence pairs with the '{TruncationStrategy.LONGEST_FIRST.value}' "
                "truncation strategy. So the returned list will always be empty even if some "
                "tokens have been removed."
            )
            len_ids, len_pair = len(ids), len(pair_ids) if pair_ids else 0
            first_remove = min(abs(len_pair - len_ids), num_tokens_to_remove)
            second_remove = num_tokens_to_remove - first_remove

            if len_ids > len_pair:
                ids_to_move = first_remove + second_remove // 2
                pair_ids_to_move = second_remove - second_remove // 2
            else:
                ids_to_move = second_remove // 2
                pair_ids_to_move = first_remove + second_remove - (second_remove // 2)

            if self.truncation_side == "right":
                ids = ids[:-ids_to_move] if ids_to_move > 0 else ids
                pair_ids = pair_ids[:-pair_ids_to_move] if pair_ids and pair_ids_to_move > 0 else pair_ids
            else:
                ids = ids[ids_to_move:]
                pair_ids = pair_ids[pair_ids_to_move:] if pair_ids else None

        # ONLY_SECOND
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and pair_ids:
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            if self.truncation_side == "right":
                overflowing_tokens = pair_ids[-window_len:]
                pair_ids = pair_ids[:-num_tokens_to_remove]
            else:
                overflowing_tokens = pair_ids[:window_len]
                pair_ids = pair_ids[num_tokens_to_remove:]

        return ids, pair_ids, overflowing_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        This method dynamically builds the token type IDs based on the tokenizer's configuration attributes:
        - `token_type_ids_pattern`: Pattern to use ("all_zeros" or "bert_style")
        - `token_type_ids_include_special_tokens`: Whether to account for special tokens in length calculation

        Args:
            token_ids_0 (`list[int]`):
                List of IDs.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: Token type IDs according to the configured pattern.

        Examples:
            ```python
            # All zeros pattern (default, used by RoBERTa, BART, etc.)
            tokenizer.token_type_ids_pattern = "all_zeros"
            # Returns: [0, 0, 0, ...] for both sequences

            # BERT-style pattern (first sequence gets 0s, second gets 1s)
            tokenizer.token_type_ids_pattern = "bert_style"
            # Returns: [0, 0, 0, ..., 1, 1, 1, ...] for sequence pairs
            ```
        """
        # Calculate lengths - account for special tokens if configured
        if self.token_type_ids_include_special_tokens:
            # Build the full sequence to get accurate length
            if token_ids_1 is None:
                sequence = self.build_inputs_with_special_tokens(token_ids_0)
                seq0_len = len(sequence)
                seq1_len = 0
            else:
                full_sequence = self.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
                # Approximate split - this works for most tokenizers
                # For more complex cases, subclasses should still override
                seq0_with_special = self.build_inputs_with_special_tokens(token_ids_0)
                seq0_len = len(seq0_with_special)
                seq1_len = len(full_sequence) - seq0_len
        else:
            # Use raw token lengths
            seq0_len = len(token_ids_0)
            seq1_len = len(token_ids_1) if token_ids_1 is not None else 0

        # Build token type IDs based on pattern
        if self.special_tokens_pattern == "prefix_suffix":
            total_len = len(getattr(self, "prefix_tokens", [])) + len(token_ids_0)
            if token_ids_1 is not None:
                total_len += len(token_ids_1)
            total_len += len(getattr(self, "suffix_tokens", []))
            return [0] * total_len

        if self.token_type_ids_pattern == "bert_style" and token_ids_1 is not None:
            # BERT-style: first sequence gets 0s, second sequence gets 1s
            return [0] * seq0_len + [1] * seq1_len
        else:
            # All zeros pattern (default): everything gets 0s
            return [0] * (seq0_len + seq1_len)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        """
        Default implementation for common vocabulary saving patterns.
        Saves self.encoder/self.vocab as JSON, optionally with self.bpe_ranks as merges.
        Returns empty tuple if no vocabulary exists.

        Override this method if your tokenizer needs custom saving logic (e.g., SentencePiece models,
        multiple vocabulary files, or special file formats).

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `tuple[str, ...]`: Paths to the files saved, or empty tuple if no files saved.
        """
        import json
        import os

        vocab_attr = getattr(self, "encoder", None) or getattr(self, "vocab", None)
        if vocab_attr is None:
            return ()

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ()

        vocab_files_names = getattr(self, "vocab_files_names", {})
        prefix = f"{filename_prefix}-" if filename_prefix else ""

        # Save vocabulary
        vocab_file = os.path.join(save_directory, prefix + vocab_files_names.get("vocab_file", "vocab.json"))
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(vocab_attr, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # Save BPE merges if present
        bpe_ranks = getattr(self, "bpe_ranks", None)
        if bpe_ranks is None:
            return (vocab_file,)

        merge_file = os.path.join(save_directory, prefix + vocab_files_names.get("merges_file", "merges.txt"))
        with open(merge_file, "w", encoding="utf-8") as writer:
            if getattr(self, "add_bpe_version_header", False):
                writer.write("#version: 0.2\n")

            index = 0
            for bpe_tokens, token_index in sorted(bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return (vocab_file, merge_file)


# Backward compatibility alias
PreTrainedTokenizer = PythonBackend
