# Copyright 2026 The HuggingFace Inc. team.
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
Serves the 0.x `tokenizers.Tokenizer` API, which [`TokenizersBackend`] and the model tokenizers are written against,
from tokenizers>=1.

tokenizers 1.x splits the 0.x `Tokenizer` into an editable spec (`Tokenizer`) and the `CompiledTokenizer` built from
it, takes padding and truncation per call, and returns encodings holding only ids, type ids and an attention mask. The
rest of a 0.x encoding (special tokens mask, sequence and word ids, overflowing windows) is rebuilt here from the
post-processor template, which tokenizers 1.x keeps as plain token ids. Offsets can't be rebuilt.
"""

import json
from functools import cached_property

from tokenizers import Padding, Tokenizer, Truncation


NO_OFFSETS = "Offsets are not available with tokenizers>=1 yet."
NO_WORD_IDS = "Word ids are only available with tokenizers>=1 for pretokenized inputs (`is_split_into_words=True`)."
_COMPONENTS = frozenset({"normalizer", "pre_tokenizer", "post_processor", "decoder", "model"})
_DEFAULT_TEMPLATE = {"single": [{"seq": "A"}], "pair": [{"seq": "A"}, {"seq": "B", "type_id": 1}]}


class Encoding:
    """
    The 0.x `tokenizers.Encoding` API, without offsets. An encoding from the native path rebuilds its special tokens
    mask, sequence ids and overflowing windows in Python on first access, by encoding its input again.
    """

    def __init__(
        self,
        ids,
        type_ids,
        attention_mask,
        tokenizer,
        special_tokens_mask=None,
        sequence_ids=None,
        word_ids=None,
        overflowing=(),
        source=None,
    ):
        self.ids = ids
        self.type_ids = type_ids
        self.attention_mask = attention_mask
        self._tokenizer = tokenizer
        self._special_tokens_mask = special_tokens_mask
        self._sequence_ids = sequence_ids
        self._word_ids = word_ids
        self._overflowing = list(overflowing)
        self._source = source

    def __len__(self) -> int:
        return len(self.ids)

    def _resolved(self) -> "Encoding":
        if self._source is not None:
            detailed = self._tokenizer._assemble(*self._source)[0]
            self._source = None
            self._special_tokens_mask = detailed._special_tokens_mask
            self._sequence_ids = detailed._sequence_ids
            self._overflowing = detailed._overflowing
        return self

    @cached_property
    def tokens(self) -> list[str]:
        return self._tokenizer._compiled.decode_tokens(self.ids, skip_special_tokens=False)

    @property
    def special_tokens_mask(self) -> list[int]:
        return self._resolved()._special_tokens_mask

    @property
    def sequence_ids(self) -> list[int | None]:
        return self._resolved()._sequence_ids

    @property
    def word_ids(self) -> list[int | None]:
        if self._word_ids is None:
            raise NotImplementedError(NO_WORD_IDS)
        return self._word_ids

    @property
    def overflowing(self) -> list["Encoding"]:
        return self._resolved()._overflowing

    @property
    def n_sequences(self) -> int:
        return len({index for index in self.sequence_ids if index is not None})

    @property
    def offsets(self):
        raise NotImplementedError(NO_OFFSETS)

    def token_to_sequence(self, token_index: int) -> int | None:
        return self.sequence_ids[token_index]

    def token_to_word(self, token_index: int) -> int | None:
        return self.word_ids[token_index]

    def word_to_tokens(self, word_index: int, sequence_index: int = 0) -> tuple[int, int] | None:
        tokens = [
            index
            for index, (word, sequence) in enumerate(zip(self.word_ids, self.sequence_ids))
            if word == word_index and sequence == sequence_index
        ]
        return (tokens[0], tokens[-1] + 1) if tokens else None

    def token_to_chars(self, token_index: int):
        raise NotImplementedError(NO_OFFSETS)

    def char_to_token(self, char_pos: int, sequence_index: int = 0):
        raise NotImplementedError(NO_OFFSETS)

    def word_to_chars(self, word_index: int, sequence_index: int = 0):
        raise NotImplementedError(NO_OFFSETS)

    def char_to_word(self, char_pos: int, sequence_index: int = 0):
        raise NotImplementedError(NO_OFFSETS)


def _windows(length: int, keep: int, stride: int, direction: str) -> list[tuple[int, int]]:
    """
    The `(start, stop)` ranges 0.x `Encoding.truncate` cuts a sequence of `length` tokens into: the first range is
    kept, the others overflow, and consecutive ranges share `stride` tokens.
    """
    if keep >= length:
        return [(0, length)]
    if keep == 0:
        return [(0, 0), (0, length)]
    if stride >= keep:
        raise ValueError(f"`stride` must be strictly less than the {keep} tokens left after truncation")
    ranges = []
    if direction == "right":
        for start in range(0, length, keep - stride):
            ranges.append((start, min(start + keep, length)))
            if ranges[-1][1] == length:
                break
    else:
        for stop in range(length, 0, stride - keep):
            ranges.append((max(stop - keep, 0), stop))
            if ranges[-1][0] == 0:
                break
    return ranges


def _truncation_lengths(lengths: list[int], truncation: dict | None, num_special_tokens: int) -> list[int]:
    """How many tokens of each sequence 0.x and 1.x truncation keep."""
    if truncation is None:
        return lengths
    budget = max(truncation["max_length"] - num_special_tokens, 0)
    excess = sum(lengths) - budget
    if excess <= 0:
        return lengths
    strategy = truncation["strategy"]
    if strategy == "longest_first":
        if len(lengths) == 1:
            return [budget]
        n1, n2 = lengths
        swap = n1 > n2
        if swap:
            n1, n2 = n2, n1
        n2 = n1 if n1 > budget else max(n1, budget - n1)
        if n1 + n2 > budget:
            n1 = budget // 2
            n2 = n1 + budget % 2
        return [n2, n1] if swap else [n1, n2]
    target = 0 if strategy == "only_first" else 1
    if target >= len(lengths):
        raise ValueError("Truncation error: Second sequence not provided")
    if lengths[target] <= excess:
        raise ValueError("Truncation error: Sequence to truncate too short to respect the provided max_length")
    return [length - excess if index == target else length for index, length in enumerate(lengths)]


class V0TokenizerFacade:
    """
    The 0.x `tokenizers.Tokenizer` API over a tokenizers>=1 spec and the `CompiledTokenizer` built from it.

    Component writes and added tokens go to the spec and drop the compiled tokenizer, which is rebuilt on next use.
    Padding and truncation are kept here, as 0.x dicts, and passed to every encode call.
    """

    def __init__(self, spec: Tokenizer):
        self._spec = spec
        self.encode_special_tokens = False
        truncation, padding = spec.truncation, spec.padding
        self._truncation = truncation and {
            "max_length": truncation.max_length,
            "stride": 0,
            "strategy": truncation.strategy,
            "direction": truncation.direction,
        }
        self._padding = padding and {
            "length": padding.length,
            "pad_to_multiple_of": padding.pad_to_multiple_of,
            "pad_id": padding.pad_id,
            "pad_type_id": padding.pad_type_id,
            "pad_token": padding.pad_token,
            "direction": padding.direction,
        }

    def __getattr__(self, name):
        if name in _COMPONENTS:
            return getattr(self._spec, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in _COMPONENTS:
            setattr(self._spec, name, value)
            self._recompile()
        else:
            super().__setattr__(name, value)

    @cached_property
    def _compiled(self):
        return self._spec.compile()

    def _recompile(self):
        self.__dict__.pop("_compiled", None)

    @property
    def truncation(self) -> dict | None:
        return self._truncation and dict(self._truncation)

    @property
    def padding(self) -> dict | None:
        return self._padding and dict(self._padding)

    def enable_truncation(self, max_length, stride=0, strategy="longest_first", direction="right"):
        self._truncation = {"max_length": max_length, "stride": stride, "strategy": strategy, "direction": direction}
        self._spec.truncation = Truncation(max_length, strategy=strategy, direction=direction)

    def no_truncation(self):
        self._truncation = self._spec.truncation = None

    def enable_padding(
        self, direction="right", pad_id=0, pad_type_id=0, pad_token="[PAD]", length=None, pad_to_multiple_of=None
    ):
        self._padding = {
            "length": length,
            "pad_to_multiple_of": pad_to_multiple_of,
            "pad_id": pad_id,
            "pad_type_id": pad_type_id,
            "pad_token": pad_token,
            "direction": direction,
        }
        self._spec.padding = Padding(**self._padding)

    def no_padding(self):
        self._padding = self._spec.padding = None

    def _template(self) -> dict:
        post_processor = self._spec.post_processor
        return _DEFAULT_TEMPLATE if post_processor is None else json.loads(post_processor.__getstate__())

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        pieces = self._template()["pair" if is_pair else "single"]
        return sum(len(piece["ids"]) for piece in pieces if "ids" in piece)

    def get_vocab(self, with_added_tokens=True) -> dict[str, int]:
        return self._compiled.get_vocab(with_added_tokens=with_added_tokens)

    def get_vocab_size(self, with_added_tokens=True) -> int:
        return self._compiled.get_vocab_size(with_added_tokens=with_added_tokens)

    def get_added_tokens_decoder(self):
        return self._compiled.added_tokens

    def token_to_id(self, token: str) -> int | None:
        return self._compiled.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        return self._compiled.id_to_token(id)

    def add_tokens(self, tokens) -> int:
        added = self._spec.add_tokens(tokens)
        self._recompile()
        return added

    def add_special_tokens(self, tokens) -> int:
        added = self._spec.add_special_tokens(tokens)
        self._recompile()
        return added

    def decode(self, ids, skip_special_tokens=True) -> str:
        return self._compiled.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def decode_batch(self, sequences, skip_special_tokens=True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in sequences]

    def save(self, path, pretty=True):
        self._spec.save(path, pretty=pretty)

    def to_str(self, pretty=False) -> str:
        return self._spec.to_str(pretty=pretty)

    def encode(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True) -> Encoding:
        item = sequence if pair is None else (sequence, pair)
        return self.encode_batch([item], is_pretokenized=is_pretokenized, add_special_tokens=add_special_tokens)[0]

    def encode_batch(self, input, is_pretokenized=False, add_special_tokens=True) -> list[Encoding]:
        inputs = list(input)
        if is_pretokenized:
            return self._assemble(
                inputs, add_special_tokens, True, self.encode_special_tokens, self.truncation, self.padding
            )

        truncation, padding = self.truncation, self.padding
        encodings = self._compiled.encode_batch(
            inputs,
            add_special_tokens=add_special_tokens,
            encode_special_tokens=self.encode_special_tokens,
            padding=padding and Padding(**padding),
            truncation=truncation
            and Truncation(
                truncation["max_length"], strategy=truncation["strategy"], direction=truncation["direction"]
            ),
        )
        return [
            Encoding(
                encoding.ids,
                encoding.type_ids,
                encoding.attention_mask,
                self,
                source=(
                    [item],
                    add_special_tokens,
                    False,
                    self.encode_special_tokens,
                    truncation,
                    padding and {**padding, "length": len(encoding.ids), "pad_to_multiple_of": None},
                ),
            )
            for item, encoding in zip(inputs, encodings)
        ]

    def _assemble(
        self, inputs, add_special_tokens, is_pretokenized, encode_special_tokens, truncation, padding
    ) -> list[Encoding]:
        """
        Encodes `inputs` the way 0.x does, in Python: each sequence (or each word, when pretokenized) without special
        tokens, then truncation into overflowing windows, the post-processor template, and padding.
        """
        items = [item if isinstance(item, tuple) else (item,) for item in inputs]
        texts = [text for item in items for sequence in item for text in (sequence if is_pretokenized else [sequence])]
        encoded = iter(
            self._compiled.encode_batch(
                texts,
                add_special_tokens=False,
                encode_special_tokens=encode_special_tokens,
                padding=None,
                truncation=None,
            )
            if texts
            else []
        )

        def take(sequence) -> tuple[list[int], list[int] | None]:
            if not is_pretokenized:
                return next(encoded).ids, None
            ids, word_ids = [], []
            for word_index in range(len(sequence)):
                word = next(encoded).ids
                ids += word
                word_ids += [word_index] * len(word)
            return ids, word_ids

        template = self._template()
        encodings = [
            self._post_process([take(sequence) for sequence in item], template, add_special_tokens, truncation)
            for item in items
        ]
        if padding is not None:
            length = padding["length"] or max((len(encoding) for encoding in encodings), default=0)
            if (multiple := padding["pad_to_multiple_of"]) and length % multiple:
                length += multiple - length % multiple
            for encoding in encodings:
                for window in (encoding, *encoding._overflowing):
                    self._pad(window, length, padding)
        return encodings

    def _post_process(self, sequences, template, add_special_tokens, truncation) -> Encoding:
        pieces = template["pair" if len(sequences) == 2 else "single"]
        if not add_special_tokens:
            pieces = [piece for piece in pieces if "seq" in piece]
        num_special_tokens = sum(len(piece["ids"]) for piece in pieces if "ids" in piece)
        lengths = _truncation_lengths([len(ids) for ids, _ in sequences], truncation, num_special_tokens)
        stride, direction = (truncation["stride"], truncation["direction"]) if truncation else (0, "right")
        windows = [_windows(len(ids), keep, stride, direction) for (ids, _), keep in zip(sequences, lengths)]

        # The order 0.x `Encoding.merge_with` lists the overflowing windows of a pair in
        if len(windows) == 1:
            combinations = [(window,) for window in windows[0]]
        else:
            first, second = windows
            combinations = [(first[0], second[0])]
            combinations += [(f, s) for f in first[1:] for s in second]
            combinations += [(first[0], s) for s in second[1:]]

        main, *overflowing = [self._apply_template(sequences, ranges, pieces) for ranges in combinations]
        main._overflowing = overflowing
        return main

    def _apply_template(self, sequences, ranges, pieces) -> Encoding:
        ids, type_ids, special_tokens_mask, sequence_ids, word_ids = [], [], [], [], []
        has_word_ids = sequences[0][1] is not None
        for piece in pieces:
            type_id = piece.get("type_id", 0)
            if "ids" in piece:
                ids += piece["ids"]
                special_tokens_mask += [1] * len(piece["ids"])
                sequence_ids += [None] * len(piece["ids"])
                word_ids += [None] * len(piece["ids"])
            else:
                index = "AB".index(piece["seq"])
                (sequence, words), (start, stop) = sequences[index], ranges[index]
                ids += sequence[start:stop]
                special_tokens_mask += [0] * (stop - start)
                sequence_ids += [index] * (stop - start)
                word_ids += words[start:stop] if has_word_ids else []
            type_ids += [type_id] * (len(ids) - len(type_ids))
        return Encoding(
            ids,
            type_ids,
            [1] * len(ids),
            self,
            special_tokens_mask=special_tokens_mask,
            sequence_ids=sequence_ids,
            word_ids=word_ids if has_word_ids else None,
        )

    @staticmethod
    def _pad(encoding: Encoding, length: int, padding: dict):
        missing = length - len(encoding.ids)
        if missing <= 0:
            return
        fields = {
            "ids": [padding["pad_id"]] * missing,
            "type_ids": [padding["pad_type_id"]] * missing,
            "attention_mask": [0] * missing,
            "_special_tokens_mask": [1] * missing,
            "_sequence_ids": [None] * missing,
        }
        if encoding._word_ids is not None:
            fields["_word_ids"] = [None] * missing
        for name, pad in fields.items():
            value = getattr(encoding, name)
            setattr(encoding, name, pad + value if padding["direction"] == "left" else value + pad)
        encoding.__dict__.pop("tokens", None)
