# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Decoding of generated token ids for response parsing, so that control tokens the model spelled out with ordinary
tokens can't act as delimiters."""

from __future__ import annotations

from functools import lru_cache
from itertools import groupby
from typing import Any

from tokenizers.decoders import DecodeStream

from ..chat_template_utils import _get_control_token_patterns


# Inserted between the characters of spelled-out control tokens so no delimiter can match them, and stripped from
# the parser's output. A noncharacter, which Unicode reserves for internal use, so model output doesn't contain it.
MARKER = "\ufdd2"


@lru_cache
def _control_token_prefixes(control_tokens: tuple[str, ...]) -> frozenset[str]:
    """Every proper prefix of the control tokens, e.g. `<`, `<|`, ..., `<|end|` for `<|end|>`."""
    return frozenset(token[:length] for token in control_tokens for length in range(1, len(token)))


def strip_markers(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(MARKER, "")
    if isinstance(value, list):
        return [strip_markers(item) for item in value]
    if isinstance(value, dict):
        return {strip_markers(key): strip_markers(item) for key, item in value.items()}
    return value


class TokenIdDecoder:
    """Incrementally decodes generated token ids into the text the response parser scans.

    A control token generated as its id decodes to its usual spelling, as in text mode. In the text decoded from
    other tokens, control-token spellings are defused with `MARKER` between their characters, and text that could
    still become one is held back until later ids decide. Control tokens are the ones
    `apply_chat_template(..., sanitize_control_tokens=True)` protects.
    """

    def __init__(self, tokenizer):
        # A tokenizer can have no chat template, one, or several named ones (e.g. "default" and "tool_use")
        chat_template = tokenizer.chat_template
        templates = list(chat_template.values()) if isinstance(chat_template, dict) else [chat_template or ""]
        control_tokens = tuple(
            sorted({token for template in templates for token in tokenizer._get_control_tokens(template)})
        )
        self._control_token_ids = {
            token_id: added_token.content
            for token_id, added_token in tokenizer.added_tokens_decoder.items()
            if added_token.content in control_tokens
        }
        self._control_tokens_re, _ = _get_control_token_patterns(control_tokens)
        self._prefixes = _control_token_prefixes(control_tokens)
        self._longest_prefix = max(map(len, self._prefixes), default=0)
        self._backend = tokenizer.backend_tokenizer
        self._stream = DecodeStream(skip_special_tokens=False)
        self._pending: list[int] = []  # ids `DecodeStream` holds back as an incomplete character
        self._tail = ""  # text held back as a possible control-token spelling

    def feed(self, token_ids: list[int]) -> str:
        """Decode the next token ids, and return the text they add for the parser. It can be empty while text is held
        back, either as an incomplete character or as the possible start of a spelled-out control token."""
        new_text = []
        for is_control, run in groupby(token_ids, self._control_token_ids.__contains__):
            if is_control:
                for token_id in run:
                    spelling = self._control_token_ids[token_id]
                    # Decoding a control token also releases any incomplete character held back before it
                    decoded = self._stream.step(self._backend, token_id) or ""
                    new_text.append(self._defuse(decoded.removesuffix(spelling), final=True) + spelling)
                self._pending = []
                continue
            run = list(run)
            decoded = self._stream.step(self._backend, run)
            if decoded is None:  # the run ends in the middle of a multi-byte character
                self._pending += run
            else:
                self._pending = []
                new_text.append(self._defuse(decoded))
        return "".join(new_text)

    def flush(self, end_of_stream: bool = True) -> str:
        """Emit the text held back as a possible control-token spelling and, at the end of the stream, any
        incomplete character too."""
        text = ""
        if end_of_stream and self._pending:
            text, self._pending = self._backend.decode(self._pending, skip_special_tokens=False), []
        return self._defuse(text, final=True)

    def _defuse(self, text: str, final: bool = False) -> str:
        """Defuse control-token spellings in `text`, holding back a tail that could still become one unless `final`."""
        text, self._tail = self._tail + text, ""
        if not final:
            held = next(
                (
                    length
                    for length in range(min(self._longest_prefix, len(text)), 0, -1)
                    if text[-length:] in self._prefixes
                ),
                0,
            )
            cut = len(text) - held
            if held:
                # Never cut through a complete spelling, so that it is defused whole once the tail resolves
                for match in self._control_tokens_re.finditer(text, max(0, cut - self._longest_prefix)):
                    if match.start() < cut < match.end():
                        cut = match.start()
                        break
            text, self._tail = text[:cut], text[cut:]
        return self._control_tokens_re.sub(lambda match: MARKER.join(match.group()), text)
