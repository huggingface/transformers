# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
Robust TTS input normalizer for non-semantic text normalization.

Goals
-----
1. Only perform robustness cleanup; do not expand numbers, units, dates, amounts, or other semantic content.
2. Prefer protecting high-risk tokens to avoid corrupting `.map`, `app.js.map`, `v2.3.1`, URLs, email addresses,
   @mentions, and #hashtags.
3. Preserve the contents of `[]` and `{}` so they remain available for future sound events and control tags.
4. Replace structural symbols instead of deleting them:
   - Convert `【】`, `〖〗`, `『』`, and `「」` in structural positions into sentence boundaries.
   - Split `《》` only in standalone title or section-name contexts; keep embedded titles unchanged.
   - Convert `——`, `--`, and `——...` into sentence boundaries.
   - Convert flow connectors such as `->`, `=>`, and `→` into Chinese commas to avoid feeding unstable raw symbols to
     TTS.
5. Lightly normalize common social-platform noise:
   - `......` and `……` -> `。`
   - `？？？！！！` -> `？！`
   - `！！！` -> `！`
6. Handle spaces according to script type:
   - Inside Western text spans, collapse consecutive spaces to one space.
   - Inside Chinese-character and Japanese-kana spans, remove spaces.
   - Between Chinese characters or Japanese kana and Latin-like or protected tokens, preserve or add one space.
   - Between Chinese characters or Japanese kana and plain digits, do not force a space.
7. Lightly handle Markdown and newlines:
   - `[text](url)` -> `text url`
   - Remove heading `#`, quote `>`, and list prefixes.
   - Convert newlines into sentence boundaries, `。`.

Non-goals
---------
1. Do not decide how content should be spoken.
2. Do not remove the contents of `[]` or `{}`.
3. Do not interpret HTML, SSML, or semantic tags.
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------
# Base constants and regular expressions
# ---------------------------

# Scripts that do not rely on whitespace tokenization: Chinese characters and Japanese kana.
_CJK_CHARS = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"
_CJK = f"[{_CJK_CHARS}]"

# Placeholder used for protected spans.
_PROT = r"___PROT\d+___"

# High-risk tokens that should be protected from normalization.
_URL_RE = re.compile(r"https?://[^\s\u3000，。！？；、）】》〉」』]+")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}")
_REDDIT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])#(?!\s)[^\s#]+")

# `.map` / `.env` / `.gitignore`
_DOT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+")

# `app.js.map`, `index.d.ts`, `v2.3.1`, `foo/bar-baz.py`, and similar tokens.
_FILELIKE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[A-Za-z0-9._/+:-]*[A-Za-z])"
    r"(?=[A-Za-z0-9._/+:-]*[._/+:-])"
    r"[A-Za-z0-9](?:[A-Za-z0-9._/+:-]*[A-Za-z0-9])?"
    r"(?![A-Za-z0-9_])"
)

# Tokens that participate in adding spaces at mixed CJK/Latin boundaries. They must contain at least one Latin letter
# or be protected placeholders.
_LATINISH = rf"(?:{_PROT}|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)"

# Zero-width characters.
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")


# ---------------------------
# Public entry point
# ---------------------------


def normalize_tts_text(text: str) -> str:
    """Apply robust normalization to TTS input text."""
    text = _base_cleanup(text)
    text = _normalize_markdown_and_lines(text)
    text, protected = _protect_spans(text)

    text = _normalize_spaces(text)
    text = _normalize_structural_punctuation(text)
    text = _normalize_repeated_punctuation(text)
    text = _normalize_spaces(text)

    text = _restore_spans(text, protected)
    return text.strip()


# ---------------------------
# Normalization rules
# ---------------------------


def _base_cleanup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = _ZERO_WIDTH_RE.sub("", text)

    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t " or not cat.startswith("C"):
            cleaned.append(ch)
    return "".join(cleaned)


def _normalize_markdown_and_lines(text: str) -> str:
    # Markdown links: [text](url) -> text url.
    text = re.sub(r"\[([^\[\]]+?)\]\((https?://[^)\s]+)\)", r"\1 \2", text)

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        line = re.sub(r"^#{1,6}\s+", "", line)  # Heading.
        line = re.sub(r"^>\s+", "", line)  # Block quote.
        line = re.sub(r"^[-*+]\s+", "", line)  # Unordered list.
        line = re.sub(r"^\d+[.)]\s+", "", line)  # Ordered list.
        lines.append(line)

    return "。".join(lines) if lines else ""


def _protect_spans(text: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def repl(match: re.Match[str]) -> str:
        idx = len(protected)
        protected.append(match.group(0))
        return f"___PROT{idx}___"

    for pattern in (
        _URL_RE,
        _EMAIL_RE,
        _MENTION_RE,
        _REDDIT_RE,
        _HASHTAG_RE,
        _DOT_TOKEN_RE,
        _FILELIKE_RE,
    ):
        text = pattern.sub(repl, text)

    return text, protected


def _restore_spans(text: str, protected: list[str]) -> str:
    for idx, original in enumerate(protected):
        text = text.replace(f"___PROT{idx}___", original)
    return text


def _normalize_spaces(text: str) -> str:
    # Normalize whitespace.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)

    # Remove spaces inside Chinese/Japanese spans.
    text = re.sub(rf"({_CJK})\s+(?={_CJK})", r"\1", text)

    # Remove spaces between Chinese/Japanese text and plain digits; do not force spaces there.
    text = re.sub(rf"({_CJK})\s+(?=\d)", r"\1", text)
    text = re.sub(rf"(\d)\s+(?={_CJK})", r"\1", text)

    # Keep or add one space between Chinese/Japanese text and Latin-like or protected tokens.
    text = re.sub(rf"({_CJK})(?=({_LATINISH}))", r"\1 ", text)
    text = re.sub(rf"(({_LATINISH}))(?={_CJK})", r"\1 ", text)

    # Collapse repeated spaces again.
    text = re.sub(r" {2,}", " ", text)

    # Do not keep spaces around CJK punctuation.
    text = re.sub(r"\s+([，。！？；：、”’」』】）》])", r"\1", text)
    text = re.sub(r"([（【「『《“‘])\s+", r"\1", text)
    text = re.sub(r"([，。！？；：、])\s*", r"\1", text)

    # Do not keep spaces before ASCII punctuation; do not force changes to following English spacing.
    text = re.sub(r"\s+([,.;!?])", r"\1", text)

    return re.sub(r" {2,}", " ", text).strip()


def _normalize_structural_punctuation(text: str) -> str:
    # Structural brackets: unwrap them in structural positions and convert them to sentence boundaries.
    # Run two passes so consecutive blocks can converge.
    for _ in range(2):
        text = re.sub(
            r"(^|[。！？!?；;]\s*)[【〖『「]([^】〗』」]+)[】〗』」]\s*",
            r"\1\2。",
            text,
        )

    # Only process standalone titles in 《》; leave embedded titles unchanged.
    # Example: News.《Product launch》——Starting now! -> News.Product launch.Starting now!
    text = re.sub(
        r"(^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))",
        r"\1\2",
        text,
    )

    # Flow or mapping arrows: convert to a Chinese comma, preserving link structure while avoiding raw `->` in TTS.
    text = re.sub(
        r"\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*",
        "，",
        text,
    )

    # Long dashes or repeated hyphens: convert to sentence boundaries.
    text = re.sub(r"\s*(?:—|–|―|-){2,}\s*", "。", text)

    return text


def _normalize_repeated_punctuation(text: str) -> str:
    # Ellipses and repeated periods.
    text = re.sub(r"(?:\.{3,}|…{2,}|……+)", "。", text)

    # Repeated punctuation of the same type.
    text = re.sub(r"[。．]{2,}", "。", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[!！]{2,}", "！", text)
    text = re.sub(r"[?？]{2,}", "？", text)

    # Mixed question/exclamation marks: converge to ?!.
    def _mixed_qe(match: re.Match[str]) -> str:
        s = match.group(0)
        has_q = any(ch in s for ch in "?？")
        has_e = any(ch in s for ch in "!！")
        if has_q and has_e:
            return "？！"
        return "？" if has_q else "！"

    text = re.sub(r"[!?！？]{2,}", _mixed_qe, text)
    return text
