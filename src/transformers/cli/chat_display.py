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
"""Streaming markdown display for `transformers chat`.

The terminal is the scrollbar: finished markdown blocks are printed once into the terminal's normal
scrollback and never repainted, while only the block still being streamed lives in a small transient
`rich.live.Live` region at the bottom. When that in-progress block is taller than the window budget,
the region shows its newest lines (the generation point), so incoming tokens are always visible.

This replaces re-rendering the whole accumulated message inside `Live` on every token, which capped
the visible response at one screen (`vertical_overflow="ellipsis"` froze long responses at their
first screenful until the stream ended) and rewrote the entire message to the terminal on each
token.

A block boundary is a blank line outside a code fence. Blocks whose rendering depends on later
lines are kept together as one run and only committed when the run is over:

- fenced code (``` or ~~~): blank lines inside the fence are not boundaries
- lists: a blank line followed by another list item or an indented line continues the same list
  (markdown merges them, which can renumber or re-space every item)
- indented code: adjacent indented blocks separated by blank lines merge into one code block

Known limitation: reference-style link definitions (`[1]: https://...`) that appear after the
paragraph using them cannot apply to already-committed lines.
"""

import re
from collections.abc import Callable

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import Markdown
from rich.segment import Segment, Segments
from rich.style import Style


# A fence marker: optional indentation, then 3+ backticks or tildes. Indented fences occur inside
# list items. Misreading a non-fence line as an opener only delays commits (never corrupts output).
_FENCE_OPEN_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")
# A list item: -, *, + or 1. / 1) markers, at up to 3 spaces of indentation.
_LIST_ITEM_RE = re.compile(r"^ {0,3}(?:[-*+]|\d{1,9}[.)])(?:\s|$)")
# A single word in angle brackets, e.g. <think> or </think>, which markdown would treat as HTML.
_TAG_RE = re.compile(r"<(/*)(\w*)>")


def escape_tags(line: str) -> str:
    """Escape single-word `<tags>` (e.g. `<think>` -> `\\<think\\>`) so markdown renders them."""
    return _TAG_RE.sub(r"\<\1\2\>", line)


def hard_break(line: str) -> str:
    """Append the markdown hard-break marker (two trailing spaces) to a rendered-prose line.

    Chat models emit a single newline to mean "new line", but markdown treats a single newline in
    prose as a space, so every completed prose line gets a hard break.
    """
    if line and not line.isspace():
        return line + "  "
    return line


class _TailWindow:
    """Renders the in-progress markdown block, cropped to its newest lines.

    When the block is taller than the window budget, the oldest lines are dropped and a dim
    ellipsis row marks that the block continues above; the hidden lines are printed in full when
    the block is committed.
    """

    def __init__(self, stream: "MarkdownStream") -> None:
        self._stream = stream

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        source = self._stream._tail_source
        if not source:
            return
        lines = self._stream._render_markdown_lines(source)
        if not lines:
            return
        if self._stream._blocks_printed:
            # the seam blank line that _print_block will emit when this block commits
            yield Segment.line()
        max_lines = max(4, console.size.height - 6)
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
            yield Segment("…", Style(dim=True))
            yield Segment.line()
        for line in lines:
            yield from line
            yield Segment.line()


class MarkdownStream:
    """Streams a markdown message to a rich console without ever exceeding the screen.

    Feed raw text deltas with [`~cli.chat_display.MarkdownStream.append`] and call
    [`~cli.chat_display.MarkdownStream.finalize`] when the stream ends (also on errors or
    interrupts: the text streamed so far is then completed into scrollback). The final terminal
    content is identical to printing the fully accumulated message once.

    Args:
        console (`rich.console.Console`):
            The console to print to. On a non-terminal console the live region is skipped and
            blocks are printed as they complete.
        code_theme (`str`, *optional*, defaults to `"github-dark"`):
            Pygments theme for code blocks.
        refresh_per_second (`float`, *optional*, defaults to 8.0):
            Repaint rate of the live tail region.
        line_transforms (`list[Callable[[str], str]]`, *optional*):
            Transforms applied to each completed source line outside code fences, in order.
            Defaults to `[escape_tags, hard_break]`.
    """

    def __init__(
        self,
        console: Console,
        code_theme: str = "github-dark",
        refresh_per_second: float = 8.0,
        line_transforms: list[Callable[[str], str]] | None = None,
    ) -> None:
        self._console = console
        self._code_theme = code_theme
        self._refresh_per_second = refresh_per_second
        self._line_transforms = [escape_tags, hard_break] if line_transforms is None else line_transforms

        self._raw_buffer = ""  # raw text not yet split into complete lines
        self._fence: tuple[str, int] | None = None  # (fence char, marker length) when inside a fence
        self._run_lines: list[str] = []  # prepared lines of the current (uncommitted) run
        self._run_kind: str | None = None  # "list" | "indented" | "other"
        self._pending_blanks = 0  # blank lines seen since the last non-blank line
        self._blocks_printed = False
        self._finalized = False

        self._tail_source = ""  # snapshot read by the live refresh thread
        self._live: Live | None = None

    def append(self, text: str) -> None:
        """Add a raw text delta to the stream."""
        if self._finalized:
            raise RuntimeError("Cannot append to a finalized MarkdownStream.")
        self._raw_buffer += text
        *complete_lines, self._raw_buffer = self._raw_buffer.split("\n")
        for line in complete_lines:
            self._feed_line(line.rstrip("\r"))
        self._tail_source = self._compose_tail_source()
        self._ensure_live()

    def finalize(self) -> None:
        """Print everything still in the tail and release the live region. Idempotent."""
        if self._finalized:
            return
        self._finalized = True
        if self._live is not None:
            self._live.stop()  # transient: erases the tail region
            self._live = None
        remaining = self._compose_tail_source().strip("\n")
        if remaining:
            self._print_block(remaining)
        self._raw_buffer = ""
        self._run_lines = []
        self._pending_blanks = 0
        self._tail_source = ""

    def __enter__(self) -> "MarkdownStream":
        return self

    def __exit__(self, *exc_info) -> None:
        self.finalize()

    # --- source segmentation ---

    def _feed_line(self, raw_line: str) -> None:
        """Consume one complete source line, committing the previous run when a new one starts."""
        in_fence = self._fence is not None
        line = self._prepare_line(raw_line, in_fence)
        blank = line.strip() == ""

        if in_fence:
            self._append_to_run(line)
        elif blank:
            self._pending_blanks += 1
        elif self._run_lines and (self._pending_blanks == 0 or self._continues_run(line)):
            self._append_to_run(line)
        else:
            if self._run_lines:
                self._print_block("\n".join(self._run_lines))
                self._run_lines = []
            self._pending_blanks = 0
            self._run_kind = self._classify(line)
            self._append_to_run(line)

    def _append_to_run(self, line: str) -> None:
        self._run_lines.extend([""] * self._pending_blanks)
        self._pending_blanks = 0
        self._run_lines.append(line)
        self._track_fence(line)

    def _continues_run(self, line: str) -> bool:
        """Whether a blank-separated line still belongs to the current run."""
        if self._run_kind == "list":
            return bool(_LIST_ITEM_RE.match(line)) or line[:1] in (" ", "\t")
        if self._run_kind == "indented":
            return line[:1] in (" ", "\t")
        return False

    @staticmethod
    def _classify(line: str) -> str:
        if _LIST_ITEM_RE.match(line):
            return "list"
        if line[:1] in (" ", "\t"):
            return "indented"
        return "other"

    def _track_fence(self, line: str) -> None:
        if self._fence is None:
            match = _FENCE_OPEN_RE.match(line)
            if match:
                marker = match.group(2)
                self._fence = (marker[0], len(marker))
        else:
            char, length = self._fence
            if re.match(rf"^\s*{re.escape(char)}{{{length},}}\s*$", line):
                self._fence = None

    def _prepare_line(self, line: str, in_fence: bool) -> str:
        if in_fence or _FENCE_OPEN_RE.match(line):
            return line
        for transform in self._line_transforms:
            line = transform(line)
        return line

    def _compose_tail_source(self) -> str:
        lines = list(self._run_lines)
        partial = self._prepare_line(self._raw_buffer.rstrip("\r"), self._fence is not None)
        if partial.strip():
            lines.extend([""] * self._pending_blanks)
            lines.append(partial)
        return "\n".join(lines)

    # --- rendering ---

    def _render_markdown_lines(self, source: str) -> list[list[Segment]]:
        """Render markdown to segment lines, stripping unstyled blank edge lines (block margins)."""
        markdown = Markdown(source, code_theme=self._code_theme)
        lines = self._console.render_lines(markdown, self._console.options, pad=False)

        def is_margin(line: list[Segment]) -> bool:
            return all(not segment.text.strip() and not segment.style for segment in line)

        while lines and is_margin(lines[0]):
            lines.pop(0)
        while lines and is_margin(lines[-1]):
            lines.pop()
        return lines

    def _print_block(self, source: str) -> None:
        """Commit a finished block: print it permanently above the live region."""
        lines = self._render_markdown_lines(source)
        if not lines:
            return
        segments: list[Segment] = []
        for line in lines:
            segments.extend(line)
            segments.append(Segment.line())
        if self._blocks_printed:
            self._console.print()
        self._console.print(Segments(segments))
        self._blocks_printed = True

    def _ensure_live(self) -> None:
        if self._live is None and self._console.is_terminal:
            self._live = Live(
                _TailWindow(self),
                console=self._console,
                transient=True,
                refresh_per_second=self._refresh_per_second,
                vertical_overflow="crop",
            )
            self._live.start()
