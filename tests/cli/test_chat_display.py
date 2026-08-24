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
import io
import os
import re
import sys
import textwrap

import pytest
from rich.console import Console
from rich.markdown import Markdown

from transformers.cli.chat_display import MarkdownStream, escape_tags, hard_break


WIDTH = 60


def make_console(force_terminal=False):
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=force_terminal,
        width=WIDTH,
        color_system="truecolor" if force_terminal else None,
    )
    return console, buffer


def normalize(text):
    """Erase differences that are invisible on a terminal.

    Trailing plain-space padding per line, OSC-8 hyperlink ids (a process-global counter), and
    blank lines at the edges of the whole message (the streaming path trims them).
    """
    text = "\n".join(re.sub(r" +$", "", line) for line in text.split("\n"))
    text = re.sub(r"(\x1b\]8;id=)\d+(;)", r"\1X\2", text)
    return text.strip("\n") + "\n"


def oneshot_render(source, force_terminal=False):
    console, buffer = make_console(force_terminal)
    console.print(Markdown(source, code_theme="github-dark"))
    return buffer.getvalue()


def stream_render(source, chunk_size, force_terminal=False):
    console, buffer = make_console(force_terminal)
    stream = MarkdownStream(console=console, line_transforms=[])
    for start in range(0, len(source), chunk_size):
        stream.append(source[start : start + chunk_size])
    stream.finalize()
    return buffer.getvalue()


FIDELITY_CASES = {
    "paragraphs": "First paragraph with words that wrap the sixty column width.\n\nSecond paragraph.\n",
    "headings": "# Title\n\nBody text.\n\n## Section\n\nMore body text.\n",
    "fence": "Intro:\n\n```python\nfor i in range(3):\n    print(i)\n```\n\nAfter code.\n",
    "fence_with_blank_lines": "```text\nfirst\n\nsecond\n\n\nthird\n```\n\nAfter.\n",
    "fence_unclosed": "Some text.\n\n```python\nx = 1\ny = 2\n",
    "tilde_fence_with_backticks": "~~~\n```\nnot a nested fence\n```\n~~~\n\nAfter.\n",
    "bullets": "- one\n- two\n- three\n\nAfter the list.\n",
    "ordered_renumber": "\n".join(f"{i}. item number {i}" for i in range(1, 13)) + "\n\nAfter.\n",
    "adjacent_ordered_lists": "1. first\n2. second\n\n1. third\n2. fourth\n\nAfter.\n",
    "adjacent_bullet_lists": "- one\n- two\n\n- three\n\nAfter.\n",
    "loose_list": "- item one\n\n  continuation paragraph of item one\n\n- item two\n\nAfter.\n",
    "nested_list": "- top\n  - nested one\n  - nested two\n- top two\n\nAfter.\n",
    "list_with_fence": "- item\n\n  ```\n  code in item\n\n  more after a blank\n  ```\n\n- next item\n\nAfter.\n",
    "table": "Table:\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 33 | 44 |\n\nAfter the table.\n",
    "quote": "> quoted line\n> second quoted line\n\nAfter the quote.\n",
    "hr_middle": "Before.\n\n---\n\nAfter.\n",
    "hr_at_end": "Before.\n\n---\n",
    "setext_heading": "Title\n---\n\nAfter.\n",
    "indented_code_adjacent": "Paragraph.\n\n    code line a\n\n    code line b\n\nAfter.\n",
    "links_and_inline": "See [rich](https://github.com/Textualize/rich) with **bold** and `code`.\n\nAfter.\n",
    "mixed_document": (
        "# Answer\n\nA paragraph that wraps around the sixty column line width.\n\n"
        "```python\nprint('hello')\n```\n\n- alpha\n- beta\n\n1. first\n2. second\n\n"
        "> a quote\n\n| x | y |\n|---|---|\n| 1 | 2 |\n\nFinal words.\n"
    ),
}


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
@pytest.mark.parametrize("case", FIDELITY_CASES)
def test_stream_matches_oneshot_render(case, chunk_size):
    """The streamed transcript must be identical to rendering the full message once."""
    source = FIDELITY_CASES[case]
    assert normalize(stream_render(source, chunk_size)) == normalize(oneshot_render(source))


@pytest.mark.parametrize("case", FIDELITY_CASES)
def test_stream_matches_oneshot_render_ansi(case, monkeypatch):
    """Same fidelity guarantee with colors on (the live region itself is exercised in the pty test)."""
    monkeypatch.setattr(MarkdownStream, "_ensure_live", lambda self: None)
    source = FIDELITY_CASES[case]
    streamed = stream_render(source, chunk_size=5, force_terminal=True)
    assert normalize(streamed) == normalize(oneshot_render(source, force_terminal=True))


def test_blocks_commit_progressively():
    """Finished blocks reach the output while later blocks are still streaming."""
    console, buffer = make_console()
    stream = MarkdownStream(console=console, line_transforms=[])
    stream.append("First block done.\n\nSecond block done.\n\nThird block began\nand continues stre")
    mid = buffer.getvalue()
    assert "First block done." in mid
    assert "Second block done." in mid
    assert "Third" not in mid  # still streaming: only in the live tail
    stream.finalize()
    assert "and continues stre" in buffer.getvalue()


def test_list_runs_are_not_committed_early():
    """A blank line inside a list is not a commit point: the next item may renumber every row."""
    console, buffer = make_console()
    stream = MarkdownStream(console=console, line_transforms=[])
    stream.append("intro paragraph\n\n1. one\n2. two\n\n")
    assert "one" not in buffer.getvalue()  # the list may still continue
    stream.append("3. three\n\nafter paragraph\n")
    assert "three" in buffer.getvalue()  # "after paragraph" ended the list run
    stream.finalize()


def test_unclosed_fence_is_never_committed_early():
    console, buffer = make_console()
    stream = MarkdownStream(console=console, line_transforms=[])
    stream.append("```python\ncode one\n\ncode two\n\n")
    assert "code one" not in buffer.getvalue()
    stream.finalize()
    assert "code one" in buffer.getvalue()
    assert "code two" in buffer.getvalue()


def test_escape_tags():
    assert escape_tags("a <think> b </think> c") == r"a \<think\> b \</think\> c"
    assert escape_tags("<not a tag, has spaces>") == "<not a tag, has spaces>"


def test_hard_break():
    assert hard_break("some prose") == "some prose  "
    assert hard_break("") == ""
    assert hard_break("   ") == "   "


def test_transforms_apply_only_outside_fences():
    console, buffer = make_console()
    stream = MarkdownStream(console=console)
    stream.append("<think>\nplanning\n</think>\n\n```html\n<div>\n</div>\n```\n")
    stream.finalize()
    output = buffer.getvalue()
    assert "<think>" in output  # escaped in source, so rendered literally instead of as HTML
    assert "<div>" in output  # inside the fence: not escaped
    assert "\\<div" not in output


def test_single_newlines_render_as_line_breaks():
    console, buffer = make_console()
    stream = MarkdownStream(console=console)
    stream.append("alpha line\nbeta line\n")
    stream.finalize()
    lines = [line.strip() for line in buffer.getvalue().splitlines()]
    assert "alpha line" in lines
    assert "beta line" in lines


def test_finalize_is_idempotent_and_append_after_finalize_raises():
    console, buffer = make_console()
    stream = MarkdownStream(console=console, line_transforms=[])
    stream.append("hello\n")
    stream.finalize()
    first = buffer.getvalue()
    stream.finalize()
    assert buffer.getvalue() == first
    with pytest.raises(RuntimeError):
        stream.append("more")


def test_interrupted_stream_matches_partial_oneshot():
    """finalize() after an interrupt leaves exactly the partial message in scrollback."""
    source = FIDELITY_CASES["mixed_document"]
    partial = source[: len(source) // 2]
    console, buffer = make_console()
    stream = MarkdownStream(console=console, line_transforms=[])
    stream.append(partial)
    stream.finalize()
    assert normalize(buffer.getvalue()) == normalize(oneshot_render(partial))


def test_empty_stream_prints_nothing():
    console, buffer = make_console()
    with MarkdownStream(console=console) as stream:
        pass
    assert buffer.getvalue() == ""
    console2, buffer2 = make_console()
    with MarkdownStream(console=console2) as stream:
        stream.append("   \n\n  ")
    assert buffer2.getvalue().strip() == ""


PTY_CHILD_SCRIPT = textwrap.dedent(
    """
    import asyncio
    import time
    from types import SimpleNamespace

    from transformers.cli.chat import RichInterface


    def make_token(text):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)], usage=None
        )


    async def fake_stream():
        text = "".join(f"NUM{i:02d} alpha beta gamma\\n" for i in range(1, 41))
        for i in range(0, len(text), 8):
            yield make_token(text[i : i + 8])
            await asyncio.sleep(0.003)
        time.sleep(1.6)  # lets the parent snapshot the mid-stream terminal state
        yield make_token("\\nTHE-VERY-END\\n")


    async def main():
        interface = RichInterface(model_id="test/model", user_id="user", base_url="http://localhost:1")

        async def stream():
            return fake_stream()

        await interface.stream_output(stream())


    asyncio.run(main())
    """
)


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="requires a pty")
def test_streaming_follows_newest_content_in_a_real_terminal(tmp_path):
    """In a 24x80 terminal, a 40-line response must keep its newest lines visible while streaming.

    The pre-rework display froze at the first screenful (`Live` with `vertical_overflow="ellipsis"`)
    and only revealed the rest of the message after the stream ended.
    """
    import fcntl
    import select
    import struct
    import subprocess
    import termios
    import time

    script = tmp_path / "pty_child.py"
    script.write_text(PTY_CHILD_SCRIPT)

    master, slave = os.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    env = dict(os.environ, TERM="xterm-256color", COLUMNS="80", LINES="24")
    proc = subprocess.Popen(
        [sys.executable, str(script)], stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True
    )
    os.close(slave)

    chunks = []  # (monotonic timestamp, bytes)
    deadline = time.monotonic() + 60
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master], [], [], 0.25)
            if master in readable:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                chunks.append((time.monotonic(), data))
            elif proc.poll() is not None:
                break
    finally:
        os.close(master)
        proc.wait(timeout=30)

    output = b"".join(data for _, data in chunks)
    end_time = next(t for t, data in chunks if b"THE-VERY-END" in data)
    mid_output = b"".join(data for t, data in chunks if t < end_time - 0.5)

    # while the stream was still open, the newest lines had already been drawn
    assert b"NUM39" in mid_output and b"NUM40" in mid_output
    # and the display never froze on the first screenful behind a crop marker line
    assert b"\n..." not in mid_output
    # the finished transcript contains the whole message, exactly once committed
    final_text = output.decode("utf-8", errors="replace")
    for i in range(1, 41):
        assert f"NUM{i:02d}" in final_text
    assert "THE-VERY-END" in final_text
