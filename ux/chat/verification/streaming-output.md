# Checklist: streaming output

Covers `streaming/response-streaming.md`, `streaming/markdown-rendering.md`,
`streaming/the-stats-line.md`, `streaming/length-limit-continuation.md`. Setup for all items
unless noted: stub server + `transformers chat test/model http://127.0.0.1:8123/v1` in a 24×80
terminal (see [README.md](README.md)). "long reply" = a message containing `long`.

Commit for recorded results: `4b27c4c7915b5672ab4e25349c5c2e209d25956c`.

| ID | Claim | Steps | Expected | Pri | Terminal | Result |
| --- | --- | --- | --- | --- | --- | --- |
| SO-01 | Reply header prints before the first token | Send any message; watch the first instant | `<model_id>:` line appears alone, then tokens below it | P2 | any | pass (machine, 4b27c4c) |
| SO-02 | Newest lines stay visible while a long reply streams | Send a long reply; watch mid-stream | The bottom of the screen always shows the newest rows (ROW38, ROW39… as they arrive) | P1 | 24×80 | pass (machine, 4b27c4c) |
| SO-03 | The display never freezes behind a crop marker | Same as SO-02 | No `...` line; tokens keep arriving visibly for the whole stream | P1 | 24×80 | pass (machine, 4b27c4c) |
| SO-04 | Finished blocks reach scrollback mid-stream | Send a multi-paragraph reply; scroll up before it ends | Early paragraphs are in scrollback while later ones still stream | P1 | 24×80 | pass (machine, 4b27c4c) |
| SO-05 | The settled transcript equals a one-shot render | Any reply; compare against rendering the same text at once | Identical, including block spacing (fidelity suite, 4 chunk sizes × 21 documents) | P1 | any | pass (machine, 4b27c4c) |
| SO-06 | Code fences stream inside the live tail and commit whole | Ask for a long code block | Fence renders as one block; blank lines inside it never split it; `…` marker above when taller than the window | P1 | 24×80 | pass (machine, 4b27c4c) |
| SO-07 | Lists never commit early and renumber correctly | Ask for a 12-item numbered list | Items re-align when numbers widen; final list identical to one-shot render | P1 | any | pass (machine, 4b27c4c) |
| SO-08 | Tags render literally outside fences, raw inside | Reply containing `<think>` prose and `<div>` in a fence | `<think>` visible as text; `<div>` in code without backslashes | P2 | any | pass (machine, 4b27c4c) |
| SO-09 | Single newlines are line breaks | Reply with `alpha line\nbeta line` | Two rows, not one joined sentence | P2 | any | pass (machine, 4b27c4c) |
| SO-10 | Stats line appears when the server reports usage | Any stub reply | Dim `42 tokens in …s (… tok/s)` between blank lines, then prompt | P2 | any | pass (machine, 4b27c4c) |
| SO-11 | Stats line absent without usage | Edit stub: drop the usage field | Blank line then prompt; no stats | P3 | any | — |
| SO-12 | Length-limit flow | Edit stub: `finish_reason: "length"` | Yellow notice, `Continue generating? (y/N)`, `y` streams a canned continue turn, Enter declines | P2 | any | — |
| SO-13 | Settling shows no visible flicker | Real server, long reply; watch the instant it ends | Live region is replaced by identical committed lines with no flash | P2 | *hand*: iTerm2, Windows Terminal, GNOME Terminal, kitty | — |
| SO-14 | Scroll-up mid-stream stays pinned | Long reply; scroll up mid-stream; wait | Viewport stays where put; new output accumulates below; scroll down to rejoin | P2 | *hand*: same matrix | — |
| SO-15 | Resize mid-stream | Long reply; make the window narrower mid-stream | Tail re-wraps to new width; committed lines keep old wrap; no corruption | P3 | *hand*: any two | — |
