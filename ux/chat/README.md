# transformers chat product description

A written description of the user experience of `transformers chat`: what the user sees in the
terminal, what they can do, and exactly what happens when they do it.

## Purpose

`transformers chat` is, from the user's point of view, a large state chart. The user moves through
it with the product's inputs: typed lines, chat commands beginning with `!`, Enter, Ctrl+C and
Ctrl+D, terminal scrolling, and window resizes. Most of that behavior is defined implicitly, spread
across the session loop in `chat.py`, the streaming display in `chat_display.py`, the rich library's
rendering pipeline, and the tests. There is no single place that says, in plain language, "when the
user does X, this is what happens, and this is what happens if they do Y halfway through."

This project is that place. It describes the full experience a user has in a `transformers chat`
session connected to a `transformers serve` backend, in the default configuration, with nothing
customized, in a modern terminal emulator.

The documents are for people who need to understand or change the product: designers, engineers,
writers, testers, and anyone evaluating whether a behavior is intentional. They are also built to be
a training and evaluation harness: together with the checklists in `verification/`, they specify the
observable behavior of the streaming display precisely enough that an implementation can be written
against them from scratch and graded item by item — the autoscroll rework on this branch was built
and verified exactly that way. They are written from the outside in. They describe the experience,
not the implementation.

### What this is not

- Not API documentation. That lives in `docs/source/en/` (see `conversations.md` and `serving.md`)
  and in the docstrings of `transformers chat --help`.
- Not organized by package. `chat.py`, `chat_display.py`, and `rich` are not described separately.
  A single behavior is described once, wherever the user encounters it.
- Not a technical design document. Where a technical detail is critical to understanding the
  experience, it appears in a block quote labeled `Technical note:` and nowhere else.

## Conventions

- Describe the experience, not the code. "The newest lines of the reply stay visible at the bottom
  of the screen" rather than "the tail renderable slices the last N segment lines".
- Technical detail goes in block quotes, prefixed with `Technical note:`. Use it only when the
  mechanism changes what the user would expect.
- Use sentence case for headings.
- Name the vocabulary consistently. The [glossary](glossary.md) is the source of truth for terms
  like *turn*, *stream*, *block*, *commit*, *tail*, *scrollback*, and *settle*.
- Every document ends with the commit of `huggingface/transformers` it was verified against and a
  list of open questions.
- When a behavior is surprising, say so and say why it is that way if the reason is known. Do not
  smooth it over.

## The work to be done

Each document describes one feature. Features are large things (the streaming of a response, with
its follow-the-newest-lines scrolling) or small things (the tokens-per-second line after a reply),
but each is described in full, including its edge cases and its interactions with other features.

### Document template

Every feature document follows the same skeleton so that documents are comparable and nothing is
skipped.

1. **Summary.** One paragraph describing the feature abstractly. For example: "After a reply
   finishes streaming, a dim one-line summary reports how many tokens arrived and how fast."
2. **The simple case.** The common path in prose.
3. **The interaction, event by event.** The five phases of a turn: **composing** (typing at the
   prompt), **ending at once** (input handled locally, no request made), **waiting** (request sent,
   no tokens yet), **streaming** (tokens arriving), and **settling** (the reply is finalized and
   the prompt returns). What starts the turn and what is captured, what happens if it ends at once,
   what is decided the moment the request is sent, what updates live while tokens arrive, and what
   is committed at the end. Include a small state diagram (Mermaid `stateDiagram-v2`) of the states
   the user passes through.
4. **Modifiers.** A table of the product's variant axes — generation settings (`!set`, CLI flags),
   the system prompt, terminal size, and whether output is a real terminal — and what each one does
   when set at the start and when changed *during* the interaction.
5. **Cancel and interrupt.** The same checklist in every document:
   - Ctrl+C (the user's explicit abort)
   - The user doing something else mid-way: typing while the reply is streaming, scrolling the
     terminal, resizing the window
   - The environment failing: connection to the server lost, request fails or times out
   - The process going away: the terminal window is closed, the chat process is killed
   - The input channel changing: Ctrl+D or end of input, output piped rather than a terminal
6. **Interactions with other systems.** The product's cross-cutting concerns, in a fixed order:
   generation settings; chat history; the server and model state; terminal capabilities; saved
   chats.
7. **Edge cases.** Anything a user could notice that is not covered above.
8. **Open questions and verification.** The `huggingface/transformers` commit the document was
   verified against, and any behavior that could not be confirmed.

Item 5 matters most. Asking the same interrupt questions of every feature is how gaps and
inconsistencies are found — three of the defects in [bug-triage.md](bug-triage.md) were found by
item 5 alone.

### Method

For each document:

1. Read the session loop in `src/transformers/cli/chat.py` and the streaming display in
   `src/transformers/cli/chat_display.py`.
2. Read the matching tests in `tests/cli/`. `test_chat_display.py` is close to an executable
   specification of the streaming edge cases: the render-fidelity suite and the pty test encode the
   two central promises of the display.
3. Draft the document.
4. Try anything ambiguous in the running product: `transformers serve` in one shell,
   `transformers chat <model>` in another — or the scripted stub server in
   [verification/README.md](verification/README.md) when the behavior must be reproduced exactly
   (timing, interrupts, terminal size). Tests settle "what happens"; the running product settles
   how it feels, what is visible while the interaction is in progress, and what the timing is like.
5. Record the commit verified against.

### Verification

Drafting reads the code; verification watches the product. The `verification/` directory holds one
checklist per cluster of documents, each item a single observable claim with setup, steps, expected
result, a priority, and the terminal it needs. A tester runs them in a real terminal (or with the
scripted pty rig described in `verification/README.md`), records `pass`, `fail`, or `blocked` in
the Result column, and files every failure in `bug-triage.md` with the item's ID. A document moves
from `drafted` to `verified` in the coverage table only when every P1 and P2 item for it has passed
or been filed.

`bug-triage.md` is the other half: every behavior the documents flagged as a likely defect,
deduplicated, with reproduction steps, the reason in the code, a severity, and the decision the
product team needs to make. Entries confirmed in the running product carry a Status line.

### Order of work

1. **Pilot: [the stats line](streaming/the-stats-line.md).** Small and self-contained. Used to
   settle the template, tone, and depth.
2. **Foundations: [the terminal medium](foundations/the-terminal-medium.md) and
   [the session](foundations/the-session.md).** Everything else refers to them.
3. **[Response streaming](streaming/response-streaming.md).** The bulk of the experience and the
   hardest part — the scrolling behavior this project exists to pin down. Written third so the
   template is already proven.
4. **Everything else.** Once the template and two exemplars exist, the remaining documents can be
   drafted in parallel, followed by a consistency pass and a verification pass across the whole
   set.

Progress is tracked in the [coverage table](#coverage) below.

### Scope decisions

- **The `transformers serve` backend.** Excluded. The chat client's experience depends on it only
  through the events it emits (model-loading progress, token chunks, usage counts, errors), which
  are described where the user sees them. The server's own surface (its API, its logs, its model
  management) deserves its own description project later.
- **Generation settings.** Described inside each document's Modifiers table where they change that
  feature, and once in [cross-cutting/generation-settings.md](cross-cutting/generation-settings.md)
  for the shared model of how settings are set, merged, and displayed. A settings appendix per
  document would drift.
- **Markdown rendering.** Described once in
  [streaming/markdown-rendering.md](streaming/markdown-rendering.md) rather than in every feature
  that prints markdown, because the same pipeline renders help text, replies, and load notices.
- **Interaction shape.** The unit of interaction is a turn and its phases are composing, ending at
  once, waiting, streaming, settling. The interrupt list and the order of cross-cutting concerns
  are fixed as written in the document template above.
- **Numbered rules.** These are prose documents, not numbered specifications. Stable heading
  anchors are enough for cross-references.

## Structure

```
README.md                        this file
goal.md                          the standing instructions for whoever drafts
AGENTS.md, CLAUDE.md             entry points for agents: read README.md, then goal.md
glossary.md                      shared vocabulary
bug-triage.md                    suspected defects collected from every document, with repro steps
                                 and decisions needed

verification/
  README.md                      how to run a hand-verification pass and record results,
                                 including the scripted stub-server rig
  streaming-output.md            checklist for streaming/
  session-and-commands.md        checklist for foundations/, session/ and cross-cutting/

foundations/
  the-terminal-medium.md         scrollback, the viewport, the live region, why a terminal
                                 cannot autoscroll like a web page
  the-session.md                 the session loop: startup, turns, history, exit

streaming/
  response-streaming.md          a reply arriving token by token, and what "scrolling" means
                                 while it does
  markdown-rendering.md          how model text becomes styled terminal output
  the-stats-line.md              the dim tokens-per-second line after a reply (pilot)
  length-limit-continuation.md   the token-limit notice and the continue? prompt

session/
  input-and-the-prompt.md        the prompt, typing, history recall, submitting
  commands.md                    !help, !status, !clear, !save, !set, !example, !exit,
                                 and invalid commands
  model-loading.md               the progress display while the server loads a model
  connection-and-errors.md       health checks, unreachable servers, mid-stream failures

cross-cutting/
  generation-settings.md         how settings flow from CLI flags and !set into every turn
```

## Coverage

Status is one of `not started`, `drafted`, or `verified`.

| Document | Status |
| --- | --- |
| glossary.md | drafted |
| bug-triage.md | drafted |
| verification/streaming-output.md | drafted |
| verification/session-and-commands.md | drafted |
| foundations/the-terminal-medium.md | drafted |
| foundations/the-session.md | drafted |
| streaming/response-streaming.md | drafted (all machine items pass; emulator hand pass SO-13/SO-14 pending) |
| streaming/markdown-rendering.md | verified |
| streaming/the-stats-line.md | verified |
| streaming/length-limit-continuation.md | drafted |
| session/input-and-the-prompt.md | verified |
| session/commands.md | drafted |
| session/model-loading.md | drafted |
| session/connection-and-errors.md | drafted |
| cross-cutting/generation-settings.md | drafted |

## Reference

The source of truth is `huggingface/transformers` at `https://github.com/huggingface/transformers`.
The relevant locations are:

- `src/transformers/cli/chat.py`: the surface this project describes — the session loop, the
  prompt, the commands, the turn
- `src/transformers/cli/chat_display.py`: where the streaming interaction state lives — block
  segmentation, the commit rule, the live tail
- `src/transformers/cli/serving/`: the server whose events shape loading, streaming, and errors
- `tests/cli/test_chat_display.py`: behavioral tests — the render-fidelity suite and the pty
  streaming test are executable specifications
- `tests/cli/test_chat.py`: tests for command helpers and history persistence
- the `rich` library (`Live`, `Markdown`, `Console`): the rendering subsystem that shapes what is
  physically possible on screen
