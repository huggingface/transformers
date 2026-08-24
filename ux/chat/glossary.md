# Glossary

The source of truth for the vocabulary of this project. Documents use these words with exactly
these meanings; when two candidate words exist, the one listed here wins.

- **session** — one run of `transformers chat`, from launch to exit. A session holds one
  conversation at a time and one set of generation settings.
- **turn** — one round of interaction: the user submits a line, the product responds (or handles
  it locally), and the prompt returns. The five phases of a turn are **composing**, **ending at
  once**, **waiting**, **streaming**, and **settling** (see the document template in
  [README.md](README.md)).
- **prompt** — the `<username>:` line at which the user types. "At the prompt" means the session
  is waiting for input.
- **message** — one entry in the conversation: the user's submitted text or the model's reply.
- **reply** — the model's message for a turn, as the user sees it rendered.
- **stream** — the reply's arrival token by token. "Mid-stream" means tokens are still arriving.
- **chunk** — one increment of streamed text as it arrives. Chunks have no visual identity; lines
  and blocks do.
- **block** — a top-level unit of the reply's markdown: a paragraph, a heading, a fenced code
  block, a list, a table, a quote, a rule. Blocks are separated by blank lines in the source text.
- **run** — one or more blocks that must be treated as a single unit while streaming because
  later text can change how earlier text renders: a list that may gain items, adjacent indented
  code, an unclosed code fence.
- **commit** — the moment a finished run is printed permanently into scrollback. Committed lines
  are never repainted.
- **tail** — the not-yet-committed end of the reply: the run still being streamed. The tail is
  the only part of the reply that repaints.
- **live region** — the area at the bottom of the screen where the tail is drawn. It is erased
  when the reply settles and its content is committed.
- **follow** — the display's promise that the newest lines of the tail stay visible at the bottom
  of the live region while streaming; the terminal analog of a chat window sticking to the bottom.
- **window budget** — the maximum height of the live region: the screen height minus six rows,
  and at least four rows.
- **ellipsis row** — the dim `…` drawn at the top of the live region when the tail is taller than
  the window budget, meaning: this block continues above and will be printed in full when it
  commits.
- **scrollback** — the terminal's own history of lines that have scrolled off the top of the
  screen. Scrollback belongs to the terminal emulator, not to the product; committed lines live
  there.
- **viewport** — the part of scrollback plus screen the terminal currently shows. Scrolling moves
  the viewport; it never changes what the product prints.
- **settle** — the end of the streaming phase: the live region is erased, the tail is committed,
  the stats line may print, the conversation records the reply, and the prompt returns.
- **stats line** — the dim `N tokens in X.Xs (Y.Y tok/s)` line printed after a reply settles, when
  the server reported usage.
- **hard break** — the rendering rule that a single newline in the model's prose starts a new
  line on screen (standard markdown would join the lines with a space).
- **fence** — a markdown code fence: a line of three or more backticks or tildes opening a code
  block, closed by a matching line. Text inside a fence renders verbatim, with syntax coloring.
- **generation settings** — the knobs sent with every request: `max_new_tokens`, `do_sample`,
  temperature, and the rest. Set at launch by flags, changed with `!set`, shown by `!status`.
- **system prompt** — the optional instruction message installed at the start of the conversation
  with `--system-prompt`.
- **command** — an input line starting with `!`, handled locally by the session instead of being
  sent to the model.
