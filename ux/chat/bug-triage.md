# Bug triage

Every behavior the documents flagged as a likely defect, deduplicated. Severity: P1 breaks a core
flow, P2 hurts a real use, P3 is cosmetic. Status `fixed` entries name the commit; they are kept
here because they are part of this corpus's harness role — an implementation built against these
documents on the pre-rework code must fix exactly BT-01…BT-07, and the checklists in
`verification/` grade that.

---

**BT-01 · Long replies froze at their first screenful while streaming** — P1 — **Status: fixed**
in `4b27c4c7915b5672ab4e25349c5c2e209d25956c`; confirmed pre-fix in a scripted 24×80 pty.
*Repro (pre-fix):* ask for anything longer than the window; watch the display stop at ~23 lines
with a `...` marker; new tokens never appear until the stream ends, then the whole reply dumps at
once. *Reason:* the entire accumulated reply was re-rendered inside a bottom-of-screen live
region, whose overflow policy can only crop at the bottom — the newest text was the part cropped.
*Decision taken:* commit finished blocks to scrollback; keep only the in-progress block live,
showing its newest lines.

**BT-02 · Every token rewrote the entire reply to the terminal** — P2 — **Status: fixed** in
`4b27c4c`. *Repro (pre-fix):* stream a 50-line reply; ~368 KB is written for ~1.5 KB of text
(~240×), with whole-screen repaints and visible flicker on some emulators; cost grew with the
square of reply length. *Reason:* full re-render per chunk with a forced refresh. *Decision
taken:* render each block once at commit; repaint only the tail, at most ~8 times a second.

**BT-03 · The reply was unreachable by scrolling until it finished** — P1 — **Status: fixed** in
`4b27c4c`. *Repro (pre-fix):* scroll up mid-reply — none of the reply is in scrollback; combined
with BT-01, most of a long reply was unreadable *anywhere* until settling. *Reason:* live regions
are not scrollback. *Decision taken:* committed blocks become ordinary printed lines immediately.

**BT-04 · The display's frame-rate limit was bypassed** — P3 — **Status: fixed** in `4b27c4c`.
*Reason:* each chunk forced an immediate repaint, so the configured 4 fps cap did nothing;
repaints now ride the refresh clock.

**BT-05 · Chat history stored the display's escaped text, not the model's** — P2 — **Status:
fixed** in `4b27c4c`. *Repro (pre-fix):* have the model emit `<think>`; `!save`; the file
contains `\<think\>`, and the escaped text is also what got sent back as context on later turns.
*Reason:* escaping was applied before accumulation. *Decision taken:* history keeps raw text;
escaping is display-only.

**BT-06 · Tag escaping corrupted fenced code** — P2 — **Status: fixed** in `4b27c4c`. *Repro
(pre-fix):* ask for an HTML snippet in a code block; `<div>` renders as `\<div\>` inside the
fence. *Reason:* the escape ran on every line, fence or not. *Decision taken:* fence-aware line
preparation.

**BT-07 · Tags split across stream chunks escaped the escaping** — P3 — **Status: fixed** in
`4b27c4c`. *Reason:* the escape ran per network chunk, so `<thi` + `nk>` matched nothing.
*Decision taken:* prepare per completed line instead.

---

**BT-08 · Ctrl+C during generation kills the whole session with a traceback** — P1 — **Status:
open**; confirmed in the scripted rig (exit code −2, `CancelledError` chained into
`KeyboardInterrupt`). *Repro:* interrupt any streaming reply. The streamed-so-far text is
committed and readable (post-rework), but the session — and its conversation — is gone, under a
multi-screen traceback. *Reason:* the asyncio runner converts SIGINT into a task cancellation;
the loop's own `except KeyboardInterrupt` never fires for it. *Decision needed:* Ctrl+C
mid-stream should stop the generation and return to the prompt with the partial reply in history
(every comparable chat CLI's behavior); a second Ctrl+C can exit. Requires running the stream as
a cancellable task and handling the interrupt at the runner boundary.

**BT-09 · Ctrl+C at the prompt does nothing — then the next Enter destroys the session** — P1 —
**Status: open**; confirmed in the scripted rig. *Repro:* press Ctrl+C at the prompt (nothing
happens), type a message, press Enter: the message is not sent; the session dies with the BT-08
traceback. *Reason:* the deferred cancellation from the swallowed SIGINT lands at the next
`await` — the send. *Decision needed:* same rework as BT-08; at the prompt, Ctrl+C should clear
the line or exit cleanly, immediately.

**BT-10 · Ctrl+D ends the session with an `EOFError` traceback** — P2 — **Status: open**;
confirmed in the scripted rig (exit code 1). *Decision needed:* treat end-of-input as `!exit`.

**BT-11 · Every failure surfaces as a traceback** — P2 — **Status: open**. Unreachable default
server (the one case with a good message — it is printed *inside* a `ValueError` traceback),
custom endpoints failing at load, server errors during load, connection loss mid-stream, HTTP
errors mid-turn. *Decision needed:* catch at the session boundary; print the message, keep the
session when the conversation can survive (mid-turn failures), exit cleanly when it cannot
(startup).

**BT-12 · Tag escaping still leaks into inline code and indented code blocks** — P3 — **Status:
open**. `` `<think>` `` shows a stray backslash; a `<div>` inside four-space-indented code gains
backslashes. *Reason:* the escape is line-level and only fences suppress it. *Decision needed:*
accept as a documented edge (models use fences), or teach preparation about inline spans and
indented code.

**BT-13 · `!save NAME` is advertised but unreachable, and the help promises the wrong format** —
P2 — **Status: open**. *Repro:* `!save mychat` → red "not a valid command" plus the help, which
itself documents `!save {SAVE_NAME}` and a `.yaml` default; an argument-less `!save` writes
`.json`. *Reason:* the command matcher requires fewer than two words, so the name branch can
never run; the help string predates the format. *Decision needed:* honor the name, fix the help.

**BT-14 · The canned continue message ends with a stray typographic quote** — P3 — **Status:
open**. `Please continue. Do not repeat text.”` — echoed on screen and stored in history.

**BT-15 · An empty input line is sent to the model as an empty message** — P3 — **Status: open**.
*Decision needed:* re-prompt on empty input.

**BT-16 · A dead duplicate of the command handler drifts from the real one** — P3 (engineering
hygiene with user-facing risk) — **Status: open**. A never-called function re-implements the
command handling with different behavior (its `!save NAME` works); the next editor may fix the
wrong copy. *Decision needed:* delete or adopt it.

**BT-17 · `!example` silently discards the system prompt** — P2 — **Status: open**. `!clear`
rebuilds the conversation with the system prompt; `!example` resets it to empty. *Decision
needed:* make `!example` rebuild like `!clear`.

**BT-18 · `!example` with an unknown name still sends a request** — P2 — **Status: open**. The
error message prints, then control falls through to the send: the model answers the previous
conversation again (or an empty one). *Reason:* the command branch does not return to the prompt
like every other command. *Decision needed:* make invalid `!example` end the turn at once.
