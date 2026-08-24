# Standing instructions for drafting

You are writing or revising documents in the `transformers chat` product description. Read
[README.md](README.md) first; it defines the project, the document template, and the coverage
table. These are the standing instructions that stay true while you work.

## The one rule

Describe what a user sees and does, in the order they see and do it. If you catch yourself naming a
class, a function, or a library outside a `Technical note:` block quote, rewrite the sentence from
the user's side of the glass.

## Sources, in order of authority

1. **The running product.** `transformers serve` plus `transformers chat <model>` in a real
   terminal, or the scripted rig in [verification/README.md](verification/README.md) when you need
   exact terminal sizes, timing, or interrupts. What the product does wins over what any document
   or comment says it does.
2. **The tests.** `tests/cli/test_chat_display.py` encodes the streaming display's two promises
   (the final transcript equals the one-shot render; the newest lines stay visible while
   streaming). `tests/cli/test_chat.py` covers command helpers. A behavior asserted by a test may
   be described as fact.
3. **The code.** `src/transformers/cli/chat.py` and `src/transformers/cli/chat_display.py`.
   Reading code is how you find behaviors to verify, not how you verify them: code-only claims go
   in the document's Open questions until tried.

## While drafting

- Follow the eight-part skeleton in README.md exactly, including the fixed interrupt checklist in
  part 5 and the fixed cross-cutting order in part 6. Identical lists in every document is the
  method: a gap in the grid is a finding.
- Use glossary terms exactly ([glossary.md](glossary.md)). If you need a new term, add it to the
  glossary in the same change.
- Every surprising behavior gets a sentence saying it is surprising. If it looks like a defect,
  add an entry to [bug-triage.md](bug-triage.md) — repro steps, reason in the code, severity,
  decision needed — and cite its ID from the document.
- Every claim you verified, verify on one commit, and write that commit at the bottom of the
  document. Claims you could not verify go under Open questions, phrased as questions.
- Keep Mermaid diagrams small: states the user can tell apart, nothing internal.

## When changing the product

If you change `chat.py` or `chat_display.py`, the corresponding documents are part of the diff:
update them and re-run the affected checklist items in `verification/`, then update the coverage
table if a document's status changed. The description and the product must not drift apart — a
stale document here is worse than none.

## Updating coverage

The Structure block and the Coverage table in README.md are exact mirrors of the files on disk.
Adding, renaming, or deleting a document means updating both in the same change.
