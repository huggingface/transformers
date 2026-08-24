# Agents: start here

This directory is a product description of `transformers chat` — prose documents describing the
user experience, used as a specification and verification harness.

1. Read [README.md](README.md): purpose, document template, coverage.
2. Read [goal.md](goal.md): the standing instructions for drafting and revising.
3. Use [glossary.md](glossary.md) vocabulary in everything you write here.

If you are changing the chat CLI itself (`src/transformers/cli/chat.py`,
`src/transformers/cli/chat_display.py`), treat the documents here as part of your diff: update them
with the behavior change and re-run the matching checklist items in `verification/`.
