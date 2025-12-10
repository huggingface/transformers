# Claude Code Instructions

## Git Workflow

**IMPORTANT: Do not push directly to upstream (huggingface/transformers).**

- `origin` = `lyfegame/transformers` (fork) - push here
- `upstream` = `huggingface/transformers` - DO NOT push here

When committing and pushing changes:
1. Always push to `origin` (the fork), not `upstream`
2. The user will manually create PRs to upstream

```bash
# Correct - push to fork
git push origin <branch-name>

# WRONG - never do this
git push upstream <branch-name>
```
