# Rename to sarah - Plan and helper

This folder contains a helper script to perform a repository-wide rename from `transformers`/`sarah` to `sarah`.

IMPORTANT: This is a potentially breaking change. The script in this folder performs a dry-run by default and will not change files unless you explicitly pass `--apply` and confirm.

Workflow summary

1. Create and switch to the branch (already created): `git checkout -b rename-to-sarah`.
2. Generate an annotated list of occurrences and a preview of replacements:
   ```powershell
   python rename_plan/apply_patch.py --dry-run
   ```
3. Review the generated files in `rename_plan/preview/`.
4. If you want to perform only safe non-code replacements (docs, READMEs, CI), run:
   ```powershell
   python rename_plan/apply_patch.py --apply --mode noncode
   ```
5. To prepare the full package rename (including moving `src/transformers` -> `src/sarah` and updating imports), run:
   ```powershell
   python rename_plan/apply_patch.py --apply --mode full
   ```
   This will perform more invasive edits and create a compatibility shim at `src/transformers/__init__.py`.

Files produced by the script (in dry-run):
- `rename_plan/preview/transformers_matches.txt` - list of files and lines containing `transformers`.
- `rename_plan/preview/sarah_matches.txt` - list of files and lines containing `sarah`.
- `rename_plan/preview/proposed_changes.patch` - unified diff of proposed changes (only when `--apply` is used with `--generate-patch`).

Notes
- Default package name used by the script: `sarah` (lowercase). Change by passing `--pkg sarah`.
- The script will NOT commit changes; it will modify files in-place only when `--apply` is passed. Use `git status` to review.
- I will not push or open a PR; you must review locally and push when ready.

If you want me to proceed further (apply non-code changes now on the branch), reply `APPLY_NONCODE` and I will apply them locally on the branch (but will not push).