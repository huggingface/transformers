# Model integration contracts (proof)

Vendored from [huggingface/model-integration-contracts](https://github.com/huggingface/model-integration-contracts) (`example/`, fb6c686) for the Transformers CI proof. Do not edit here; change the source repository and re-vendor.

- `run.py`, `select_contracts.py`, `schemas/`: the contract runner.
- `tests/contracts/fleet-lock.json`: pinned contracts, fixtures, checkpoints, and environment. Contracts and fixtures are downloaded from the Hub (`hf-internal-testing/*`) at their pinned revisions and checked against their hashes. No `local_patches`: open framework bugs are narrowed `known_failures` entries (an error message to match, or outputs to skip).
- `tests/contracts/expectations/<contract>/{cpu-x86_64,cpu-x86_64-bf16,cpu-arm64}/`: reviewed baselines. CI uses `cpu-x86_64` (and `cpu-x86_64-bf16` for bfloat16 fixtures); `cpu-arm64` (Apple silicon) lets a failure reproduce on a Mac with the same command.

```sh
python utils/contracts/run.py --framework transformers --fixtures-only --lock tests/contracts/fleet-lock.json
git diff --name-only origin/main...HEAD | python utils/contracts/run.py --framework transformers --fixtures-only --lock tests/contracts/fleet-lock.json --changed-files -
```
