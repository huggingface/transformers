# Model integration contracts (proof)

Vendored from [huggingface/model-integration-contracts](https://github.com/huggingface/model-integration-contracts) (`example/`, commit b303626) for the Transformers CI proof (phase 1: does the fixture baseline hold on the PR CI runner and image?). Do not edit here; change the source repository and re-vendor.

- `run.py`, `select_contracts.py`, `patches.py`, `schemas/`: the contract runner.
- `transformers/tests/contracts/fleet.lock`: pinned contracts, fixtures, checkpoints, and environment; the contracts and fixtures are downloaded from the Hub (`hf-internal-testing/*`) by revision and checked against their hashes.
- `transformers/tests/contracts/expectations/<contract>/{cpu-x86_64,cpu-x86_64-bf16}/`: reviewed baselines.

Run by `.github/workflows/contracts-probe.yml` on pushes to `contracts-proof`.
