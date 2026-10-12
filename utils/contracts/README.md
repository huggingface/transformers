# Model integration contracts (proof)

The CI workflow installs the immutable `TMIC_SHA` of
[huggingface/transformers-mic](https://github.com/huggingface/transformers-mic) and
uses `tmic check` and `tmic report`. The old vendored scripts are retained as a
reference for the original proof; CI executes the installed package.


- `tests/contracts/fleet-lock.json`: pinned contracts, fixtures, checkpoints, and environment. Contracts and fixtures are downloaded from the Hub (`hf-internal-testing/*`) at their pinned revisions and checked against their hashes. No `local_patches`: open framework bugs are narrowed `known_failures` entries (an error message to match, or outputs to skip).
- `tests/contracts/expectations/<contract>/{cpu-x86_64,cpu-x86_64-bf16,cpu-arm64}/`: reviewed baselines. CI uses `cpu-x86_64` (and `cpu-x86_64-bf16` for bfloat16 fixtures); `cpu-arm64` (Apple silicon) lets a failure reproduce on a Mac with the same command.

```sh
tmic check --fixtures-only --lock tests/contracts/fleet-lock.json
git diff --name-only origin/main...HEAD > /tmp/mic-changed-files.txt
tmic check --fixtures-only --lock tests/contracts/fleet-lock.json --changed-files /tmp/mic-changed-files.txt
```

`tests/contracts/prototype-fleet.lock` adds 17 `tmic`-generated architecture
contracts, checked against published Hub pins on CPU and GPU before export.
These run in default, eager and bfloat16 variants. The original six-contract
fleet remains the historical regression control. Both fleets use changed-file
selection on PRs and run in full on pushes to `contracts-proof`.

The prototype CPU expectations were measured on the GPU worker's x86 CPU under
the pinned runtime. This branch also checks them on the dedicated CPU CI runner;
passing the GPU-worker CPU check alone does not establish that result.
