# Model integration contracts (proof)

The PR CI workflow installs the immutable `TMIC_SHA` of
[huggingface/transformers-mic](https://github.com/huggingface/transformers-mic) and
uses `tmic check` and `tmic report`. The old vendored scripts are retained as a
reference for the original proof; CI executes the installed package.

The first PR wave selects the BF16 variant of 23 published prototype contracts
and one historical SmolLM control. Changed files select affected contracts;
shared Transformers code selects the whole wave. This is an observational job:
failures produce a warning and a job summary but do not block the PR. Download
the `contracts` artifact for per-case JSONL, the runner log and
`contracts-observation.json` with status counts, selection reasons and source
metadata. A run with no selected contracts has no validation result. The
current self-hosted runner is limited to PRs from repository members and
collaborators. External fork PRs need an isolated runner path before inclusion.
The job uses the current `transformers-torch-light` image's PyTorch and
scientific packages; it does not replace them with the historical recording
versions. The artifact records installed versions, and each case reports
environment differences from its reviewed baseline.

- `tests/contracts/fleet-lock.json`: pinned contracts, fixtures, checkpoints, and environment. Contracts and fixtures are downloaded from the Hub (`hf-internal-testing/*`) at their pinned revisions and checked against their hashes. No `local_patches`: open framework bugs are narrowed `known_failures` entries (an error message to match, or outputs to skip).
- `tests/contracts/expectations/<contract>/{cpu-x86_64,cpu-x86_64-bf16,cpu-arm64}/`: reviewed baselines. CI uses `cpu-x86_64` (and `cpu-x86_64-bf16` for bfloat16 fixtures); `cpu-arm64` (Apple silicon) lets a failure reproduce on a Mac with the same command.

```sh
tmic check --fixtures-only --lock tests/contracts/fleet-lock.json
git diff --name-only origin/main...HEAD > /tmp/mic-changed-files.txt
tmic check --fixtures-only --lock tests/contracts/fleet-lock.json --changed-files /tmp/mic-changed-files.txt
```

`tests/contracts/prototype-fleet.lock` adds 23 `tmic`-generated architecture
contracts, checked against published Hub pins on CPU and GPU before export.
The prototype lock uses `mic-*` contract keys to avoid case collisions with
historical names such as `BERT`. The current PR wave runs only their BF16
variants; default and eager results remain historical proof evidence. The
original six-contract fleet remains historical regression evidence, with only
the SmolLM control included in this wave. Pushes to `contracts-proof` run all
24 selected cases and check that none were missed.

The prototype CPU expectations were measured on the GPU worker's x86 CPU under
the pinned runtime. This branch also checks them on the dedicated CPU CI runner;
passing the GPU-worker CPU check alone does not establish that result.
