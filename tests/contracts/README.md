# Model integration contracts

Model integration contracts (MICs) check model behavior against reviewed baselines
using small, fast model fixtures. PR CI selects relevant cases from changes to
model and processor code and reports differences in logs, summaries and OpenTelemetry.
Baseline failures are observational and do not fail the CI job.

The fleet locks in this directory pin model repositories and revisions in the
[`huggingface-mic`](https://huggingface.co/huggingface-mic) Hub organization.
Each repository contains model weights, `integration.json`, usage examples and
framework expectations under
`integrations/<framework>/expectations/<platform>/<target>[.<variant>].json`.

The runner is maintained in
[`transformers-ci`](https://github.com/huggingface/transformers-ci).
To run MICs locally from the Transformers checkout:

```bash
pip install "transformers-ci[contracts,otel] @ git+https://github.com/huggingface/transformers-ci@main"
run-model-contracts --fixtures-only --lock tests/contracts/prototype-fleet.lock --variant bfloat16
run-model-contracts --fixtures-only --lock tests/contracts/fleet-lock.json \
  --contract SmolLM --target mic-SmolLM-slim
```

CI compares outputs against the pinned expectations; it never records new baselines.
Review measured behavior changes before updating expectations or lock revisions.
