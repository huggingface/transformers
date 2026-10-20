# Model integration contracts

These locks pin consolidated `huggingface-mic/<name>-mic` Hub model repositories.
Each snapshot contains slim weights, `integration.json`, runnable usage and
`integrations/<framework>/expectations/<platform>/<target>[.<variant>].json`.

The runner lives in `huggingface/transformers-ci` (`transformersci.contracts`).
PR CI selects cases from changed model and processor dependencies, compares the
reviewed baselines and emits OTEL case spans. Failures remain observational:
they appear in logs, summaries and telemetry while the GitHub job stays green.

```bash
run-model-contracts --fixtures-only --lock tests/contracts/prototype-fleet.lock --variant bfloat16
run-model-contracts --fixtures-only --lock tests/contracts/fleet-lock.json \
  --contract SmolLM --target mic-SmolLM-slim
```

The explicit fixture `target` preserves historical baseline names after repo
consolidation. `.mic-repo.json` separates contract and fixture content inventories;
commit pins also freeze the framework expectations. Update pins only after
measuring and reviewing changes. The runner never records baselines in CI.

These locks require the consolidated-repo runner in transformers-ci. Land that
runner before switching CI consumers to this lock format.
