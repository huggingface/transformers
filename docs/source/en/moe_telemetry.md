<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# MoE telemetry

Use MoE telemetry to monitor router health during training without changing model outputs or exposing per-token expert assignments through the default [`Trainer`] API.

The first version focuses on trainer-friendly scalar metrics:

- entropy
- normalized entropy
- load coefficient of variation (CV)
- max-load ratio
- active experts
- dead experts

These metrics are logged through the standard [`Trainer`] callback path, so experiment trackers continue to receive ordinary flat scalar dictionaries. Exact expert assignments remain internal to the model unless a separate replay or debug feature explicitly exposes them.

## Logging router health with a callback

The intended implementation is a built-in [`TrainerCallback`] that:

- reads router activity from the model without changing default model outputs
- prefers exact selected expert indices when a router surfaces them internally
- falls back to router-logit-derived top-k assignments when exact indices are not available
- emits flat scalar metrics through the normal trainer logging path
- keeps routing telemetry memory-safe by aggregating expert counts immediately instead of storing full routing tensors

```python
from transformers import MoERouterHealthCallback, Trainer


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[MoERouterHealthCallback()],
)
```

The callback aggregates per-layer expert counts during forwards, then emits a flat `dict[str, float]` during trainer logging. That keeps the trainer interface unchanged while making the metrics usable with standard experiment trackers.

The built-in callback uses a reduction policy rather than blindly reducing over all distributed ranks:

- normal distributed replicas: reduce counts across the world process group so trainer logs show global metrics
- tensor-parallel MoE models: do not implicitly world-reduce replicated router counts
- local-only debugging: disable implicit reduction explicitly

This keeps the default behavior intuitive for common `Trainer` usage while avoiding overcounting in `tp_plan`-based MoE runs.

## Distributed and DeepEP-style settings

The metric definitions are based on routing assignments, not transport internals.

For distributed MoE systems:

1. compute local expert counts from routing decisions
2. optionally reduce those counts across the expert group
3. derive health metrics from the reduced counts

This is why the callback design reduces per-expert counts, not transport-specific state. Backends such as standard expert parallel or DeepEP can reduce those counts before computing the final scalar metrics, while keeping the metric API itself backend-agnostic.

Use local metrics when you want rank-local visibility. Use reduced counts when you want trainer-facing global health metrics. In particular, tensor-parallel MoE models may replicate routing state across ranks, so global world-size reduction is not always the correct default.

## Related docs

- [Callbacks](./trainer_callbacks)
- [Experts backends](./experts_interface)
- [Expert parallelism](./expert_parallelism)
