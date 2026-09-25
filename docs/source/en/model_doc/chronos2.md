<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not
be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-31.*

# Chronos-2

## Overview

Chronos-2 is a 120M-parameter, encoder-only time-series foundation model introduced in
[Chronos-2: From Univariate to Universal Forecasting](https://huggingface.co/papers/2510.15821) by Abdul Fatir
Ansari et al. It produces direct multi-step quantile forecasts for univariate, multivariate, and
covariate-informed tasks. Context values are divided into patches and processed by alternating time attention and
group attention. Rows with the same group ID can exchange information, which allows target and covariate series to
form one forecasting task.

The original implementation is available in the
[Amazon Chronos forecasting repository](https://github.com/amazon-science/chronos-forecasting), and the released
checkpoint is [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2). The native implementation adapts
Apache-2.0-licensed Amazon source and retains its prepared-tensor model behavior. High-level DataFrame handling,
categorical encoding, ragged-series batching, and task packing remain available in the external
`chronos-forecasting` package and are not duplicated in Transformers.

## Loading the released checkpoint

The released checkpoint's configuration declares `model_type="t5"` for historical compatibility. Load that
unchanged checkpoint with the explicit native class:

```python
import torch

from transformers import Chronos2Model


model = Chronos2Model.from_pretrained("amazon/chronos-2", dtype=torch.float32)
model.eval()
```

The explicit class recognizes the legacy configuration and does not require `trust_remote_code=True`. Generic Auto
classes read the legacy `model_type` first and therefore resolve the unchanged checkpoint as T5. Convert it once to
write the native `model_type="chronos2"` configuration while preserving every tensor unchanged:

```bash
python src/transformers/models/chronos2/convert_chronos2_original_to_hf.py \
    --checkpoint amazon/chronos-2 \
    --revision 29ec3766d36d6f73f0696f85560a422f50e8498c \
    --output_path ./chronos-2-transformers
```

The converter accepts either a local checkpoint directory or a Hub model ID. It strictly checks tensor keys,
shapes, dtypes, values, missing keys, and unexpected keys before writing a new directory. It never uploads files and
refuses to overwrite an existing output path. Add `--verify-original` to compare a deterministic grouped forecast
against the optional `chronos-forecasting` package.

The converted directory works with the time-series Auto class:

```python
from transformers import AutoModelForTimeSeriesPrediction


model = AutoModelForTimeSeriesPrediction.from_pretrained("./chronos-2-transformers")
```

## Prepared tensor API

`Chronos2Model` operates on already prepared rank-2 tensors. The batch axis represents individual target or
covariate series, not conventional independent examples:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `context` | `(num_series, history_length)` | Historical values for every target and covariate row. |
| `context_mask` | `(num_series, history_length)` | Optional observed-value mask (`1` observed, `0` missing). When omitted, it is inferred from NaNs. |
| `group_ids` | `(num_series,)` | Rows with the same ID form one task and share information through group attention. When omitted, every row is independent. |
| `future_covariates` | `(num_series, future_length)` | Known future values. Use NaNs in target or past-only-covariate rows whose future is unknown. |
| `future_covariates_mask` | `(num_series, future_length)` | Optional known-value mask. When omitted, it is inferred from NaNs. |
| `future_target` | `(num_series, future_length)` | Optional training targets used to compute quantile loss. |
| `future_target_mask` | `(num_series, future_length)` | Optional observed-target mask, inferred from NaNs when omitted. |

`num_output_patches` controls the direct forecast length. The output `quantile_preds` has shape
`(num_series, num_quantiles, num_output_patches * output_patch_size)`. It contains predictions for every input row,
including rows used as known covariates. Callers should retain their target-row indices and discard forecasts for
covariate-only rows.

The native API intentionally does not accept DataFrames, dictionaries of named variables, categorical values, or
ragged lists. Convert those inputs into the tensors above, or use `Chronos2Pipeline` from `chronos-forecasting` when
that preprocessing is desired.

## Univariate forecasting

Rows are independent when `group_ids` is omitted. This example forecasts 32 steps, which is two 16-step output
patches for the released checkpoint:

```python
import torch

from transformers import Chronos2Model


model = Chronos2Model.from_pretrained("amazon/chronos-2").eval()
context = torch.stack(
    [
        torch.sin(torch.arange(96, dtype=torch.float32) / 8),
        torch.cos(torch.arange(96, dtype=torch.float32) / 12),
    ]
).to(model.device)

with torch.no_grad():
    outputs = model(context=context, num_output_patches=2)

quantile_forecasts = outputs.quantile_preds  # (2, 21, 32)
median_index = model.config.chronos_config["quantiles"].index(0.5)
median_forecasts = quantile_forecasts[:, median_index]
```

## Multivariate and covariate-informed forecasting

Give related rows the same group ID to enable multivariate forecasting. Known future covariates use finite values;
unknown future targets use NaNs. For example, the first two rows below are targets and the third row is a known
temperature covariate:

```python
import math

import torch


history_length = 96
prediction_length = 24
steps = torch.arange(history_length, dtype=torch.float32, device=model.device)
context = torch.stack([torch.sin(steps / 8), torch.cos(steps / 12), 15 + steps / 48])
group_ids = torch.zeros(3, dtype=torch.long, device=model.device)

future_covariates = torch.full(
    (3, prediction_length),
    torch.nan,
    dtype=torch.float32,
    device=model.device,
)
future_covariates[2] = torch.linspace(17, 19, prediction_length, device=model.device)
output_patch_size = model.config.chronos_config["output_patch_size"]
num_output_patches = math.ceil(prediction_length / output_patch_size)

with torch.no_grad():
    outputs = model(
        context=context,
        group_ids=group_ids,
        future_covariates=future_covariates,
        num_output_patches=num_output_patches,
    )

target_forecasts = outputs.quantile_preds[:2, :, :prediction_length]
```

A past-only covariate is represented the same way, with historical values in `context` and NaNs across its future
row. Its forecast is ignored downstream. Multiple unrelated tasks can share a call by assigning a distinct group ID
to each task. Group ID values themselves have no ordering or numerical meaning.

## Quantiles and long horizons

The released model predicts 21 trained quantiles from `0.01` through `0.99`. `forward` returns those levels in the
order stored in `model.config.chronos_config["quantiles"]`; it does not interpolate arbitrary requested levels.

For a horizon longer than one configured direct chunk, use the tensor-only [`~Chronos2Model.predict`] method. It
retains the original quantile-path unrolling heuristic and returns the same
`(num_series, num_quantiles, prediction_length)` layout:

```python
long_future_covariates = torch.full(
    (3, 2048),
    torch.nan,
    dtype=torch.float32,
    device=model.device,
)
long_future_covariates[2] = torch.linspace(19, 28, 2048, device=model.device)

with torch.no_grad():
    long_forecast = model.predict(
        context=context,
        prediction_length=2048,
        group_ids=group_ids,
        future_covariates=long_future_covariates,
    )
```

`predict` infers missing and known values from NaNs and therefore does not take explicit context or future masks.
Long-horizon forecasting repeatedly appends selected quantile paths, preserves known future covariate values while
sliding the context, and recombines paths into the model's trained quantiles. Forecast quality can degrade beyond the
horizon used to train the checkpoint. When supplied, `future_covariates` should cover the full requested prediction
horizon. Like the original Chronos-2 pipeline, `predict` returns a CPU `float32` tensor regardless of the model device
or dtype.

## Chronos2Config

[[autodoc]] Chronos2Config

## Chronos2Model

[[autodoc]] Chronos2Model
    - encode
    - forward
    - predict

## Chronos2PreTrainedModel

[[autodoc]] Chronos2PreTrainedModel

## Chronos2Output

[[autodoc]] Chronos2Output
