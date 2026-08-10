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

# WeatherNext 2

[WeatherNext 2](https://huggingface.co/papers/2506.10772) is a global medium-range weather forecasting model
from Google DeepMind and Google Research. It advances the state of the atmosphere by 6 hours per forward pass, on a 0.25° latitude/
longitude grid (or 1° for the mini checkpoints), and predicts 13 pressure levels of temperature, geopotential, wind and
humidity along with surface fields, precipitation and a set of tropical-cyclone diagnostics.

Architecturally it is an encode–process–decode graph network. The lat/lon grid is encoded, projected onto an
icosahedral mesh through a graph network, processed by a transformer whose attention is restricted to a local
neighbourhood on that mesh, projected back onto the grid, and decoded. There are no positional encodings anywhere:
position is carried entirely by the mesh geometry and by the attention mask.

What makes it a *Functional Generative Network* is how it becomes probabilistic. Rather than injecting a noise field
into the input, or running a diffusion sampler, the model draws a single 32-dimensional noise vector per ensemble
member and uses it to modulate the scale and offset of **every** normalization layer in the network. One draw gives one
self-consistent forecast; an ensemble is just several draws. In [`WeatherNext2ForWeatherForecasting`] the ensemble is
the batch dimension.

This model was contributed by the Hugging Face team. The original code can be found
[here](https://github.com/google-deepmind/weathernext).

Released checkpoints, all sharing this architecture:

| Checkpoint | Resolution | Params | Notes |
|---|---|---|---|
| [kashif/weathernext2-mini](https://huggingface.co/kashif/weathernext2-mini) | 1° | 56.7M | runs on modest hardware; `main` and `<2023` revisions |
| [kashif/weathernext2](https://huggingface.co/kashif/weathernext2) | 0.25° | 183.8M | also predicts 100m winds |
| [kashif/weathernext-cyclones](https://huggingface.co/kashif/weathernext-cyclones) | 0.25° | 183.8M | operational cyclone model; `main`, `<2024`, `<2023` revisions |

The 0.25° repositories hold all four independently trained ensemble members: member 1 at the root and the rest in
`model2`/`model3`/`model4` subfolders, so `from_pretrained(..., subfolder="model3")` selects one. The examples below
use the Mini checkpoint because it runs anywhere; the 0.25° models need roughly 50 GB per ensemble member.

## Usage

The model itself works in a normalized space, and [`WeatherNext2Processor`] owns everything physical: the per-variable
normalization statistics, the calendar-derived forcings, and the residual connection that turns the model's output back
into an atmospheric state.

```python
import numpy as np
import torch
from transformers import WeatherNext2ForWeatherForecasting, WeatherNext2Processor

model = WeatherNext2ForWeatherForecasting.from_pretrained("kashif/weathernext2-mini").eval()
processor = WeatherNext2Processor.from_pretrained("kashif/weathernext2-mini")

# `state` maps each input variable to its values. Time-varying variables are
# [batch, num_input_timesteps, (levels,) lat, lon]; static ones are [lat, lon].
state = ...
valid_time = np.array([np.datetime64("2024-10-07T06:00:00").astype("datetime64[s]").astype(np.int64)])

inputs = processor(state, seconds_since_epoch=valid_time)
with torch.no_grad():
    outputs = model(**inputs, generator=torch.Generator().manual_seed(0))

forecast = processor.postprocess(outputs.prediction, state)
print(forecast["2m_temperature"].shape)  # (1, 181, 360)
```

### Ensembles

A forecast ensemble is produced by repeating the same inputs along the batch axis and letting each member get its own
noise draw. Because the members are independent, they can also be split across devices.

```python
members = 8
inputs = processor(state, seconds_since_epoch=valid_time)
inputs = {key: value.repeat(members, *([1] * (value.ndim - 1))) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, generator=torch.Generator().manual_seed(0))
```

### Autoregressive rollout

Each 6-hour step draws fresh noise. [`~WeatherNext2Processor.advance_state`] assembles the next conditioning window:
it drops the oldest frame, appends the forecast, recomputes the clock variables, and discards the targets that are not
also inputs (precipitation and the cyclone diagnostics).

```python
step_seconds = processor.time_step_hours * 3600
for step in range(1, 21):  # 5 days
    inputs = processor(state, seconds_since_epoch=valid_time)
    with torch.no_grad():
        outputs = model(**inputs)
    forecast = processor.postprocess(outputs.prediction, state)
    valid_time = valid_time + step_seconds
    state = processor.advance_state(state, forecast, valid_time)
```

## Notes

- The mesh, both grid↔mesh graphs and the attention mask follow deterministically from the configuration, so they are
  computed at instantiation rather than stored in the checkpoint. At 0.25° this takes tens of seconds the first time;
  the result is cached under `HF_HOME`.
- Attention is computed over the three block-diagonals induced by the reverse Cuthill-McKee ordering of the mesh. This
  is exactly equivalent to masking the full attention matrix, but avoids materializing a mask that would be 1.7 GB at
  0.25°.
- `eager`, `sdpa` and `flex_attention` are all supported and agree to within float noise. Flash Attention is not:
  its kernels only express causal and sliding-window patterns, and this mask is neither.
- The 0.25° checkpoints need roughly an H100's worth of memory for a single ensemble member. The 1° mini checkpoints
  run comfortably on a much smaller GPU.
- Model weights are released by Google DeepMind under CC-BY-4.0, separately from the Apache-2.0 code.

## WeatherNext2Config

[[autodoc]] WeatherNext2Config

## WeatherNext2Processor

[[autodoc]] WeatherNext2Processor
    - __call__
    - postprocess
    - advance_state
    - compute_forcings

## WeatherNext2Model

[[autodoc]] WeatherNext2Model
    - forward

## WeatherNext2ForWeatherForecasting

[[autodoc]] WeatherNext2ForWeatherForecasting
    - forward
