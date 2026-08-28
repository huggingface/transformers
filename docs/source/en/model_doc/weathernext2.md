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
*This model was published in HF papers on 2025-06-12 and contributed to Hugging Face Transformers on 2026-08-28.*

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

The 0.25° releases are four separate training runs rather than one checkpoint. Each repository holds all four: run 1 at
the root and the rest in `model2`/`model3`/`model4` subfolders, so `from_pretrained(..., subfolder="model3")` selects
one. Combining them into a multi-model ensemble - loading each in turn and pooling its members - is left to the caller,
as it is upstream. The examples below use the Mini checkpoint because it runs anywhere; the 0.25° models need roughly
50 GB per ensemble member.

## Getting initial conditions

Unlike text or images, the input here is a physical atmospheric state that has to come from somewhere. The checkpoints
are fine-tuned to be initialized from ECMWF HRES *analysis* - the operational best estimate of the current atmosphere -
rather than from reanalysis. [earth2studio](https://github.com/NVIDIA/earth2studio) wraps the public sources:

```python
from earth2studio.data import IFS

ifs = IFS(source="ecmwf")            # HRES analysis, 0.25°, updated four times a day
data = ifs(times, ["t2m", "msl", "u10m", "v10m", "z500", "t850"])
```

Three things need care when assembling the state:

- **Latitude order.** earth2studio returns latitudes descending (90 to -90); WeatherNext 2 expects them ascending.
  Getting this backwards produces a plausible-looking forecast with badly degraded skill rather than an error.
- **Two frames.** The model conditions on the current analysis and the one 6 hours earlier, in that order.
- **Missing fields.** `geopotential_at_surface` is static and is not in the IFS lexicon - take it from any reanalysis
  file, since it never changes. `sea_surface_temperature` is likewise absent; skin temperature masked to the ocean is
  a workable substitute, and ERA5 through [`earth2studio.data.ARCO`] is better when the initialization time is old
  enough to be covered.

The clock forcings are derived by [`~WeatherNext2FeatureExtractor.compute_forcings`], so they never need fetching.
For historical cases, ERA5 via ARCO or [WeatherBench 2](https://sites.research.google/weatherbench/) is simpler than
analysis, at the cost of a small mismatch with what these checkpoints were tuned for.

## Usage

The model itself works in a normalized space, and [`WeatherNext2FeatureExtractor`] owns everything physical: the per-variable
normalization statistics, the calendar-derived forcings, and the residual connection that turns the model's output back
into an atmospheric state.

```python
import numpy as np
import torch
from transformers import WeatherNext2ForWeatherForecasting, WeatherNext2FeatureExtractor

model = WeatherNext2ForWeatherForecasting.from_pretrained("kashif/weathernext2-mini", device_map="auto").eval()
processor = WeatherNext2FeatureExtractor.from_pretrained("kashif/weathernext2-mini")

# `state` maps each input variable to its values. Time-varying variables are
# [batch, num_input_timesteps, (levels,) lat, lon]; static ones are [lat, lon].
state = ...
valid_time = np.array([np.datetime64("2024-10-07T06:00:00").astype("datetime64[s]").astype(np.int64)])

inputs = processor(state, seconds_since_epoch=valid_time).to(model.device)
with torch.no_grad():
    outputs = model(**inputs, generator=torch.Generator().manual_seed(0))

forecast = processor.postprocess(outputs.prediction, state)
print(forecast["2m_temperature"].shape)  # (1, 181, 360)
```

The noise is drawn on the generator's device and moved to the model's, so a plain CPU `torch.Generator` gives the
same ensemble on any accelerator.

### Ensembles

Each member is one draw of the 32-dimensional noise vector through the same weights, so an ensemble is a batch: stack
one vector per member and run a single forward.

```python
members = 8
inputs = processor(state, seconds_since_epoch=valid_time).to(model.device)
batched = {key: value.repeat(members, *([1] * (value.ndim - 1))) for key, value in inputs.items()}

noise = torch.stack(
    [
        torch.randn(model.config.noise_channels, generator=torch.Generator().manual_seed(member))
        for member in range(members)
    ]
)
with torch.no_grad():
    outputs = model(**batched, noise=noise.to(model.device))
```

Seeding per member rather than drawing from one stream means the first `n` members are the same however large the
ensemble is - the property the original implementation gets from `jax.random.fold_in`. Passing `generator=` instead of
`noise=` lets the model draw every member from one stream, which is shorter to write but gives up that property.

At 0.25° a member needs roughly 50 GB, so there the members have to go one at a time. That is also what the reference
implementation does, one member per device, looping or sharding rather than batching:

```python
predictions = []
for member in range(members):
    noise = torch.randn(1, model.config.noise_channels, generator=torch.Generator().manual_seed(member))
    with torch.no_grad():
        predictions.append(model(**inputs, noise=noise.to(model.device)).prediction)
```

Both give the same ensemble, agreeing to within float noise rather than bit-exactly, since a batched matmul reduces in
a different order than a batch of one.

Note that the batch axis serves double duty: it carries ensemble members here, and independent initialization times
when several forecasts are run together. The reference implementation keeps these as separate `sample` and `batch`
dimensions.

### Autoregressive rollout

Each 6-hour step draws fresh noise. [`~WeatherNext2FeatureExtractor.advance_state`] assembles the next conditioning window:
it drops the oldest frame, appends the forecast, recomputes the clock variables, and discards the targets that are not
also inputs (precipitation and the cyclone diagnostics).

[`WeatherNext2FeatureExtractor`] hands back whatever array type it was given, so moving the state onto the accelerator
once keeps the whole loop there and avoids a host transfer per step.

```python
state = {name: torch.as_tensor(values).to(model.device) for name, values in state.items()}

step_seconds = processor.time_step_hours * 3600
for step in range(1, 21):  # 5 days
    inputs = processor(state, seconds_since_epoch=valid_time)
    with torch.no_grad():
        outputs = model(**inputs)
    forecast = processor.postprocess(outputs.prediction, state)
    # `valid_time` is the time this forecast is valid at, so it stamps the frame being appended.
    # Only then does it move on to the step after.
    state = processor.advance_state(state, forecast, valid_time)
    valid_time = valid_time + step_seconds
```

Passing numpy arrays instead works exactly the same way and returns numpy, at the cost of two transfers per step.

### Tropical cyclones

`kashif/weathernext-cyclones` predicts 17 cyclone diagnostics alongside the usual atmospheric
fields: a per-gridpoint existence probability, `cyclone_exists_gaussian_unit_mode`, plus intensity,
radius of maximum winds, and the twelve wind radii, one per quadrant at 34, 50 and 64 knots. They
come out of [`~WeatherNext2FeatureExtractor.postprocess`] under the names the upstream tracker in
[weathernext](https://github.com/google-deepmind/weathernext) already looks for, so a rollout can be
tracked without renaming anything:

```python
import numpy as np
import xarray
from weathernext.cyclones import direct_tracker_6h_v1_config

# `frames` holds one postprocessed forecast per step, and `analysis` the state it started from.
cyclone_fields = [name for name in processor.target_variables if name.startswith("cyclone_")]
gridded = xarray.Dataset(
    {
        name: (("time", "lat", "lon"), np.stack([analysis[name]] + [frame[name] for frame in frames]))
        for name in cyclone_fields
    },
    coords={
        # lead time, not valid time, and lead 0 has to be present
        "time": np.arange(0, 6 * (len(frames) + 1), 6).astype("timedelta64[h]").astype("timedelta64[ns]"),
        "lat": latitudes,
        "lon": longitudes,
        "init_time": np.datetime64("2024-10-07T00:00:00"),
    },
)

config = direct_tracker_6h_v1_config.get_config()
tracker = config.tracker_constructor(**config.tracker_kwargs)
tracks = tracker(gridded_ds=gridded, initial_storms_df=initial_storms, do_cyclogenesis=True)
```

`initial_storms_df` is the frame the tracks are initialized from, one row per storm present at lead
time 0 with `track_id`, `lat`, `lon` and `valid_time`; upstream builds it from IBTrACS observations.
Storms that appear later are picked up by cyclogenesis from the existence field.

Tracking does not have to use the cyclone head at all. earth2studio's `TCTrackerWuDuan` finds
cyclones from vorticity and pressure, so it needs only `10m_u_component_of_wind`,
`10m_v_component_of_wind`, `mean_sea_level_pressure` and the 850 hPa winds, and runs on any of these
checkpoints.

### Training

[`WeatherNext2ForWeatherForecasting`] does not take `labels` and returns no loss, because the loss
WeatherNext 2 is trained on is not a per-example one. It is *fair CRPS* over an ensemble:

```
CRPS = mean over members of MAE(member, target) - 0.5 * mean over member pairs of MAE(member, member)
```

The second term rewards spread, so a model that hedges by predicting the ensemble mean is penalised,
and a single prediction is not enough to score: each optimizer step needs several members, which is
what the batch axis carries. The MAE is weighted by grid-cell area and by pressure level, and summed
over variables.

Two things to know when writing that loss:

- It is computed in the model's own normalized residual space, not in physical units. That is what
  `prediction` already is, so compare it against targets encoded the same way rather than calling
  [`~WeatherNext2FeatureExtractor.postprocess`] first.
- Wherever a target is missing (sea surface temperature over land) *both* terms have to be masked.
  Masking only the accuracy term lets the loss be driven arbitrarily negative by pushing members
  apart at points that are never scored.

Gradients reach every parameter except the six of `model.mesh_to_grid.mesh_node_update`. The decoder
updates the mesh nodes as well as the grid points, but the forecasting head reads only the grid, so
those weights cannot move. This matches the original implementation, which names the same value
`unused_updated_latent_mesh_data`.

## Notes

- The mesh, both grid↔mesh graphs and the attention mask follow deterministically from the configuration, but nothing
  about them is learned and nothing about them ever changes, so they are computed once when a checkpoint is converted
  and stored in it as buffers. Loading a model needs no geometry library; a model built from a config rather than
  loaded gets a placeholder in their place and will not forecast anything meaningful until weights are loaded. The
  derivation lives in `convert_weathernext2_original_checkpoint.py`, needs `scipy` and `trimesh`, and takes about two
  minutes at 0.25°.
- Attention is computed over the three block-diagonals induced by the reverse Cuthill-McKee ordering of the mesh. This
  is exactly equivalent to masking the full attention matrix, but avoids materializing a mask that would be 1.7 GB at
  0.25°.
- `eager`, `sdpa` and `flex_attention` are all supported and agree to within float noise; `sdpa` is the default and
  the fastest. Flash Attention is not supported: it cannot take an arbitrary attention mask.
- The 0.25° checkpoints need roughly an H100's worth of memory for a single ensemble member. The 1° mini checkpoints
  run comfortably on a much smaller GPU.
- Model weights are released by Google DeepMind under CC-BY-4.0, separately from the Apache-2.0 code.

## WeatherNext2Config

[[autodoc]] WeatherNext2Config

## WeatherNext2FeatureExtractor

[[autodoc]] WeatherNext2FeatureExtractor
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
