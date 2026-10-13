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
*This model was published in HF papers on 2025-06-12 and contributed to Hugging Face Transformers on 2026-10-12.*

# WeatherNext 2

[WeatherNext 2](https://huggingface.co/papers/2506.10772) is a global medium-range weather forecasting model
from Google DeepMind and Google Research. It advances the state of the atmosphere by 6 hours per forward pass, on a 0.25° latitude/
longitude grid (or 1° for the mini checkpoints), and predicts 13 pressure levels of temperature, geopotential, wind and
humidity along with surface fields, precipitation and a set of tropical-cyclone diagnostics.

The model encodes the grid, projects it onto an icosahedral mesh, applies local transformer attention, and decodes
back onto the grid. Geometry carries positional information. Each ensemble member is defined by a 32-dimensional
noise vector that conditions the normalization layers.

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
as it is upstream. The examples below use the smaller Mini checkpoint.

## Getting initial conditions

Unlike text or images, the input here is a physical atmospheric state that has to come from somewhere. The checkpoints
are fine-tuned to be initialized from the *analysis* of the ECMWF high-resolution forecast (HRES) - the operational
best estimate of the current atmosphere - rather than from reanalysis.
[earth2studio](https://github.com/NVIDIA/earth2studio) wraps the public sources:

```python
from earth2studio.data import IFS

ifs = IFS(source="ecmwf")            # HRES analysis, 0.25°, updated four times a day
data = ifs(times, ["t2m", "msl", "u10m", "v10m", "z500", "t850"])
```

Three things need care when assembling the state:

- **Latitude order.** earth2studio returns latitudes descending (90 to -90); WeatherNext 2 expects them ascending.
  Getting this backwards produces a plausible-looking forecast with badly degraded skill rather than an error.
- **Two frames.** The model conditions on two chronological frames, six hours apart.
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
into an atmospheric state. It accepts NumPy arrays or tensors and returns PyTorch tensors by default.

The official WeatherNext 2 demo publishes a 1° example batch in the `dm_graphcast` bucket. It is also the sample used
by Earth2Studio's WeatherNext 2 wrapper. Install the packages used to read it with
`pip install google-cloud-storage h5netcdf xarray`.

```python
import numpy as np
import torch
import xarray
from google.cloud import storage
from transformers import WeatherNext2ForWeatherForecasting, WeatherNext2FeatureExtractor

model = WeatherNext2ForWeatherForecasting.from_pretrained("kashif/weathernext2-mini", device_map="auto")
processor = WeatherNext2FeatureExtractor.from_pretrained("kashif/weathernext2-mini")

# Download the three-frame HRES example batch used by Earth2Studio.
data_path = "weathernext2/dataset/source-hres_forecast_init-2024-10-07 00:00:00_res-1.0_levels-13_steps-01.nc"
data_file = "weathernext2-example.nc"
client = storage.Client.create_anonymous_client()
bucket = client.get_bucket("dm_graphcast")
bucket.blob(data_path).download_to_filename(data_file)
example_batch = xarray.load_dataset(data_file).compute()

# The first two frames are the conditioning state. The processor derives the
# clock forcings itself, so only physical fields are copied from the dataset.
initial = example_batch.isel(time=slice(0, processor.num_input_timesteps))
state = {}
for name in processor.input_variables:
    if name in processor.forcing_variables:
        continue
    field = initial[name]
    if name in processor.static_variables:
        field = field.isel({dim: -1 for dim in ("batch", "time") if dim in field.dims}, drop=True)
        field = field.transpose("lat", "lon")
    elif name in processor.atmospheric_variables:
        field = field.sel(level=processor.pressure_levels).transpose("batch", "time", "level", "lat", "lon")
    else:
        field = field.transpose("batch", "time", "lat", "lon")
    state[name] = field.values.astype(np.float32)

# Derive the clock forcings for the two conditioning frames.
conditioning_times = initial.datetime.values.astype("datetime64[s]").astype(np.int64)
state.update(processor.compute_forcings(conditioning_times))

# The third frame is the valid time of the forecast.
valid_time = example_batch.datetime.isel(time=processor.num_input_timesteps).values
valid_time = valid_time.astype("datetime64[s]").astype(np.int64)

inputs = processor(state, seconds_since_epoch=valid_time).to(device=model.device, dtype=model.dtype)
with torch.no_grad():
    outputs = model(**inputs, generator=torch.Generator().manual_seed(0))

forecast = processor.postprocess(outputs.prediction, state)
print(forecast["2m_temperature"].shape)  # torch.Size([1, 181, 360])
```

### Ensembles

Each member is one draw of the 32-dimensional noise vector through the same weights, so an ensemble is a batch.
[`~WeatherNext2GenerationMixin.generate`] builds it: `num_members` repeats every initial condition and gives each member
its own noise.

```python
outputs = model.generate(
    state=state,
    feature_extractor=processor,
    seconds_since_epoch=valid_time,
    num_steps=1,
    num_members=8,
    generator=torch.Generator().manual_seed(0),
)
print(outputs.forecasts[0]["2m_temperature"].shape)  # torch.Size([8, 181, 360])
```

For a direct model forward, the processor can repeat the normalized inputs instead:

```python
inputs = processor(state, seconds_since_epoch=valid_time, num_members=8).to(device=model.device, dtype=model.dtype)
with torch.no_grad():
    outputs = model(**inputs, generator=torch.Generator().manual_seed(0))
```

To run members sequentially, pass `batch_size=1` to `generate`. This limits the rows evaluated together without
changing their order or noise draws. The returned forecasts and conditioning states still hold the full ensemble.

### Autoregressive rollout

[`~WeatherNext2GenerationMixin.generate`] prepares inputs, runs the model, postprocesses each forecast and advances
the conditioning state. Each 6-hour step draws fresh noise.

```python
state = {name: torch.as_tensor(values).to(model.device) for name, values in state.items()}

outputs = model.generate(
    state=state,
    feature_extractor=processor,
    seconds_since_epoch=valid_time,
    num_steps=20,  # 5 days
)
frames = outputs.forecasts
state = outputs.state
```

`seconds_since_epoch` is the valid time of the first generated forecast, not the last conditioning frame.

### Tropical cyclones

Install the optional packages used in this example with `pip install xarray weathernext`.

`kashif/weathernext-cyclones` predicts 17 cyclone diagnostics, including existence probability, intensity and wind
radii. Postprocessed field names match the upstream [weathernext](https://github.com/google-deepmind/weathernext) tracker:

```python
import numpy as np
import xarray
from weathernext.cyclones import direct_tracker_6h_v1_config

# `frames` holds one postprocessed forecast per step, and `analysis` the state it started from.
cyclone_fields = [name for name in processor.target_variables if name.startswith("cyclone_")]
gridded = xarray.Dataset(
    {
        name: (
            ("time", "lat", "lon"),
            np.stack([np.asarray(analysis[name])] + [frame[name][0].cpu().numpy() for frame in frames]),
        )
        for name in cyclone_fields
    },
    coords={
        # lead time, not valid time, and lead 0 has to be present
        "time": np.arange(0, 6 * (len(frames) + 1), 6).astype("timedelta64[h]").astype("timedelta64[ns]"),
        "lat": processor.latitudes.numpy(),
        "lon": processor.longitudes.numpy(),
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

### Training

[`WeatherNext2ForWeatherForecasting`] returns predictions without a built-in loss. Training uses ensemble fair CRPS
in normalized residual space, with grid-cell and pressure-level weighting. Mask missing targets in both the accuracy
and ensemble-spread terms.

## Notes

- Member `m` of initial condition `i` is row `i * num_members + m`; `num_members` defaults to `config.num_members`.
  For fixed per-member draws, pass `noise` shaped `(num_steps, initial_batch_size * num_members, noise_channels)` to
  `generate`. The same noise can be reused across batched and sequential execution; separately seeded members are
  needed to preserve a prefix when changing the ensemble size.
- Put the conditioning state on the model's device to avoid host transfers during rollout. NumPy states are also
  accepted. The next conditioning window drops the oldest frame, appends the forecast and recomputes clock forcings.
- With `use_kernels=True`, grid operations run in tiles of [`WeatherNext2Config.chunk_size`] rows, following
  [Faster-WeatherNext](https://github.com/Raymondlol/Faster-WeatherNext). The regular PyTorch path is not tiled;
  `chunk_size=None` disables kernel tiling.
- Exact mesh geometry and grid-to-mesh and mesh-to-grid graphs are stored as frozen parameters in `model.geometry`.
  The feature extractor loads the mesh mask from `attention_mask.safetensors` and includes it in the model inputs.
  Loading needs no geometry library; a model built from a config gets placeholder geometry.
  The conversion script derives the geometry using `scipy` and `trimesh`.
- Noise is sampled on the generator's device and transferred only when needed. A CPU `torch.Generator` preserves
  the same noise draws across accelerators; this does not guarantee bitwise-identical forecasts across devices.
- Attention and graph aggregation accumulate in fp32. Lower-precision inference is not reference-equivalent and
  should be validated against the application's forecast metrics. The released checkpoints use fp32; for optional
  bf16 inference, load with `dtype=torch.bfloat16` and cast processor outputs to `model.dtype`.
- Attention is computed over the three block-diagonals induced by the reverse Cuthill-McKee ordering of the mesh. This
  is exactly equivalent to masking the full attention matrix, but avoids materializing a mask that would be 1.7 GB at
  0.25°.
- `eager`, `sdpa` and `flex_attention` are all supported and agree to within float noise; `sdpa` is the default and
  the fastest. Flash Attention is not supported: it cannot take an arbitrary attention mask.
- The 1° Mini checkpoints need substantially less memory than the 0.25° checkpoints. Use `generate(batch_size=1)`
  to bound per-forward working memory. Retaining the full ensemble's state and forecasts still scales with its size.
- The forecasting head reads only the grid output, so `model.mesh_to_grid.mesh_node_update` receives no gradient,
  matching the unused final mesh update in the original implementation.
- Model weights are released by Google DeepMind under CC-BY-4.0, separately from the Apache-2.0 code.

## Faster inference with a fused kernel

Passing `use_kernels=True` to [`~PreTrainedModel.from_pretrained`] enables tiled grid encoders, graph networks and the
forecasting head, along with fused banded mesh attention and shared mask preparation. These are loaded from the Hub via the
[`kernels`](https://github.com/huggingface/kernels) library, which visits only the tiles of the mask that hold a key. It is
inference-only and runs on CUDA, ROCm and XPU; on CPU, under autograd, or without the kernel installed the model falls back
to the PyTorch implementation. Make sure the model is on an accelerator when kernelization happens (e.g. with
`device_map`), and keep the default `sdpa` attention: with `flex_attention` the layer receives a `BlockMask`, which
the kernel cannot read, and it raises a `ValueError`.

```python
from transformers import WeatherNext2ForWeatherForecasting

model = WeatherNext2ForWeatherForecasting.from_pretrained(
    "kashif/weathernext2", device_map="auto", use_kernels=True
)
```

## WeatherNext2Config

[[autodoc]] WeatherNext2Config

## WeatherNext2FeatureExtractor

[[autodoc]] WeatherNext2FeatureExtractor
    - __call__
    - postprocess
    - advance_state
    - compute_forcings

## WeatherNext2GenerationMixin

[[autodoc]] WeatherNext2GenerationMixin
    - generate

## WeatherNext2GenerationOutput

[[autodoc]] WeatherNext2GenerationOutput

## WeatherNext2Model

[[autodoc]] WeatherNext2Model
    - forward

## WeatherNext2ForWeatherForecasting

[[autodoc]] WeatherNext2ForWeatherForecasting
    - forward
