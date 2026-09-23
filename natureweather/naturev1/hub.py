# Copyright 2026 Nathan. Apache-2.0.
"""
Publishing to the Hugging Face Hub, so a stranger can run this in three lines.

:class:`naturev1.CheckpointManager` already mirrors ``latest.pt`` mid-run, which is what saves a Colab
session from a dead VM. That is a backup, not a release: a bare tensor file tells nobody what
architecture it belongs to, what the inputs mean, or what the numbers are in.

A release needs four things beside the weights, and this module writes all of them:

* **the config**, so the model can be rebuilt without guessing a hidden size;
* **the normalization statistics**, without which every prediction decodes to a plausible wrong number --
  a forecast in units of somebody else's standard deviation;
* **a model card** stating what it was trained on, what it scores, and what it must not be used for;
* **a loader** that takes the repo id and hands back a working model.

That last one is the difference between a file and a release. :func:`from_pretrained` reconstructs the
config, rebuilds the mesh, loads the weights and returns something you can call.

The card is deliberately blunt about limits. A weather model that gets shared without its scorecard
invites someone to trust it during a storm, and the honest state of a run -- including "this has not
beaten persistence yet" -- belongs on the front page, not in a footnote.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from .model import NatureConfig, NatureV1


CARD_TEMPLATE = """---
license: apache-2.0
library_name: naturev1
tags:
  - weather
  - forecasting
  - hurricane
  - geometric-deep-learning
  - ihelix
---

# {name}

A probabilistic weather and tropical-cyclone model built on **iHELIX**, which treats every input as
geolocated samples on a manifold rather than as a rectangle of pixels.

```bash
pip install "naturev1[all]"
```

```python
from naturev1 import from_pretrained

model, config, stats = from_pretrained("{repo_id}")
```

## What it predicts

Every output is a distribution, never a bare number.

| Output | Form |
|---|---|
| Track | {track_modes} weighted scenarios, each a full trajectory with a tilting uncertainty ellipse |
| Landfall | probability per lead time, with a timing sigma |
| Intensity | max wind and min pressure, 90% intervals, Saffir-Simpson category |
| Eyewall | peak wind, radius of maximum wind, 34/50/64 kt radii per quadrant |
| Rapid intensification | probability at 25/30/35 kt, expected 24 h change, likeliest onset |
| Surface fields | {num_fields} fields with per-point variance, at {num_leads} lead times |
| Weather type | calibrated 7-way categorical |
| ENSO | Nino 3.4 index with phase |

Lead times: {lead_times} hours.

## Architecture

{parameters:,} parameters ({parameters_m:.1f}M). {num_layers} braided blocks on a {latent_points}-point
Fibonacci mesh, each block combining:

- **geodesic attention** over metric-space neighbours, with per-head physical radii from
  {min_radius:.0f} to {max_radius:.0f} km -- lengths, not array steps, so they keep their meaning when
  the sampling changes;
- **index attention**, content-addressed retrieval of distant regions through a hierarchy, which reaches
  the far side of the planet in one step rather than many mesh hops;
- **a gated delta rule** along time only, with a fixed-size state, so frame 1000 of a rollout costs what
  frame 1 costs.

Measured properties: sample-order invariance is exact (0.000e+00), and refining the grid converges
(0.455% -> 0.034%) because attention is quadrature-weighted rather than a plain sum.

## Training data

{training_data}

## Scores

{scores}

## Limitations

{limitations}

**Not for operational use.** The National Hurricane Center is the authoritative source for tropical
cyclone forecasts, and your national meteorological service for everything else.

## Citation

Built on [iHELIX](https://pypi.org/project/ihelix/) and [naturev1](https://pypi.org/project/naturev1/).
Made by {author}.
"""

DEFAULT_LIMITATIONS = """- Trained on surface variables at 1.5 degrees unless stated otherwise above. Medium-range skill lives
  in the 500 hPa geopotential height; a surface-only configuration cannot see it. Use
  `naturev1.upper` to add the 13 pressure levels.
- At 1.5 degrees a grid cell is about 165 km. A hurricane eyewall is tens of km, so eyewall and RMW
  outputs are inferred from the broader circulation rather than resolved.
- There is no genesis head: every storm output is conditioned on a centre you supply. It answers
  "given a storm here, where does it go", not "will a storm form".
- Forecast timing is quantised to the lead times listed above."""


def save_for_hub(
    model: NatureV1,
    directory: str | Path,
    repo_id: str,
    name: str = "NatureV1",
    normalizer=None,
    scorecard=None,
    training_data: str = "ERA5 reanalysis (WeatherBench 2) for the backbone; HURDAT2 best tracks for the storm heads.",
    limitations: str = DEFAULT_LIMITATIONS,
    author: str = "Nathan",
) -> Path:
    """
    Write weights, config, statistics and a model card into a directory ready to upload.

    Args:
        normalizer: the :class:`naturev1.Normalizer` the model was trained with. Strongly recommended --
            without it, predictions decode with whatever statistics the next person happens to compute,
            which produces numbers that look fine and are wrong.
        scorecard: a :class:`naturev1.Scorecard`. Its table goes on the card, including the baselines.
            An untrained or unscored model says so in plain words rather than leaving the section blank.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = model.config

    torch.save(model.state_dict(), directory / "model.safetensors.pt")
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2, default=list))
    if normalizer is not None:
        (directory / "normalizer.json").write_text(json.dumps(normalizer.to_dict(), indent=2))

    if scorecard is not None and getattr(scorecard, "scores", None):
        scores = "```\n" + scorecard.table() + "\n\n" + scorecard.verdict() + "\n```"
    else:
        scores = ("**Not yet scored.** Nothing here has been measured against persistence or "
                  "climatology, so no claim is made about forecast skill. Run `naturev1.score_model` "
                  "on held-out years before trusting any output.")

    card = CARD_TEMPLATE.format(
        name=name, repo_id=repo_id,
        parameters=model.num_parameters(), parameters_m=model.num_parameters() / 1e6,
        num_layers=config.num_layers, latent_points=config.latent_points,
        min_radius=config.min_radius_km, max_radius=config.max_radius_km,
        track_modes=config.track_modes, num_leads=config.num_leads,
        num_fields=len(__import__("naturev1").SURFACE_FIELDS),
        lead_times=", ".join(str(h) for h in config.lead_times_hours),
        training_data=training_data, scores=scores, limitations=limitations, author=author,
    )
    (directory / "README.md").write_text(card)
    return directory


def push_to_hub(
    model: NatureV1,
    repo_id: str,
    directory: str | Path = "./hub_export",
    private: bool = False,
    token: str | None = None,
    **card,
) -> str:
    """
    Write the release and upload it. Needs ``HF_TOKEN`` in the environment or ``token=``.

    Returns:
        The repository URL.
    """
    from huggingface_hub import HfApi

    path = save_for_hub(model, directory, repo_id, **card)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(folder_path=str(path), repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"


def from_pretrained(repo_id: str, device: str | None = None, token: str | None = None):
    """
    Rebuild a published model in one call: config, mesh, weights and statistics.

    The mesh is reconstructed from the config rather than downloaded, because it is a deterministic
    function of ``latent_points`` -- shipping it would be shipping something regenerable, and a mesh that
    disagreed with the config would be a silent, unfindable bug.

    Returns:
        ``(model, config, normalizer_or_None)``.
    """
    from huggingface_hub import hf_hub_download

    from ihelix import fibonacci_sphere

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", repo_type="model", token=token)
    raw = json.loads(Path(config_path).read_text())
    known = {f for f in NatureConfig.__dataclass_fields__}
    values = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in known}
    config = NatureConfig(**values)

    mesh = fibonacci_sphere(config.latent_points, num_neighbours=config.latent_neighbours,
                            cluster_size=config.latent_cluster)
    model = NatureV1(config, mesh)

    weights = hf_hub_download(repo_id=repo_id, filename="model.safetensors.pt", repo_type="model", token=token)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model = model.to(device).eval()

    normalizer = None
    try:
        stats = hf_hub_download(repo_id=repo_id, filename="normalizer.json", repo_type="model", token=token)
        from .era5 import Normalizer

        normalizer = Normalizer.from_dict(json.loads(Path(stats).read_text()))
    except Exception:
        print("[hub] no normalizer.json in this repo -- predictions will decode in normalized units.")

    return model, config, normalizer
