# VidEoMT DINOv2 Conversion Progress

Last updated: 2026-03-27

Existing Hub checkpoint already present:

| Checkpoint | Hub repo | Status |
| --- | --- | --- |
| `yt_2019_vit_small_52.8.pth` | `tue-mps/videomt-dinov2-small-ytvis2019` | Already converted before this run; model card refreshed during this run |

Remaining DINOv2 checkpoints targeted in this run:

| Checkpoint | Hub repo | Status | Notes |
| --- | --- | --- | --- |
| `yt_2019_vit_base_58.2.pth` | `tue-mps/videomt-dinov2-base-ytvis2019` | Done | Pushed to Hub and verified against the upstream implementation |
| `yt_2019_vit_large_68.6.pth` | `tue-mps/videomt-dinov2-large-ytvis2019` | Done | Pushed to Hub and verified against the upstream implementation |
| `yt_2021_vit_large_63.1.pth` | `tue-mps/videomt-dinov2-large-ytvis2021` | Done | Pushed to Hub and verified against the upstream implementation |
| `yt_2022_vit_large_42.6.pth` | `tue-mps/videomt-dinov2-large-ytvis2022` | Done | Pushed to Hub and verified against the upstream implementation |
| `ovis_vit_large_52.5.pth` | `tue-mps/videomt-dinov2-large-ovis` | Done | Pushed to Hub and verified against the upstream implementation |
| `vipseg_vit_large_55.2.pth` | `tue-mps/videomt-dinov2-large-vipseg` | Done | Pushed to Hub and verified against the upstream implementation; registry image size corrected to 1280 during this run |
| `vspw_vit_large_95.0_64.9.pth` | `tue-mps/videomt-dinov2-large-vspw` | Done | Pushed to Hub and verified against the upstream implementation |

Final status:

All remaining DINOv2-based VidEoMT checkpoints were converted, pushed to the `tue-mps` organization on the Hub, and checked for the expected `README.md`, `config.json`, `model.safetensors`, and `video_preprocessor_config.json` files.

Execution note:

The local repo does not declare a `[project]` table in `pyproject.toml`, so `uv` commands are being run as `uv run --no-project --python .venv/bin/python ...`.
