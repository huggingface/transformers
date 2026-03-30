# SAM 3.1 Progress

Updated on 2026-03-28 while working in `/Users/nielsrogge/Documents/python_projecten/transformers`.

## Policy reminder

- If this work later becomes a `huggingface/transformers` PR, issue coordination is required first.
- Breaching the repository's agent contribution policy can lead to automatic banning.

## Current status

- The `sam3_1_video` implementation is now driven from [modular_sam3_1_video.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/sam3_1_video/modular_sam3_1_video.py), and the generated [modeling_sam3_1_video.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/sam3_1_video/modeling_sam3_1_video.py) / [configuration_sam3_1_video.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/sam3_1_video/configuration_sam3_1_video.py) have been regenerated from it.
- The real `facebook/sam3.1` checkpoint was downloaded locally as `/Users/nielsrogge/Documents/python_projecten/transformers/.cache/sam3.1_multiplex.pt`.
- The image converter [convert_sam3_to_hf.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/sam3/convert_sam3_to_hf.py) now supports merged SAM 3.1 checkpoints by extracting the detector weights from `sam3.1_multiplex.pt`.
- The conversion script [convert_sam3_1_video_to_hf.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/sam3_1_video/convert_sam3_1_video_to_hf.py) now converts that checkpoint into the Transformers `Sam3_1VideoModel` with zero missing keys and zero unexpected keys.
- The image converter now verifies both save/load/forward consistency and preprocessing parity against the upstream SAM 3 image preprocessing pipeline by default.
- The converter now verifies against the upstream SAM 3.1 implementation by default and prints explicit success checkpoints while converting.
- The converter's default parity path succeeds against the upstream SAM 3.1 repo at `/Users/nielsrogge/Documents/python_projecten/sam3`.
- The SAM 3.1 video converter now also supports an optional `push_to_hub` path with explicit CLI flags and fast-fail `repo_id` validation.
- A first dedicated `sam3_1_video` test module now exists at [test_modeling_sam3_1_video.py](/Users/nielsrogge/Documents/python_projecten/transformers/tests/models/sam3_1_video/test_modeling_sam3_1_video.py).
- SAM 3.1 docs now exist at [sam3_1.md](/Users/nielsrogge/Documents/python_projecten/transformers/docs/source/en/model_doc/sam3_1.md) and [sam3_1_video.md](/Users/nielsrogge/Documents/python_projecten/transformers/docs/source/en/model_doc/sam3_1_video.md), and both pages are registered in [docs/source/en/_toctree.yml](/Users/nielsrogge/Documents/python_projecten/transformers/docs/source/en/_toctree.yml).
- `make check-repo` now passes in the existing `.venv-codex` environment.
- Technically, the current SAM 3.1 patch set is in PR-shape for a first review pass; the main blocker is now procedural rather than code-quality-related: we still need explicit issue coordination before opening an AI-assisted PR against `huggingface/transformers`.

## What changed in this round

- Re-ran the real SAM 3.1 image conversion from `.cache/sam3.1_multiplex.pt` into `/tmp/sam3_1_image_hf` and confirmed the default verification path still passes end-to-end.
- Re-ran the real SAM 3.1 video conversion from `.cache/sam3.1_multiplex.pt` into `/tmp/sam3_1_video_hf` and confirmed upstream parity still passes end-to-end against `/Users/nielsrogge/Documents/python_projecten/sam3`.
- Installed the missing repo-check dependencies in the working `.venv-codex` environment so local checkers can at least start correctly:
- `mistral-common==1.10.0`
- `ty`
- Fixed the doc model TOC ordering through the repo utility so the new SAM 3.1 docs stay in the expected sort order.
- Cleaned up two stale `ty` suppression comments in [src/transformers/generation/utils.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/generation/utils.py) that were being reported by the current local checker run.
- Replaced the dynamic `logging.Logger` monkey-patch suppressions in [src/transformers/utils/logging.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/utils/logging.py) with `cast(Any, logging.Logger)` assignments so the local checker setup no longer reports those lines.
- Moved the `sam3_1_video` source of truth to modular and kept regenerating from it with `utils/modular_model_converter.py sam3_1_video`.
- Kept the tri-head backbone / multiplex tracker structure aligned with the upstream standalone multiplex tracker checkpoint instead of the higher-level detector+predictor wrapper.
- Fixed the interactive SAM positional-embedding path so prompted mask decoding uses the prompt encoder's dense positional embedding, matching the upstream interactive decoder path.
- Fixed the interactive SAM IoU head so it respects `iou_prediction_use_sigmoid` instead of forcing sigmoid behavior inherited from the old tracker-video code.
- Kept the separate tracker `image_pe_layer` weights from the original checkpoint intact for the non-interactive tracker path.
- Reworked the parity helper so it no longer imports the full upstream `sam3.model_builder` stack. Instead it:
- builds the upstream 3.1 tracker model directly from the needed upstream modules
- installs local stub modules for optional `triton`, `torchvision`, and `timm` surfaces that block CPU-only verification on this machine
- patches the upstream fused `addmm_act` helper with a float32 CPU fallback for parity-only runs
- remaps the merged checkpoint into the upstream tracker-only load format (`tracker.model.*` plus `detector.backbone.*`)
- Loosened parity tolerances slightly to `1e-3` because the CPU fallback path introduces small numerical drift relative to the upstream fused path.
- Changed the converter UX so `convert_checkpoint(...)` defaults to `verify=True`, the CLI defaults to verification unless `--no_verify` is passed, and successful conversion steps are printed explicitly.
- Added the `sam3_1_video` entry to [src/transformers/models/__init__.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/__init__.py) so the model family is exposed alongside the other SAM 3 packages.
- Fixed `Sam3_1VideoConfig` save/load round-tripping by accepting JSON list values for `image_mean` / `image_std` and normalizing them back to tuples in `__post_init__`.
- Fixed a real modular transformer bug caught by Ruff: `memory_pos` was deleted and then passed into the decoupled transformer layers.
- Added a first focused `sam3_1_video` test module that covers:
- config save/load round-trip
- point-prompt forward with the propagation head
- box-prompt forward
- model save/load round-trip
- custom image size handling
- prompt-required error handling
- low-precision dtype smoke coverage when supported by the current device
- Extended [convert_sam3_to_hf.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/sam3/convert_sam3_to_hf.py) so it:
- auto-detects merged SAM 3.1 checkpoints via `detector.` prefixes
- extracts only detector weights and drops multiplex-only neck branches for the image path
- zero-initializes the unused 0.5x FPN layer for deterministic SAM 3.1 detector loading
- writes `processor_config.json` without instantiating `Sam3Processor`, which avoids the unrelated local `mistral_common` import problem
- defaults to `verify=True`, with `--no_verify` as the CLI opt-out
- Added save/load/forward verification for the image converter. This does not currently compare against the upstream OSS image runtime directly; instead it verifies that the converted checkpoint saves, reloads, and reproduces identical deterministic outputs in Transformers. I investigated upstream full-forward parity, but the current upstream image runtime is not a clean apples-to-apples comparator for the existing `Sam3Model` path.
- Added SAM 3.1 image and video docs, plus `_toctree.yml` entries, mirroring the existing SAM 3 / SAM 3 Video documentation layout.
- Added preprocessing parity verification to [convert_sam3_to_hf.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/sam3/convert_sam3_to_hf.py) by reading the upstream [sam3_image_processor.py](/Users/nielsrogge/Documents/python_projecten/sam3/sam3/model/sam3_image_processor.py) source, asserting the expected transform recipe is still present, and comparing deterministic pixel values against the equivalent upstream torchvision transform at `1e-6` tolerance.
- Added optional Hub upload support to [convert_sam3_1_video_to_hf.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/sam3_1_video/convert_sam3_1_video_to_hf.py) through `push_to_hub` / `repo_id`, with a fast-fail `ValueError` when `repo_id` is missing.
- Expanded the `Sam3_1VideoConfig` docstring in [modular_sam3_1_video.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/sam3_1_video/modular_sam3_1_video.py), added `@auto_docstring(checkpoint="facebook/sam3.1")`, and updated [utils/check_config_attributes.py](/Users/nielsrogge/Documents/python_projecten/transformers/utils/check_config_attributes.py) so the new SAM 3.1 configs satisfy the repo's config docstring and config attribute checkers.
- Exported `Sam3_1ViTModel` from the generated SAM 3.1 video modeling module, documented it in [sam3_1_video.md](/Users/nielsrogge/Documents/python_projecten/transformers/docs/source/en/model_doc/sam3_1_video.md), and registered the `sam3_1` doc page label in [configuration_auto.py](/Users/nielsrogge/Documents/python_projecten/transformers/src/transformers/models/auto/configuration_auto.py).
- Added `all_model_classes` plus a direct `Sam3_1ViTModel` backbone forward smoke test to [test_modeling_sam3_1_video.py](/Users/nielsrogge/Documents/python_projecten/transformers/tests/models/sam3_1_video/test_modeling_sam3_1_video.py), and updated [utils/check_repo.py](/Users/nielsrogge/Documents/python_projecten/transformers/utils/check_repo.py) so this backbone is treated like the other intentionally non-auto-configured building-block models.
- Updated `Sam3_1VideoModel.forward()` to accept `**kwargs`, which was required to make the repo checker happy with the public model forward signature.
- Ran the mandatory duplicate-work checks for a potential `huggingface/transformers` PR with `gh`:
- `gh issue list --repo huggingface/transformers --search "sam 3.1" --state all --limit 20`
- `gh issue list --repo huggingface/transformers --search "sam3.1" --state all --limit 20`
- `gh search issues '"SAM 3.1" OR "Object Multiplex"' --repo huggingface/transformers --state open/closed`
- `gh pr list --repo huggingface/transformers --state open --limit 300 --json ... | jq ...`
- `gh search prs '"SAM 3.1" OR "Object Multiplex"' --repo huggingface/transformers --state open`
- Result: I did not find an existing SAM 3.1 support issue or an overlapping open PR in `huggingface/transformers`, so the remaining PR blocker is the absence of coordination evidence rather than duplicate work.

## Verified locally

- `python -m py_compile` succeeds for the modular file, generated config/model files, and converter.
- Loading the converted checkpoint into `Sam3_1VideoModel(Sam3_1VideoConfig())` yields:
- `missing 0 []`
- `unexpected 0 []`
- Running the video converter now succeeds with default verification enabled:
- `convert_checkpoint(checkpoint_path='.cache/sam3.1_multiplex.pt', sam3_repo_path='/Users/nielsrogge/Documents/python_projecten/sam3')`
- CLI re-run:
- `PYTHONPATH=src ./.venv-codex/bin/python src/transformers/models/sam3_1_video/convert_sam3_1_video_to_hf.py --checkpoint_path .cache/sam3.1_multiplex.pt --output_dir /tmp/sam3_1_video_hf --sam3_repo_path /Users/nielsrogge/Documents/python_projecten/sam3`
- result: `Verification passed: the converted Transformers model matches the upstream SAM 3.1 implementation.`
- Running the image converter on the real merged SAM 3.1 checkpoint now succeeds with default verification enabled:
- `convert_sam3_checkpoint(checkpoint_path='.cache/sam3.1_multiplex.pt', output_path='<tmpdir>')`
- CLI re-run:
- `PYTHONPATH=src ./.venv-codex/bin/python src/transformers/models/sam3/convert_sam3_to_hf.py --checkpoint_path .cache/sam3.1_multiplex.pt --output_path /tmp/sam3_1_image_hf --sam3_repo_path /Users/nielsrogge/Documents/python_projecten/sam3`
- result:
- `Verification passed: the saved SAM3 checkpoint reloads and matches the in-memory model, and image preprocessing matches the upstream SAM 3 implementation.`
- The new focused SAM 3.1 tests pass locally:
- `PYTHONPATH=src ./.venv-codex/bin/python -m pytest tests/models/sam3_1_video/test_modeling_sam3_1_video.py -q`
- result: `8 passed`
- Existing SAM 3 model tests still pass locally:
- `PYTHONPATH=src ./.venv-codex/bin/python -m pytest tests/models/sam3/test_modeling_sam3.py -q`
- result: `193 passed, 153 skipped`
- Existing SAM 3 video integration tests are still clean in this environment:
- `PYTHONPATH=src ./.venv-codex/bin/python -m pytest tests/models/sam3_video/test_modeling_sam3_video.py -q`
- result: `5 skipped`
- `make style` succeeds in the existing `.venv-codex` environment.
- `make check-repo` now passes in `.venv-codex` with `All 21 checks passed.` The `Type annotations` phase still prints four existing `ty` diagnostics from untouched repo code, but in this checker configuration the target still returns success and is not currently blocked by the SAM 3.1 changes.
- The new `push_to_hub` validation path is also exercised locally:
- calling `convert_checkpoint(..., push_to_hub=True, repo_id=None)` raises `ValueError: repo_id must be provided when push_to_hub=True`
- `git status --short` was clean before updating this progress note.

## Remaining work

1. Open or coordinate on a `huggingface/transformers` issue for SAM 3.1 support before any PR is opened, and wait for clear maintainer/issue-thread approval. This is required for AI-assisted contributions and skipping it can lead to automatic banning.
2. Once issue coordination exists, decide whether the first PR should keep the current scope (SAM 3.1 image conversion + `sam3_1_video`) or be narrowed further for reviewability.
3. Decide whether to add a higher-level SAM 3.1 processor / session API before trying to mirror the full `tests/models/sam3_video/test_modeling_sam3_video.py` integration coverage.
4. Decide whether to add a dedicated SAM 3.1 image test module or keep relying on the real image conversion verification plus the existing `sam3` model test suite.
5. Decide whether to publish converted SAM 3.1 image/video checkpoints to the Hub so the docs can use `from_pretrained(...)` directly instead of requiring local conversion first.
6. If we want full upstream image-model parity later, build a closer apples-to-apples forward comparator for `Sam3Model` than the current preprocessing-plus-save/load verification path.
