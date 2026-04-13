# SAM 3.1 Support Plan

Progress note written on 2026-03-28 after inspecting the current `sam3`, `sam3_video`, `sam3_tracker`, `sam3_tracker_video` Transformers implementations and comparing them against the upstream SAM 3.1 release in `/Users/nielsrogge/Documents/python_projecten/sam3`.

## Policy note

- If this work turns into a `transformers` PR, the repository's agent contribution policy requires coordination on the relevant issue first.
- Violating that policy can lead to automatic banning, so the implementation work should stay in planning / local prototyping mode until coordination is clear.

## Current findings

- SAM 3.1 is not a drop-in replacement for the current SAM 3 video stack in Transformers.
- Upstream changed the video path to Object Multiplex buckets, added a new multiplex predictor stack, and introduced a merged `sam3.1_multiplex.pt` checkpoint with custom remapping / non-strict loading behavior.
- The image path is much closer to existing SAM 3 than the video path is.
- Upstream `build_sam3_image_model()` still downloads `version="sam3"`, not `version="sam3.1"`.
- The 3.1 detector used inside the new video stack is still `Sam3Image`-based; the biggest detector-side change is a tri-head neck that keeps the original detector neck as the main `convs` branch and adds `interactive_convs` and `propagation_convs`.
- That strongly suggests `Sam3Model` is still the right class for SAM 3.1 image support, and that the main image-side work is likely checkpoint extraction / conversion rather than a new detector architecture.
- Full SAM 3.1 video support is a new architecture from the Transformers point of view.
- The current Transformers `sam3_video` and `sam3_tracker_video` code assume per-object tracking, not multiplex buckets.

## Current repo structure implications

- `src/transformers/models/sam3/` is only partially modular today. `modular_sam3.py` covers the image processor, but `modeling_sam3.py` is still hand-written.
- `src/transformers/models/sam3_tracker/` is modular-based.
- `src/transformers/models/sam3_tracker_video/` is modular-based.
- `src/transformers/models/sam3_video/` is currently hand-written and does not have a modular file.
- Because of that, the cleanest way to use modular for SAM 3.1 is probably to add new 3.1-specific video / tracker folders rather than first refactoring all existing SAM 3 code into modular form.

## Recommended repository strategy

- Keep SAM 3.1 image support inside the existing `src/transformers/models/sam3/` folder unless checkpoint inspection proves a real detector architecture mismatch.
- Add a new video family under `src/transformers/models/sam3_1_video/`.
- Only add a standalone `src/transformers/models/sam3_1_tracker_video/` folder if we decide that exposing the multiplex tracker directly is worth the extra API surface and maintenance cost.
- If the standalone multiplex tracker is not needed publicly, keep tracker internals private to `sam3_1_video` and expose only the end-to-end video model / processor.
- Use modular for new 3.1 tracker / video code where inheritance is helpful:
- A new `sam3_1_tracker_video` modular file could inherit from `sam3_tracker_video` pieces and override config, state management, and feature plumbing.
- A new `sam3_1_video` modular file can still import / inherit from current hand-written `sam3_video` classes even though `sam3_video` itself is not modular yet.
- Avoid refactoring current `sam3_video` to modular as a prerequisite unless code duplication becomes severe; that is extra scope.

## Work breakdown

### 1. Checkpoint reconnaissance and mapping

- Download or inspect the local `facebook/sam3.1` checkpoint and enumerate key prefixes and tensor shapes.
- Separate detector weights, tri-head extra weights, and multiplex tracker weights.
- Confirm whether the detector branch can be loaded into the current `Sam3Model` after conversion while ignoring video-only branches.
- Draft exact mapping rules for:
- `sam3.1` image extraction into `Sam3Model`
- `sam3.1` video conversion into a new `Sam3_1VideoModel`

### 2. Image support in existing `sam3`

- Extend `src/transformers/models/sam3/convert_sam3_to_hf.py` so it can understand 3.1 checkpoint roots such as `detector.` or legacy `sam3_model.`.
- Teach the converter to ignore or drop `interactive_convs` / `propagation_convs` when building an image-only detector checkpoint.
- Validate whether `modeling_sam3.py` can stay unchanged.
- If it cannot, make the smallest compatible detector-side change instead of creating a new 3.1 image model family.
- Decide whether `facebook/sam3.1` should load directly into `Sam3Model`, or whether a separate converted detector repo id is cleaner for the first pass.
- Add image integration tests for 3.1 loading and output parity checks against upstream image inference.

### 3. New video family for SAM 3.1

- Add `Sam3_1VideoConfig` with multiplex-specific fields such as:
- `multiplex_count`
- `max_num_objects`
- dynamic bucket / planning settings
- updated detection / reconditioning knobs
- Implement a detector vision path that can expose detector, interactive, and propagation features from one backbone pass.
- Decide whether to:
- add a tri-head detector helper under the new video folder, or
- minimally extend shared SAM 3 vision utilities in a reusable way
- Implement multiplex inference session / state management:
- bucketized object ids
- add / remove / refine prompt flows
- propagation lifecycle
- cancellation / reset behavior
- cached per-frame tri-head features
- Implement the multiplex tracker core or wrapper.
- Decide whether this lives in a standalone `sam3_1_tracker_video` public model or as private components inside `sam3_1_video`.
- Implement the new video processor and post-processing logic.
- Add a new conversion script for SAM 3.1 video checkpoints.

### 4. Optional standalone multiplex tracker family

- Only do this if there is a strong product / API reason to expose the multiplex tracker itself.
- If yes, add `src/transformers/models/sam3_1_tracker_video/` with modular inheritance from `sam3_tracker_video`.
- Keep the tracker API aligned with the new 3.1 video model so there is not a split maintenance story.
- If no, skip this folder for the initial support pass and keep tracker internals private.

### 5. Registrations, docs, and tests

- Register any new model types in:
- `src/transformers/models/auto/configuration_auto.py`
- `src/transformers/models/auto/modeling_auto.py`
- `src/transformers/models/auto/image_processing_auto.py`
- `src/transformers/models/auto/processing_auto.py`
- `src/transformers/models/auto/tokenization_auto.py`
- `src/transformers/models/auto/video_processing_auto.py`
- package `__init__.py` files
- Update docs pages and `_toctree.yml`.
- Add unit tests and integration tests mirroring current SAM 3 image / tracker / video coverage.
- If a new folder uses modular, run the modular converter and then `make fix-repo`.
- Finish with `make style` and targeted tests, then broader checks as needed.

## Open design decisions

- Should SAM 3.1 image support be wired to the existing `Sam3Model` only, or do we want a new config / model type for bookkeeping?
- Do we want a public standalone `sam3_1_tracker_video` class, or is `sam3_1_video` enough for the first milestone?
- Is it acceptable for the first implementation to support only the end-to-end multiplex video model and not every internal upstream class?
- Should the user-facing processor / session API stay close to current `Sam3VideoProcessor`, or should it follow the upstream unified predictor flow more closely?
- Do we want to refactor shared SAM 3 video utilities into reusable modular-friendly helpers before adding 3.1, or only after the first working version lands?

## Suggested implementation order

1. Inspect a real `sam3.1` checkpoint and write the concrete key-mapping rules.
2. Land image support first by reusing `Sam3Model` if checkpoint extraction works.
3. Prototype the new `sam3_1_video` config and conversion path.
4. Implement multiplex state / session logic and tri-head detector feature reuse.
5. Decide whether a standalone `sam3_1_tracker_video` public class is still needed.
6. Add registrations, docs, and tests.
7. Run `make fix-repo`, `make style`, and the relevant model tests.

## Immediate next step

- Inspect the actual local `sam3.1` checkpoint state dict and draft the exact conversion rules for image extraction and new video conversion before touching implementation files.
