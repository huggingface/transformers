# RF-DETR Contribution Progress

## Task
Implement RF-DETR in Transformers based on `/Users/nielsrogge/Documents/python_projecten/rf-detr/src/rfdetr`, with focus on:
- modular implementation,
- modeling file,
- conversion script for checkpoint conversion,
- successful conversion and output parity check for `RFDETRSmall` on dummy inputs.

## Progress
- [x] Inspected upstream RF-DETR (`rfdetr`) architecture and checkpoint key structure.
- [x] Inspected existing HF LW-DETR implementation and conversion utilities for reusable components.
- [x] Added new RF-DETR model package scaffold:
  - `src/transformers/models/rf_detr/modular_rf_detr.py`
  - `src/transformers/models/rf_detr/modeling_rf_detr.py`
  - `src/transformers/models/rf_detr/convert_rf_detr_to_hf.py`
  - `src/transformers/models/rf_detr/configuration_rf_detr.py`
  - `src/transformers/models/rf_detr/__init__.py` (lazy module pattern)
- [x] Implemented a windowed DINOv2-style backbone for RF-DETR on top of existing HF DINOv2-with-registers components.
- [x] Hooked RF-DETR model classes to reuse LW-DETR decoder/projector path.
- [x] Implemented conversion script key remapping + decoder qkv splitting.
- [x] Wired RF-DETR into Transformers auto mappings:
  - `src/transformers/models/auto/configuration_auto.py`
  - `src/transformers/models/auto/modeling_auto.py`
  - `src/transformers/models/__init__.py`
- [x] Verified API instantiation works with `AutoConfig`, `AutoModel`, `AutoModelForObjectDetection`, and `AutoBackbone`.
- [x] Validated end-to-end conversion/parity on an RFDETRSmall-style checkpoint artifact generated from upstream `rfdetr` model args.
- [x] Re-ran conversion/parity after follow-up fixes (still matching).
- [x] Follow-up robustness fixes:
  - converter imports from `modeling_rf_detr` (instead of direct `modular_rf_detr` import),
  - `RfDetrWindowedDinov2Config.window_block_indexes` default aligned with upstream behavior (`list(range(num_hidden_layers))`).
- [x] Added converter compatibility for real upstream checkpoints where args use `dinov2_patch_size`/`dinov2_num_windows` instead of `patch_size`/`num_windows`.
- [x] Added converter support for inferring `num_labels` from checkpoint tensors (handles stale `args.num_classes` values in released checkpoints).
- [x] Added original-model verification fallbacks for missing `positional_encoding_size` in checkpoint args.
- [x] Added Hugging Face Hub upload support to converter:
  - new CLI flag: `--push_to_hub`,
  - optional `--repo_id`,
  - default target repo id is inferred as `nielsr/<checkpoint-name>`.
- [x] Completed RF-DETR model documentation page:
  - added `docs/source/en/model_doc/rf_detr.md` (patterned after DETR/LW-DETR docs),
  - included the HF paper link: `https://huggingface.co/papers/2511.09554`,
  - added usage examples (`Pipeline`, `AutoModelForObjectDetection`) and autodoc sections for RF-DETR config/model/backbone/outputs.
- [x] Wired RF-DETR docs page into navigation:
  - added `model_doc/rf_detr` entry to `docs/source/en/_toctree.yml` (Vision models section).
- [x] Ran modular conversion to regenerate generated files from the modular source:
  - command: `python utils/modular_model_converter.py rf_detr`
  - regenerated files: `src/transformers/models/rf_detr/modeling_rf_detr.py`, `src/transformers/models/rf_detr/configuration_rf_detr.py`
- [x] Fixed modular-to-generated compatibility issues discovered after modular conversion:
  - expanded config init signatures in modular configs so generated `configuration_rf_detr.py` is valid,
  - introduced RF-prefixed wrapper classes in modular (`RfDetrPreTrainedModel`, `RfDetrDecoder`, `RfDetrMultiScaleProjector`, etc.) so generated `modeling_rf_detr.py` has no unresolved `LwDetr*` symbols.
- [x] Re-verified generated files after modular conversion:
  - `py_compile` succeeds for generated RF-DETR files,
  - `ruff check` passes on modular/generated/conversion files,
  - `AutoConfig`, `AutoModel`, `AutoModelForObjectDetection`, `AutoBackbone` instantiation and forward pass work.
- [x] Validate end-to-end conversion/parity run on the real pre-trained `RFDETRSmall` checkpoint.
- [x] Run lint checks on modified files (`ruff check` on RF-DETR + auto-mapping touched files).
- [x] Added RF-DETR modeling tests:
  - created `tests/models/rf_detr/__init__.py`,
  - added `tests/models/rf_detr/test_modeling_rf_detr.py` based on `tests/models/lw_detr/test_modeling_lw_detr.py`,
  - adapted imports/config/model names for RF-DETR (`RfDetr*` classes/configs),
  - skipped backbone attention-output test since RF-DINOv2 backbone does not expose attentions.
- [x] Fixed RF-DETR test/runtime issues discovered while running pytest:
  - added missing `is_training` attributes in RF-DINOv2 backbone/model tester helpers for `ModelTesterMixin` compatibility,
  - set `num_register_tokens=4` in RF test configs to avoid empty-tensor state dict entries that break `test_torch_save_load`,
  - mapped `"RfDetrForObjectDetection"` to `LwDetrForObjectDetectionLoss` in `src/transformers/loss/loss_utils.py` so RF-DETR uses the correct detection loss implementation (instead of generic `ForObjectDetectionLoss`).
- [x] Verified RF-DETR tests with `uv` and existing `.venv`:
  - command: `source .venv/bin/activate && uv run --no-project --python .venv/bin/python pytest -q tests/models/rf_detr/test_modeling_rf_detr.py`,
  - result: `190 passed, 144 skipped, 14 warnings` (no failures).
- [x] Verified style on touched files:
  - command: `source .venv/bin/activate && uv run --no-project --python .venv/bin/python ruff check src/transformers/loss/loss_utils.py tests/models/rf_detr/test_modeling_rf_detr.py`,
  - result: `All checks passed!`.
- [x] Extended RF-DETR conversion script to support model-name based conversion from Hub checkpoints:
  - added `--model_name` (mutually exclusive with `--checkpoint_path`),
  - added `--checkpoint_repo_id` (default: `nielsr/rf-detr-checkpoints`),
  - added `hf_hub_download` flow to fetch original RF-DETR checkpoints from Hub before conversion.
- [x] Added robust checkpoint-args normalization for multi-checkpoint compatibility:
  - fallback defaults per supported object-detection variant (`nano`, `small`, `medium`, `large`, `base`, `base-2`, `base-o365`),
  - support for checkpoints missing `args` (e.g. `rf-detr-large-2026.pth`),
  - inference of missing `patch_size` from patch-embedding tensor shape,
  - inference/override of `resolution` from positional-embedding grid shape,
  - improved backbone depth handling by using `vit_encoder_num_layers` when available.
- [x] Verified model-name conversion end-to-end on all currently supported object-detection model names:
  - command:
    `source .venv/bin/activate && for model in nano small medium large base base-2 base-o365; do uv run --no-project --python .venv/bin/python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --model_name \"$model\" --checkpoint_repo_id nielsr/rf-detr-checkpoints --pytorch_dump_folder_path \"/tmp/rf-detr-${model}-hf\" || exit 1; done`
  - result: all variants converted successfully with `Missing keys: 0` and `Unexpected keys: 0`.
- [x] Verified `large-2026` alias support:
  - command:
    `source .venv/bin/activate && uv run --no-project --python .venv/bin/python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --model_name large-2026 --checkpoint_repo_id nielsr/rf-detr-checkpoints --pytorch_dump_folder_path /tmp/rf-detr-large2026-alias`
  - result: successful conversion to HF format.
- [x] Added RF-DETR instance-segmentation modeling support in modular RF-DETR:
  - implemented `RfDetrDepthwiseConvBlock`, `RfDetrMLPBlock`, and `RfDetrSegmentationHead`,
  - added `RfDetrInstanceSegmentationOutput`,
  - added `RfDetrForInstanceSegmentation` with mask prediction head on top of RF-DETR decoder outputs,
  - extended `RfDetrConfig` with `mask_downsample_ratio` and `segmentation_bottleneck_ratio`.
- [x] Regenerated RF-DETR generated files from modular source after segmentation changes:
  - command: `source .venv/bin/activate && uv run --no-project --python .venv/bin/python utils/modular_model_converter.py rf_detr`,
  - regenerated files include `src/transformers/models/rf_detr/modeling_rf_detr.py` and `src/transformers/models/rf_detr/configuration_rf_detr.py`.
- [x] Extended RF-DETR conversion script to support instance segmentation conversion for `rf-detr-seg-small.pt`:
  - added checkpoint/task resolution for both object detection and instance segmentation model names,
  - added segmentation checkpoint defaults (`seg-small`) for checkpoints without embedded `args`,
  - added segmentation key remapping for `segmentation_head.*` weights,
  - converter now instantiates `RfDetrForInstanceSegmentation` when needed and verifies mask parity.
- [x] Verified conversion + original parity for `rf-detr-seg-small.pt` from `nielsr/rf-detr-checkpoints`:
  - command:
    `source .venv/bin/activate && uv run --no-project --python .venv/bin/python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --model_name seg-small --checkpoint_repo_id nielsr/rf-detr-checkpoints --pytorch_dump_folder_path /tmp/rf-detr-seg-small-hf --verify_with_original --original_repo_path /Users/nielsrogge/Documents/python_projecten/rf-detr`,
  - result: successful conversion with `Missing keys: 0` and `Unexpected keys: 0`.
- [x] Verified lint/smoke checks on updated RF-DETR files:
  - `ruff check` passes on modular/generated/converter files,
  - import smoke test confirms `from transformers.models.rf_detr import RfDetrForInstanceSegmentation`.
- [x] Extended RF-DETR modeling tests with instance-segmentation coverage:
  - updated `tests/models/rf_detr/test_modeling_rf_detr.py` to import `RfDetrForInstanceSegmentation`,
  - added `create_and_check_rf_detr_instance_segmentation_head_model` with shape checks for `logits`, `pred_boxes`, and `pred_masks`,
  - added explicit check that passing `labels` raises `NotImplementedError` (current model limitation),
  - added `test_rf_detr_instance_segmentation_head_model`.
- [x] Verified RF-DETR test suite with existing `.venv` after segmentation test updates:
  - command:
    `source .venv/bin/activate && uv run --no-project --python .venv/bin/python pytest -q tests/models/rf_detr/test_modeling_rf_detr.py`,
  - result: `191 passed, 144 skipped, 14 warnings` (no failures).
- [x] Verified lint for the updated test file:
  - command:
    `source .venv/bin/activate && uv run --no-project --python .venv/bin/python ruff check tests/models/rf_detr/test_modeling_rf_detr.py`,
  - result: `All checks passed!`.
- [x] Added `RfDetrImageProcessor` implementation:
  - created `src/transformers/models/rf_detr/image_processing_rf_detr.py`,
  - implemented RF-DETR-native preprocessing (`to_tensor`-equivalent conversion, [0,1] validation, ImageNet normalization, fixed square resize),
  - implemented `post_process_object_detection` with RF-DETR/LW-DETR sigmoid top-k behavior,
  - implemented `post_process_instance_segmentation` with top-k mask gather, resizing, and thresholding behavior matching upstream.
- [x] Wired RF-DETR image processing into Transformers mappings/imports:
  - added `"rf_detr"` to `src/transformers/models/auto/image_processing_auto.py`,
  - updated `src/transformers/models/rf_detr/__init__.py` to include `image_processing_rf_detr` in TYPE_CHECKING imports.
- [x] Extended `src/transformers/models/rf_detr/convert_rf_detr_to_hf.py` for image processor support:
  - converter now builds/saves `RfDetrImageProcessor` alongside model/config,
  - converter now pushes image processor with `--push_to_hub`,
  - converter now configures image processor `size` and `num_top_queries` from checkpoint args/config.
- [x] Added conversion-time preprocessing parity check against upstream RF-DETR:
  - deterministic dummy uint8 image preprocessing compared between upstream torchvision path and HF `RfDetrImageProcessor`,
  - converter prints slices and `max_abs_preprocess_diff`,
  - converter raises on mismatch.
- [x] Added conversion-time postprocessing parity checks against upstream `PostProcess`:
  - compares detection postprocess scores/boxes/labels on same upstream model outputs,
  - compares segmentation postprocess scores/boxes/labels/masks on same upstream model outputs,
  - converter raises on mismatch.
- [x] Verified updated conversion flow end-to-end on local released checkpoints:
  - object detection checkpoint: `/Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-small.pth`,
  - instance segmentation checkpoint: `/Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-small.pt`,
  - both runs passed with `Missing keys: 0` and `Unexpected keys: 0`.
- [x] Verified image processor loading:
  - `from transformers import RfDetrImageProcessor` works,
  - `AutoImageProcessor.from_pretrained(...)` on converted checkpoint resolves to `RfDetrImageProcessor`.
- [x] Verified style/syntax checks for image processor/converter integration:
  - `ruff check` passes on touched files,
  - `py_compile` passes on `image_processing_rf_detr.py` and `convert_rf_detr_to_hf.py`.
- [x] Re-ran RF-DETR model tests after image processor/converter integration:
  - command:
    `source .venv/bin/activate && uv run --no-project --python .venv/bin/python pytest -q tests/models/rf_detr/test_modeling_rf_detr.py`,
  - result: `191 passed, 144 skipped, 14 warnings`.
- [x] Added RF-DETR fast image processor:
  - created `src/transformers/models/rf_detr/image_processing_rf_detr_fast.py`,
  - implemented `RfDetrImageProcessorFast` on top of `BaseImageProcessorFast`,
  - kept RF-DETR input semantics (torch tensors must already be in `[0,1]`, 3-channel RGB requirement),
  - added `post_process_object_detection` and `post_process_instance_segmentation` parity logic in fast class.
- [x] Wired RF-DETR fast image processor into imports/auto mappings:
  - updated `src/transformers/models/rf_detr/__init__.py` to include `image_processing_rf_detr_fast`,
  - updated `src/transformers/models/auto/image_processing_auto.py` mapping for `"rf_detr"` to `("RfDetrImageProcessor", "RfDetrImageProcessorFast")`.
- [x] Extended converter verification to include fast image processor:
  - `src/transformers/models/rf_detr/convert_rf_detr_to_hf.py` now instantiates `RfDetrImageProcessorFast`,
  - converter now checks slow/original and fast/original preprocess parity on deterministic dummy image,
  - converter now checks slow vs fast postprocess parity for both detection and segmentation,
  - strict exact check remains for slow processor vs original, fast checks use tight tolerance (`atol=1e-6`) for preprocessing.
- [x] Added RF-DETR image processing tests:
  - created `tests/models/rf_detr/test_image_processing_rf_detr.py`,
  - enabled `ImageProcessingTestMixin` coverage for slow+fast processor save/load and equivalence flows,
  - added explicit slow/fast equivalence tests for `post_process_object_detection` and `post_process_instance_segmentation`,
  - added explicit validation test that unnormalized torch inputs are rejected.
- [x] Verified RF-DETR image processing tests:
  - command:
    `source .venv/bin/activate && uv run --no-project --python .venv/bin/python pytest -q tests/models/rf_detr/test_image_processing_rf_detr.py`,
  - result: `23 passed, 3 skipped`.
- [x] Re-verified converter end-to-end with fast verification enabled:
  - object detection checkpoint conversion/parity:
    `/Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-small.pth`,
  - instance segmentation checkpoint conversion/parity:
    `/Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-small.pt`,
  - both runs passed with `Missing keys: 0` and `Unexpected keys: 0`.
- [x] Re-ran RF-DETR modeling + image-processing tests together after fast image-processor changes:
  - command:
    `source .venv/bin/activate && uv run --no-project --python .venv/bin/python pytest -q tests/models/rf_detr/test_modeling_rf_detr.py tests/models/rf_detr/test_image_processing_rf_detr.py`,
  - result: `214 passed, 147 skipped, 14 warnings`.
- [x] Added dataset-specific `id2label` / `label2id` population in `src/transformers/models/rf_detr/convert_rf_detr_to_hf.py`:
  - infer label dataset per checkpoint/model variant (`coco` for `nano/small/medium/large/base/base-2/seg-small`, `object365` for `base-o365`),
  - load mappings from `huggingface/label-files` (`coco-detection-id2label.json`, `object365-id2label.json`),
  - align mapping length with checkpoint head size (`num_labels`) and keep an offline-safe fallback to `LABEL_{id}` labels when label-files are unavailable.
  - smoke-verified conversion after the change:
    - `rf-detr-small.pth` loads `coco` mapping with `91` labels,
    - `rf-detr-base-o365.pth` loads `object365` mapping with `366` labels,
    - `rf-detr-seg-small.pt` loads `coco` mapping with `91` labels,
    - all conversions completed with `Missing keys: 0` and `Unexpected keys: 0`.
- [x] Extended instance-segmentation conversion support in `src/transformers/models/rf_detr/convert_rf_detr_to_hf.py` to all checkpoints mentioned in upstream RF-DETR:
  - added model-name support, aliases, filename candidates, and regex patterns for:
    `seg-preview`, `seg-nano`, `seg-small`, `seg-medium`, `seg-large`, `seg-xlarge`, `seg-2xlarge` (including `seg-xxlarge` alias/file pattern),
  - added default fallback args for all segmentation variants (for checkpoints without embedded `args`),
  - expanded COCO label mapping assignment to all segmentation variants,
  - added query-count fallback inference from `query_feat.weight` in `_prepare_checkpoint_args` to auto-correct `num_queries` when checkpoint/defaults disagree.
- [x] Verified full local segmentation conversion sweep (all upstream-mentioned segmentation checkpoints):
  - command:
    `source .venv/bin/activate && for ckpt in /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-preview.pt /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-nano.pt /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-small.pt /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-medium.pt /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-large.pt /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-xlarge.pt /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-xxlarge.pt; do uv run --no-project --python .venv/bin/python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --checkpoint_path \"$ckpt\" --pytorch_dump_folder_path \"/tmp/rf-detr-$(basename \"$ckpt\" .pt)-hf-allseg\" || exit 1; done`
  - result: all variants converted successfully with `Missing keys: 0` and `Unexpected keys: 0`.
- [x] Verified new model-name download path for additional segmentation variants:
  - `--model_name seg-nano` converts and downloads `rf-detr-seg-nano.pt` successfully,
  - `--model_name seg-xxlarge` alias resolves to canonical `seg-2xlarge` and converts `rf-detr-seg-xxlarge.pt` successfully,
  - both runs completed with `Missing keys: 0` and `Unexpected keys: 0`.
- [x] Fixed RF-DETR modeling tests after adding `RfDetrForInstanceSegmentation` to `all_model_classes`:
  - fixed tuple/dict output equivalence by adding `@can_return_tuple` to `RfDetrForInstanceSegmentation.forward` in `src/transformers/models/rf_detr/modular_rf_detr.py` and regenerating `modeling_rf_detr.py`,
  - fixed training test failure (`loss is None`) by overriding `test_training` in `tests/models/rf_detr/test_modeling_rf_detr.py` to train only `RfDetrForObjectDetection` (segmentation loss is intentionally not implemented yet).
- [x] Verified tests with existing `.venv`:
  - targeted failures:
    `source .venv/bin/activate && uv run --no-project --python .venv/bin/python pytest -q tests/models/rf_detr/test_modeling_rf_detr.py -k "test_model_outputs_equivalence or test_training"`
    -> `10 passed, 325 deselected`,
  - full RF-DETR modeling suite:
    `source .venv/bin/activate && uv run --no-project --python .venv/bin/python pytest -q tests/models/rf_detr/test_modeling_rf_detr.py`
    -> `190 passed, 145 skipped, 14 warnings`.
- [x] Updated RF-DETR docs page at `docs/source/en/model_doc/rf_detr.md`:
  - added `RfDetrForInstanceSegmentation` to model docs + autodoc sections,
  - added `RfDetrImageProcessor` and `RfDetrImageProcessorFast` autodoc sections with RF-DETR post-processing methods,
  - added an instance segmentation usage example using `AutoImageProcessor` + `RfDetrForInstanceSegmentation`,
  - updated notes to reflect current support (instance segmentation inference available; segmentation training loss not implemented yet).

## Latest Verification Snapshot
- Conversion load status: `Missing keys: 0`, `Unexpected keys: 0`.
- Converter now auto-populates `config.id2label`/`config.label2id`:
  - COCO variants load `coco-detection-id2label.json` (`91` labels),
  - Objects365 variant (`base-o365`) loads `object365-id2label.json` (`366` labels).
- Full segmentation checkpoint conversion sweep succeeded for:
  `rf-detr-seg-preview.pt`, `rf-detr-seg-nano.pt`, `rf-detr-seg-small.pt`,
  `rf-detr-seg-medium.pt`, `rf-detr-seg-large.pt`, `rf-detr-seg-xlarge.pt`, `rf-detr-seg-xxlarge.pt`.
- Numerical parity on locally generated RFDETRSmall-style dummy checkpoint: `max_abs_logits_diff ~= 8.6e-6`, `max_abs_boxes_diff = 0.0`.
- Numerical parity on real released `RFDETRSmall` checkpoint: `max_abs_logits_diff ~= 1.67e-4`, `max_abs_boxes_diff ~= 7.34e-5`.
- Numerical parity on released `RFDETRSegSmall` checkpoint: `max_abs_logits_diff ~= 1.01e-4`, `max_abs_boxes_diff ~= 1.51e-4`, `max_abs_masks_diff ~= 4.24e-3`.
- Preprocessing parity on real released `RFDETRSmall`: `max_abs_preprocess_diff = 0.0` (exact tensor match).
- Preprocessing parity on real released `RFDETRSegSmall`: `max_abs_preprocess_diff = 0.0` (exact tensor match).
- Fast preprocessing parity on real released `RFDETRSmall`:
  - `max_abs_preprocess_fast_diff ~= 9.54e-7` vs original,
  - `max_abs_slow_fast_preprocess_diff ~= 9.54e-7` (slow vs fast).
- Fast preprocessing parity on real released `RFDETRSegSmall`:
  - `max_abs_preprocess_fast_diff ~= 7.15e-7` vs original,
  - `max_abs_slow_fast_preprocess_diff ~= 7.15e-7` (slow vs fast).
- Detection postprocess parity on real released `RFDETRSmall`:
  - `max_abs_postprocess_scores_diff = 0.0`,
  - `max_abs_postprocess_boxes_diff = 0.0`,
  - `postprocess_labels_match = True`,
  - fast parity: scores/boxes diff `0.0`, labels match `True`, slow-vs-fast labels match `True`.
- Segmentation postprocess parity on real released `RFDETRSegSmall`:
  - `max_abs_postprocess_scores_diff = 0.0`,
  - `max_abs_postprocess_boxes_diff = 0.0`,
  - `postprocess_labels_match = True`,
  - `postprocess_masks_match = True`,
  - fast parity: scores/boxes diff `0.0`, labels/masks match `True`, slow-vs-fast masks match `True`.
- Printed logits/boxes slices are matching up to float tolerance.
- Printed logits/boxes/masks slices are matching up to float tolerance for segmentation conversion.
- Real checkpoint parity above was re-run after modular regeneration (same metrics), confirming generated `modeling_rf_detr.py` parity.
- RF-DETR tests (including new instance-segmentation unit coverage): `191 passed, 144 skipped, 14 warnings`.
- RF-DETR image-processing tests (slow+fast): `23 passed, 3 skipped`.
- RF-DETR combined modeling + image-processing tests after fast processor integration: `214 passed, 147 skipped, 14 warnings`.

## Notes
- The RT-DETR small checkpoint can be found at `/Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-small.pth`.
- The real pre-trained `RFDETRSmall` checkpoint is now available locally at `/Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-small.pth` and was used for conversion/parity.
- Numerical parity is currently validated via a locally generated RFDETRSmall-style checkpoint artifact (same architecture/args shape), and the converter prints matching slices with very small max abs diff (~8e-6 logits, 0 boxes).
- Verification was re-run inside the existing repository virtual environment (`.venv`) as requested.

## 2026-03-01 Follow-up
- [x] Fixed `utils/check_config_attributes.py` failure for RF-DETR config classes:
  - added `RfDetrConfig: True` to `SPECIAL_CASES_TO_ALLOW` (same handling pattern as `LwDetrConfig`, because these loss-related attributes are consumed through shared loss utilities outside `modeling_rf_detr.py`),
  - added `RfDetrWindowedDinov2Config: ["gradient_checkpointing"]` to `SPECIAL_CASES_TO_ALLOW`.
- [x] Verified with existing virtual environment:
  - command: `.venv/bin/python utils/check_config_attributes.py`
  - result: no errors.

## 2026-03-02 Refactor: Converter Cleanup
- [x] Refactored `src/transformers/models/rf_detr/convert_rf_detr_to_hf.py` to reduce duplication and improve maintainability.
- [x] Consolidated model metadata/defaults:
  - replaced large repeated per-model dict blocks with shared default templates (`_DETECTION_COMMON_DEFAULT_ARGS`, `_SEGMENTATION_COMMON_DEFAULT_ARGS`),
  - introduced compact variant tables (`_OBJECT_DETECTION_VARIANTS`, `_INSTANCE_SEGMENTATION_VARIANTS`) and generated `MODEL_SPECS` from them,
  - removed parallel metadata map maintenance in favor of direct lookups from `MODEL_SPECS` in model-name resolution, checkpoint-path inference, and filename resolution.
- [x] Simplified conversion flow:
  - added helper functions to split responsibilities:
    - `_extract_num_labels_from_state_dict`
    - `_build_model_and_processors`
    - `_convert_state_dict`
    - `_add_missing_register_tokens_if_needed`
    - `_run_postprocess`
    - `_print_max_abs_diff`
    - `_verify_postprocess_outputs`
    - `_verify_conversion_with_original`
  - replaced the long inline verification block inside `convert_rf_detr_checkpoint` with a single helper call.
- [x] Preserved behavior and verified after refactor:
  - syntax check: `.venv/bin/python -m py_compile src/transformers/models/rf_detr/convert_rf_detr_to_hf.py`
  - lint check: `ruff check src/transformers/models/rf_detr/convert_rf_detr_to_hf.py`
  - smoke conversion:
    `uv run --no-project --python .venv/bin/python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --model_name small --checkpoint_repo_id nielsr/rf-detr-checkpoints --pytorch_dump_folder_path /tmp/rf-detr-small-hf-smoke`
    -> `Missing keys: 0`, `Unexpected keys: 0`.
- [x] Line-count reduction:
  - before: `1409` lines,
  - after: `1263` lines.

## 2026-03-02 Loss Audit: RF-DETR vs LW-DETR Mapping
- [x] Audited current mapping in `src/transformers/loss/loss_utils.py`:
  - `RfDetrForObjectDetection -> LwDetrForObjectDetectionLoss` (`loss_utils.py:168`).
- [x] Compared HF `src/transformers/loss/loss_lw_detr.py` against original RF-DETR implementation in:
  - `/Users/nielsrogge/Documents/python_projecten/rf-detr/src/rfdetr/models/lwdetr.py`
  - `/Users/nielsrogge/Documents/python_projecten/rf-detr/src/rfdetr/models/matcher.py`
- [x] Findings:
  - RF-DETR upstream uses `SetCriterion` with **multiple classification-loss modes** (`ia_bce_loss`, `use_position_supervised_loss`, `use_varifocal_loss`, fallback focal) and `sum_group_losses` control (`lwdetr.py:299-330`, `lwdetr.py:343-408`, `lwdetr.py:587-589`, `lwdetr.py:879-894`).
  - HF `LwDetrForObjectDetectionLoss` currently hardcodes the **IA-BCE-style branch** (IoU-modulated BCE) and does not expose those mode switches (`loss_lw_detr.py:111-145`, `loss_lw_detr.py:305-355`).
  - RF-DETR upstream matcher uses numerically-stable `logsigmoid` formulation and optional mask costs in matcher (for segmentation); HF LW matcher uses the focal-cost formula via probability logs and the object-detection matching path (`matcher.py:98-103`, `matcher.py:108-149` vs `loss_lw_detr.py:60-79`).
  - RF-DETR upstream weights `loss_ce` with `args.cls_loss_coef`; HF `LwDetrForObjectDetectionLoss` fixes CE weight to `1` in the final weighted sum (`lwdetr.py:861-863` vs `loss_lw_detr.py:345`).
- [x] Checked real RF-DETR small checkpoint args (`/Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-small.pth`):
  - `ia_bce_loss=True`, `use_varifocal_loss=False`, `use_position_supervised_loss=False`, `sum_group_losses=False`, `cls_loss_coef=1.0`.
  - This means the current HF mapping is aligned with the **default/released small checkpoint path**, but is **not a full exact superset** of all RF-DETR loss options from upstream training code.

## 2026-03-03 Image Processor Reuse Audit (RF-DETR vs LW-DETR)
- [x] Checked current auto mappings in `src/transformers/models/auto/image_processing_auto.py`:
  - `lw_detr -> (DeformableDetrImageProcessor, DeformableDetrImageProcessorFast)`,
  - `rf_detr -> (RfDetrImageProcessor, RfDetrImageProcessorFast)`.
- [x] Verified RF-DETR conversion script parity still passes with current RF processors:
  - object detection:
    `uv run --no-project --python .venv/bin/python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --checkpoint_path /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-small.pth --original_repo_path /Users/nielsrogge/Documents/python_projecten/rf-detr --verify_with_original --pytorch_dump_folder_path /tmp/rf-detr-small-hf-verify-reuse-check`
    -> exact/near-exact preprocessing and postprocessing parity, conversion load clean.
  - instance segmentation:
    `uv run --no-project --python .venv/bin/python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --checkpoint_path /Users/nielsrogge/Documents/python_projecten/rf-detr/rf-detr-seg-small.pt --original_repo_path /Users/nielsrogge/Documents/python_projecten/rf-detr --verify_with_original --pytorch_dump_folder_path /tmp/rf-detr-seg-small-hf-verify-reuse-check`
    -> preprocessing + detection/segmentation postprocessing parity, conversion load clean.
- [x] Reuse check against LW-DETR’s current processor choice (`DeformableDetrImageProcessor` + fast), using the same conversion-script parity setup/helpers:
  - preprocessing mismatch vs original RF-DETR pipeline:
    - small: `max_abs_preprocess_diff ~= 1.7245e-2` (slow), `~8.7537e-3` (fast),
    - seg-small: `max_abs_preprocess_diff ~= 1.7233e-2` (slow), `~8.7536e-3` (fast),
    - while RF processors are exact/near-exact (`0.0` for slow, ~`1e-6` for fast).
  - object-detection postprocess matched for this check (scores/boxes/labels), but:
  - `DeformableDetrImageProcessor` does **not** implement `post_process_instance_segmentation`, so it cannot cover RF-DETR instance-segmentation inference/postprocessing requirements.
- [x] Conclusion:
  - We should **not** remap `rf_detr` to LW-DETR’s current auto image processor mapping (`DeformableDetrImageProcessor`), because it is not a drop-in replacement for RF-DETR end-to-end parity requirements (notably preprocessing exactness and missing instance-segmentation postprocess API).
  - No change made to `image_processing_auto.py` mapping.

## 2026-03-03 Image Processor Reuse Audit (RF-DETR vs DETR)
- [x] Evaluated whether `rf_detr` can reuse DETR image processors (`DetrImageProcessor`, `DetrImageProcessorFast`) instead of RF-specific processors, using conversion-script-style parity checks against original RF-DETR implementation.
- [x] Preprocessing parity check (same dummy inputs used in converter logic):
  - `small` checkpoint:
    - RF processors: exact/near-exact (`0.0` slow, `~9.54e-7` fast),
    - DETR processors: clear mismatch (`~1.7245e-2` slow, `~8.7537e-3` fast).
  - `seg-small` checkpoint:
    - RF processors: exact/near-exact (`0.0` slow, `~7.15e-7` fast),
    - DETR processors: clear mismatch (`~1.7233e-2` slow, `~8.7536e-3` fast).
- [x] Post-processing parity check vs original RF-DETR postprocessor:
  - Object detection with DETR processors is not compatible with RF-DETR semantics:
    - `small`: score diff `~9.98e-2`, box diff `~526.2`, labels mismatch.
    - `seg-small`: score diff `~9.53e-2`, box diff `~374.6`, labels mismatch.
  - Root cause is expected: DETR postprocess uses softmax/no-object behavior and different selection logic, while RF-DETR uses sigmoid + top-k over all query/class scores.
- [x] Instance-segmentation API mismatch:
  - DETR `post_process_instance_segmentation` returns panoptic-style structure (`segmentation`, `segments_info`) and does not match RF-DETR’s expected output format (`scores`, `labels`, `boxes`, `masks`) used in RF conversion verification.
- [x] Conclusion:
  - We should **not** remap `rf_detr` to DETR image processors in `src/transformers/models/auto/image_processing_auto.py`.
  - No mapping change made.

## 2026-03-05 Base Conversion Verification Fix (rf-detr-base)
- [x] Reproduced user-reported failure when running:
  - `.venv/bin/python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --model_name base --original_repo_path /Users/nielsrogge/Documents/python_projecten/rf-detr --pytorch_dump_folder_path . --verify_with_original`
  - Original failure: assertion in upstream backbone requiring input shape divisible by `56`, while converter verification used `518x518`.
- [x] Root cause analysis:
  - `_prepare_checkpoint_args` was always overriding `resolution` with the image size inferred from position embeddings.
  - For `rf-detr-base`, checkpoint args contain `resolution=560` (valid for windowed block size `56`), but position embeddings are `37x37` (`518` image size), so verification input became `518` and failed upstream assert.
  - After stopping this override, conversion load then exposed a second issue: HF config `image_size=560` mismatched checkpoint positional embeddings shape (`1370` vs expected `1601`).
- [x] Implemented fix in `src/transformers/models/rf_detr/convert_rf_detr_to_hf.py`:
  - `_prepare_checkpoint_args`: only infer/assign `resolution` from position embeddings when `resolution` is missing.
  - Added a split between:
    - preprocessing/verification resolution (kept from checkpoint args, e.g. `560` for base),
    - backbone config image size (inferred from position embeddings when available, e.g. `518` for base) so checkpoint tensors load without shape mismatch.
  - `build_rf_detr_config_from_checkpoint` now accepts `backbone_image_size`.
  - `_build_model_and_processors` now accepts `processor_image_size` so saved image processors use checkpoint resolution.
  - `build_original_rfdetr_model` now accepts `state_dict` and infers `positional_encoding_size` from checkpoint position embeddings for verification model reconstruction.
- [x] Verified fix end-to-end (existing `.venv`):
  - Command:
    `.venv/bin/python src/transformers/models/rf_detr/convert_rf_detr_to_hf.py --model_name base --original_repo_path /Users/nielsrogge/Documents/python_projecten/rf-detr --pytorch_dump_folder_path . --verify_with_original`
  - Result: conversion succeeded (`Missing keys: 0`, `Unexpected keys: 0`) and verification completed successfully against original implementation.
  - Key parity metrics:
    - `max_abs_preprocess_diff=0.0`
    - `max_abs_logits_diff=8.86917e-05`
    - `max_abs_boxes_diff=1.153946e-04`
    - postprocess diffs (`scores`/`boxes`) `=0.0`, labels match `True`.
- [x] Regression check:
  - Re-ran `small` conversion/verification with existing `.venv` and parity still passes (`Missing keys: 0`, `Unexpected keys: 0`; preprocessing and postprocess checks pass).

## 2026-03-05 Full Model-Name Conversion Sweep
- [x] Verified conversion for **all supported `--model_name` values** using existing `.venv` with `--verify_with_original`, outputting converted artifacts under `/tmp/rf-detr-convert-all` and per-model logs under `/tmp/rf-detr-convert-all/logs`.
- [x] Object detection models:
  - `nano`, `small`, `medium`, `large`, `base`, `base-2`, `base-o365` -> all pass.
- [x] Instance segmentation models:
  - `seg-preview`, `seg-nano`, `seg-small`, `seg-medium`, `seg-large`, `seg-xlarge`, `seg-2xlarge` -> all pass.
- [x] Fix applied during sweep:
  - Initial `base-o365` verification failed in original-model reconstruction with:
    - `ValueError: peft.__spec__ is None`
  - Updated PEFT compatibility shim in `src/transformers/models/rf_detr/convert_rf_detr_to_hf.py` to set:
    - `peft.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)`
  - Re-ran `base-o365`; it then passed.

## 2026-03-05 RF-DETR fp16 Fine-Tuning Loss Dtype Fix
- [x] Reproduced mixed-precision loss failure in the RF-DETR training path (`RfDetrForObjectDetection` -> `LwDetrForObjectDetectionLoss`) with a minimal train-step script under `torch.autocast(..., dtype=torch.float16)`:
  - Error:
    - `RuntimeError: Index put requires the source and destination dtypes match, got Half for the destination and Float for the source.`
  - Failing line before fix:
    - `pos_weights[pos_ind] = t` in `src/transformers/loss/loss_lw_detr.py`.
- [x] Root cause:
  - Under mixed precision, `t = prob[pos_ind].pow(alpha) * pos_ious.pow(1 - alpha)` may be upcast to `float32` while `pos_weights`/`neg_weights` stay in lower precision, causing indexed assignment dtype mismatch.
- [x] Fix implemented in `src/transformers/loss/loss_lw_detr.py`:
  - Replaced list-based advanced index with tuple index (`pos_ind = (*idx, target_classes_o)`).
  - Added explicit dtype alignment before indexed assignment:
    - `t = torch.clamp(t, 0.01).to(pos_weights.dtype).detach()`
    - `neg_weights[pos_ind] = (1 - t).to(neg_weights.dtype)`
- [x] Verification:
  - Mixed precision forward-loss path now succeeds where it previously crashed.
  - End-to-end single train step (`forward + loss + backward + optimizer.step`) succeeds on MPS with fallback enabled for unrelated backend coverage:
    - `PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python <minimal_rfdetr_autocast_train_step_script>`
  - Note:
    - Without fallback, backward currently fails on MPS due unrelated missing op (`aten::grid_sampler_2d_backward`), not due RF-DETR loss dtype handling.
