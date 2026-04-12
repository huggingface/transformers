# PR 44320 review follow-up

Review scope: comments from the April 10, 2026 review submitted with "Sorry, found a few other smaller things. Shouldn't be anything big (and also some repeating stuff)".

## In progress

- [x] Switch the SAM3-LiteText doc example from `requests` to `httpx`.
- [x] Add the missing `sam3_lite_text` auto image processor mapping and cover it with a unit test.
- [x] Apply the requested SAM3-LiteText modeling cleanups in `modular_sam3_lite_text.py`.
- [x] Tighten the SAM3-LiteText tests based on the review feedback.

## Validation

- [x] Regenerated the standalone SAM3-LiteText files from `modular_sam3_lite_text.py`.
- [x] `ruff check --fix` and `ruff format` passed on the touched Python files.
- [x] `pytest tests/models/auto/test_image_processing_auto.py -k sam3_lite_text -v`
- [x] `pytest tests/models/sam3_lite_text/test_modeling_sam3_lite_text.py -v` -> `100 passed, 104 skipped`
- [x] Reverted 167 unintended repo-wide file modifications so the remaining diff stays limited to the SAM3-LiteText review follow-up.

## Needs reply / no code change expected

- [x] `docs/source/en/model_doc/nomic_bert.md`: "Unrelated but welcome :D" appears informational only.
- [x] `docs/source/en/model_doc/sam3_lite_text.md`: "Are there any plans to move these to another repo?" likely needs a PR reply, not a code change.
- [x] `src/transformers/models/sam3_lite_text/modular_sam3_lite_text.py`: "Just to be sure we don't need position ids here..." likely needs rationale, not a code change.

## Suggested PR replies

- `nomic_bert`: no action needed; treat as an informational note.
- `Are there any plans to move these to another repo?`: answer based on the current release plan for the checkpoints/docs, no code change needed.
- `position ids`: explain that the text positional embedding is strictly sequence-length based and the padding behavior is handled by the attention mask, so explicit `position_ids` are not needed here.
