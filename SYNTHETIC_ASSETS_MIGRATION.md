# Synthetic test assets migration

Branch: `tarek/synthetic-test-assets`

Replaces every third-party-hosted media fixture reachable from the test suite with a
procedurally generated, CC0 equivalent from
[`hf-internal-testing/transformers-synthetic-assets`](https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets).
The generator scripts are not committed here: one hardcodes a local path, so they belong in the
dataset repo alongside the assets they produce (worth uploading, as they are the provenance for
the CC0 claim).

## What changed

| | |
|---|---|
| URLs remapped | 45 (41 from the asset manifest + 4 leftovers on the same hosts) |
| Occurrences rewritten | 319 |
| Files touched | 148 |
| Non-HF media refs left in `tests/` | **0** |

Hosts removed from the test path: `cdn.britannica.com`, `thumbs.dreamstime.com`,
`llava-vl.github.io`, `aria-vl.github.io`, `paddle-model-ecology.bj.bcebos.com`,
`qianwen-res.oss-*.aliyuncs.com`, `clip-cn-beijing.oss-cn-beijing.aliyuncs.com`,
`nineplanets.org`, `i.postimg.cc`, `image.slidesharecdn.com`, `templates.invoicehome.com`,
`www.kxan.com`, plus the re-hosted copies under `spaces/impira/docquery`,
`ydshieh/mistral3-test-data` and `huggingface/documentation-images`.

## Follow-up required: refresh pinned expectations on accelerator hardware

The synthetic assets are visually and semantically *analogous*, not identical, so any test
that pins model output computed from the old pixels/samples now needs re-recording. Each is
marked in-tree with `# TODO(synthetic-assets):` — grep for it.

**83 tests across 40 files** (97 tests touch a synthetic asset in total;
the other 14 assert only on shapes/dtypes and should pass unchanged).

Verified locally on CPU: `mgp_str::test_inference` — the shape assert and the
`generated_text == "ticket"` assert both still pass (the synthetic image has "ticket" drawn
on it); only the pinned logits slice differs. That is the expected shape of the remaining work.

| File | Tests needing a refreshed expectation |
|---|---|
| `tests/models/aria/test_modeling_aria.py` | `test_small_model_integration_test`, `test_small_model_integration_test_batch`, `test_small_model_integration_test_llama_batched`, `test_small_model_integration_test_llama_batched_regression`, `test_small_model_integration_test_llama_single` |
| `tests/models/aya_vision/test_modeling_aya_vision.py` | `test_small_model_integration_batched_generate`, `test_small_model_integration_batched_generate_multi_image` |
| `tests/models/chameleon/test_modeling_chameleon.py` | `test_model_7b`, `test_model_7b_batched`, `test_model_7b_multi_image` |
| `tests/models/chinese_clip/test_modeling_chinese_clip.py` | `test_inference` |
| `tests/models/chmv2/test_modeling_chmv2.py` | `test_inference_depth_estimation` |
| `tests/models/cohere2_vision/test_modeling_cohere2_vision.py` | `test_model_integration_batched_generate`, `test_model_integration_batched_generate_multi_image` |
| `tests/models/cosmos3_omni/test_modeling_cosmos3_omni.py` | `test_small_model_integration`, `test_small_model_integration_batched` |
| `tests/models/diffusion_gemma/test_modeling_diffusion_gemma.py` | `test_diffusion_gemma_chat_template_image`, `test_diffusion_gemma_forward_with_image`, `test_diffusion_gemma_generate_with_image_batched`, `test_diffusion_gemma_generate_with_image_batched_long`, `test_minified_diffusion_gemma_forward_with_image`, `test_minified_diffusion_gemma_generate_with_image_batched`, `test_minified_diffusion_gemma_generate_with_image_batched_long` |
| `tests/models/edgetam/test_modeling_edgetam.py` | `test_inference_mask_generation_batched_images_multi_points` |
| `tests/models/fast_vlm/test_modeling_fast_vlm.py` | `test_small_model_integration_test`, `test_small_model_integration_test_batch` |
| `tests/models/higgs_audio_v2/test_modeling_higgs_audio_v2.py` | `test_multi_speaker_voice_cloning` |
| `tests/models/idefics2/test_modeling_idefics2.py` | `test_integration_test`, `test_integration_test_4bit` |
| `tests/models/idefics3/test_modeling_idefics3.py` | `test_integration_test`, `test_integration_test_4bit` |
| `tests/models/instructblip/test_modeling_instructblip.py` | `test_inference_flant5_xl`, `test_inference_vicuna_7b` |
| `tests/models/internvl/test_modeling_internvl.py` | `test_llama_small_model_integration_batched_generate`, `test_llama_small_model_integration_batched_generate_multi_image`, `test_llama_small_model_integration_interleaved_images_videos`, `test_qwen2_small_model_integration_batched_generate`, `test_qwen2_small_model_integration_batched_generate_multi_image`, `test_qwen2_small_model_integration_interleaved_images_videos` |
| `tests/models/janus/test_modeling_janus.py` | `test_model_text_generation`, `test_model_text_generation_batched`, `test_model_text_generation_with_multi_image` |
| `tests/models/lfm2_vl/test_modeling_lfm2_vl.py` | `test_integration_test_batched`, `test_integration_test_batched`, `test_integration_test_high_resolution`, `test_integration_test_high_resolution` |
| `tests/models/llava/test_modeling_llava.py` | `test_small_model_integration_test`, `test_small_model_integration_test_batch`, `test_small_model_integration_test_llama_batched`, `test_small_model_integration_test_llama_batched_regression`, `test_small_model_integration_test_llama_single` |
| `tests/models/mgp_str/test_modeling_mgp_str.py` | `test_inference` |
| `tests/models/mistral3/test_modeling_mistral3.py` | `test_mistral3_integration_batched_generate`, `test_mistral3_integration_batched_generate_multi_image` |
| `tests/models/mllama/test_modeling_mllama.py` | `test_11b_model_integration_batched_generate`, `test_11b_model_integration_forward`, `test_11b_model_integration_generate`, `test_11b_model_integration_multi_image_generate` |
| `tests/models/omdet_turbo/test_modeling_omdet_turbo.py` | `test_inference_object_detection_head_batched` |
| `tests/models/paddleocr_vl/test_modeling_paddleocr_vl.py` | `test_small_model_integration_test`, `test_small_model_integration_test_batch`, `test_small_model_integration_test_batch_flashatt2`, `test_small_model_integration_test_flashatt2` |
| `tests/models/phi4_multimodal/test_modeling_phi4_multimodal.py` | `test_multi_image_vision_text_generation` |
| `tests/models/pp_chart2table/test_modeling_pp_chart2table.py` | `test_small_model_integration_test_pp_chart2table`, `test_small_model_integration_test_pp_chart2table_batched` |
| `tests/models/prompt_depth_anything/test_modeling_prompt_depth_anything.py` | `test_inference`, `test_inference_wo_prompt_depth` |
| `tests/models/qwen2_5_omni/test_modeling_qwen2_5_omni.py` | `test_small_model_integration_test_w_audio` |
| `tests/models/qwen2_audio/test_modeling_qwen2_audio.py` | `test_small_model_integration_test_batch` |
| `tests/models/qwen3_asr/test_modeling_qwen3_asr.py` | `test_fixture_batch_matches`, `test_fixture_timestamps_batched` |
| `tests/models/qwen3_omni_moe/test_modeling_qwen3_omni_moe.py` | `test_small_model_integration_test_multiturn`, `test_small_model_integration_test_w_audio` |
| `tests/models/qwen3_vl/test_modeling_qwen3_vl.py` | `test_small_model_integration_test` |
| `tests/models/sam/test_modeling_sam.py` | `test_inference_mask_generation_batched_image_one_point` |
| `tests/models/sam2/test_modeling_sam2.py` | `test_inference_mask_generation_batched_images_multi_points` |
| `tests/models/sam3_tracker/test_modeling_sam3_tracker.py` | `test_inference_mask_generation_batched_images_multi_points` |
| `tests/models/smolvlm/test_modeling_smolvlm.py` | `test_integration_test`, `test_integration_test_video` |
| `tests/pipelines/test_pipelines_document_question_answering.py` | `test_small_model_pt`, `test_small_model_pt_bf16` |
| `tests/pipelines/test_pipelines_image_text_to_text.py` | `test_model_pt_chat_template` |

## Follow-up: doctest output

These docstrings/docs pin output after a synthetic-asset example and run in the nightly
`Doctests` workflow. They carry no in-tree marker on purpose: a comment inside a docstring
either breaks the doctest or leaks into the published docs.

| File | Line | Pinned output |
|---|---|---|
| `src/transformers/models/aria/modeling_aria.py` | 1080 | `Assistant: There are buildings, trees, lights, and water visible in this image.` |
| `src/transformers/models/aria/modular_aria.py` | 996 | `Assistant: There are buildings, trees, lights, and water visible in this image.` |
| `src/transformers/models/idefics2/modeling_idefics2.py` | 1047 | `['In this image, we can see the city of New York, and more specifically the Statue of Liberty. In th` |
| `src/transformers/models/idefics3/modeling_idefics3.py` | 793 | `Assistant: There are buildings, trees, lights, and water visible in this image.` |
| `src/transformers/models/pp_doclayout_v2/modeling_pp_doclayout_v2.py` | 2407 | `Order 1: text: 0.99 [335.39, 184.26, 896.49, 654.48]` |
| `src/transformers/models/pp_doclayout_v2/modular_pp_doclayout_v2.py` | 871 | `Order 1: text: 0.99 [335.39, 184.26, 896.49, 654.48]` |
| `src/transformers/models/pp_doclayout_v3/modeling_pp_doclayout_v3.py` | 2060 | `Order 1: text: 0.99 [334.95, 184.78, 897.25, 654.83]` |
| `src/transformers/models/pp_doclayout_v3/modular_pp_doclayout_v3.py` | 1354 | `Order 1: text: 0.99 [334.95, 184.78, 897.25, 654.83]` |
| `src/transformers/models/pp_formulanet/modeling_pp_formulanet.py` | 1077 | `['\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}\\int_{\\mu}^{\\infty}d\\omega\\int_{C_{+}}d ` |
| `src/transformers/models/pp_formulanet/modular_pp_formulanet.py` | 466 | `['\\zeta_{0}(\\nu)=-\\frac{\\nu\\varrho^{-2\\nu}}{\\pi}\\int_{\\mu}^{\\infty}d\\omega\\int_{C_{+}}d ` |
| `src/transformers/models/pp_lcnet/modeling_pp_lcnet.py` | 347 | `wireless_table` |
| `src/transformers/models/pp_lcnet/modular_pp_lcnet.py` | 514 | `wireless_table` |
| `src/transformers/models/qwen2_audio/modeling_qwen2_audio.py` | 821 | `"Generate the caption in English: Glass is breaking."` |
| `src/transformers/pipelines/any_to_any.py` | 99 | `[{'input_text': [{'role': 'user',` |
| `src/transformers/pipelines/image_text_to_text.py` | 84 | `[{'input_text': [{'role': 'user',` |

## Notes

- `utils/fetch_hub_objects_for_ci.py` now prefetches the synthetic dataset (42 of 43 files) and
  the stale `TODO: copy those to our hf-internal-testing dataset` list is gone.
- `audio/glass_breaking.mp3` is deliberately **not** prefetched: `url_to_local_path` keys on the
  basename and `dummy-audio-samples` ships a *different* `glass_breaking.mp3`, so prefetching
  both would let one silently shadow the other. It is fetched at test time instead.
- Added `_check_basename_collisions()` to that file so a future basename clash fails loudly
  instead of surfacing as a baffling assertion error. It allowlists the one pre-existing
  collision (`000000039769.png` under both `coco_panoptic/` and `val2017/`).
- Out of scope: ~600 third-party media refs remain in `src/` docstrings and `docs/` prose
  (COCO, ilankelman, wikimedia, ...). None are reachable from `tests/`.
- `make fix-repo`, `make style` and `make check-repo` pass. The single `check-repo` typing
  failure (`modeling_utils.py:3111`) is pre-existing on `main` and unrelated.
