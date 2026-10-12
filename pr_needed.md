# Integration follow-ups

## vLLM (`vllm_swissai`, `vllm/model_executor/models/apertus_mm.py`)

Rechecked on 2026-10-12 against the current remote heads of the Apertus 1.5 branches:
`origin/fix/apertus-1.5-draft-model-image-token` (8f2584e35, 2026-09-12) and `origin/apertus-1-5` (806a65923,
2026-08-13). Both heads are unchanged since the previous check. Line numbers refer to the first branch.

vLLM currently matches the checkpoint published on `main` of `swiss-ai/Apertus-v1.5-8B` (old config layout). All
items below apply as soon as vLLM runs against the transformers PR version and the updated checkpoint, so they
have to land together with the checkpoint update.

### 1. Renamed sub-config keys (lines 474, 479)

`config.vision_tokenizer_config` and `config.audio_tokenizer_config` no longer exist.

```python
# now
_init_component_model(config.vision_tokenizer_config, model_cls=Apertus1p5VisionTokenizerModel)
_init_component_model(config.audio_tokenizer_config)
# needed
_init_component_model(config.vision_config, model_cls=Apertus1p5VisionTokenizerModel)
_init_component_model(config.audio_config)
```

Origin: transformers commit "refactor(apertus1p5): standardize multimodal sub-config names" (Cyril's review).

### 2. Vision tokenizer `encode` returns an output object (line 547)

`Apertus1p5VisionTokenizerModel.encode` returns `Apertus1p5VisionTokenizerEncoderOutput`, not a tensor, so
`image_codes.flatten()` fails.

```python
# now
image_codes = vision_tower.encode(...)
# needed
image_codes = vision_tower.encode(...).image_tokens
```

Origin: transformers commit "Simplify Apertus model interfaces and config validation" (zucchini-nlp, modular `:434`).

### 3. Audio tower is now the encoder-only WavTokenizer (lines 109-115, 478-480)

The composite `audio_config.model_type` is now `wavtokenizer_encoder`, so `_init_component_model(config.audio_config)`
(`AutoConfig.for_model` + `AutoModel.from_config`) builds `WavTokenizerEncoderModel` (encoder and quantizer only)
instead of the full codec. With the existing weight mapper (`model.audio_tokenizer.` to `audio_tower.`), the keys
`audio_tower.encoder.*` and `audio_tower.quantizer.*` line up directly, and `.encode(...).audio_codes` is unchanged.

No code change expected; verify that the audio tower loads without missing or unexpected keys.

Origin: Apertus now uses `AutoModel` with a dedicated `WavTokenizerEncoderConfig` (zucchini-nlp modular `:38`,
first half of eustlb's encoder/decoder split request).

### 4. `image_processor.max_pixels` no longer exists (lines 172, 213)

The pixel budget lives in the `size` dict. The attribute only exists today because the published processor config
still has the old `min_pixels`/`max_pixels` keys; after the checkpoint update this raises `AttributeError` in the
token budget and the dummy inputs.

```python
# now
image_processor.max_pixels
# needed
image_processor.size["max_pixels"]
```

Origin: transformers commit "Align Apertus image preprocessing with training" (zucchini-nlp, image processor `:88`).

### 5. `replace_image_token` reads `image_sizes`, not `image_grids` (lines 286-296)

The processor derives the grid from `image_sizes` and the spatial factor, so passing `image_grids` raises
`KeyError: 'image_sizes'`. The local grid computation can go.

```python
# now
image_grids = [[height // image_processor.spatial_factor, width // image_processor.spatial_factor]]
image_replacement = processor.replace_image_token({"image_grids": image_grids}, 0)
# needed
image_replacement = processor.replace_image_token({"image_sizes": [[height, width]]}, 0)
```

Origin: removal of `image_grids` from the image processor (zucchini-nlp, image processor `:145`).

## Checkpoint (`swiss-ai/Apertus-v1.5-8B` and `swiss-ai/Apertus-v1.5-70B`) changes these depend on

Checked on 2026-10-12 in the [8B checkpoint PR #5](https://huggingface.co/swiss-ai/Apertus-v1.5-8B/discussions/5)
(revision `909676f895e367025071cf809c488092649a2765`) and
[70B checkpoint PR #8](https://huggingface.co/swiss-ai/Apertus-v1.5-70B/discussions/8)
(revision `1169769c728931831ccf188709bc50a9aa109190`). Hub `main` still uses the older sub-config layout;
completed items below refer to the PR revisions, not the published `main` branches.

- [x] `config.json`: renamed `vision_config` / `audio_config` keys are present in both PRs.
- [x] `config.json`: `dtype = "bfloat16"` is present in both PRs; audio tokenizer dtype is `float32`.
- [ ] `processor_config.json`: replace the image processor's legacy `min_pixels` / `max_pixels` fields with
  `size = {"min_pixels": 16384, "max_pixels": 1960000}`. Both PRs still omit `size` and store the old minimum
  of 65536. Keep `do_resize = true` and `spatial_factor = 16`.

- [ ] Update `config.json` in the [8B checkpoint PR #5](https://huggingface.co/swiss-ai/Apertus-v1.5-8B/discussions/5)
  and [70B checkpoint PR #8](https://huggingface.co/swiss-ai/Apertus-v1.5-70B/discussions/8): change
  `audio_config.model_type` from `"wavtokenizer"` to `"wavtokenizer_encoder"`. Keep the top-level
  `model_type` as `"apertus1p5"`.

Both PR configs still contain the old audio type. Current `Apertus1p5Config` rejects them; changing only
this field makes both pass config validation. Their indexes already contain 62 encoder tensors and four
quantizer tensors under `model.audio_tokenizer.`, in a separate audio shard, with no decoder tensors.
The metadata corrections require no weight renaming. On 2026-10-12, the local 8B weights passed the slow
integration tests on CPU using a temporary copy of the corrected metadata, including multimodal generation
and standalone text-model loading. The original checkpoint was left unchanged. Validate the final Hub PR
revisions, including the 70B checkpoint, before publishing.

This adaptation comes from the earlier encoder-only audio integration, not the subsequent docstring cleanup.
The converter already writes the correct type and image processor size. The standalone full WavTokenizer checkpoint must retain
`model_type = "wavtokenizer"`; it supports loading either the full codec or its encoder.

## WavTokenizer encoder/decoder split: waiting for upstream

[Reviewer request](https://github.com/huggingface/transformers/pull/47662#discussion_r4093394327): independently
loadable encoder and decoder models, composed through `AutoModel` in the full codec.

- Done: dedicated encoder model/config, loading the encoder from a full checkpoint, and Apertus using the
  encoder through `AutoModel`.
- Pending: decoder model/config, full-codec composition, shared codebook loading, converter compatibility,
  documentation, and tests for loading either half independently.
- Prerequisite: [#49087](https://github.com/huggingface/transformers/pull/49087), still open on 2026-10-12 and
  marked blocked by [#49138](https://github.com/huggingface/transformers/pull/49138). Wait for upstream;
  do not carry separate shared-loader changes in this integration.

The codebook is stored as buffers. Sharing it between encoder and decoder requires loading paths that use
`get_parameter_or_buffer()`: the current parameter-only lookups fail with `embed is not an nn.Parameter`.
This does not affect the current encoder-only loading path. Preserve existing full-checkpoint compatibility
and Apertus audio tensor names when completing the split; update this note with any additional migration
requirements established by the merged upstream implementation.

## Upstream `huggingface/transformers` issues (not part of the Apertus PR)

Found during the 2026-10-09 review of the Apertus branch. Each affects other models too; Apertus either works
around it or is not hit with the released checkpoint. Rechecked on 2026-10-12 against upstream main
`e4f8d92a61a091e5d70510f1e354680f2a1268cd`: all three relevant paths remain unchanged. Issues 2 and 3 were
also reproduced locally; issue 1 was inspected without a GPU on this machine.

### 1. Shared multimodal pre-encoding ignores the model's device

`GenerationMixin._prepare_multimodal_encoder_kwargs_for_generation` (`generation/utils.py:1013`, from #45783) calls
`get_{image,video}_features` with the raw `model_kwargs` before `prepare_inputs_for_generation` moves inputs to the
model device (step 6 at `generation/utils.py:770`, which explicitly allows callers to keep inputs elsewhere).

- Repro: tiny Llava moved to MPS/GPU, `generate(input_ids=..., pixel_values=<cpu tensor>)` raises
  `RuntimeError: ... input(device='cpu') and weight(device='mps:0') must be on the same device`.
- Worked before for models that encode media inside `forward`.
- Apertus works around it by moving inputs to the tokenizer's device in `Apertus1p5VisionTokenizerModel.encode`.
- Possible fix: move the encoder kwargs to the model (or encoder) device before calling `get_*_features`.

### 2. `generate` mutates caller-provided `mm_encoder_outputs` in place

`_expand_multimodal_outputs` (`generation/utils.py:1149`) assigns the beam-expanded list back to
`modalily_outputs.pooler_output` on the object the caller passed in.

- Repro: `feats = model.get_image_features(...)`, then
  `model.generate(..., mm_encoder_outputs={"image": feats}, num_beams=2)`; afterwards `len(feats.pooler_output)`
  has doubled, and reusing `feats` raises a token/feature count mismatch or silently drops entries.
- Possible fix: shallow-copy the output (and the dict) before expanding.

### 3. Tokenizer init kwargs leak into audio kwargs

`ProcessorMixin._merge_kwargs` (`processing_utils.py:1615-1624`) fills the defaults of every modality, audio
included, from `tokenizer.init_kwargs`. A tokenizer whose saved config contains `pad_to_multiple_of` (for example
GPT-2 based ones) therefore passes `pad_to_multiple_of=None` and `max_length=None` to the feature extractor.

- `WavTokenizerFeatureExtractor` and `EncodecFeatureExtractor` accept `max_length`, but not `pad_to_multiple_of`, and raise
  `TypeError: __call__() got an unexpected keyword argument 'pad_to_multiple_of'` after a processor save/reload.
- Not hit by the Apertus checkpoint: its tokenizer config does not store `pad_to_multiple_of` (checked before and
  after a re-save).
- Possible fix: only apply tokenizer init kwargs to `text_kwargs`, or filter against the target component's
  accepted arguments.
