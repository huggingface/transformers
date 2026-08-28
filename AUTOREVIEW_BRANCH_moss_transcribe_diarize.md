--- Deep Review: BRANCH modular_playground (moss_transcribe_diarize) ---

Breaching the repository agent contribution guidelines can result in automatic banning.

SUMMARY:
New `moss_transcribe_diarize` modular ports Whisper+VQ-adaptor+Qwen3 diarized ASR on top of GlmAsr/AudioFlamingo3. `check_modular_conversion` is clean. Highest-severity gaps: original checkpoint keys (`whisper_encoder` / `vq_adaptor`) are not renamed by the load registry (Fsys1), `_skip_keys_device_placement` is a string that becomes a char-set at init (Fsys3/F2), and there is no numerical integration/parity test (Fsys2).

DIMENSIONS:
- D1 modular-reuse — clean: inherits GlmAsr/AudioFlamingo3; `check_modular_conversion` exit 0
- D2 decorator-stack — findings: F5 (`return_dict=True` in new forwards)
- D3 attention-qkv — not triggered: no new attention module
- D4 torch-compile — findings: F4 (`bincount().tolist()` graph break)
- D5 fidelity-parity — findings: Fsys2
- D6 dead-branches — findings: F3 (`input_features_mask` unused)
- D7 config-discipline — clean: `@strict` typed config; `adaptor_input_dim` derived in `__post_init__`
- D8 layer-types — not triggered: no new decoder layers
- D9 signature-kwargs — findings: F3
- D10 bc-deprecation — not triggered: new model surface
- D11 cross-model-blast — not triggered: no shared-pattern edit of siblings
- D12 test-standards — findings: Fsys2
- D13 naming — clean: `MossTranscribeDiarize*` prefix on public classes
- D14 minimal-surface — clean: no `logger.warning` in modular
- D15 anti-premature-abstraction — clean: chunk loop stays in-file
- D16 loading-saving — findings: Fsys1
- D17 state-hoisting — clean: no module attrs written in forward
- D18 init-weights — not triggered: no custom `_init_weights`
- D19 tied-weights — clean: `_tied_weights_keys` dict form on CG
- D20 dtype-cast — clean: `.to(self.dtype)` on packed features only; no sensitive cast churn
- D21 conversion-hub — findings: Fsys1
- D22 vision-postprocess — not triggered: no vision post_process
- D23 optional-backends — clean: `@requires(backends=("torch",))` on processor
- D24 module-containers — clean: no getattr/setattr layer lists
- D25 path-pattern-plans — not triggered: no tp_plan edits
- D26 return-type-honesty — clean: annotated `tuple | ModelOutput` with `@can_return_tuple`
- D27 comment-hygiene — clean: merge/unpad comments state the trigger
- D28 docs-metadata — findings: F6 (inherited GLM-ASR example); no `docs/source/en/model_doc` page
- D29 occam-measurement — clean: single audio path via chunk map
- D30 escalation — not triggered: no tokenizer-owner collision
- D31 framework-mechanisms — findings: Fsys1, Fsys3
- D32 feature-isolation — findings: F7 (vendored original_code / extra package / MODULAR_PROMPT)
- D33 hybrid-cache — not triggered: no new cache/mask logic
- D34 mask-plumbing — not triggered: no new masking_utils kwargs
- D35 tokenizer — not triggered: reuses Qwen2Tokenizer via auto map

REVIEWER REMARKS:
Ledger: 0 reviewer comments (local untracked model; no inline review comments). Pass 2 covered R∅: 0 addressed, 0 outstanding (0+0 = 0).

Systemic checks run: 11/11; Fsys count = 3
Symbol ledger: 25 symbols (S1–S25) + 1 generated-files collective row.
Symbol ledger covered S1–S25: F citations + neutrals = 25 (F+D = 25).

FINDINGS:
FINDING Fsys1
FILE src/transformers/models/moss_transcribe_diarize/modular_moss_transcribe_diarize.py LINE —
SEVERITY: warning
CATEGORY: conversion
CODE: `_checkpoint_conversion_mapping = {`
NOTE: Class `_checkpoint_conversion_mapping` (whisper_encoder→audio_tower, vq_adaptor→multi_modal_projector) is not read by `get_model_conversion_mapping` / `from_pretrained`. Registry aliases (`moss_transcribe_diarize`→`qwen2_audio`, `MossTranscribeDiarizeModel`→`AudioFlamingo3Model`) leave original keys unchanged. Original index uses `model.whisper_encoder.*` and `model.vq_adaptor.*`. Move the renames into `conversion_mapping.py` as `WeightRenaming` rules and delete the class attribute.

FINDING Fsys2
FILE tests/models/moss_transcribe_diarize/ LINE —
SEVERITY: warning
CATEGORY: tests
CODE: `(no Integration / allclose / EXPECTED / assert_close matches under tests/models/moss_transcribe_diarize/)`
NOTE: Tester coverage is ALM mixin + processor shape/equality checks only. No numerical logits/decoded-text parity against the reference or a tiny deterministic checkpoint. Add a slow integration test with a stated provenance slice (or tiny hub checkpoint summary stats).

FINDING Fsys3
FILE src/transformers/models/moss_transcribe_diarize/modular_moss_transcribe_diarize.py LINE —
SEVERITY: warning
CATEGORY: flags
CODE: `_skip_keys_device_placement = "past_key_values"` / `_no_split_modules = ["Qwen3DecoderLayer", "WhisperEncoderLayer"]`
NOTE: Flag surface is wrong for this composite: string skip-keys and a non-existent split module name. See F1 and F2.

FINDING F1
FILE src/transformers/models/moss_transcribe_diarize/modular_moss_transcribe_diarize.py LINE 347
SEVERITY: warning
CATEGORY: flags
CODE: `_skip_keys_device_placement = "past_key_values"`
NOTE: `PreTrainedModel.__init__` does `set(self._skip_keys_device_placement or [])`. A string becomes a set of characters. Parents use a list. After init the instance contains single-char keys plus `past_key_values` from children.

MRE:
```python
from transformers import MossTranscribeDiarizeConfig, MossTranscribeDiarizeForConditionalGeneration

cfg = MossTranscribeDiarizeConfig(
    text_config={
        "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
        "head_dim": 8, "max_position_embeddings": 32,
    },
    audio_config={
        "num_mel_bins": 80, "d_model": 16, "encoder_layers": 1,
        "encoder_attention_heads": 2, "encoder_ffn_dim": 32, "max_source_positions": 16,
    },
    audio_merge_size=4, audio_token_id=0,
)
m = MossTranscribeDiarizeForConditionalGeneration(cfg)
chars = [x for x in m._skip_keys_device_placement if len(x) == 1]
assert chars, "expected set('past_key_values') pollution"
print(sorted(chars))
```

FINDING F2
FILE src/transformers/models/moss_transcribe_diarize/modular_moss_transcribe_diarize.py LINE 346
SEVERITY: warning
CATEGORY: flags
CODE: `_no_split_modules = ["Qwen3DecoderLayer", "WhisperEncoderLayer"]`
NOTE: Audio tower is `Qwen2AudioEncoder` (`Qwen2AudioEncoderLayer`). No `WhisperEncoderLayer` modules exist. Child towers merge the correct names, but the declared `WhisperEncoderLayer` entry is dead. Use `Qwen2AudioEncoderLayer` or omit and rely on submodel `_no_split_modules`.

FINDING F3
FILE src/transformers/models/moss_transcribe_diarize/modular_moss_transcribe_diarize.py LINE 430
SEVERITY: warning
CATEGORY: docs
CODE: `input_features_mask: Optional[torch.Tensor] = None,`
NOTE: `MossTranscribeDiarizeModel.forward` accepts `input_features_mask` but never uses it. Processor strips it from `model_input_names`. Instantiating the model prints auto_docstring ERROR that the arg is undocumented. Drop the parameter or document and wire it.

FINDING F4
FILE src/transformers/models/moss_transcribe_diarize/modular_moss_transcribe_diarize.py LINE 402
SEVERITY: warning
CATEGORY: logic
CODE: `for n_chunks in torch.bincount(audio_chunk_mapping).tolist():`
NOTE: `get_audio_features` breaks `torch.compile(..., fullgraph=True)` (`GuardOnDataDependentSymNode` / dynamic `bincount`). `_can_compile_fullgraph` is False (honest). Prefer a compile-safe pack (fixed max chunks, or Python lengths from the processor) if compile coverage is required.

FINDING F5
FILE src/transformers/models/moss_transcribe_diarize/modular_moss_transcribe_diarize.py LINE 385
SEVERITY: warning
CATEGORY: decorators
CODE: `audio_outputs = self.audio_tower(input_features, return_dict=True, **kwargs)`
NOTE: New-model forward path hard-codes `return_dict=True` (also at Model.forward→get_audio_features call). Decorator stack owns BC; remove the literal and use typed internal returns.

FINDING F6
FILE src/transformers/models/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py LINE 322
SEVERITY: warning
CATEGORY: docs
CODE: `>>> model_id = "zai-org/GLM-ASR-Nano-2512"`
NOTE: Modular does not override `ForConditionalGeneration.forward`, so the converter inlines GlmAsr’s docstring/example pointing at GLM-ASR. Override forward (or supply a custom docstring) with the MOSS checkpoint and `audio_feature_lengths` / `audio_chunk_mapping` inputs.

FINDING F7
FILE scripts/moss_transcribe_diarize/original_code/ / src/transformers/moss_transcribe_diarize/ / src/transformers/models/moss_transcribe_diarize/moss_transcribe_diarize_MODULAR_PROMPT LINE —
SEVERITY: warning
CATEGORY: policy
CODE: `(deliverable-scope paths present next to the model package)`
NOTE: Vendored `scripts/.../original_code` (weights, tokenizer, original modeling), duplicate `src/transformers/moss_transcribe_diarize/`, and `moss_transcribe_diarize_MODULAR_PROMPT` are working materials. Remove from the PR deliverable; keep only `models/moss_transcribe_diarize/` + tests + registry/docs.

SUGGESTED CHECKS:
- `PYTHONPATH=src python utils/check_modular_conversion.py --files src/transformers/models/moss_transcribe_diarize/modular_moss_transcribe_diarize.py`
- `PYTHONPATH=src python -c "from transformers.conversion_mapping import get_model_conversion_mapping; ..."` asserting `model.whisper_encoder.conv1.weight` → `model.audio_tower.conv1.weight`
- `PYTHONPATH=src pytest tests/models/moss_transcribe_diarize/ -q`
- `RUN_SLOW=1 PYTHONPATH=src pytest tests/models/moss_transcribe_diarize/ -k Integration -xvs` (once added)

VERDICT: Request changes
REASON: Fsys1 blocks original-checkpoint loading; Fsys3/F1 corrupt device-map skip keys; Fsys2 leaves no numerical source-of-truth test.
