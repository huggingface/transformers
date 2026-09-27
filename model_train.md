# Standard Qwen3-TTS Training

## Summary

Support Base-model fine-tuning through both `model(**batch).loss.backward()` and the standard `Trainer`. Precompute audio codes and speaker embeddings, and save the result as a Base checkpoint.

Use **CSM’s two-stage teacher forcing** as the architectural reference and **VibeVoice’s public loss and preprocessing interface** as the API reference. Preserve Qwen’s objective: first-codebook loss plus `0.3 × residual-codebook loss`, as used in its [fine-tuning script](https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/sft_12hz.py).

The current CPU probe confirms that ordinary `forward` requires missing generation state. Its embedding-only loss backpropagates through the talker but does not train the text projection or code predictor.

## Forward and Loss

- Implement changes in `modular_qwen3_tts.py`, then regenerate modeling/configuration files with the modular converter.
- Give `forward` independent text and audio inputs: `input_ids` and `attention_mask` shaped `(B, T)`, `audio_codes` shaped `(B, F, K)`, `audio_attention_mask` shaped `(B, F)`, and `speaker_embeddings` shaped `(B, H)`.
- Accept unshifted `labels` shaped `(B, F, K)`, with `-100` marking ignored targets. `labels=None` returns logits without computing training loss, including when the model is in evaluation mode.
- Assemble the non-streaming Base sequence inside the model: role/language-control prefix, speaker conditioning, complete text, codec BOS, ground-truth audio frames, and codec EOS. Derive positions and padding from actual lengths; use configured token IDs and codebook counts.
- Compute text embeddings through the existing text projection and sum the appropriate codebook embeddings at each audio position. Frozen speaker embeddings remain conditioning inputs.
- Expand labels onto the assembled sequence, masking the prefix and padding. Supervise EOS only in the first codebook, for examples containing supervised first-codebook targets. Shift the temporal targets exactly once using the library loss helper.
- For frame `t`, condition the code predictor on the talker state immediately **before** that frame. Teacher-force its sequence with that state followed by codes `0` through `K-2`; predict codes `1` through `K-1`.
- Apply each residual head to its corresponding position using views of the existing concatenated `lm_head.weight`. Preserve all parameter names and shapes, including the optional dimension projection.
- Add `code_predictor_loss_weight=0.3` to the top-level config. Return total `loss`, `talker_loss`, `code_predictor_loss`, primary logits, and standard hidden-state/cache fields. Primary logits follow the assembled sequence; document that alignment.
- Handle fully ignored targets with differentiable zero losses. Reject malformed shapes, invalid code IDs, and empty target audio with clear errors.

## Preprocessing and Generation

- Extend `Qwen3TTSProcessor.__call__` with precomputed `audio_codes`, `speaker_embeddings`, and `output_labels=True`. It formats text, pads text and audio independently, creates masks, and returns labels. Learned embedding computation stays inside the model.
- Use the existing audio tokenizer and speaker feature extraction during offline preprocessing, in evaluation mode without gradients. Store ordinary detached tensors suitable for training.
- Add a documented collator that batches these cached examples through the processor. Keep model-specific prefix token IDs in the model config, avoiding a second configuration copy in the processor.
- Move residual-code sampling and frame collection into `Qwen3TTSGenerationMixin`, following CSM’s generation-loop structure. `forward` must never call `generate`.
- Remove the use of nested `hidden_states` to transport generated codes. Generation collects codes explicitly while forward returns conventional hidden states.
- Keep the main model's input embedding accessors aligned with codec `input_ids`, matching CSM and the
  Qwen3-Omni talker. Text inputs continue to use the separate text embedding accessor and projection.
- Preserve existing generation arguments, streaming and non-streaming behavior, and the generated-code output consumed by `processor.decode`. Existing checkpoint conversion and loading must retain identical parameter mappings.

## Trainer and Documentation

- Use `accepts_loss_kwargs=False`: the two losses have different target counts and are independently averaged. Trainer should average the combined microbatch losses during gradient accumulation rather than apply one shared token denominator.
- Document `label_smoothing_factor=0`, `prediction_loss_only=True`, and `remove_unused_columns=False` for the example’s collator.
- Add a complete example in `qwen3_tts.md`: preprocessing, batching, manual forward/backward/optimizer step, Trainer training, saving, reloading, and generation using the reference-speaker prompt.
- Freeze the speaker encoder explicitly in the training recipe. Train the talker, text embeddings/projection, codec embeddings/head, and code predictor.
- Keep Base checkpoint structure after saving. CustomVoice export, training other variants, audio-tokenizer training, and single-codebook work are outside this change. Initial training uses Auto language conditioning and non-streaming text preparation.

## Verification

- **Alignment and causality:** compare both losses with independently calculated targets; verify first/last frames, EOS, padding, and ignored labels. Changing a target frame must not change the preceding talker state or predictions that precede that code.
- **Gradient coverage:** confirm finite gradients and optimizer updates across both prediction stages and the text projection; confirm frozen speaker/tokenizer parameters remain unchanged.
- **Teacher-forcing equivalence:** compare parallel residual prediction with sequential prediction supplied the same ground-truth history, using the matching head at every step.
- **Batching and Trainer:** test unequal text/audio lengths, one-frame examples, ignored targets, evaluation loss, and accumulation against an equivalent manual microbatch loop.
- **Configurations:** exercise matching and differing talker/predictor dimensions, and multiple codebook counts. Correct tiny test fixtures so special tokens and nested codebook counts are valid.
- **Compatibility:** verify unchanged state-dict keys/shapes, strict converted-checkpoint loading, save/reload, and generation regressions for Base, CustomVoice, and VoiceDesign. Re-enable common forward tests wherever their assumptions now apply.
- Run focused model/processor tests, modular consistency, docstring checks, and repository checks. The current environment is CPU-only; real-checkpoint backward and GPU mixed-precision validation remain required before claiming training readiness.
