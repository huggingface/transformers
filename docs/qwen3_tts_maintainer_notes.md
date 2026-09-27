# Qwen3-TTS Maintainer Notes

## Code predictor output head

The Qwen3-TTS code predictor generates codec groups sequentially, while Higgs predicts all audio codebooks in parallel from one hidden state.

To align the output-head structure with the Higgs-style packed projection, `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` now uses a single linear layer:

```python
nn.Linear(config.hidden_size, (config.num_code_groups - 1) * config.vocab_size, bias=False)
```

During forward, the packed logits are reshaped to:

```python
[..., num_code_groups - 1, vocab_size]
```

and the active `generation_steps` slice is returned:

```python
logits = logits[..., generation_steps, :]
```

This preserves the existing sequential generation contract, where `generate()` receives logits of shape `[..., vocab_size]` for the current codec group, while keeping the projection reusable and structurally closer to other multi-codebook TTS models.

## Modular converter: class naming and base-class resolution in the multi-codebook decoder

This was the longest-running problem in the multi-codebook tokenizer, and it is worth recording because
the failure mode is not obvious from either the modular file or the generated file on its own.

### Background

The modular converter does not import the parent class at runtime. When a class in a `modular_*.py` file
inherits from a class in another model, the converter *inlines the parent's full source* into the generated
`modeling_*.py`, rewriting the parent's model prefix to the current model's prefix. The class in the
generated file is therefore a flattened, standalone copy — not a subclass.

That is the intended design (every generated model file must be readable and runnable in isolation), but it
means two things stop being true that a normal Python developer assumes:

1. The name you write in the modular file is not necessarily the only name that ends up in the output.
2. The base class you write is not necessarily the base class the output ends up with.

Both bit us in the decoder.

### Problem 1 — the name we chose was not the name the converter produced

The multi-codebook decoder reuses several components from the Qwen3-Omni-MoE Code2Wav stack. In the modular
file these were declared with short, model-local names:

```python
class Qwen3TTSTokenizerMultiCodebookAttention(Qwen3OmniMoeCode2WavAttention): ...
class Qwen3TTSTokenizerMultiCodebookMlp(Qwen3OmniMoeCode2WavMlp): ...
class Qwen3TTSTokenizerMultiCodebookDecoderBlock(Qwen3OmniMoeCode2WavDecoderBlock): ...
```

The converter renames an inlined parent by substituting the *model prefix only*, preserving the rest of the
parent's name. `Qwen3OmniMoeCode2WavAttention` therefore becomes
`Qwen3TTSTokenizerMultiCodebookCode2WavAttention` — which is not the name we declared. The generated file
ends up carrying both the name we asked for and the name the converter derived, for each reused component,
plus transitive helpers that were never declared in the modular file at all.

The same shape appears on the encoder side, where the inlined Mimi encoder and our own encoder wrapper
resolve to closely-related names that are easy to confuse when reading the generated file.

The practical lesson: when reusing a component whose upstream name contains a sub-namespace (here,
`Code2Wav`), the local name must be chosen with the converter's prefix-substitution rule in mind, not with
what reads nicest in the modular file. Naming and code generation are coupled.

### Problem 2 — inlining changed the effective base class, which invalidated a call site

Several reused components are plain `nn.Module`s upstream. Because the decoder tree they were being inlined
into is rooted at a `PreTrainedModel`, the flattening step resolved some of them against a `PreTrainedModel`
base rather than `nn.Module`.

That promotion is not cosmetic. `PreTrainedModel.__init__` has the signature `(config, *inputs, **kwargs)`
and validates the config it receives against the class's declared `config_class`. Our nested decoder
submodules are constructed positionally with a *sub-config* (the Code2Wav config), not the top-level
tokenizer config. So a construction that is perfectly valid for an `nn.Module` became an invalid call once
the same class was resolved as a `PreTrainedModel` — the argument was rejected on config-class validation,
and the error surfaced at the call site in the decoder rather than at the class that had actually changed.

This is what made the issue hard to diagnose: the modular file looked correct, the generated file looked
plausible, and the traceback pointed somewhere other than the cause.

### How it was resolved

Three changes, applied together:

1. **A dedicated intermediate base class for the decoder side.**
   `Qwen3TTSTokenizerMultiCodebookCode2WavPreTrainedModel` sits between the top-level pretrained model and
   every decoder-side module, and declares `config_class = Qwen3TTSTokenizerMultiCodebookCode2WavConfig`
   along with its own `_no_split_modules`. This gives the decoder subtree a base whose config contract
   matches the sub-config it is actually constructed with, instead of inheriting a base that expects the
   composite config.

2. **Construct nested pretrained submodules via `_from_config` rather than the constructor.**
   `Qwen3TTSTokenizerMultiCodebookDecoderTransformerModel._from_config(config)` (and likewise for the
   encoder and decoder in the top-level model) goes through the documented factory path, which resolves the
   config against the class properly, instead of relying on positional `__init__` arguments that only
   happen to work for a plain module.

3. **Own the classes where inlining was fighting us.**
   `Qwen3TTSTokenizerMultiCodebookDecoderTransformerModel` is written out explicitly in the modular file
   rather than inherited. Everything around it is still a one-line `pass` subclass, so reuse is preserved
   where it works; the one class whose expansion did not line up is maintained directly. Trading a small
   amount of duplication for a correct, readable generated file was the right call for that single class.

### Takeaway

With a code-generating build step, the source you write and the source that runs are two different artifacts,
and correctness has to be verified against the generated one. Reviewing only the modular file would never
have surfaced either problem. The fix was not a clever workaround but making the config/base-class contract
explicit at every level of the module tree, so that flattening produces something structurally valid rather
than something that merely looks right.

## A silently detuned rotary base, and the test that could not have caught it

This is the most instructive failure in the port, because the bug was invisible at every level
a normal review looks at: the modular file was right, the generated file was right, the weights
loaded, the shapes matched, and the model produced fluent-looking output.

### The symptom

The integration test compared generated codec codes against the original implementation. Row 0 of
the code matrix matched exactly — all sixteen codebooks. Row 1 matched through column 5 and
diverged from column 6 onwards. Row 2 was entirely different.

Two features of that pattern turned out to matter. First, the break was at row 1, not row 0.
Second, within row 1 the first few codes were right and the later ones were wrong.

### Why the obvious explanations were wrong

The initial hypothesis was numerical: the code predictor is known to produce near-tied logits, so
a tiny difference in accumulation order flips a greedy `argmax`, and from there two sequences
diverge completely. That story is self-consistent and it is what the earlier version of the test
encoded — it compared only the first six codes and carried a comment explaining that anything
beyond that was expected to drift.

That explanation was wrong, and believing it had already cost months: it turned a real bug into
an accepted limitation, and it justified weakening the test until the test could no longer fail.

A second hypothesis, that running in float32 rather than bfloat16 would resolve the ties, was
also wrong. It changed which trajectory the model took without changing whether it matched.

### Finding it

What settled it was refusing to reason about the sequence and measuring the pipeline instead.
Both implementations were run side by side with hooks recording, at every generation step, the
inputs each submodule received and the outputs it returned. Because the two live in different
environments, this meant dumping tensors from one and comparing them in the other.

That immediately narrowed the search:

- The talker's hidden states agreed to ~1e-5 — floating point noise. The talker was fine.
- At the first decode step the code predictor received **bit-identical** input yet returned logits
  differing by 4e-2, growing to 3.5 by its fifteenth step.

An error that starts above the noise floor and grows with position is not accumulation noise. It
is a systematically wrong rotary embedding: the rotation angle is proportional to position, so the
discrepancy scales with how far into the sequence you are. That is also why row 0 matched — at
position 0 the rotation is the identity regardless of how it is parameterized.

The cause was one dropped config key. The original stores the code predictor's `rope_theta` as a
standalone field with no accompanying `rope_scaling`, while the conversion script only merged
`rope_theta` *inside* a branch guarded on `rope_scaling` being present. The key was therefore
never copied, and the config class default silently took over — 500000 instead of 1000000. Every
frequency in the code predictor's rotary embedding was wrong by a factor of two in its base.

After the fix, the logit error is flat at ~1e-5 across all positions, and the full code matrix
matches the reference exactly for both single and batched inputs.

### The deeper problem: fixtures that cannot fail

While tracing this, a second issue surfaced. The main model's expected outputs were being produced
by a script that ran *our own implementation* and saved its output as the expected values. Its
docstring argued this was necessary, on the grounds that cross-implementation comparison was
inherently unstable.

A test built that way cannot detect a porting bug. It asserts only that the code is deterministic.
The detuned rotary base sat underneath it for months precisely because nothing was positioned to
notice. This is the failure mode behind the recommendation to write integration tests against the
reference *before* refactoring toward library conventions: the tests are not paperwork to be
completed once the port is finished, they are the instrument that tells you the port is still
correct while you change it.

The fix was to make the reference implementation the only source of expected values, for the main
model and the tokenizer alike, and to delete the self-referential generator so it cannot be picked
up again by mistake.

### Two smaller traps found on the way

**Greedy decoding is not the model's configured mode.** The upstream examples always sample, with a
repetition penalty. Forcing greedy decoding for reproducibility pushed one prompt into a
degenerate loop where it emitted the same code vector until it hit the token cap and never
produced an end-of-sequence token. The resolution was to keep greedy decoding — a fixture must not
depend on RNG — but to pin a short, fixed horizon. Greedy decoding is prefix-deterministic, so a
short window is exactly the prefix of a longer run, and it stays clear of the degenerate tail.

**A stale checkpoint file can shadow a fresh one.** Saving a model that has grown past the shard
threshold writes numbered shards plus an index, and the framework removes previous *shards* — but
an older single-file checkpoint in the same directory does not match that pattern and survives.
Loading then prefers the single file over the index, so a conversion that had actually succeeded
appeared to produce a model with every weight missing. The lesson is procedural: convert into a
clean directory, and treat "every key missing" as a question about which file was read, not only
about the key mapping.

### Takeaway

Two things generalize. First, when output diverges, measure where it diverges rather than
reasoning about why it might; the shape of the error — constant versus growing with position —
identifies the class of bug before you have read any code. Second, a test whose expected values
came from the code under test provides confidence without providing coverage, and that is worse
than having no test at all, because it stops anyone from looking.

## The same symptom, benign this time: telling precision from the rotary bug

The section above ends by trusting a comparison of converted codes against the reference. Run that
comparison in bfloat16 *after* the rotary base was fixed and it still fails — the codes diverge, with the
same visual signature the rope bug had: row 0 agrees, a later row agrees for its first few codebooks and
then breaks. Seeing that again is alarming, because it is the exact pattern that "cost months" when it was
misread as precision. This time the reading really is precision, and the two cases are worth pinning side by
side so the next person does not reopen a closed bug — or, worse, weaken the test a second time to make it
pass.

### What was observed (CustomVoice 0.6B, greedy, identical input)

- **bfloat16**, original implementation vs converted: they diverge almost immediately — step 0's first six
  codebooks agree, then split at codebook 6; only ~13% of tokens match over the common length. But the
  *distributions* are the same: mean 887.6 / 886.0, std 568 / 577, both spanning ~[2, 2047]. Same shape,
  different draw.
- **float32**, original implementation vs converted: **every token identical — 176/176 over 11 steps × 16
  codebooks.** No divergence anywhere.

That float32 result is the whole answer. The detuned rope did *not* go away in float32 — the section above
records this explicitly ("running in float32 rather than bfloat16 ... changed which trajectory the model
took without changing whether it matched"). A systematic error survives a change of precision, because it is
baked into the numbers the model computes. This error did not survive: at float32 the two implementations
compute the identical function, so whatever is left at bfloat16 is rounding, not a wrong constant.

### The confirming test: dtype alone, nothing else

The float32 match rules out a systematic difference *between* the implementations. To prove positively that
bfloat16 can produce this much divergence on its own, take a single model — the converted one — and run it
twice on the same input, same weights, same code, changing only the storage dtype:

- converted @ bfloat16 vs converted @ float32: first five steps identical, then diverge at step 5,
  codebook 7 (1227 vs 1848).

Same weights, same graph, same input; the only variable is float32 vs bfloat16, and the greedy trajectory
still forks. That is the mechanism in isolation. In autoregressive greedy decoding a near-tied `argmax` is
decided by the last bit of a rounded logit; once one code differs the sub-talker conditions on a different
prefix and the rest of the vector is unrelated. The code predictor is *designed* to produce near-tied
logits, so it is unusually exposed to this.

(An incidental detail: the cross-implementation bfloat16 run diverges *earlier*, at step 0, than the
single-model bfloat16-vs-float32 run, at step 5. The two implementations are not bit-identical in bfloat16 —
they order a few operations differently, e.g. the equivalent-but-not-identical interleaved vs non-interleaved
mRoPE layouts, which were checked to be numerically identical for the talker's shared position ids — so the
cross-impl run carries slightly more rounding noise and tips over sooner. Both collapse to the same float32
answer, which is the point.)

### The difference between the two, stated once

Same symptom, opposite cause, and one cheap test separates them:

| | Detuned-rotary bug (above) | Precision divergence (this) |
|---|---|---|
| float32 cross-impl codes | still diverge | **exact match** |
| per-position error | grows with position (rope angle ∝ position) | at the noise floor; only `argmax` ties flip |
| config check | code-predictor `rope_theta` wrong (500000 vs 1000000) | configs identical; rope `inv_freq` bit-for-bit equal |
| fix | copy the missing config key | nothing to fix — expected |

So the discriminator is cheap and decisive: **re-run the comparison in float32.** If it becomes exact, the
divergence was bfloat16 rounding and the port is faithful. If it stays broken — especially if the
per-position error *grows* rather than sitting at ~1e-5 — it is a systematic bug, and the float32 run is
telling you to go measure the pipeline (hooks, per-submodule tensors) rather than the sequence, exactly as
the rotary section did.

### Takeaway

The bfloat16 token stream is not a faithfulness test; it is a coin flip past the first tie. Faithfulness
lives at float32, and the honest statement about a low-precision run is about *distribution* ("same range,
same statistics") not *identity* ("same tokens"). The rotary bug and this precision divergence are the two
readings of one symptom, and confusing them is expensive in both directions: calling a real bug "precision"
hid the rope for months, and calling this precision "a bug" would send someone chasing a config error that
is not there. The float32 cross-implementation match is what tells you which one you are looking at.

## Converter and integration tests

The non-slow unit tests for the main Qwen3-TTS model, processor, multi-codebook tokenizer, and single-codebook tokenizer are in place and passing.

The slow integration tests and `convert_qwen3_tts_to_hf.py` will be updated after the model code is finalized with reviewer feedback. Once the final state dict layout and processor/model contracts are agreed, the converter can be rerun, the converted checkpoint can be pushed to the Hub, and the integration tests can be pointed to that hosted checkpoint instead of a local `qwen3_tts_converted` path.

## The audio tokenizer the processor declares but never receives

The processor is written as though it owns a multi-codebook tokenizer, and at runtime it never
does. `batch_decode` and `save_audio` both dereference `self.audio_tokenizer`, and that attribute
is never set on a processor loaded from a published checkpoint, so the first thing a user does
after `generate` raises `AttributeError`. Every unit test passes anyway.

### Where the idea came from

This is not a design change. It is the unfinished half of a change already agreed with the
reviewer:

- 2026-03-27, on `modular_qwen3_tts.py`: "To simplify / compartmentalize things, we can make the
  QwenTTS Tokenizer(s) their own model(s). Similar to how Mimi is its own model and is used as a
  subconfig/model for Kyutai's STT". This produced the separate tokenizer models.
- 2026-06-11, on `docs/source/en/model_doc/qwen3_tts.md`: "The audio tokenizer should be within the
  processor. See Higgs Audio for reference."
- 2026-06-12, on `processing_qwen3_tts.py:67`: "This will be quite a large architecture change from
  what you have at the moment. We will want the audio tokenizer within the processor."
- 2026-06-14, commit `67cd3db3ee`, "add audio_tokenizer to processor": `audio_tokenizer_class` was
  declared and the decode paths were written against `self.audio_tokenizer`.

The two requests are not in tension. Being its own model and living inside the processor are the
same arrangement Mimi and Higgs use: the tokenizer keeps its own module, repo, doc page and tests,
and the parent's processor holds a *reference* to it. Nothing about the split changes, and no
tokenizer weights are added to the main checkpoint — the main repo gains roughly a hundred bytes
of address:

```json
"audio_tokenizer": {
  "audio_tokenizer_class": "Qwen3TTSTokenizerMultiCodebookModel",
  "audio_tokenizer_name_or_path": "shahvandit/qwen3-tts-tokenizer-multi-codebook-hf"
}
```

which is byte-for-byte the shape `eustlb/higgs-audio-v2-generation-3B-base` publishes.

### Why it never worked

Three things have to be in place for that attribute to exist, and none of them was:

1. `MODEL_FOR_AUDIO_TOKENIZATION_NAMES` in `models/auto/modeling_auto.py` did not list the
   multi-codebook tokenizer, so the library did not recognise it as an audio tokenizer at all.
2. `MODALITY_TO_BASE_CLASS_MAPPING["audio_tokenizer"]` in `processing_utils.py` is a hardcoded
   tuple of `("HiggsAudioV2TokenizerModel", "DacModel")`. Handing the processor anything else
   raises `TypeError` before the very next line, which would have accepted it — the model already
   subclasses `PreTrainedAudioTokenizerBase`. Registration alone is not enough; this was verified
   by adding (1) and watching the constructor still refuse.
3. `convert_qwen3_tts_to_hf.py` never attached the tokenizer before saving the processor, so the
   pointer above was never written and the published checkpoint has no way to find the codec.

Both entry points hit gate (2): direct construction, and `from_pretrained`, which resolves the
pointer and then calls `cls(*args, **valid_kwargs)`.

### Why the tests did not catch it

`test_processing_qwen3_tts.py` assigns `processor.audio_tokenizer = _build_tiny_audio_tokenizer(...)`
directly before exercising `batch_decode` and `save_audio`. Attribute assignment bypasses the
constructor, so it steps over all three gaps at once. The tests cover the decode maths and nothing
about how a user actually obtains a working processor. Worth adding a test that loads the published
checkpoint through `AutoProcessor` and decodes, which is the path that was broken.

### Two traps found while fixing it

Do not pass `audio_tokenizer=` to `Qwen3TTSProcessor.from_pretrained()`. The kwarg is forwarded
down into the sub-tokenizer's `init_kwargs`, and saving then fails with `Object of type
Qwen3TTSTokenizerMultiCodebookModel is not JSON serializable`, thrown from
`tokenization_utils_base.py`, far from the cause. The converter must use the plain constructor.

`feature_extractor_class` on the processor is the deprecated way of naming a sub-processor and
logs a warning on every load; `qwen3_tts` is already registered in `feature_extraction_auto.py`,
so the attribute is redundant. `tokenizer_class` stays — text tokenizers take a different branch
that does not warn.

### Takeaway

A class attribute that names a collaborator is a declaration of intent, not a wiring. Between
declaring `audio_tokenizer_class` and a user getting a waveform there were three separate
registries and one converter, none of which the type checker or the unit tests had any opinion
about. When a model declares that it owns something it loads at runtime, the test that matters is
the one that loads it the way a reader of the docs would.
