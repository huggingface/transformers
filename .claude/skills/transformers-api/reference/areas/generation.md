# Generation (decode, sampling, beams, stopping, streaming, chat)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Always-follow workflow](#always-follow-workflow)
- [Quickstarts](#Quickstarts)
  - [A. Decoder-only (CausalLM) minimal generation (greedy)](#a-decoder-only-causallm-minimal-generation-greedy)
  - [B. Encoder-decoder (Seq2Seq) minimal generation (greedy)](#b-encoder-decoder-seq2seq-minimal-generation-greedy)
- [Output length (do this first)](#output-length-do-this-first)
- [Decoding strategies (choose one)](#decoding-strategies-choose-one)
  - [1. Greedy (deterministic baseline)](#1-greedy-deterministic-baseline)
  - [2. Sampling (creative / diverse)](#2-sampling-creative--diverse)
  - [3. Beam search (more exhaustive, more deterministic)](#3-beam-search-more-exhaustive-more-deterministic)
  - [4. Diverse candidates (multiple outputs)](#4-diverse-candidates-multiple-outputs)
- [Chat prompting (chat templates)](#chat-prompting-chat-templates)
  - [Chat template → generate (decoder-only)](#chat-template--generate-decoder-only)
- [“Decoder-only returns the prompt too” (slice it)](#decoder-only-returns-the-prompt-too-slice-it)
- [Stopping](#stopping)
  - [1. EOS-based stopping (default)](#1-eos-based-stopping-default)
  - [2. Stop on custom condition (StoppingCriteria)](#2-stop-on-custom-condition-stoppingcriteria)
  - [3. Stop on strings (built-in: stop_strings)](#3-stop-on-strings-built-in-stop_strings)
- [Streaming](#streaming)
  - [TextIteratorStreamer (common pattern with a background thread)](#textiteratorstreamer-common-pattern-with-a-background-thread)
- [Inspecting generation internals (scores, beams, etc.)](#inspecting-generation-internals-scores-beams-etc)
- [What to change (knobs that matter most)](#what-to-change-knobs-that-matter-most)
- [Pitfalls & fixes (high-frequency)](#pitfalls--fixes-high-frequency)
- [Repo hotspots (when asked “where is this implemented?”)](#repo-hotspots-when-asked-where-is-this-implemented)
- [Verification checklist (anti-hallucination)](#verification-checklist-anti-hallucination)


## Scope

Use this page when the user’s goal is **text generation / chat behavior**:
- `.generate()` decoding strategy (greedy / sampling / beams)
- output length control (`max_new_tokens`, `min_new_tokens`, etc.)
- repetition control (`repetition_penalty`, `no_repeat_ngram_size`)
- stopping (EOS, custom stopping criteria)
- streaming (streamers)
- chat templates + generation together

---

## Minimum questions to ask

Ask only what’s required to produce a runnable snippet:
1) Model type: **decoder-only** (CausalLM) vs **encoder-decoder** (Seq2Seq)
2) Desired behavior: **deterministic** vs **creative**
3) Output constraints: length, stop condition, format (JSON, bullets, etc.)
4) Environment: `transformers` version + backend/device (CPU/CUDA/MPS)
5) If blocked: full traceback + minimal repro

---

## Always-follow workflow

1) Load model + tokenizer from the same checkpoint.
2) Prepare prompt (raw text or chat template).
3) Put tensors + model on the same device.
4) Choose a decoding strategy (greedy / sampling / beam) and set length via `max_new_tokens`.
5) Generate under `torch.inference_mode()` (PyTorch).
6) Decode, and (for decoder-only models) optionally slice off the prompt tokens.

---

## Quickstarts

### A. Decoder-only (CausalLM) minimal generation (greedy)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "distilbert/distilgpt2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

prompt = "Write a one-sentence summary of Transformers:"
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=50)

text = tok.decode(out[0], skip_special_tokens=True)
print(text)
```

### B. Encoder-decoder (Seq2Seq) minimal generation (greedy)
```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "google/flan-t5-small"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.eval()

prompt = "Translate to German: The cat is on the table."
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=50)

print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Output length (do this first)

Prefer `max_new_tokens` over `max_length`.

- `max_new_tokens`: number of tokens **generated beyond** the prompt (recommended)
- `max_length`: prompt length + generated length (often confusing)

Also consider:
- `min_new_tokens` (or `min_length` depending on model/version)
- `early_stopping` (beam search behavior)

---

## Decoding strategies (choose one)

### 1. Greedy (deterministic baseline)
Good for short, factual, structured outputs. Can repeat for long outputs.
```python
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
```

### 2. Sampling (creative / diverse)
Use when you want variation. Typical defaults:
- `do_sample=True`
- `temperature` ~ 0.7–1.0
- `top_p` ~ 0.9–0.95 (nucleus)
- optionally `top_k` ~ 40–100

```python
out = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
)
```

### 3. Beam search (more exhaustive, more deterministic)
Useful for translation/summarization; can become repetitive for open-ended chat.

```python
out = model.generate(
    **inputs,
    max_new_tokens=200,
    num_beams=4,
    do_sample=False,
    early_stopping=True,
)
```

### 4. Diverse candidates (multiple outputs)
```python
out = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    num_return_sequences=3,
)
texts = tok.batch_decode(out, skip_special_tokens=True)
for i, t in enumerate(texts, 1):
    print(f"\n--- candidate {i} ---\n{t}")
```

---

## Chat prompting (chat templates)

If the model expects chat formatting, use `apply_chat_template` (tokenizer or processor).
If you’re unsure whether the model is “chat/instruct”, check its docs/model card or your `reference/generated/*`.

### Chat template → generate (decoder-only)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain beam search in one paragraph."},
]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=180, do_sample=False)

print(tok.decode(out[0], skip_special_tokens=True))
```

---

## “Decoder-only returns the prompt too” (slice it)

For decoder-only LMs, `generate()` returns `[prompt + completion]`.
If you only want the completion tokens:

```python
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=80)

prompt_len = inputs["input_ids"].shape[-1]
completion_ids = out[0, prompt_len:]
completion_text = tok.decode(completion_ids, skip_special_tokens=True)
print(completion_text)
```

(For encoder-decoder models, the generated sequence is usually just the decoder output.)

---

## Stopping

### 1. EOS-based stopping (default)
Most models stop when `eos_token_id` is produced (or hit length limits).
If you see “never stops” behavior, verify:
- `eos_token_id` exists and is correct
- you didn’t set an incompatible `min_length` / `min_new_tokens`

### 2. Stop on custom condition (StoppingCriteria)
Use this when you need “stop when a phrase appears” or other custom termination.

```python
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, stop_ids: list[int]):
        self.stop_ids = torch.tensor(stop_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # Return shape: (batch_size, 1) — True means “stop for that row”
        stop_ids = self.stop_ids.to(input_ids.device)
        bsz, seqlen = input_ids.shape
        n = stop_ids.numel()

        if seqlen < n:
            return torch.zeros((bsz, 1), dtype=torch.bool, device=input_ids.device)

        tail = input_ids[:, -n:]  # (bsz, n)
        matched = (tail == stop_ids).all(dim=1, keepdim=True)  # (bsz, 1)
        return matched


stop_text = "\n###"
stop_ids = tok(stop_text, add_special_tokens=False)["input_ids"]
criteria = StoppingCriteriaList([StopOnTokenSequence(stop_ids)])

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        stopping_criteria=criteria,
    )

print(tok.decode(out[0], skip_special_tokens=True))
```

Notes:
- In batched generation, stopping criteria can return a `(batch_size, 1)` mask (per-sample). 
- However, generation often keeps tensor shapes fixed (e.g., padding finished rows), so you may not get compute savings unless you re-batch unfinished samples. 


### 3. Stop on strings (built-in: stop_strings)

If you want to stop when the model outputs a specific string, you can use `stop_strings`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "distilbert/distilgpt2"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

prompt = "Write a short answer, then end with a line containing ###:\n"
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        stop_strings=["\n###"],
        tokenizer=tok,  # required so stop_strings can be matched against decoded text
    )

text = tok.decode(out[0], skip_special_tokens=True)
print(text)
```

Notes:
- `stop_strings` stops generation *after* the stop string is produced.
- Pass `tokenizer=tok` so Transformers can detect the stop string correctly during generation.
- If you need the returned text *without* the stop string, trim it after decoding (e.g., `text.split("\n###")[0]`).

---

## Streaming

Use streamers when you want token-by-token (or chunk-by-chunk) output.

### TextIteratorStreamer (common pattern with a background thread)
```python
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = "distilbert/distilgpt2"
tok = AutoTokenizer.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.eval()

prompt = "Tell me a short story about a robot:"
inputs = tok(prompt, return_tensors="pt").to(model.device)
streamer = TextIteratorStreamer(tok, skip_special_tokens=True, skip_prompt=True)
generation_kwargs = dict(
    **inputs,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
    streamer=streamer,
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text_chunk in streamer:
    print(text_chunk, end="", flush=True)

thread.join()
print()
```
Pitfall: Some pipelines/deepcopies can conflict with streamer objects; if you hit errors, call `model.generate`
directly (like above) rather than wrapping in a pipeline.
---

## Inspecting generation internals (scores, beams, etc.)

If you need token-level probabilities, request structured outputs from `generate()`:

```python
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

# out.sequences: token ids
# out.scores: tuple of per-step logits (one tensor per generated step)
print(type(out))
print(out.sequences.shape, len(out.scores))
```
---

## What to change (knobs that matter most)

Length / termination:
- `max_new_tokens` (primary)
- `min_new_tokens` / `min_length`
- `eos_token_id`, `pad_token_id`
- `stopping_criteria`

Creativity / diversity:
- `do_sample`
- `temperature`
- `top_p`, `top_k`
- `typical_p` (if supported by your version/model)

Determinism / search:
- `num_beams`
- `early_stopping`
- `length_penalty`

Repetition control:
- `repetition_penalty`
- `no_repeat_ngram_size`
- `encoder_no_repeat_ngram_size` (encoder-decoder)

Multiple outputs:
- `num_return_sequences` (sampling or beams + sampling variants)

---

## Pitfalls & fixes (high-frequency)

### “It ignores temperature/top_p”
Sampling knobs only apply when `do_sample=True`.
Fix: set `do_sample=True` (and typically keep `num_beams=1` for pure sampling).

### “It stops too early / too late”
- Prefer `max_new_tokens` for length.
- Verify `eos_token_id` and that you didn’t set `min_new_tokens` too high.

### “Beam search is repetitive”
Try:
- smaller `num_beams` (e.g., 2–4)
- `repetition_penalty` or `no_repeat_ngram_size`
- or switch to sampling with moderate temperature/top_p.

### “Decoder-only output contains prompt”
Slice using `prompt_len` (see above).

### “Batched generation breaks on padding”
For decoder-only:
- ensure a pad token exists (`tok.pad_token = tok.eos_token` is common)
- consider `tok.padding_side = "left"` for batched generation

### “OOM during generation”
Route to `performance.md` for:
- `device_map="auto"`, dtype reduction, quantization
- smaller `max_new_tokens`, smaller batch size
- attention backend / KV cache strategies

---

## Repo hotspots (when asked “where is this implemented?”)

Generation configuration + defaults:
- `src/transformers/generation/configuration_utils.py`
Streaming:
- `src/transformers/generation/streamers.py`
Logits processors / warpers (repetition penalty, top-k/top-p, etc.):
- `src/transformers/generation/logits_process.py`
Pipelines wrapping generation:
- `src/transformers/pipelines/text_generation.py`
Core generate logic commonly lives under:
- `src/transformers/generation/` (search for `GenerationMixin` and `generate`)

---

## Verification checklist (anti-hallucination)

When uncertain, verify in this order:
1) `reference/generated/public_api.md` (does the symbol/kwarg exist in this version?)
2) `reference/generated/module_tree.md` (where is it implemented?)
3) `reference/generated/docs_map.md` (where is it documented?)
4) Then inspect `src/transformers/generation/...` and grep the exact name (e.g., `stop_strings`, `typical_p`, `TextIteratorStreamer`).