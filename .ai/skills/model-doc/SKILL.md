---
name: model-doc
description: Completes scaffolded model documentation pages at docs/source/en/model_doc/<model>.md by resolving the `<UPPERCASE_SNAKE>` sentinel tokens emitted by the `transformers add-new-model-like` CLI. Resolves paper links, Hub checkpoints, intro summaries, Quickstart inputs, and autodoc blocks against the model's source code under `src/transformers/models/<model>/`. Trigger when a contributor asks to "finish the docs for this new model", "fill in the model_doc template", or "complete the scaffolded docs for <model>".
---

# model-doc skill

Resolve `<UPPERCASE_SNAKE>` sentinel tokens in a scaffolded `docs/source/en/model_doc/<model>.md` file. The source of truth is the codebase under `src/transformers/models/<model>/` (modular file, modeling file, config, tokenizer/processor).

Use the [hf-papers](https://github.com/huggingface/skills/blob/main/skills/hf-cli/SKILL.md#hf-papers--interact-with-papers-on-the-hub), [hf-models](https://github.com/huggingface/skills/blob/main/skills/hf-cli/SKILL.md#hf-models--interact-with-models-on-the-hub), and [hf-collections](https://github.com/huggingface/skills/blob/main/skills/hf-cli/SKILL.md#hf-collections--interact-with-collections-on-the-hub) skills for Hub lookups, with web search as a fallback when those skills are unavailable or return no match.

## Output contract

Write the completed file and return nothing else — no summary, no resolver report, no warnings. Sentinels you cannot resolve stay in the file as `<UPPERCASE_SNAKE>` tokens. A human reviewer will fill them in.

## Scope rule

The model_doc covers only what is unique to the Transformers integration of a model. Do not duplicate other sources of truth:

| Info type | Lives in |
|---|---|
| Benchmarks, training data, license, intended use | Hub model card |
| Architecture details, ablations, paper abstract | Paper |
| Quantization backends, generation strategies, attention impls, cache classes | Central Transformers docs |
| Transformers API surface, gotchas specific to this model's integration, one runnable example | model_doc |

## Sentinel vocabulary

| Token | Filled with | Primary resolver |
|---|---|---|
| `<PAPER_URL>` | Hub papers page | `hf-papers` |
| `<INTRO_SENTENCES>` | 2–4 sentence factual summary | `hf-papers` abstract |
| `<ORG>` | Hub org or user name | `hf-models` |
| `<HUB_CHECKPOINT_URL>` | Collection, org page, or single-checkpoint URL | `hf-collections` + `hf-models` |
| `<CHECKPOINT_ID>` | Canonical checkpoint ID for the Quickstart | `hf-models` |
| `<TASK>` | Pipeline task string, e.g. `"text-generation"` | inferred from the public class list |
| `<AUTOMODEL_CLASS>` | `AutoModelForCausalLM`, etc. | inferred from the public class list |
| `<AUTO_CLASS>` | `AutoTokenizer`, `AutoProcessor`, `AutoImageProcessor`, or `AutoFeatureExtractor` | inferred from the processor class in `__all__`; replaces the hardcoded `AutoTokenizer` the scaffolder emits |
| `<MINIMAL_INPUT>` | Modality-appropriate input literal | modality table below |
| `<OUTPUT_EXPRESSION>` | Concrete value to `print()` | modality table below |

## Resolver rules

### `<PAPER_URL>` and `<INTRO_SENTENCES>`

Try `hf-papers` first, then a web search for an official technical report or release blog post; leave both sentinels in place if neither finds a source.

When a source is found, write a 2–4 sentence intro under these constraints.

- Name the most-distinguishing architectural choice the source actually states
- No benchmark claims. "State-of-the-art," "outperforms," "best," "leading" are banned — they rot fast
- No adjectives the source did not use. If the abstract does not call it "novel," neither do you
- No parameter count or training-token count unless the checkpoint ID already implies them
- Present tense, declarative, active voice. No marketing

The first sentence starts: `[ModelName](<PAPER_URL>) is`

### `<ORG>` and `<HUB_CHECKPOINT_URL>`

Use `hf-collections` to check whether a curated collection exists. Pick one variant:

- Collection exists: `You can find all the [ModelName] checkpoints under the [Org](https://hf.co/collections/...) collection.`
- No collection, multiple checkpoints under a verified org: `You can find all the [ModelName] checkpoints on the [Org](https://hf.co/Org) page.`
- No collection, single checkpoint or user-account upload: `The [ModelName] checkpoint is [org/model](https://hf.co/org/model).`

### `<CHECKPOINT_ID>`

Pick the smallest official checkpoint that produces a convincing Quickstart output. For generative modalities, bump up if the smallest is below the quality threshold:

- LLMs: prefer ≥ 1B parameters
- ASR: skip `tiny`, start at `base` or equivalent
- Image classification, embeddings, non-generative heads: smallest is fine

### `<TASK>` and `<AUTOMODEL_CLASS>`

Pick one representative task head from the model's `__all__`. Common mappings:

- `<Model>ForCausalLM` → `AutoModelForCausalLM` + `"text-generation"`
- `<Model>ForConditionalGeneration` (text-only) → `AutoModelForSeq2SeqLM` + `"text2text-generation"`
- `<Model>ForSequenceClassification` → `AutoModelForSequenceClassification` + `"text-classification"`
- `<Model>ForImageClassification` → `AutoModelForImageClassification` + `"image-classification"`
- `<Model>ForCTC` / `<Model>ForSpeechSeq2Seq` → corresponding Auto class + `"automatic-speech-recognition"`
- Multimodal `<Model>ForConditionalGeneration` → `AutoModelForImageTextToText` + `"image-text-to-text"`

If the model is not registered in any `src/transformers/models/auto/*_mapping.py`, drop the Pipeline tab — emit only the AutoModel block, no `<hfoptions>` wrapper.

### `<AUTO_CLASS>` and variable name

| Processor suffix in `__all__` | `<AUTO_CLASS>` | Variable name |
|---|---|---|
| `Tokenizer` / `TokenizerFast` | `AutoTokenizer` | `tokenizer` |
| `ImageProcessor` / `ImageProcessorFast` | `AutoImageProcessor` | `processor` |
| `FeatureExtractor` | `AutoFeatureExtractor` | `processor` |
| `Processor` (multimodal) | `AutoProcessor` | `processor` |

When `<AUTO_CLASS>` ≠ `AutoTokenizer`, also rename the `tokenizer` variable to `processor` everywhere in the Quickstart code block (import, instantiation, input prep, decode call).

### Call style: generative vs non-generative

Inspect the Quickstart task head (the single head class chosen for `<AUTOMODEL_CLASS>`):

- Generative (`ForCausalLM`, `ForConditionalGeneration`, `ForSeq2SeqLM`, `ForSpeechSeq2Seq`): replace `model(inputs)` with `model.generate(inputs, max_new_tokens=32)`.
- Non-generative (all other heads): keep `model(inputs)`.

### Badge row

Emit a badge row after the release date line and before the `# ModelName` H1. Determine which badges to include by grepping the model's modeling file:

| Badge | Detection |
|---|---|
| FlashAttention | `_supports_flash_attn_2 = True` on any model class |
| SDPA | `_supports_sdpa = True` on any model class |
| Tensor parallelism | `_tp_plan` dict defined on any model class |

```bash
grep -n "_supports_flash_attn_2\|_supports_sdpa\|_tp_plan" src/transformers/models/<model>/modeling_*.py
```

Emit only the badges that match:

```html
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>
```

### `<MINIMAL_INPUT>` and `<OUTPUT_EXPRESSION>`

Pick the row that matches the model's processor + task head. Emit exactly these values.

| Signal | `<MINIMAL_INPUT>` | `<OUTPUT_EXPRESSION>` |
|---|---|---|
| Tokenizer + `ForCausalLM` | `"Plants create energy through a process known as"` | `tokenizer.decode(outputs[0], skip_special_tokens=True)` |
| Tokenizer + `ForSeq2SeqLM` / `ForConditionalGeneration` | `"translate English to French: Hello, how are you?"` (or task-appropriate) | `tokenizer.decode(outputs[0], skip_special_tokens=True)` |
| Tokenizer + `ForSequenceClassification` | `"I love this movie!"` | `model.config.id2label[outputs.logits.argmax(-1).item()]` |
| Tokenizer + `ForTokenClassification` / `ForQuestionAnswering` | task-appropriate literal | `outputs.logits.shape` |
| ImageProcessor + `ForImageClassification` | `Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", stream=True).raw)` | `model.config.id2label[outputs.logits.argmax(-1).item()]` |
| ImageProcessor + other vision heads | same image load | `outputs.last_hidden_state.shape` or task-appropriate |
| FeatureExtractor + `ForCTC` / `ForSpeechSeq2Seq` | `load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")[0]["audio"]` | `processor.batch_decode(outputs.logits.argmax(-1))[0]` for CTC, `processor.batch_decode(outputs, skip_special_tokens=True)[0]` for seq2seq |
| Processor (multimodal image+text) | `[{"role": "user", "content": [{"type": "image", "url": "..."}, {"type": "text", "text": "What is in this image?"}]}]` with `processor.apply_chat_template(..., add_generation_prompt=True, tokenize=True, return_tensors="pt")` | `processor.decode(outputs[0], skip_special_tokens=True)` |

If the combination matches no row, leave both sentinels in place.

## Code example conventions

Apply to every code block in the doc — Quickstart, `## Usage examples`, and `## Quantization`.

- Never set `dtype` or `torch_dtype`. `from_pretrained` and `pipeline` set it automatically. Strip it out
- Always pass `device_map="auto"` to model `from_pretrained` calls. This is what makes the code work unchanged on single GPU, multi-GPU, MPS, and CPU. In any block, not just the Quickstart
- Always chain `.to(model.device)` on prepared inputs:
  ```py
  inputs = tokenizer("...", return_tensors="pt").to(model.device)
  outputs = model(inputs)
  ```
  This works on single GPU, multi-GPU, MPS, or CPU.

The Pipeline tab uses `device=0` (the `pipeline()` API convention, not `device_map`). The dtype rule still applies — no `torch_dtype` on `pipeline()` either.

## Prose conventions

- Checkpoint IDs such as `Qwen/Qwen3.5-35B-A3B` are not code — write them as plain text or Hub links, never in backticks. Reserve backticks for code: class names, method names, kwarg names, and string literals that appear in code
- Reference Transformers classes and methods using doc-builder link syntax: [`~AutoProcessor.from_pretrained`], [`~pipeline`]. The `~` drops the module prefix so only the leaf name renders as the link text

## Required sections (in this order)

1. License + MDX header — scaffolder-emitted; do not edit
2. Release date metadata line — `*This model was released on YYYY-MM-DD and added to Hugging Face Transformers on YYYY-MM-DD.*`, placed above the H1. Tooling-managed — never write or edit the dates yourself
3. Badge row — see Badge row resolver rule above
4. `# ModelName` — H1
5. Intro paragraph — resolves `<INTRO_SENTENCES>`. Links the paper inline. Does not quote the abstract
6. Checkpoints line — one of the three variants from the `<ORG>` / `<HUB_CHECKPOINT_URL>` rule
7. Quickstart — two tabs using doc-builder MDX syntax, or a single AutoModel block when no pipeline task mapping exists:
   ```
   <hfoptions id="usage">
   <hfoption id="Pipeline">…</hfoption>
   <hfoption id="AutoModel">…</hfoption>
   </hfoptions>
   ```
8. `[[autodoc]]` blocks — one `## <ClassName>` per public class, in the fixed order below

## Optional sections

Place between Quickstart and autodoc, in this order. Include only when the model genuinely needs them.

- `## Usage tips and notes` — bullets for non-obvious transformers-specific gotchas. Each bullet must name a specific kwarg, class, or behavior. Examples:
  - "Inputs should be padded on the right because the model uses absolute position embeddings"
  - "The tokenizer does not add a BOS token by default — pass `add_special_tokens=True` when batching"
  - "Static cache is not supported; use `DynamicCache` for generation"
  - "Audio inputs must be mono and resampled to 16 kHz"

- `## Usage examples` — only when the Quickstart cannot cover the case (multimodal batching, long-form audio windowing, fine-tuning recipes with non-obvious collators).

## Autodoc coverage

Public class = the union of every `__all__` list across Python files in `src/transformers/models/<model>/`, excluding any class whose name ends in `Output`.

```bash
grep -h "^__all__" src/transformers/models/<model>/*.py
```

Order, top to bottom:

1. Config class
2. Tokenizer (fast variant before slow) / Processor / ImageProcessor / FeatureExtractor
3. Base `<ModelName>Model`
4. `<ModelName>PreTrainedModel` — only if present in `__all__`; many models do not export it
5. Task heads in their `__all__` order

Each autodoc block is formatted as:
```
## ClassName

[[autodoc]] ClassName
    - forward
```

`- forward` vs `- generate`: every block gets `- forward`, except the single Quickstart task-head class — if it is generative (`ForCausalLM`, `ForConditionalGeneration`, `ForSeq2SeqLM`, `ForSpeechSeq2Seq`), it gets `- generate` instead. Exactly one `- generate` per doc.

## Workflow

Ground every resolution in `__all__` and the model's source files — do not rely on paper claims or Hub card descriptions without verifying in code. Apply the resolver rules above for each sentinel; leave unresolvable sentinels in place. Run the quality checklist before writing. Write the completed file and nothing else; never touch the release date line.

## Quality checklist

- [ ] Release date line is present above the H1, unedited
- [ ] Intro is 2–4 sentences, links the paper inline, makes no benchmark claims, uses no invented adjectives
- [ ] Checkpoints line matches one of the three variants and drops "official" when not a verified org
- [ ] Quickstart has Pipeline + AutoModel tabs in `<hfoptions>`, or a single AutoModel block when no pipeline task exists
- [ ] AutoModel tab uses exactly one task-head class and prints a concrete output (not `outputs`)
- [ ] AutoModel tab imports `<AUTO_CLASS>` (not hardcoded `AutoTokenizer`) and uses the correct variable name (`tokenizer` or `processor`) per the `<AUTO_CLASS>` rule
- [ ] AutoModel tab uses `model.generate(inputs, max_new_tokens=32)` for generative task heads and `model(inputs)` for non-generative
- [ ] Every code block follows the conventions: no `torch_dtype` / `dtype`; `device_map="auto"` on every model `from_pretrained`; `.to(model.device)` on every prepared input
- [ ] `## Usage tips and notes` is absent or every bullet names a specific transformers API
- [ ] Badge row is present; FlashAttention/SDPA/Tensor parallelism badges match what `_supports_flash_attn_2`, `_supports_sdpa`, and `_tp_plan` indicate in the modeling file
- [ ] No abstract quote, contributor line, "click sidebar" tip, `## Overview` heading, or long-form checkpoints paragraph
- [ ] Checkpoint IDs in prose are plain text or Hub links, not backtick-wrapped
- [ ] Transformers classes and methods in prose use doc-builder link syntax: [`~ClassName.method`]
- [ ] Autodoc blocks cover the full `__all__` union (minus `*Output` classes) in the fixed order
- [ ] Exactly one autodoc block uses `- generate` — the Quickstart task head if generative
