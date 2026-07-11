# `arch_svg` — architecture diagrams generated from the `transformers` source

Introspect any model in `transformers` and emit its architecture as an **SVG**, in the
spirit of Sebastian Raschka's [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
— but generated automatically from the library's own source instead of drawn by hand.

Output is a **static, self-contained SVG** (CSS only, no JS, no external fetch) — meant to be
embeddable directly in a model's README on the Hub. Modular models get **two** views
(`--mode both`, the default for `all`); standalone models get only the full view.

- **full** — a detailed, Sebastian-Raschka-style diagram read **top-to-bottom like the code**:
  `input_ids [1, 12]` → token embedding → the **expanded decoder block** (pre-norm →
  **RoPE** → self-attention with Q/K/V/O projections at real meta-tensor dims, GQA/MQA/MLA,
  QK-norm → ⊕ residual → pre-norm → MLP gate/up/down **or** Sparse MoE router→top-k→experts
  (+shared)) → final norm → LM head → softmax → `logits`, with residual skip arrows and
  tensor shapes at each stage. A left **layer-schedule strip** shows the per-layer
  `config.layer_types` pattern (e.g. sliding/full, linear/full), and per-distinct-type
  **attention-mask pattern grids** are drawn on the right. The facts panel (far right) shows
  the example checkpoint id, model_type, dims, heads/kv, experts, etc.
- **diff** — *only what this model changes* vs. the parent(s) it inherits from in its
  `modular_<name>.py`. A clean modular model produces a tiny, legible diff; a bloated one
  produces a large diff. **The diff size is a direct, automatic measure of how modular each
  model actually is.** A "changes by class" panel lists every overridden / added / deleted
  member.

> Compared to interactive viewers like hfviewer.com, this is deliberately **static**: one
> deterministic `.svg` per model, safe to commit and embed as a model-card fingerprint.

The headline feature is **exploiting modular `transformers`**: many models are defined as a
`modular_<name>.py` that inherits from another model (`class GemmaModel(LlamaModel)`), and
the diff view shows exactly the override/add/delete payload of that inheritance.

No network, no weights. Models are built on `torch.device("meta")` from their **default
config class** (`Config()` — no `config.json` download required), so the whole zoo renders
in seconds at zero GPU/memory cost. Anything that can't be built on meta falls back to
config-only parsing; anything that can't be parsed at all is recorded in `report.json`
rather than crashing the batch.

## Usage

```bash
python -m arch_svg one  --model llama  --mode full --out out/   # detailed Raschka-style block
python -m arch_svg one  --model gemma  --mode diff --out out/   # Gemma's diff vs Llama
python -m arch_svg all  --mode both    --out out/ --jobs 10      # whole library, BOTH views + index.html
python -m arch_svg all  --mode both    --out out/ --limit 30     # quick subset
python -m arch_svg list                                          # list discovered models
```

`all` (mode `both`) writes `out/<model>.svg` (full) **and** `out/<model>.diff.svg` (diff) per
model, an `out/index.html` contact sheet (each card shows both thumbnails, with a live filter
box and a "largest diffs" leaderboard), and `out/report.json` with per-model status
(`ok` / `built-from-config-only` / `failed` + traceback), the resolved parent, and the count
of overridden / added / deleted / new nodes.

Open `out/index.html` in a browser — SVGs use CSS variables and honor `prefers-color-scheme`,
so light/dark is a one-line swap. (Note: `librsvg`/Quick Look don't support CSS custom
properties, so a *browser* is the intended viewer; opening a single `.svg` in some image
apps may show it un-themed.)

## How it works

```
arch_svg/
  discover.py    # enumerate models/* (import-free; stat the tree + the config mapping)
  modular.py     # ← the interesting part: libcst diff of modular_*.py vs its parent(s)
  introspect.py  # default-config + meta-device build → normalized ArchModel (+ detectors)
  layout.py      # ArchModel/ArchDiff → positioned boxes (standardized vocabulary)
  render.py      # boxes → SVG string (themeable <style>, deterministic output)
  gallery.py     # batch driver, index.html, report.json (keeps going on failure)
  cli.py         # python -m arch_svg
```

`modular.py` does **not** re-run `utils/modular_model_converter.py` (that exists to
*generate* standalone modeling files). Instead it mirrors the converter's inheritance
semantics with `libcst` to recover the diff payload directly:

- a method/attr present in **both** the modular class and its named parent → **overridden**
- present **only** in the modular class → **added**
- a deletion sentinel (`attr = AttributeError(...)` or `raise AttributeError`) → **deleted**

The parent model is read from the modular file's `from ..<model>.modeling_<model> import …`
imports — the same import-following the converter performs.

### Standardized visual vocabulary

Following the "**standardize, don't abstract**" tenet, every component kind maps to a fixed
shape/color, so the same block looks identical across all 450+ diagrams:

| color | component | | color | component |
|---|---|---|---|---|
| 🔵 blue | token embedding | | 🟠 orange | MoE / experts |
| 🟦 cyan | attention | | 🟣 purple | MLP / dense FFN |
| 🟢 green | Mamba / SSM | | ⚪ grey | norm (RMS/Layer) |
| 🔴 red | LM head | | 🟡 yellow | RoPE |

In **diff mode** the parent architecture is drawn ghosted, and only changed regions are
highlighted: **green = added, amber = overridden, red = deleted**. A "changes by class"
panel enumerates every modular class and exactly which methods/attrs it touches.

## What diff mode revealed about modularity

Run over the full library (454 models with modeling code): **249 modular, 205 standalone**;
374 built on meta, 72 config-only, 8 unrenderable.

**Llama is the spine of the library.** 46 models inherit (transitively, via their dominant
parent) from `llama` — then `vit` (10), `llava` (9), `mixtral` (9), `clip` (7),
`wav2vec2` (7). A handful of base models account for most of the zoo.

**The cleanest modular models are nearly free.** Diff "size" =
`overridden + added + deleted + 3·new_classes`. Median modular diff is **39**; the
exemplars are tiny:

| model | parent | diff size |
|---|---|---|
| `glm` | llama | 2 |
| `qwen3` | qwen2 | 4 |
| `qwen2` | llama | 5 |
| `granite` | llama | 7 |
| `ijepa` | vit | 7 |

9 models are **pure-override** (only re-tune existing methods, add *nothing* and create *no*
new classes) — the gold standard for modular: `bitnet`, `camembert`, `ernie4_5`, `glm`,
`granite`, `ijepa`, `qwen2`, `qwen3`, `xlm_roberta`.

**The largest diffs flag where modularity breaks down (refactor candidates).** 30 models
have diff size > 100:

| model | parent | size | reading |
|---|---|---|---|
| `qwen2_5_omni` | qwen2_5_vl | 429 | omni model bolts on audio+talker towers — mostly *new* classes |
| `qwen3_omni_moe` | qwen3_moe | 310 | same pattern |
| `gemma3n` | gemma3 | 213 | adds a full audio encoder + per-layer laurel/altup |
| `rt_detr` | detr | 197 | **0 overridden, 137 added** — inherits the *name* but rebuilds the model |
| `maskformer` | detr | 143 | **0 overridden, 86 added** — same: near-zero reuse of the parent |

Two distinct shapes of "large diff" show up, and the diagram tells them apart at a glance:

1. **Legitimately large** — multimodal/omni models (`qwen2_5_omni`, `phi4_multimodal`,
   `florence2`) that genuinely add vision/audio sub-towers. Big, but the additions are real
   new architecture.
2. **Suspicious** — models with **0 overridden + many added** (`rt_detr`, `maskformer`).
   They declare a parent but override none of its behaviour; the inheritance buys almost
   nothing and is closer to copy-paste-with-a-base-class. These are the clearest candidates
   to either lean harder on the parent or stop pretending to inherit from it.

### A note on philosophy (and why there are 8 failures + 72 config-only)

This tool deliberately contains **no per-model escapes or special-cases**. The resolvers
(model-type normalization, class-name→region mapping, sequence-valued config formatting) are
all generic. When a model *doesn't* render cleanly, that is treated as a **finding, not a bug
to paper over**:

- the **8 failures** are all composite meta-models (`encoder_decoder`, `rag`, `musicgen`,
  `vision_encoder_decoder`, …) that can't be instantiated from defaults;
- the **72 config-only** models are ones whose default config can't build on meta;
- a **huge diff** means the modeling code deviates from the standard.

The right response to a model that doesn't fit the standard vocabulary is to **standardize the
model**, not to add a workaround here. In that sense this gallery doubles as a linter for
modularity and standardization across the library.

## Regenerating the gallery

```bash
python -m arch_svg all --mode diff --out arch_svg/out      --jobs 10
python -m arch_svg all --mode full --out arch_svg/out_full --jobs 10
```

Output is deterministic (stable ordering, integer coordinates) so the SVGs are diff-able in
git.
