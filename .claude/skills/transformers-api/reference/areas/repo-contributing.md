# Repo navigation & contributing (where is X implemented? + PR hygiene)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Decision guide](#decision-guide)
- [Quickstarts](#quickstarts)
  - [1) Locate an implementation (public API → module → file)](#1-locate-an-implementation-public-api--module--file)
  - [2) Set up a dev environment (editable install)](#2-set-up-a-dev-environment-editable-install)
  - [3) Run the smallest relevant tests](#3-run-the-smallest-relevant-tests)
  - [4) Run style/quality checks (make targets)](#4-run-stylequality-checks-make-targets)
  - [5) Run repo consistency checks (make repo-consistency)](#5-run-repo-consistency-checks-make-repo-consistency)
  - [6) Build docs locally (doc-builder)](#6-build-docs-locally-doc-builder)
  - [7) Model contributions (modular approach + checklist)](#7-model-contributions-modular-approach--checklist)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)
- [Repo hotspots](#repo-hotspots)
- [Verify / locate in repo](#verify--locate-in-repo)

---

## Scope

Use this page when the user wants to:
- find “**where is X implemented?**” (exact file/class/function)
- understand the **repo layout** (`src/`, `tests/`, `docs/`, `examples/`)
- make a **small targeted change** and open a PR safely
- add/update **docs**, **tests**, or a **model**

---

## Minimum questions to ask

Ask only what you need (0–5 questions):
1) The **symbol/name** (class/function/arg) OR the **behavior** (what changed / what’s wrong)
2) Is the request “**where is it**” or “**change it**” or “**add it**”?
3) Which backend matters (PyTorch/TF/JAX) and which area (pipelines/generation/trainer/tokenizers/processors)?
4) Do they have a **repro** or failing test? (ideal)
5) Are they changing **public API** or internal behavior only?

---

## Decision guide

### If the question is “Where is X implemented?”
Use this ladder (don’t guess):
1) Confirm the public symbol exists → `reference/generated/public_api.md`
2) Map it to a file path → `reference/generated/module_tree.md`
3) Grep the repo for the symbol / error substring / config key
4) Find the tests that cover it, then adjust minimally

### If the goal is “Change X” (bug fix / behavior change)
1) Reproduce (minimal script) OR write a failing test first
2) Make the smallest code change
3) Run the smallest relevant tests
4) Run `make fixup` and fix remaining issues
5) Open PR with a clear title and minimal diff

### If the goal is “Add X” (new model / new feature)
1) Prefer the modular approach when available (keeps contributions maintainable)
2) Add code + docs + tests together
3) Run repo consistency checks so required registries/indexes don’t get missed
4) Keep the PR as small and focused as possible

---

## Quickstarts

### 1. Locate an implementation (public API → module → file)

Follow this sequence:

1) **Does the symbol exist publicly?**  
   Open: `reference/generated/public_api.md`

2) **Where is it implemented?**  
   Open: `reference/generated/module_tree.md`  
   - Identify the owning module/file under `src/transformers/`
   - Note adjacent files in the same folder (helpers/configs/variants)

3) **Grep keywords**  
   Use 1–3 high-signal search terms:
   - exact symbol name (e.g., `set_attn_implementation`)
   - error substring from traceback
   - config key (e.g., `attn_implementation`, `torch_dtype`)

---

### 2. Set up a dev environment (editable install)

```bash
git clone https://github.com/<your-handle>/transformers.git
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git
git checkout -b my-descriptive-branch
```

Before opening a PR (or if a maintainer asks), rebase your branch on upstream:

```bash
git fetch upstream
git rebase upstream/main
```

Editable install in a virtualenv:

```bash
pip install -e ".[dev]"
```

If that fails (optional deps can be heavy), install PyTorch first, then:

```bash
pip install -e ".[quality]"
```

If Transformers was already installed in that env, uninstall it first:

```bash
pip uninstall transformers
```

---

### 3. Run the smallest relevant tests

Run only what you touched first:

```bash
pytest tests/<TEST_TO_RUN>.py
```

Iterate faster with keyword filtering:

```bash
pytest -k "keyword_here" tests/<TEST_TO_RUN>.py
```

#### Match CI’s test selection (tests_fetcher)

Transformers CI selects tests impacted by your PR diff. You can reproduce that selection locally by running the same helper script CI uses.

```bash
python utils/tests_fetcher.py
```

This creates a `test_list.txt` file with the tests to run; execute them like this: 

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

If you add/modify `@slow` tests, run them explicitly. By default, slow tests are skipped; set `RUN_SLOW=yes` to enable them — note this can download **many gigabytes** of models (disk + bandwidth required). 

```bash
RUN_SLOW=yes python -m pytest -n auto --dist=loadfile -s -v tests/<SLOW_TEST_FILE_OR_DIR>
```

Accepted variant (you’ll also see this form used in some docs/CI contexts):

```bash
RUN_SLOW=1 pytest tests/<SLOW_TEST_FILE_OR_DIR>
```


---

### 4. Run style/quality checks (make targets)

Full formatting:

```bash
make style
```

Quality checks:

```bash
make quality
```

Fast path for PR iteration (targets modified files and also runs repo consistency):

```bash
make fixup
```

---

### 5. Run repo consistency checks (make repo-consistency)

Run:

```bash
make repo-consistency
```

If it fails on copies / generated-content checks, run:

```bash
make fix-copies
```

Then rerun:

```bash
make repo-consistency
```

---

### 6. Build docs locally (hf-doc-builder)

If you modified anything under `docs/source`, make sure the documentation can still be built.

Install the documentation builder:

```bash
pip install hf-doc-builder
```

Run the following command from the root of the repository:

```bash
doc-builder build transformers docs/source/en --build_dir ~/tmp/test-build
```

Inspect the output under `~/tmp/test-build`.

---

### 7. Model contributions (modular approach + checklist)

For vision-language / multimodal models (images/videos), follow the official Transformers contribution checklist. 

#### Required checklist (vision-language / multimodal)

1) **Implement a modular file**
- Prefer the modular architecture pattern: create `modular_<model_name>.py`.
- Use the CLI to scaffold a modular skeleton:
  - `transformers add-new-model-like` 
- Verify the modular file with:
~~~bash
python utils/modular_model_converter.py <model_name>
~~~
This generates the derived files (`modeling_*.py`, `configuration_*.py`, etc.) and CI enforces that they match the modular source. 

2) **Add a fast image processor (for image/video models)**
- If your model processes images, add a fast image processor that inherits from `BaseImageProcessorFast` (torch/torchvision-based) for better performance. 

3) **Create a weight conversion script**
- Add `convert_<model_name>_to_hf.py` to convert original checkpoints to the Hugging Face format (load, map keys, save), including usage examples in the script. 

4) **Add integration tests with exact output matching**
- Add an `IntegrationTest` that runs end-to-end processing + modeling with **exact output matching** (generated text for generative models; logits for non-generative models).
- Use real checkpoints + real inputs (consider 4-bit / half precision if the checkpoint is large for CI). 

5) **Update documentation**
- Add or update `docs/source/en/model_doc/<model_name>.md` with usage examples, model description + paper link, and basic usage with `Pipeline` and `AutoModel`.
- Add the model to the appropriate TOC files. 

6) **Look for reusable patterns**
- Reuse established patterns from similar models (LLaVA, Idefics2, Fuyu, etc.) and avoid reinventing core components. 

Before pushing, run:
~~~bash
make fixup
~~~

---

## Knobs that matter (3–8)

1) Keep PRs **small** (avoid drive-by refactors)
2) Repro-first: failing test or minimal repro before changing logic
3) Run the **smallest relevant tests** first, then expand
4) Always run `make fixup` before pushing
5) For new models/features: run `make repo-consistency`
6) If docs changed: run `doc-builder build ...`
7) If slow tests changed/added: run `RUN_SLOW=1 pytest ...`
8) When changing public API: verify docs + exports + tests

---

## Pitfalls & fixes

- Can’t find where something is defined:
  - confirm in `public_api.md`, then locate via `module_tree.md`, then grep
- CI fails on formatting/lint:
  - run `make fixup`, then rerun failing checks
- Repo consistency fails:
  - run `make repo-consistency`; if it points to copy checks, try `make fix-copies`
- Docs build fails:
  - run `doc-builder build transformers docs/source/ --build_dir ...` and fix missing toctree/refs

---

## Repo hotspots

- Core library: `src/transformers/`
- Models: `src/transformers/models/`
- Pipelines: `src/transformers/pipelines/`
- Generation: `src/transformers/generation/`
- Trainer: `src/transformers/trainer.py` (+ related modules)
- Tests: `tests/` (model tests usually under `tests/models/<model_name>/`)
- Docs: `docs/source/` (English content commonly under `docs/source/en/`)
- Examples: `examples/`

---

## Verify / locate in repo

When uncertain, use Skill verification indexes:
- “Does this symbol/arg exist?” → `reference/generated/public_api.md`
- “Where is it implemented?” → `reference/generated/module_tree.md`
