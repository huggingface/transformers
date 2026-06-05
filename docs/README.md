<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Writing docs

The Transformers docs live under `docs/source/<lang>/` and are built with [doc-builder](https://github.com/huggingface/doc-builder).

> [!TIP]
> You usually don't need to build the docs locally. When you open a PR that touches files under `docs/`, a bot builds a preview and posts the link as a comment on the PR.

Build locally for a faster iteration loop, or while drafting changes before you open a PR.This is especially useful for large documentation updates.

## Building and previewing locally

Install the quality dependencies and [doc-builder](https://github.com/huggingface/doc-builder) from the root of the repository. The `[quality]` extra is enough for docs-only work. Use `[dev]` if your change also needs the full development dependency set.

```bash
pip install -e ".[quality]"
pip install git+https://github.com/huggingface/doc-builder
```

To build the Markdown files into a temporary folder you can inspect in any Markdown editor:

```bash
doc-builder build transformers docs/source/en/ --build_dir ~/tmp/test-build
```

To live-preview the docs in a browser at [http://localhost:5173](http://localhost:5173), install [watchdog](https://github.com/gorakhargosh/watchdog) and run `preview`:

```bash
pip install watchdog
doc-builder preview transformers docs/source/en/
```

> [!WARNING]
> `preview` only picks up files that exist when it starts. After you add a brand-new page, update `_toctree.yml` and restart `preview`.

Don't commit the built output. Only changes under `docs/source/` are reviewed.

## Adding a new file to the docs

Pages live in `docs/source/<lang>/` as Markdown (`.md`) files. The sidebar navigation lives in [_toctree.yml](https://github.com/huggingface/transformers/blob/main/docs/source/en/_toctree.yml), and a new page only shows up in the sidebar after you add it there.

1. Create the Markdown file under `docs/source/en/` (or the appropriate language directory). Match the file naming and license header of an existing page. Copying a similar page is the fastest way to start.
2. Add an entry to `docs/source/en/_toctree.yml` that points to the filename without the `.md` extension.

Each entry has two fields:

- `local`: the file path relative to `docs/source/<lang>/`, without the extension.
- `title`: the human-readable label that appears in the sidebar.

For a page nested inside a subsection, add it to the inner `sections` list:

```yaml
- isExpanded: false
  sections:
  - local: contributing
    title: Contribute to Transformers
  - local: my_new_contributor_guide
    title: My new contributor guide
  title: Contribute
```

## doc-builder syntax

doc-builder accepts standard Markdown plus a few extensions you'll see across our docs. For the full set of supported syntax (inference snippets, file-include blocks, redirects, multilingual builds, etc.), see the [doc-builder README](https://github.com/huggingface/doc-builder#writing-documentation-for-hugging-face-libraries).

### Tips and warnings

Use GitHub-style blockquote callouts for notes, tips, and warnings:

```md
> [!TIP]
> Use `device_map="auto"` to let Transformers place model shards across available devices.

> [!WARNING]
> `from_pretrained` downloads the full checkpoint on first use. Set `cache_dir` to control where it lands.
```

Older pages may use the legacy `<Tip>` component. Prefer blockquotes for new content.

### Internal links to classes and functions

Wrap a class, function, or method name in square brackets and backticks to link to its doc page. doc-builder resolves the link automatically:

```md
Use [`AutoModel`] to load a model from a checkpoint, then call [`~PreTrainedModel.from_pretrained`].
```

A few variations:

- Prefix the name with `~` to render only the last component (`from_pretrained`) instead of the full path (`PreTrainedModel.from_pretrained`).
- For objects nested in a submodule, include the path inside the backticks, like `utils.ModelOutput`.
- The same syntax links to objects in other Hugging Face libraries, e.g. `accelerate.Accelerator`.

### Tabbed options

To show alternative snippets (CLI vs. Python, different backends, etc.) as tabs, use `<hfoptions>`:

````md
<hfoptions id="install">
<hfoption id="pip">

```bash
pip install transformers
```

</hfoption>
<hfoption id="uv">

```bash
uv pip install transformers
```

</hfoption>
</hfoptions>

````

### Auto-generated API reference

> [!IMPORTANT]
> Always leave a blank line after `[[autodoc]]` so the CI checks pass.

Use `[[autodoc]]` to render the docstring of a class or function. The marker pulls the description, arguments, and (for classes) every public method:

```md
## AutoModel

[[autodoc]] AutoModel
```

To restrict the output to specific methods, list them as a bulleted sub-list:

```md
[[autodoc]] BertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
```

To pull in a method that isn't documented by default (for example, `__call__`), start the list with `all` and then add the extras:

```md
[[autodoc]] BertTokenizer
    - all
    - __call__
```

### Testable code blocks

Tag Python fences with `runnable` (optionally followed by `:<label>`) to mark them as testable examples. doc-builder strips the annotation from the rendered output:

````md
```py runnable:quickstart
from transformers import pipeline

pipe = pipeline("sentiment-analysis")
print(pipe("I love this!"))
```
````

## Writing docstrings

Most docstrings are automatically generated in the library source code, especially `modeling_*.py`, `configuration_*.py`, processing, and tokenizer files. For model classes and forward methods, use the [@auto_docstring](./source/en/auto_docstring.md) decorator when it applies. It keeps shared arguments and returns consistent docstrings without repeating full `Args:` blocks in every model file.

Use hand-written docstrings, following the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), for objects that are not covered by `@auto_docstring`, or when a method needs model-specific behavior documented.

### Styling

Run `make style` to format docstrings and code examples with [Ruff](https://docs.astral.sh/ruff/).

The script can fail on syntax errors. Commit before running `make style` so you can revert if something goes wrong.

## Adding an image

Don't commit images, videos, or other binary assets to the repository because they bloat the repository. Host them on the Hub instead and reference them by URL. The standard home for documentation media is the [huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images) dataset. For external PRs, attach the images to the PR and ask a Hugging Face maintainer to migrate them to the dataset.
