<!--Copyright 2026 IBM and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-16.*

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# GraniteDocling

[GraniteDocling](https://huggingface.co/papers/2408.09869) is a vision-language model for document conversion. Given a page image, it generates [DocLang](https://doclang.ai/) (`<doclang>`): the layout elements in reading order with their bounding boxes, the text, tables, formulas and code they contain. DocLang is the markup; [Docling](https://github.com/docling-project/docling) (`docling-core`) parses a prediction into a document you can export to Markdown, HTML, JSON, and other formats.

The page is split into tiles of 512x512 pixels laid out on the grid that best matches its aspect ratio, plus a thumbnail of the whole page when there is more than one tile. A vision encoder embeds every tile, and a pixel-shuffle connector maps vision patches to image tokens. Intermediate vision encoder states are projected as well and added to the image tokens after the first decoder layers. A dense Granite-style text decoder then generates the DocLang.

GraniteDocling is released as two models. The 500M model uses 64 image tokens per tile and can switch a dense page to 256; a router chooses that per page. The 2B model uses a pixel-shuffle ratio of 2, which is 256 image tokens per tile, the same density as the 500M fine path.

Both models are published under the [docling-project](https://huggingface.co/docling-project) organization.

GraniteDocling is a different architecture from [granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M), which keeps loading through [Idefics3](./idefics3).

> [!TIP]
> This model was contributed by the [Docling team](https://huggingface.co/docling-project).
>
> Click on the GraniteDocling models in the right sidebar for more examples of how to apply GraniteDocling to different document conversion tasks.

## Usage tips

- Prompt the model through the chat template. `<doclang>` converts the full page. Append a bracket list to request a subset of elements, for example `<doclang> [<ocr>]` or `<doclang> [<ocr>, <layout>, <picture>]`.
- Decode with `skip_special_tokens=False` if you will parse the string as DocLang. Dedicated markup tokens are otherwise stripped.
- On the 500M model, pass `fine_route=True` for dense pages (small print, long tables, multi-column text). That path uses four times as many image tokens, so the prompt is longer and inference is slower. The 2B model already uses that token density and has no second path, so `fine_route=True` raises an error for it.
- The 500M model includes a density router, which decides that for you: [`~GraniteDoclingModel.predict_fine_route`] returns which pages of a batch need the fine path.

The examples below convert page 1 of the [Docling Technical Report](https://arxiv.org/pdf/2408.09869) (`arXiv:2408.09869`). Change only the text prompt to switch task; the model returns DocLang filtered to the requested elements.

| Task | Prompt |
|---|---|
| Full page | `<doclang>` |
| OCR | `<doclang> [<ocr>]` |
| Layout | `<doclang> [<layout>]` |
| Tables | `<doclang> [<table>]` |
| Formulas | `<doclang> [<formula>]` |
| Code | `<doclang> [<code>]` |
| Charts | `<doclang> [<chart>]` |
| Pictures | `<doclang> [<picture>]` |
| Combined | `<doclang> [<ocr>, <layout>, <picture>]` |

### Full page

`<doclang>` converts the whole page: layout, text, tables, formulas, code, charts, and pictures.

<hfoptions id="usage">

<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(
    task="image-text-to-text",
    model="docling-project/granite-for-docling-500m",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/docling-project/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png"},
            {"type": "text", "text": "<doclang>"},
        ],
    }
]
pipe(text=messages, max_new_tokens=4096, return_full_text=False, skip_special_tokens=False)
```

</hfoption>

<hfoption id="AutoModel">

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "docling-project/granite-for-docling-500m"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/docling-project/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png"},
            {"type": "text", "text": "<doclang>"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=4096)
# Keep the special tokens: they are the DocLang markup. Drop the end-of-text token that closes the generation.
doclang = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
doclang = doclang.removesuffix(processor.tokenizer.eos_token).strip()
print(doclang)
```

</hfoption>

</hfoptions>

To convert only some elements, change the text prompt to one of the table above, for example `<doclang> [<ocr>, <layout>, <picture>]`. Everything else stays the same.

### Dense pages: the fine path

The fine connector path spends four times as many image tokens per tile, which helps on small print, long tables and multi-column text. Select it per call with `fine_route=True`:

```python
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={"fine_route": True},
).to(model.device)
output = model.generate(**inputs, max_new_tokens=4096)
```

The 500M model (`config.density_router_hidden_size` is set) predicts which pages need the fine path from its default image tokens:

```python
needs_fine = model.model.predict_fine_route(inputs["pixel_values"])  # (batch_size,) bool
if needs_fine[0]:
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"fine_route": True},
    ).to(model.device)
output = model.generate(**inputs, max_new_tokens=4096)
```

### Batched inference

Pages get different tile grids, so `pixel_values` is padded to the largest tile count in the batch with all-zero tiles that the model discards.

```python
pages = [
    "https://huggingface.co/docling-project/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png",
    "https://huggingface.co/docling-project/granite-for-docling-500m/resolve/main/docling_technical_report_p1.png",
]
conversations = [
    [{"role": "user", "content": [{"type": "image", "url": url}, {"type": "text", "text": "<doclang>"}]}]
    for url in pages
]
inputs = processor.apply_chat_template(
    conversations,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={"padding": True},
).to(model.device)

output = model.generate(**inputs, max_new_tokens=4096)
for row in output:
    doclang = processor.decode(row[inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    print(doclang.removesuffix(processor.tokenizer.eos_token).strip())
```

Add `"fine_route": True` to `processor_kwargs` to route the whole batch through the fine path; `tile_fine_mask` in the inputs then marks the tiles that take it.

### Export DocLang with Docling

`docling-core` parses the `doclang` string from the AutoModel example into a `DoclingDocument`, which you can serialize to whatever you need.

```python
from pathlib import Path
from docling_core.transforms.deserializer.doclang import DocLangDocDeserializer

document = DocLangDocDeserializer().deserialize_str(doclang)

print(document.export_to_markdown())
print(document.export_to_html())
print(document.export_to_dict())

document.save_as_markdown(Path("page.md"))
document.save_as_html(Path("page.html"))
document.save_as_json(Path("page.json"))
```

The [DocLang](https://doclang.ai/) spec and `doclang` toolkit (validate / pack) live at [doclang-project/doclang](https://github.com/doclang-project/doclang). The deserializer and the `DoclingDocument` exporters live in [`docling-core`](https://github.com/docling-project/docling-core).

## Notes

- A full page is long: budget `max_new_tokens` in the thousands, otherwise the DocLang is cut off silently.
- Tiles are 512x512, laid out on the grid that best matches the page's aspect ratio, capped by `max_patches` (default 32) and 16 tiles per side. A page that fits in one tile gets no thumbnail.
- On the 500M model the fine path quadruples the image tokens per tile, from 64 to 256. Use it for dense pages only, or let the router decide with [`~GraniteDoclingModel.predict_fine_route`]. The 2B model uses 256 image tokens per tile throughout (pixel-shuffle ratio 2).

## GraniteDoclingConfig

[[autodoc]] GraniteDoclingConfig

## GraniteDoclingTextConfig

[[autodoc]] GraniteDoclingTextConfig

## GraniteDoclingVisionConfig

[[autodoc]] GraniteDoclingVisionConfig

## GraniteDoclingImageProcessor

[[autodoc]] GraniteDoclingImageProcessor
    - preprocess

## GraniteDoclingImageProcessorPil

[[autodoc]] GraniteDoclingImageProcessorPil
    - preprocess

## GraniteDoclingProcessor

[[autodoc]] GraniteDoclingProcessor
    - __call__

## GraniteDoclingVisionModel

[[autodoc]] GraniteDoclingVisionModel
    - forward

## GraniteDoclingModel

[[autodoc]] GraniteDoclingModel
    - forward
    - get_image_features
    - predict_fine_route

## GraniteDoclingTextModel

[[autodoc]] GraniteDoclingTextModel
    - forward

## GraniteDoclingForConditionalGeneration

[[autodoc]] GraniteDoclingForConditionalGeneration
    - forward
    - get_image_features
