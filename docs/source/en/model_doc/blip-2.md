<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-01-30 and added to Hugging Face Transformers on 2023-02-09 and contributed by [nielsr](https://huggingface.co/nielsr).*

# BLIP-2

[BLIP-2](https://huggingface.co/papers/2301.12597) bootstraps vision-language pre-training using frozen image encoders and large language models. It employs a lightweight, 12-layer Transformer encoder to bridge the modality gap, achieving state-of-the-art results on various vision-language tasks. Specifically, BLIP-2 surpasses Flamingo by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. The model also demonstrates strong zero-shot image-to-text generation capabilities following natural language instructions.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip2-opt-2.7b", dtype="auto")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
pipeline(question="What is shown in this image?", image=url)
```

</hfoption>
<hfoption id="AutoModel">

```py
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b", dtype="auto")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

question = "Question: What is shown in this image? Answer:"
inputs = processor(images=image, text=question, return_tensors="pt")

output = model.generate(**inputs)
print(processor.batch_decode(output, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## Usage tips

- BLIP-2 generates conditional text from images and optional text prompts. Use the [`generate`] method at inference time.
- Use [`Blip2Processor`] to prepare images for the model and decode predicted token IDs back to text.
- BLIP models after release v4.46 raise warnings about adding `processor.num_query_tokens = {{num_query_tokens}}` and expanding model embeddings to add special `<image>` tokens. Add these attributes to the processor if you own the model checkpoint, or open a PR if you don't. Adding these attributes means BLIP adds the required query tokens per image and expands text with `<image>` placeholders. Usually around 500 tokens per image, so ensure text isn't truncated to avoid embedding merge failures. Get attributes from `model.config.num_query_tokens` and expand model embeddings following this [link](https://huggingface.co/docs/transformers/model_doc/blip2#blip2processor).

## Blip2Config

[[autodoc]] Blip2Config

## Blip2VisionConfig

[[autodoc]] Blip2VisionConfig

## Blip2QFormerConfig

[[autodoc]] Blip2QFormerConfig

## Blip2Processor

[[autodoc]] Blip2Processor

## Blip2VisionModel

[[autodoc]] Blip2VisionModel
    - forward

## Blip2QFormerModel

[[autodoc]] Blip2QFormerModel
    - forward

## Blip2Model

[[autodoc]] Blip2Model
    - forward
    - get_text_features
    - get_image_features
    - get_qformer_features

## Blip2ForConditionalGeneration

[[autodoc]] Blip2ForConditionalGeneration
    - forward
    - generate

## Blip2ForImageTextRetrieval

[[autodoc]] Blip2ForImageTextRetrieval
    - forward

## Blip2TextModelWithProjection

[[autodoc]] Blip2TextModelWithProjection

## Blip2VisionModelWithProjection

[[autodoc]] Blip2VisionModelWithProjection

