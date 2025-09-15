<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-05-27 and added to Hugging Face Transformers on 2023-01-03.*



<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="Hugging Face" src="https://img.shields.io/badge/huggingface-GIT-blue?logo=huggingface">
    <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  </div>
</div>

# GIT

**[GIT](https://huggingface.co/papers/2205.14100) (Generative Image-to-Text Transformer)** is a multimodal model developed by Microsoft that generates natural language descriptions from images. It is a decoder-only Transformer that leverages [CLIP's](https://huggingface.co/docs/transformers/main/en/model_doc/clip) vision encoder to condition the model on vision inputs in addition to text.
GIT extends the standard Transformer architecture to jointly process visual and textual information: image features from a vision backbone are fed into a text decoder to produce captions or answers. Unlike older captioning systems that relied on handcrafted fusion modules, GIT treats the task as a straightforward sequence-to-sequence generation problem, making the design simpler and more scalable. It was trained on large collections of image–text pairs and can be used for tasks like image captioning, visual question answering, and other vision-language applications.
The model obtains state-of-the-art results on image captioning and visual question answering benchmarks.

You can find all the original GIT checkpoints under the [GIT collection](https://huggingface.co/collections/microsoft/git-6601c19e9a0401ea1f8ab8c1).

> Click on the GIT models in the right sidebar for more examples of how to apply GIT to different vision and language tasks.

---

## Usage

The example below demonstrates how to generate captions from images or answer questions about images with [`pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

### Using `pipeline` for image captioning

```py
from transformers import pipeline

captioner = pipeline("image-to-text", model="microsoft/git-base")

result = captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
print(result[0]['generated_text'])
````

### Using `pipeline` for visual question answering

```python
from transformers import pipeline

vqa = pipeline("visual-question-answering", model="microsoft/git-base")

result = vqa(
    image="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png",
    question="How many birds are in the picture?"
)
print(result[0]['generated_text'])
```

</hfoption>
<hfoption id="AutoModel">

### Using `AutoProcessor` and `AutoModelForCausalLM`

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
generated_ids = model.generate(pixel_values=inputs["pixel_values"], max_length=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(caption)
```

</hfoption>
<hfoption id="transformers-cli">

### Using `transformers-cli` pipeline

```bash
transformers-cli pipeline \
  --task "image-to-text" \
  --model "microsoft/git-base" \
  --input "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
```

</hfoption>
</hfoptions>

---

## Quantization

Quantization reduces the memory burden of large models by representing the weights in lower precision. Refer to the [Quantization overview](../quantization/overview) for more available quantization backends.
The example below uses dynamic INT8 quantization with [🤗 Optimum](https://huggingface.co/docs/optimum) to only quantize the weights to INT8.

```python
from transformers import AutoProcessor
from optimum.intel import INCModelForCausalLM
from PIL import Image
import requests

# Load processor
processor = AutoProcessor.from_pretrained("microsoft/git-base")

# Load model with dynamic INT8 quantization
model = INCModelForCausalLM.from_pretrained(
    "microsoft/git-base",
    quantization_config="dynamic"
)

# Prepare input
url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

# Run generation
generated_ids = model.generate(**inputs, max_length=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(caption)
```

---

## Attention Visualization

GIT supports returning text-token attentions by setting `output_attentions=True` during the forward pass. However, the `AttentionMaskVisualizer` utility currently only supports text-only models.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests

# Load processor and model
processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

# Example input
url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, text="Describe this image:", return_tensors="pt")

# Forward pass with attentions
outputs = model(**inputs, output_attentions=True)

# Pick the first layer, first head
attn = outputs.attentions[0][0, 0].detach().cpu().numpy()  # shape: (seq_len, seq_len)

# Plot attention heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(attn, cmap="viridis")
plt.title("Attention map (Layer 0, Head 0)")
plt.xlabel("Key positions")
plt.ylabel("Query positions")
plt.show()
```

> This returns raw attention maps for each transformer layer.
> Cross-modal attention (image ↔ text) is not fully visualizable with current tools; manual plotting is required to inspect attentions.

---

## Notes

* **Multimodal input**: GIT takes both images and text prompts. Clear, concise prompts (e.g., *"Describe this image:"*) guide generation.
* **Image preprocessing**: Use the `GitProcessor` for resizing, normalization, and tokenization. Supplying raw images may cause errors or poor results.
* **Generation behavior**: GIT can hallucinate or misinterpret content, especially for small details or fine-grained reasoning tasks. Verify outputs before using in critical applications.
* **Efficiency**: Larger checkpoints (e.g., `git-large`) are resource intensive. For experimentation or limited hardware, use `git-base` or quantization.
* **Attention maps**: `output_attentions=True` works in forward passes, but cross-modal attentions are not fully visualizable.
* **Biases**: Captions may reflect social or cultural biases from web data. Use responsibly.
* **License**: Check the license of the model checkpoint (e.g., `microsoft/git-base`) for compliance.

---

## API

### GitVisionConfig

[[autodoc]] GitVisionConfig

### GitVisionModel

[[autodoc]] GitVisionModel
    - forward

### GitConfig

[[autodoc]] GitConfig
    - all

### GitProcessor

[[autodoc]] GitProcessor
    - __call__

### GitModel

[[autodoc]] GitModel
    - forward

### GitForCausalLM

[[autodoc]] GitForCausalLM
    - forward

---

## Resources

* [GIT paper](https://arxiv.org/abs/2205.14100)
* [Microsoft GIT collection on Hugging Face](https://huggingface.co/collections/microsoft/git-6601c19e9a0401ea1f8ab8c1)
