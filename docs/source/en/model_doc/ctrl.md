<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-09-11 and added to Hugging Face Transformers on 2020-11-16 and contributed by [keskarnitishr](https://huggingface.co/keskarnitishr).*

# CTRL

[CTRL](https://huggingface.co/papers/1909.05858) is a 1.63 billion-parameter conditional transformer language model designed to generate text based on control codes. These codes guide the style, content, and task-specific behavior of the generated text, leveraging unsupervised learning while offering explicit control. The model can also predict the most likely data sources for a given sequence, enabling model-based source attribution.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-classification", model="salesforce/ctrl", dtype="auto")
pipeline("Plants are amazing because they can create energy from the sun.")
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("Salesforce/ctrl", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")

inputs = tokenizer("Plants are amazing because they can create energy from the sun.", return_tensors="pt")
outputs = model(**inputs)
predicted_class_id = outputs.logits.argmax(dim=-1).item()
label = model.config.id2label[predicted_class_id]
print(f"Predicted label: {label}")
```

</hfoption>
</hfoptions>

## Usage tips

- CTRL uses control codes to generate text. Start generations with specific words, sentences, or links to generate coherent text. Check the original implementation for details.
- Pad inputs on the right. CTRL uses absolute position embeddings.
- PyTorch models accept `past_key_values` as input. These are previously computed key/value attention pairs. Using `past_key_values` prevents re-computing pre-computed values during text generation. See the [`~CTRLModel.forward`] method for usage details.

## CTRLConfig

[[autodoc]] CTRLConfig

## CTRLTokenizer

[[autodoc]] CTRLTokenizer
    - save_vocabulary

## CTRLModel

[[autodoc]] CTRLModel
    - forward

## CTRLLMHeadModel

[[autodoc]] CTRLLMHeadModel
    - forward

## CTRLForSequenceClassification

[[autodoc]] CTRLForSequenceClassification
    - forward

