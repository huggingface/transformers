<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-17.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# OpenAI Privacy Filter

TODO: Actually add stuff from the model card here.

OpenAI Privacy Filter is an encoder model for token classification over privacy-sensitive spans. It uses a GPT-style backbone with bidirectional local attention (sliding window attention), mixture-of-experts layers, and a token classification head for named entity recognition-style privacy labels.

The example below demonstrates how to detect privacy-sensitive tokens with [`Pipeline`] or the [`AutoModelForTokenClassification`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

classifier = pipeline(
    task="token-classification",
    model="openai/privacy-filter",
)
classifier("My name is Alice Smith")
```

</hfoption>
<hfoption id="AutoModelForTokenClassification">

```py
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")
model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter", device_map="auto")

inputs = tokenizer("My name is Alice Smith", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

predicted_token_class_ids = outputs.logits.argmax(dim=-1)
predicted_token_classes = [model.config.id2label[token_id.item()] for token_id in predicted_token_class_ids[0]]
print(predicted_token_classes)
```

</hfoption>
</hfoptions>

## Resources

- [Token classification task guide](../tasks/token_classification)

## OpenAIPrivacyFilterConfig

[[autodoc]] OpenAIPrivacyFilterConfig

## OpenAIPrivacyFilterModel

[[autodoc]] OpenAIPrivacyFilterModel
    - forward

## OpenAIPrivacyFilterForTokenClassification

[[autodoc]] OpenAIPrivacyFilterForTokenClassification
    - forward
