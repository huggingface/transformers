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
*This model was released on 2022-05-12 and added to Hugging Face Transformers on 2023-02-10 and contributed by [jvamvas](https://huggingface.co/jvamvas).*

# X-MOD

[X-MOD](https://huggingface.co/papers/2022.naacl-main.255) extends multilingual masked language models by incorporating language-specific modular components, known as language adapters, during pre-training. These adapters are frozen during fine-tuning. The model addresses the curse of multilinguality by increasing the model's capacity without increasing the number of trainable parameters per language. Experiments on natural language inference, named entity recognition, and question answering demonstrate that X-MOD reduces negative interference between languages and enhances both monolingual and cross-lingual performance. Additionally, it allows for the addition of new languages post-hoc without performance degradation.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
config = AutoConfig.from_pretrained("facebook/xmod-base")
config.is_decoder = True
model = AutoModelForCausalLM.from_pretrained("facebook/xmod-base", config=config, dtype="auto")
model.set_default_language("en_XX")

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
```

</hfoption>
</hfoptions>

## XmodConfig

[[autodoc]] XmodConfig

## XmodModel

[[autodoc]] XmodModel
    - forward

## XmodForCausalLM

[[autodoc]] XmodForCausalLM
    - forward

## XmodForMaskedLM

[[autodoc]] XmodForMaskedLM
    - forward

## XmodForSequenceClassification

[[autodoc]] XmodForSequenceClassification
    - forward

## XmodForMultipleChoice

[[autodoc]] XmodForMultipleChoice
    - forward

## XmodForTokenClassification

[[autodoc]] XmodForTokenClassification
    - forward

## XmodForQuestionAnswering

[[autodoc]] XmodForQuestionAnswering
    - forward

