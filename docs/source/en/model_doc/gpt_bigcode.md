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
*This model was released on 2023-01-09 and added to Hugging Face Transformers on 2023-04-10.*

# GPTBigCode

[GPTBigCode](https://huggingface.co/papers/2301.03988) is an optimized GPT2 model with Multi-Query Attention, developed by the BigCode project. The report details progress in PII redaction, model architecture de-risking, and data preprocessing. Training 1.1B parameter models on Java, JavaScript, and Python subsets of The Stack, the model outperforms larger open-source multilingual code generation models on the MultiPL-E benchmark. Key findings include the effectiveness of aggressive near-duplicate filtering and the negative impact of selecting files from highly starred repositories. Models are released under an OpenRAIL license.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="bigcode/gpt_bigcode-santacoder", dtype="auto",)
pipeline("def fibonacci(n):")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigcode/gpt_bigcode-santacoder")
model = AutoModelForCausalLM.from_pretrained("bigcode/gpt_bigcode-santacoder", dtype="auto",)

inputs = tokenizer("def fibonacci(n):", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## GPTBigCodeConfig

[[autodoc]] GPTBigCodeConfig

## GPTBigCodeModel

[[autodoc]] GPTBigCodeModel
    - forward

## GPTBigCodeForCausalLM

[[autodoc]] GPTBigCodeForCausalLM
    - forward

## GPTBigCodeForSequenceClassification

[[autodoc]] GPTBigCodeForSequenceClassification
    - forward

## GPTBigCodeForTokenClassification

[[autodoc]] GPTBigCodeForTokenClassification
    - forward

