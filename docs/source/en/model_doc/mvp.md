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
*This model was released on 2022-06-24 and added to Hugging Face Transformers on 2022-06-29 and contributed by [StevenTang](https://huggingface.co/StevenTang).*

# MVP

[MVP: Multi-task Supervised Pre-training for Natural Language Generation](https://huggingface.co/papers/2206.12131) a natural language generation model trained using supervised multi-task pre-training. The authors compile MVPCorpus, a large dataset of 77 datasets covering 11 NLG tasks, and unify all data into a text-to-text format for pre-training. MVP also uses task-specific soft prompts to enhance performance on individual tasks. Experiments show MVP achieves state-of-the-art results on 13 of 17 benchmark datasets, demonstrating strong effectiveness and generality across diverse NLG tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="summarization", model="RUCAIBox/mvp", dtype="auto")
pipeline("""
Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.
"""
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("RUCAIBox/mvp", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp")

text="""
Plants are remarkable organisms that produce their own food using a method called photosynthesis.
This process involves converting sunlight, carbon dioxide, and water into glucose, which provides energy for growth.
Plants play a crucial role in sustaining life on Earth by generating oxygen and serving as the foundation of most ecosystems.
"""
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfopton>
</hfoptions>

## MvpConfig

[[autodoc]] MvpConfig

## MvpTokenizer

[[autodoc]] MvpTokenizer

## MvpTokenizerFast

[[autodoc]] MvpTokenizerFast

## MvpModel

[[autodoc]] MvpModel
    - forward

## MvpForConditionalGeneration

[[autodoc]] MvpForConditionalGeneration
    - forward

## MvpForSequenceClassification

[[autodoc]] MvpForSequenceClassification
    - forward

## MvpForQuestionAnswering

[[autodoc]] MvpForQuestionAnswering
    - forward

## MvpForCausalLM

[[autodoc]] MvpForCausalLM
    - forward
