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
*This model was released on 2020-10-23 and added to Hugging Face Transformers on 2020-11-27.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BARThez

[BARThez](https://huggingface.co/papers/2010.12321) is a [BART](./bart) model designed for French language tasks. Unlike existing French BERT models, BARThez includes a pretrained encoder-decoder, allowing it to generate text as well. This model is also available as a multilingual variant, mBARThez, by continuing pretraining multilingual BART on a French corpus.

You can find all of the original BARThez checkpoints under the [BARThez](https://huggingface.co/collections/dascim/barthez-670920b569a07aa53e3b6887) collection.

> [!TIP]
> This model was contributed by [moussakam](https://huggingface.co/moussakam).
> Refer to the [BART](./bart) docs for more usage examples.


The example below demonstrates how to predict the `<mask>` token with [`Pipeline`], [`AutoModel`], and from the command line.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="moussaKam/barthez",
    dtype=torch.float16,
    device=0
)
pipeline("Les plantes produisent <mask> grâce à un processus appelé photosynthèse.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "moussaKam/barthez",
)
model = AutoModelForMaskedLM.from_pretrained(
    "moussaKam/barthez",
    dtype=torch.float16,
    device_map="auto",
)
inputs = tokenizer("Les plantes produisent <mask> grâce à un processus appelé photosynthèse.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Les plantes produisent <mask> grâce à un processus appelé photosynthèse." | transformers run --task fill-mask --model moussaKam/barthez --device 0
```

</hfoption>
</hfoptions>

## BarthezTokenizer

[[autodoc]] BarthezTokenizer

## BarthezTokenizerFast

[[autodoc]] BarthezTokenizerFast
