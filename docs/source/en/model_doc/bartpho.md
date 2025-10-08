<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2021-09-20 and added to Hugging Face Transformers on 2021-10-18 and contributed by [dqnguyen](https://huggingface.co/dqnguyen).*

# BARTpho

[BARTpho](https://huggingface.co/papers/2109.09701) introduces two versions—BARTpho_word and BARTpho_syllable—as the first large-scale monolingual sequence-to-sequence models pre-trained for Vietnamese. Leveraging the "large" architecture and pre-training scheme of BART, BARTpho excels in generative NLP tasks. Evaluations on Vietnamese text summarization demonstrate that BARTpho surpasses mBART, setting a new state-of-the-art. The model is released to support future research and applications in generative Vietnamese NLP.

This model was contributed by [dqnguyen](https://huggingface.co/dqnguyen). The original code can be found [here](https://github.com/VinAIResearch/BARTpho).

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline("text2text-generation", model="vinai/bartpho-syllable", dtype="auto")
pipeline("Thực vật tạo ra năng lượng thông qua một quá trình được gọi là")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-syllable", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

inputs = tokenizer("Thực vật tạo ra năng lượng thông qua một quá trình được gọi là", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## BartphoTokenizer

[[autodoc]] BartphoTokenizer
