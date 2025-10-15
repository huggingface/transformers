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
*This model was released on 2022-08-08 and added to Hugging Face Transformers on 2022-09-02 and contributed by [zphang](https://huggingface.co/zphang).*

# PEGASUS-X

[PEGASUS-X](https://huggingface.co/papers/2208.04347) extends the PEGASUS model for long input summarization by incorporating staggered block-local attention with global tokens in the encoder and additional pretraining on long sequences. This approach enables handling inputs up to 16K tokens while maintaining strong performance and efficiency, comparable to larger models, without requiring model parallelism or a significant increase in parameters.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="google/pegasus-x-large", dtype="auto")
pipeline("Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet. Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems. These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-x-large", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")

text="""
Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
"""
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- PEGASUS-X uses the [`PegasusTokenizer`].

## PegasusXConfig

[[autodoc]] PegasusXConfig

## PegasusXModel

[[autodoc]] PegasusXModel
    - forward

## PegasusXForConditionalGeneration

[[autodoc]] PegasusXForConditionalGeneration
    - forward

