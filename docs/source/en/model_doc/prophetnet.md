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
*This model was released on 2020-01-13 and added to Hugging Face Transformers on 2020-11-16.*

# ProphetNet

[ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://huggingface.co/papers/2001.04063) is an encoder-decoder model that employs future n-gram prediction and an n-stream self-attention mechanism. Unlike traditional models that predict the next token, ProphetNet predicts the next n tokens simultaneously, enhancing its ability to plan for future tokens and reducing overfitting on local correlations. Pre-trained on both base (16GB) and large (160GB) datasets, ProphetNet outperforms other models on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for abstractive summarization and question generation tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="microsoft/prophetnet-large-uncased", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
model = AutoModelForCausalLM.from_pretrained("microsoft/prophetnet-large-uncased", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## ProphetNetConfig

[[autodoc]] ProphetNetConfig

## ProphetNetTokenizer

[[autodoc]] ProphetNetTokenizer

## ProphetNet specific outputs

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput

## ProphetNetModel

[[autodoc]] ProphetNetModel
    - forward

## ProphetNetEncoder

[[autodoc]] ProphetNetEncoder
    - forward

## ProphetNetDecoder

[[autodoc]] ProphetNetDecoder
    - forward

## ProphetNetForConditionalGeneration

[[autodoc]] ProphetNetForConditionalGeneration
    - forward

## ProphetNetForCausalLM

[[autodoc]] ProphetNetForCausalLM
    - forward

