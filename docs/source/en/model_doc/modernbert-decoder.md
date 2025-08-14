<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# ModernBERT Decoder

ModernBERT Decoder has the same architecture as [ModernBERT](https://huggingface.co/papers/2412.13663) but it is trained from scratch with a causal language modeling objective from the [Ettin paper](https://huggingface.co/papers/2507.11412). This allows for using the same architecture to compare encoders and decoders. This model is the decoder architecture implementation of ModernBERT, designed for autoregressive text generation tasks.

ModernBERT Decoder uses sliding window attention and rotary positional embeddings for efficiency and to handle longer sequences.

You can find all the original ModernBERT Decoder checkpoints under the [jhu-clsp](https://huggingface.co/collections/jhu-clsp/encoders-vs-decoders-the-ettin-suite-686303e16142257eed8e6aeb) collection.

> [!TIP]
> This model was contributed by [orionw](https://huggingface.co/orionweller).
>
> Click on the ModernBERT Decoder models in the right sidebar for more examples of how to apply ModernBERT Decoder to different text generation tasks.

The example below demonstrates how to use ModernBERT Decoder for text generation with [`Pipeline`], [`AutoModel`] (with and without quantization), and from the command line. 

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

generator = pipeline(
    task="text-generation",
    model="jhu-clsp/ettin-decoder-17m",
    dtype=torch.float16,
    device=0
)
generator("The future of artificial intelligence is", max_length=50, num_return_sequences=1)

# For sequence classification
classifier = pipeline(
    task="text-classification",
    model="jhu-clsp/ettin-decoder-17m",
    dtype=torch.float16,
    device=0
)
classifier("This movie is really great!")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-17m")
model = AutoModelForCausalLM.from_pretrained(
    "jhu-clsp/ettin-decoder-17m",
    dtype=torch.float16,
    device_map="auto",
)

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

# For sequence classification
from transformers import AutoModelForSequenceClassification

classifier_model = AutoModelForSequenceClassification.from_pretrained(
    "jhu-clsp/ettin-decoder-17m",
    dtype=torch.float16,
    device_map="auto",
    num_labels=2
)

text = "This movie is really great!"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = classifier_model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)

print(f"Predicted class: {predicted_class.item()}")
print(f"Prediction probabilities: {predictions}")
```

</hfoption>

<hfoption id="AutoModel (w/quantization)">

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-decoder-1b")
model = AutoModelForCausalLM.from_pretrained(
    "jhu-clsp/ettin-decoder-1b",
    dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config
)

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
```
</hfoption>

<hfoption id="transformers CLI">

```bash
echo "The future of artificial intelligence is" | transformers run --task text-generation --model jhu-clsp/ettin-decoder-17m --device 0
```

</hfoption>
</hfoptions>


## ModernBertDecoderConfig

[[autodoc]] ModernBertDecoderConfig

<frameworkcontent>
<pt>

## ModernBertDecoderModel

[[autodoc]] ModernBertDecoderModel
    - forward

## ModernBertDecoderForCausalLM

[[autodoc]] ModernBertDecoderForCausalLM
    - forward

## ModernBertDecoderForSequenceClassification

[[autodoc]] ModernBertDecoderForSequenceClassification
    - forward

</pt>
</frameworkcontent>
