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
*This model was released on 2023-09-07 and added to Hugging Face Transformers on 2023-09-12 and contributed by [ArthurZ](https://huggingface.co/ArthurZ).*

# Persimmon

[Persimmon](https://www.adept.ai/blog/persimmon-8b) is an open-source, 8–9.3 billion parameter decoder-only transformer designed for flexible and efficient use, supporting a context length of 16K tokens—four times that of LLaMA2. It uses squared ReLU activations, rotary positional encodings, and decoupled input/output embeddings, with 70k extra embeddings reserved for future multimodal extensions. Trained on 737 billion tokens (75% text, 25% code) using an improved FlashAttention implementation, it achieves LLaMA2-level performance with only 37% of the training data. The release also includes optimized, fast inference code that balances speed and flexibility, capable of sampling ~56 tokens per second on a single 80GB A100 GPU.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="adept/persimmon-8b-base", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("adept/persimmon-8b-base")
model = AutoModelForCausalLM.from_pretrained("adept/persimmon-8b-base", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- Persimmon models were trained using `bfloat16`, but original inference uses `float16`. Hub checkpoints use `dtype='float16'`. The [`AutoModel`] API casts checkpoints from `torch.float32` to `torch.float16`.
- Online weight dtype matters only when using `dtype="auto"`. The model downloads first (using checkpoint dtype), then casts to torch's default dtype (`torch.float32`). Specify your desired dtype or it defaults to `torch.float32`.
- Don't fine-tune in `float16`. It produces NaN values. Fine-tune in `bfloat16` instead.
- Clone the original repository to convert the model: `git clone https://github.com/persimmon-ai-labs/adept-inference`.
- Persimmon uses a sentencepiece-based tokenizer with a Unigram model. It supports bytefallback (available in `tokenizers==0.14.0` for the fast tokenizer). [`LlamaTokenizer`] wraps sentencepiece as a standard wrapper.
- Use this prompt format for chat mode: `f"human: {prompt}\n\nadept:"`.

## PersimmonConfig

[[autodoc]] PersimmonConfig

## PersimmonModel

[[autodoc]] PersimmonModel
    - forward

## PersimmonForCausalLM

[[autodoc]] PersimmonForCausalLM
    - forward

## PersimmonForSequenceClassification

[[autodoc]] PersimmonForSequenceClassification
    - forward

## PersimmonForTokenClassification

[[autodoc]] PersimmonForTokenClassification
    - forward
