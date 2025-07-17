<!--
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

<!-- Add badges -->
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="License: Apache-2.0" src="https://img.shields.io/github/license/huggingface/transformers.svg" />
        <img alt="Model: CTRL" src="https://img.shields.io/badge/model-ctrl-blue.svg" />
    </div>
</div>

# [CTRL](https://arxiv.org/abs/1909.05858)

CTRL (Conditional Transformer Language Model) is a large language model developed by Salesforce Research that enables **controllable text generation**.  
What makes it unique is its use of **control codes**—special prefixes like `Reviews:`, `Books:`, `Legal:`, etc.—that guide the model to produce text in specific domains or styles.
CTRL model was proposed in [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://huggingface.co/papers/1909.05858) by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and
Richard Socher. It's a causal (unidirectional) transformer pre-trained using language modeling on a very large corpus
of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

CTRL was trained on a large corpus of structured datasets, including Wikipedia, web data, Amazon reviews, and more.

You can find all the original CTRL checkpoints under the [CTRL model page on Hugging Face](https://huggingface.co/ctrl).

> [!TIP]
> This model was contributed by [salesforce](https://huggingface.co/salesforce).  
> Click on the [CTRL](https://huggingface.co/ctrl) model in the right sidebar for more examples of how to apply CTRL to different text generation tasks.

The example below demonstrates how to use CTRL for text generation using `pipeline`, `AutoModel`, and `transformers-cli`.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

generator = pipeline("text-generation", model="ctrl")
output = generator("Reviews: This product was", max_length=50, do_sample=True)
print(output[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ctrl")
model = AutoModelForCausalLM.from_pretrained("ctrl")

inputs = tokenizer("Books: Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers-cli">

```bash
transformers-cli run text-generation \
  --model_name_or_path=ctrl \
  --prompt "Legal: The contract states" \
  --max_length 50 \
  --do_sample
```

</hfoption>
</hfoptions>

<hfoption id="Quantization">

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ctrl")
model = AutoModelForCausalLM.from_pretrained(
    "ctrl",
    load_in_8bit=True,
    device_map="auto"
)
```

</hfoption>
</hfoptions>

<!-- Attention visualizer is not currently supported for CTRL, but section is added for future compatibility. -->

<!-- Not applicable for CTRL as it does not support attention mask visualization yet. -->

<div class="flex justify-center">
    <img src="" />
</div>

## Notes

- CTRL relies on **control codes** to guide generation to specific domains like reviews, books, or legal text.
- Using an appropriate prefix such as `Books:` or `Reviews:` is crucial for meaningful output.
- This model is **not compatible** with attention visualization tools.

```py
# Control code example
prompt = "Books: Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Resources

- [CTRL paper (ArXiv)](https://arxiv.org/abs/1909.05858)
- [Salesforce CTRL GitHub](https://github.com/salesforce/ctrl)
- [CTRL on Hugging Face](https://huggingface.co/ctrl)