---
license: apache-2.0
tags:
- text-generation
- causal-lm
- control-codes
library_name: transformers
model_type: ctrl
pipeline_tag: text-generation
---

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

## Usage

<hfoptions>

<hfoption id="pipeline">

```python
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