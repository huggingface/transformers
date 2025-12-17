<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Assisted decoding

Assisted decoding speeds up text generation by allowing a helper propose candidate tokens before the main model commits to them. The main model verifies the candidate tokens in one forward pass. The helper is fast and cheap and can replace dozens of more expensive forward passes by the main model.

This guide covers assisted decoding methods in Transformers.

## Speculative decoding

[Speculative decoding](https://hf.co/papers/2211.17192) uses a smaller assistant model to draft candidate tokens. The main model checks these tokens in one pass. Validated tokens enter the final output and rejected tokens trigger standard sampling. Generation is faster because the main model runs fewer expensive forward passes.

The method works best when the assistant model is significantly smaller than the main model and uses the same tokenizer. Speculative decoding supports greedy search and sampling but not batched inputs.

Pass `assistant_model` to [`~GenerationMixin.generate`]. Set `do_sample=True` to resample if token validation fails.

<hfoptions id="spec-decoding">
<hfoption id="greedy search">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine'
```

The `assistant_model` argument is also available in the [`Pipeline`] API.

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B",
    assistant_model="meta-llama/Llama-3.2-1B",
    dtype="auto"
)
pipeline("Hugging Face is an open-source company, ", max_new_tokens=50, do_sample=False)
```

</hfoption>
<hfoption id="sampling">

Set `temperature` to control randomness. Lower temperatures often improve latency.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that is dedicated to creating a better world through technology.'
```

</hfoption>
</hfoptions>

## Prompt lookup decoding

Prompt lookup decoding doesn't need an assistant model. It finds overlapping n-grams in the prompt to propose candidate tokens. If no match exists, it falls back to normal autoregressive decoding. This suits input-grounded tasks like summarization and translation because candidate tokens often mirror local patterns in the source text.

Pass `prompt_lookup_num_tokens` to [`~GenerationMixin.generate`]. This sets how many tokens the algorithm tries to copy from earlier in the prompt when it detects a repeated pattern.

<hfoptions id="prompt-lookup-decoding">
<hfoption id="greedy decoding">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, prompt_lookup_num_tokens=5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine learning models. It offers a variety of tools'
```

</hfoption>
<hfoption id="sampling">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, prompt_lookup_num_tokens=5, do_sample=True, temperature=0.5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine learning models. It offers a variety of tools'
```

</hfoption>
</hfoptions>

## Self-speculative decoding

Self-speculative decoding uses a model's intermediate layers as the assistant to propose candidate tokens. If the proposal matches, the model exits early and the remaining layers verify or correct the tokens.

Because it's all one model, weights and caches are shared, which boosts speed without extra memory overhead. This technique only works for models trained to support early-exit logits from intermediate layers.

Pass `assistant_early_exit` to [`~GenerationMixin.generate`] to set the exit layer.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/layerskip-llama3.2-1B")
model = AutoModelForCausalLM.from_pretrained("facebook/layerskip-llama3.2-1B", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_early_exit=4, do_sample=False, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## Universal assisted decoding

Universal assisted decoding (UAD) makes speculative decoding possible even when the main and assistant models have different tokenizers. It lets you pair any small assistant model with the main model. Candidate tokens are re-encoded and the algorithm computes the longest common subsequence so the continuation stays aligned.

Pass `tokenizer`, `assistant_tokenizer`, and `assistant_model` to [`~GenerationMixin.generate`] to enable UAD.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assistant_tokenizer = AutoTokenizer.from_pretrained("double7/vicuna-68m")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", dtype="auto")
assistant_model = AutoModelForCausalLM.from_pretrained("double7/vicuna-68m", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that is dedicated to creating a better world through technology.'
```

## Resources

- Read the [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation) blog post for more context about text generation latency and assisted generation.