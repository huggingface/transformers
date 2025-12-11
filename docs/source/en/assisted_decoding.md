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

Advanced decoding methods aim at either tackling specific generation quality issues (e.g. repetition) or at improving the generation throughput in certain situations. These techniques are more complex, and may not work correctly with all models.

## Speculative decoding

[Speculative](https://hf.co/papers/2211.17192) or assistive decoding isn't a search or sampling strategy. Instead, speculative decoding adds a second smaller model to generate candidate tokens. The main model verifies the candidate tokens in a single `forward` pass, which speeds up the decoding process overall. This method is especially useful for LLMs where it can be more costly and slower to generate tokens. Refer to the [speculative decoding](./llm_optims#speculative-decoding) guide to learn more.

Currently, only greedy search and multinomial sampling are supported with speculative decoding. Batched inputs aren't supported either.

Enable speculative decoding with the `assistant_model` parameter. You'll notice the fastest speed up with an assistant model that is much smaller than the main model. Add `do_sample=True` to enable token validation with resampling.

<hfoptions id="spec-decoding">
<hfoption id="greedy search">

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine'
```

Speculative decoding is also supported in [`Pipeline`] with the `assistant_model` parameter.

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B",
    assistant_model="meta-llama/Llama-3.2-1B",
    dtype=torch.bfloat16
)
pipe_output = pipe("Once upon a time, ", max_new_tokens=50, do_sample=False)
pipe_output[0]["generated_text"]
```

</hfoption>
<hfoption id="multinomial sampling">

Add the `temperature` parameter to control sampling randomness. For speculative decoding, a lower temperature may improve latency.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that is dedicated to creating a better world through technology.'
```

</hfoption>
</hfoptions>

## Prompt lookup decoding

[Prompt lookup decoding](./llm_optims#prompt-lookup-decoding) is a variant of speculative decoding that uses overlapping n-grams as the candidate tokens. It works well for input-grounded tasks such as summarization. Refer to the [prompt lookup decoding](./llm_optims#prompt-lookup-decoding) guide to learn more.

Enable prompt lookup decoding with the `prompt_lookup_num_tokens` parameter.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype=torch.float16).to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", dtype=torch.float16).to(device)
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

outputs = model.generate(**inputs, assistant_model=assistant_model, max_new_tokens=20, prompt_lookup_num_tokens=5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine learning models. It offers a variety of tools'
```

## Self-speculative decoding

Early exiting uses the earlier hidden states from the language modeling head as inputs, effectively skipping layers to yield a lower quality output. The lower quality output is used as the assistant output and self-speculation is applied to fix the output using the remaining layers. The final generated result from this self-speculative method is the same (or has the same distribution) as the original models generation.

The assistant model is also part of the target model, so the caches and weights can be shared, resulting in lower memory requirements.

For a model trained with early exit, pass `assistant_early_exit` to [`~GenerationMixin.generate`].

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = "Alice and Bob"
checkpoint = "facebook/layerskip-llama3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(checkpoint)
outputs = model.generate(**inputs, assistant_early_exit=4, do_sample=False, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## Universal assisted decoding

Universal assisted decoding (UAD) enables the main and assistant models to use different tokenizers. The main models input tokens are re-encoded into assistant model tokens. Candidate tokens are generated in the assistant encoding which are re-encoded into the main model candidate tokens. The candidate tokens are verified as explained in [speculative decoding](#speculative-decoding).

Re-encoding involves decoding token ids into text and encoding the text with a different tokenizer. To prevent tokenization discrepancies during re-encoding, UAD finds the longest common sub-sequence between the source and target encodings to ensure the new tokens include the correct prompt suffix.

Add the `tokenizer` and `assistant_tokenizer` parameters to [`~GenerationMixin.generate`] to enable UAD.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = "Alice and Bob"

assistant_tokenizer = AutoTokenizer.from_pretrained("double7/vicuna-68m")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
inputs = tokenizer(prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")
assistant_model = AutoModelForCausalLM.from_pretrained("double7/vicuna-68m")
outputs = model.generate(**inputs, assistant_model=assistant_model, tokenizer=tokenizer, assistant_tokenizer=assistant_tokenizer)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']
```