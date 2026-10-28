<!--Copyright 2026 The HuggingFace Team. All rights reserved.
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
*This model was contributed to Hugging Face Transformers on 2026-10-28.*

# Dory

Dory is a looped (latent-recurrent) language model from NVIDIA. It uses a [normalized-GPT](https://huggingface.co/papers/2410.01131) architecture with interleaved sliding-window and full attention layers. By applying a shared block of layers repeatedly, Dory can scale its intelligence by spending more compute per token. This compute can be adjusted by changing the loop count during inference with the same checkpoint.

Find all official checkpoints in the [Dory collection](https://huggingface.co/collections/nvidia/nemotron-labs-dory).

## Quickstart

Generate text with [`Pipeline`] or [`AutoModelForCausalLM`]. Both examples use the chat template, which enables thinking by default, and the recommended sampling settings.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
)
pipe([{"role": "user", "content": "Name the largest planet in our solar system."}], max_new_tokens=512)
```

</hfoption>
<hfoption id="AutoModelForCausalLM">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Name the largest planet in our solar system."}],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- Set `n_recurrent_loops` to balance compute and accuracy. The released checkpoints default to 2. Try 1 for lower compute per token or 4 for higher accuracy. More loops can shorten responses, so measure end-to-end latency.
- Set `recurrent_kv_cache_mode="last_loop"` to share KV history across loops instead of keeping separate histories with the default `per_loop`. Cache memory then stays independent of loop count.

## DoryConfig

[[autodoc]] DoryConfig

## DoryModel

[[autodoc]] DoryModel
    - forward

## DoryForCausalLM

[[autodoc]] DoryForCausalLM
    - forward
