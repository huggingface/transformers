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

# KV cache strategies

The key-value (KV) vectors are used to calculate attention scores. For autoregressive models, KV scores are calculated *every* time because the model predicts one token at a time. Each prediction depends on the previous tokens, which means the model performs the same computations each time.

A KV *cache* stores these calculations so they can be reused without recomputing them. Efficient caching is crucial for optimizing model performance because it reduces computation time and improves response rates. Refer to the [Caching](./cache_explanation) doc for a more detailed explanation about how a cache works.

Transformers offers several [`Cache`] classes that implement different caching mechanisms. Some of these [`Cache`] classes are optimized to save memory while others are designed to maximize generation speed. Refer to the table below to compare cache types and use it to help you select the best cache for your use case.

| Cache Type             | Supports sliding layers  | Supports offloading | Supports torch.compile() | Expected memory usage |
|------------------------|--------------------------|---------------------|--------------------------|-----------------------|
| Dynamic Cache          |           Yes            |          Yes        |           No             |         Medium        |
| Static Cache           |           Yes            |          Yes        |           Yes            |         High          |
| Quantized Cache        |           No             |          No         |           No             |         Low           |

This guide introduces you to the different [`Cache`] classes and shows you how to use them for generation.

## Default cache

The [`DynamicCache`] is the default cache class for all models. It allows the cache size to grow dynamically in order to store an increasing number of keys and values as generation progresses.

Note that for models using sliding window attention (Mistral, Gemma2,...) or chunked attention (Llama4), the cache will stop growing when the layers using these types of attention have reached their maximum size (the sliding window or chunk size).

Disable the cache by configuring `use_cache=False` in [`~GenerationMixin.generate`].

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=False)
```

Cache classes can also be initialized first before calling and passing it to the models [past_key_values](https://hf.co/docs/transformers/internal/generation_utils#transformers.generation.GenerateDecoderOnlyOutput.past_key_values) parameter. This can be useful for more fine-grained control, or more advanced usage such as context caching.

In most cases, it's easier to define the cache strategy in the [cache_implementation](https://hf.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) parameter.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

past_key_values = DynamicCache(config=model.config)
out = model.generate(**inputs, do_sample=False, max_new_tokens=20, past_key_values=past_key_values)
```

## Fixed-size cache

The default [`DynamicCache`] prevents you from taking advantage of most just-in-time (JIT) optimizations because the cache size isn't fixed. JIT optimizations enable you to maximize latency at the expense of memory usage. All of the following cache types are compatible with JIT optimizations like [torch.compile](./llm_optims#static-kv-cache-and-torchcompile) to accelerate generation. 

A fixed-size cache ([`StaticCache`]) pre-allocates a specific maximum cache size for the kv pairs. You can generate up to the maximum cache size without needing to modify it. However, having a fixed (usually large) size for the key/value states means that while generating, a lot of tokens will actually be masked as they should not take part in the attention. So this trick allows to easily `compile` the decoding stage, but it incurs a waste of tokens in the attention computation. As all things, it's then a trade-off which should be very good if you generate with several sequence of more or less the same lengths, but may be sub-optimal if you have for example 1 very large sequence, and then only short sequences (as the fix cache size would be large, a lot would be wasted for the short sequences). Make sure you understand the impact if you use it!

As for [`DynamicCache`], note that for models using sliding window attention (Mistral, Gemma2,...) or chunked attention (Llama4), the cache will never be larger than the sliding window/chunk size on layers using these types of attention, even if the maximum length specified is larger.

You can enable [`StaticCache`] by configuring `cache_implementation="static"` in [`~GenerationMixin.generate`]. This will also turn on automatic `compilation` of the decoding stage for greedy and sample decoding strategies.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="static")
tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Hello, my name is [Your Name], and I am a [Your Profession] with [Number of Years] of"
```

## Cache offloading

The KV cache can occupy a significant portion of memory and become a [bottleneck](https://hf.co/blog/llama31#inference-memory-requirements) for long-context generation. Memory efficient caches focus on trading off speed for reduced memory usage. This is especially important for large language models (LLMs) and if your hardware is memory constrained.

Offloading the cache saves GPU memory by moving the KV cache for model layers except one to the CPU. Only the current layer cache is maintained on the GPU during a models `forward` iteration over the layers. It will asynchronously prefetch the next layer's cache, and send back the current layer's cache back to the CPU after attention computation.

You may want to consider offloading if you have a small GPU and you're getting out-of-memory (OOM) errors.

> [!WARNING]
> You may notice a small degradation in generation throughput compared to a full on-device cache, depending on your model and generation choices (context size, number of generated tokens, number of beams, etc.). This is because moving the key/value states back and forth requires some work.

Offloading is available for both [`DynamicCache`] and [`StaticCache`]. You can enable it by configuring `cache_implementation="offloaded"` for the dynamic version, or `cache_implementation="offloaded_static"` for the static version, in either [`GenerationConfig`] or [`~GenerationMixin.generate`].
Additionally, you can also instantiate your own [`DynamicCache`] or [`StaticCache`] with the `offloading=True` option, and pass this cache in `generate` or your model's `forward` (for example, `past_key_values=DynamicCache(config=model.config, offloading=True)` for a dynamic cache).

Note that the 2 [`Cache`] classes mentionned above have an additional option when instantiating them directly, `offload_only_non_sliding`.
This additional argument decides if the layers using sliding window/chunk attention (if any), will be offloaded as well. Since
these layers are usually short anyway, it may be better to avoid offloading them, as offloading may incur a speed penalty. By default, this option is `False` for [`DynamicCache`], and `True` for [`StaticCache`].

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float16, device_map="auto")
inputs = tokenizer("Fun fact: The shortest", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=23, cache_implementation="offloaded")
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
Fun fact: The shortest war in history was between Britain and Zanzibar on August 27, 1896.
```

The example below shows how you can fallback to an offloaded cache if you run out of memory:

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, infer_device

def resilient_generate(model, *args, **kwargs):
    oom = False
    device = infer_device()
    torch_device_module = getattr(torch, device, torch.cuda)
    try:
        return model.generate(*args, **kwargs)
    except torch.OutOfMemoryError as e:
        print(e)
        print("retrying with cache_implementation='offloaded'")
        oom = True
    if oom:
        torch_device_module.empty_cache()
        kwargs["cache_implementation"] = "offloaded"
        return model.generate(*args, **kwargs)

ckpt = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float16, device_map="auto")
prompt = ["okay "*1000 + "Fun fact: The most"]
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
beams = { "num_beams": 40, "num_return_sequences": 20, "max_new_tokens": 23, "early_stopping": True, }
out = resilient_generate(model, **inputs, **beams)
responses = tokenizer.batch_decode(out[:,-28:], skip_special_tokens=True)
```

## Quantized cache

The [`QuantizedCache`] reduces memory requirements by quantizing the KV values to a lower precision. [`QuantizedCache`] currently supports two quantization backends:

- `hqq` supports int2, int4, and int8 datatypes.
- `quanto` supports int2 and int4 datatypes. This is the default quantization backend.

> [!WARNING]
> Quantizing the cache can harm latency if the context length is short and there is enough GPU memory available for generation without enabling cache quantization. Try to find a balance between memory efficiency and latency.

Enable [`QuantizedCache`] by configuring `cache_implementation="quantized"` in [`GenerationConfig`], and the quantization backend, as well as any additional quantization related parameters should also be passed either as a dict. You should use the default values for these additional parameters unless you're running out-of-memory. In that case, consider decreasing the residual length.

<hfoptions id="quantized-cache">

For the `hqq` backend, we recommend setting the `axis-key` and `axis-value` parameters to `1`.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantizedCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"backend": "hqq"})
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. It's a great way to express myself and rel
```

For `quanto` backend, we recommend setting the `axis-key` and `axis-value` parameters to `0`.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"nbits": 4, "backend": "quanto"})
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. It's a great way to express myself and rel
```

## Encoder-decoder cache

[`EncoderDecoderCache`] is designed for encoder-decoder models. It manages both the self-attention and cross-attention caches to ensure storage and retrieval of previous kv pairs. It is possible to individually set a different cache type for the encoder and decoder.

This cache type doesn't require any setup. It is a simple wrapper around 2 [`Cache`]s as described above, that will be used independently directly by the model.

## Model-specific caches

Some models have a unique way of storing past kv pairs or states that is not compatible with any other cache classes.

Mamba models, such as [Mamba](./model_doc/mamba), require a specific cache because the model doesn't have an attention mechanism or kv states. Thus, they are not compatible with the above [`Cache`] classes.

# Iterative generation

A cache can also work in iterative generation settings where there is back-and-forth interaction with a model (chatbots). Like regular generation, iterative generation with a cache allows a model to efficiently handle ongoing conversations without recomputing the entire context at each step.

For iterative generation with a cache, start by initializing an empty cache class and then you can feed in your new prompts. Keep track of dialogue history with a [chat template](./chat_templating).

The following example demonstrates [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). If you’re using a different chat-style model, [`~PreTrainedTokenizer.apply_chat_template`] may process messages differently. It might cut out important tokens depending on how the Jinja template is written.

For example, some models use special `<think> ... </think>` tokens during reasoning. These could get lost during re-encoding, causing indexing issues. You might need to manually remove or adjust extra tokens from the completions to keep things stable.

```py
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM, DynamicCache, StaticCache

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id)

user_prompts = ["Hello, what's your name?", "Btw, yesterday I was on a rock concert."]

past_key_values = DynamicCache(config=model.config)

messages = []
for prompt in user_prompts:
    messages.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256, past_key_values=past_key_values)
    completion = tokenizer.decode(outputs[0, input_length: ], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": completion})
```

## Prefill a cache (prefix caching)

In some situations, you may want to fill a [`Cache`] with kv pairs for a certain prefix prompt and reuse it to generate different sequences.

The example below initializes a [`StaticCache`], and then caches an initial prompt. Now you can generate several sequences from the prefilled prompt.

```py
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Init StaticCache with big enough max-length (1024 tokens for the below example)
# You can also init a DynamicCache, if that suits you better
prompt_cache = StaticCache(config=model.config, max_cache_len=1024)

INITIAL_PROMPT = "You are a helpful assistant. "
inputs_initial_prompt = tokenizer(INITIAL_PROMPT, return_tensors="pt").to(model.device.type)
# This is the common prompt cached, we need to run forward without grad to be able to copy
with torch.no_grad():
     prompt_cache = model(**inputs_initial_prompt, past_key_values = prompt_cache).past_key_values

prompts = ["Help me to write a blogpost about travelling.", "What is the capital of France?"]
responses = []
for prompt in prompts:
    new_inputs = tokenizer(INITIAL_PROMPT + prompt, return_tensors="pt").to(model.device.type)
    past_key_values = copy.deepcopy(prompt_cache)
    outputs = model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=20)
    response = tokenizer.batch_decode(outputs)[0]
    responses.append(response)

print(responses)
```
