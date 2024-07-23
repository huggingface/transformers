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

# Best Practices for Generation with Cache

Efficient caching is crucial for optimizing the performance of models in various generative tasks,
including text generation, translation, summarization and other transformer-based applications.
Effective caching helps reduce computation time and improve response rates, especially in real-time or resource-intensive applications.

Transformers support various caching methods, leveraging `Cache` classes to abstract and manage the caching logic.
This document outlines best practices for using these classes to maximize performance and efficiency.
Check out all the available `Cache` classes in the [API documentation](./internal/generation_utils.md).

## What is Cache and why we should care?

Imagine you’re having a conversation with someone, and instead of remembering what was said previously, you have to start from scratch every time you respond. This would be slow and inefficient, right? In the world of Transformer models, a similar concept applies, and that's where Caching keys and values come into play. From now on, I'll refer to the concept as KV Cache.

KV cache is needed to optimize the generation in autoregressive models, where the model predicts text token by token. This process can be slow since the model can generate only one token at a time, and each new prediction is dependent on the previous context. That means, to predict token number 1000 in the generation, you need information from the previous 999 tokens, which comes in the form of some matrix multiplications across the representations of those tokens. But to predict token number 1001, you also need the same information from the first 999 tokens, plus additional information from token number 1000. That is where key-value cache is used to optimize the sequential generation process by storing previous calculations to reuse in subsequent tokens, so they don't need to be computed again.

More concretely, key-value cache acts as a memory bank for these generative models, where the model stores key-value pairs derived from self-attention layers for previously processed tokens. By storing this information, the model can avoid redundant computations and instead retrieve keys and values of previous tokens from the cache.

### Under the Hood: How Cache Object Works in Attention Mechanism
<details>
    <summary><em>For the Curious Minds Who Like to Dive Deep</em></summary>
    <p>When utilizing a cache object in the input, the Attention module performs several critical steps to integrate past and present information seamlessly.</p>
    <p>The Attention module concatenates the current key-values with the past key-values stored in the cache. This results in an attention weights of shape `(new_tokens_length, past_kv_length + new_tokens_length)`. Essentially, the past and current key-values are combined to compute attention scores, ensuring that the model considers both previous context and new input. The concatenated key-values are used to compute the attention scores resulting in attention weights of shape `(new_tokens_length, past_kv_length + new_tokens_length)`.</p>
    <p>Therefore, when iteratively calling `forward()` instead of the `generate()` method, it’s crucial to ensure that the attention mask shape matches the combined length of past and current key-values. The attention mask should have the shape `(batch_size, past_kv_length + new_tokens_length)`. This is usually handled internally when you call `generate()` method. If you want to implement your own generation loop with Cache classes, take this into consideration and prepare the attention mask to hold values to current and past tokens.</p>
    <p>See an example below for how to implement your own generation loop.</p>
</details>



## Generate with Cache

In 🤗 Transformers, we support various Cache types to optimize the performance across different models and tasks. By default all models generate with caching
and for that we use the ["DynamicCache"] class. It allows us to dynamically grow cache size, by saving more and more keys and values as we generate. Usually all you have to do to enable or disable it is to pass in `use_cache=True` or `use_cache=False` into the `generate()` method.

Apart from that we support many cache classes, which can be set with a `cache_implementation` argument. It can be set to one of `["static", "quantized", "sliding_window", "hybrid"]`. Let's look into each cache type in more details.

### Quantized Cache

The key and value cache can occupy a large portion of memory, becoming a bottleneck for long-context generation, especially for Large Language Models.
Quantizing the cache when using `generate()` can significantly reduce memory requirements at the cost of speed.

KV Cache quantization in `transformers` is largely inspired by the paper [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache]
(https://arxiv.org/abs/2402.02750) and currently supports `quanto` and `HQQ` as backends. For more information on the inner workings see the paper.

To enable quantization of the key-value cache, one needs to indicate `cache_implementation="quantized"` in the `generation_config`.
Quantization related arguments should be passed to the `generation_config` either as a `dict` or an instance of a [`QuantizedCacheConfig`] class.
One has to indicate which quantization backend to use in the [`QuantizedCacheConfig`], the default is `quanto`.

<Tip warning={true}>

Cache quantization can be detrimental if the context length is short and there is enough GPU VRAM available to run without cache quantization.

</Tip>


```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"nbits": 4, "backend": "quanto"})
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. It's a great way to express myself and rel

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20)
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's loud and energetic. I like to listen to it when I'm feeling
```


### Static Cache

Since the ["DynamicCache"] dynamically grows with each generation step, it prevents you from taking advantage of JIT optimizations. The ["StaticCache"] pre-allocates 
a specific maximum size for the keys and values, allowing you to generate up to the maximum length without having to modify cache size. Check the below usage example.

For more examples with Static Cache and JIT compilation, take a look at (StaticCache & torchcompile)["./llm_optims.md#static-kv-cache-and-torchcompile"]

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

>>> # simply pass the cache implementation="static"
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="static")
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
```

### Sink Cache

Sink Cache was introduced in ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453). It allows you to generate long sequences of text ("infinite length" according to the paper) without any fine-tuning. That is achieved by smart handling of previous keys and values, specifically it retains a few initial tokens from the sequence, called "sink tokens". This is based on the observation that these initial tokens attract significant portion of attention scores during the generation process. Tokens that come after "sink tokens" are discarded on a sliding windowed basis, keeping only the latest `window_size` tokens.By keeping these initial tokens as "attention sinks," the model maintains stable performance even when dealing with very long texts, thus discarding most of the previous knowledge.

Unlike other cache classes, this one can't be used directly by indicating a `cache_implementation`. You have to initialize the Cache before calling on `generate()` as follows.

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("This is a long story about unicorns, fairies and magic.", return_tensors="pt").to(model.device)

>>> # get our cache, specify number of sink tokens and window size
>>> # Note that window size already includes sink tokens, so has to be larger
>>> past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=500, past_key_values=past_key_values)
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
```

### Sliding Window Cache

As the name suggests, this cache type implements a sliding window over previous keys and values, retaining only the last `sliding_window` tokens. It should be used with models like Mistral that support sliding window attention. Additionally, similar to Static Cache, this one is JIT-friendly and can be used with the same compile tecniques as Static Cache.


```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).to("cuda:0")
>>> inputs = tokenizer("Yesterday I was on a rock concert and.", return_tensors="pt").to(model.device)

>>> # can be used by passing in cache implementation
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=200, cache_implementation="sloding_window")
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
```

### Hybrid Cache

Hybrid Cache is used in specific models only. It is a combination of SlidingWindow and Static Cache, where every layer in the model operates with one of these cache types. Currently only Gemma2 supports it and uses it be default whenever you load the model. Using Hybrid Cache with other models can results in performance degradation and unexpected generation results. It can be used by passing `cache_implementation="hybrid"`, similar to the other cache classes.



## Iterative Generation with Cache

We have seen how to use each of the cache types when generating. What if you want to use cache in iterative generation setting, for example in applications like chatbots, where interactions involve multiple turns and continuous back-and-forth exchanges. Iterative generation with cache allows these systems to handle ongoing conversations effectively without reprocessing the entire context at each step. But there are some tips that you should know before you start implementing:

The general format when doing iterative generation is as below. First you have to initialize an empty cache of the type you want, and you can start feeding in new prompts iteratively. Keeping track of dialogues history and formatting can be done with chat templates, read more on that in (chat_templating)["./chat_templating.md"]

In case you are using Sink Cache, you have to crop your inputs to that maximum length because Sink Cache can generate text longer than its maximum window size, but it expects the first input to not exceed the maximum cache length.  


```python
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import (
    DynamicCache,
    SinkCache,
    StaticCache,
    SlidingWindowCache,
    QuantoQuantizedCache,
    QuantizedCacheConfig,
)

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_id)

user_prompts = ["Hello, what's your name?", "Btw, yesterday I was on a rock concert."]

past_key_values = SinkCache(window_length=30, num_sink_tokens=4)
max_cache_length = past_key_values.get_max_length()

messages = []
for prompt in user_prompts:
    messages.append({"role": "user", "content": prompt})
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    
    if isinstance(past_key_values, SinkCache):
        inputs = {k: v[:, -max_cache_length:] for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256, past_key_values=past_key_values)
    completion = tokenizer.decode(outputs[0, input_length: ], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": completion})

print(messages)
```
