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

Transformers support various caching methods, leveraging [`~Cache`] classes to abstract and manage the caching logic.
This document outlines best practices for using these classes to maximize performance and efficiency.
Check out all the available `Cache` classes in the [API documentation](./internal/generation_utils.md).

## What is Cache and why we should care?

Imagine you’re having a conversation with someone, and instead of remembering what was said previously, you have to start from scratch every time you respond. This would be slow and inefficient, right? In the world of Transformer models, a similar concept applies, and that's where Caching keys and values come into play. From now on, I'll refer to the concept as KV Cache.

KV cache is needed to optimize the generation in autoregressive models, where the model predicts text token by token. This process can be slow since the model can generate only one token at a time, and each new prediction is dependent on the previous context. That means, to predict token number 1000 in the generation, you need information from the previous 999 tokens, which comes in the form of some matrix multiplications across the representations of those tokens. But to predict token number 1001, you also need the same information from the first 999 tokens, plus additional information from token number 1000. That is where key-value cache is used to optimize the sequential generation process by storing previous calculations to reuse in subsequent tokens, so they don't need to be computed again.

More concretely, key-value cache acts as a memory bank for these generative models, where the model stores key-value pairs derived from self-attention layers for previously processed tokens. By storing this information, the model can avoid redundant computations and instead retrieve keys and values of previous tokens from the cache.

<details>
    <summary><em>For the Curious Minds Who Like to Dive Deep</em></summary>

    ### Under the Hood: How Cache Object Works in Attention Mechanism

    When utilizing a cache object in the input, the Attention module performs several critical steps to integrate past and present information seamlessly.

    The Attention module concatenates the current key-values with the past key-values stored in the cache. This results in attention weights of shape `(new_tokens_length, past_kv_length + new_tokens_length)`. Essentially, the past and current key-values are combined to compute attention scores, ensuring that the model considers both previous context and new input. The concatenated key-values are used to compute the attention scores resulting in attention weights of shape `(new_tokens_length, past_kv_length + new_tokens_length)`.

    Therefore, when iteratively calling `forward()` instead of the `generate()` method, it’s crucial to ensure that the attention mask shape matches the combined length of past and current key-values. The attention mask should have the shape `(batch_size, past_kv_length + new_tokens_length)`. This is usually handled internally when you call `generate()` method. If you want to implement your own generation loop with Cache classes, take this into consideration and prepare the attention mask to hold values to current and past tokens.

    See an example below for how to implement your own generation loop.
    
    ```python
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    past_key_values = DynamicCache()
    messages = [{"role": "user", "content": "Hello, what's your name."}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    generated_ids = inputs.input_ids
    max_new_tokens = 10
    for _ in range(max_new_tokens):
        outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)     
        # Greedily sample one next token
        next_token_ids = outputs.logits[:, -1:].argmax(-1)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)   
        # Prepare inputs for the next generation step by leaaving unprocessed tokens, in our case we have only one new token
        # and expanding attn mask for the new token, as explained above
        attention_mask = torch.cat(inputs.attention_mask)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    ```
</details>



## Generate with Cache

In 🤗 Transformers, we support various Cache types to optimize the performance across different models and tasks. By default, all models generate with caching,
with the ["~DynamicCache"] class being the default cache for most models. It allows us to dynamically grow cache size, by saving more and more keys and values as we generate. If for some reason you don't want to use caches, you can pass `use_cache=False` into the `generate()` method.

Refer to the table below to see the difference between cache types and choose the one that suits best for your use-case.

| Cache Type          | Memory Efficient | Supports torch.compile() | Initialization Recommended | Latency  |  Long Context Generation |
|---------------------|------------------|--------------------------|----------------------------|----------|--------------------------|
| Dynamic Cache       |      No          |        No                |         No                 |          |     No                   |
| Static Cache        |      No          |        Yes               |         Yes                |          |     No                   |
| Quantized Cache     |      Yes         |        No                |         No                 |          |     Yes                  |
| Sliding Window Cache|      No          |        Yes               |         Yes                |          |     No                   |
| Hybrid Cache        |      No          |        Yes               |         Yes                |          |     No                   |
| Sink Cache          |      Yes         |        No                |         Yes                |          |     Yes                  |
| Mamba Cache         |      No          |        No                |         Yes                |          |     No                   |


These cache classes can be set with a `cache_implementation` argument when generating. To learn about the available options for the cache_implementation flag, please refer to the [API Documentation](./main_classes/text_generation.md#transformers.GenerationConfig). Now, let's explore each cache type in detail and see how to use them. Note that the below examples are for decoder-only Tranformer-based models. Jump directly to ["Model-Specific Cache"]("#model-specific-cache-classes") section to know more about other architectures we support.

### Quantized Cache

The key and value cache can occupy a large portion of memory, becoming a [bottleneck for long-context generation](https://huggingface.co/blog/llama31#inference-memory-requirements), especially for Large Language Models.
Quantizing the cache when using `generate()` can significantly reduce memory requirements at the cost of speed.

KV Cache quantization in `transformers` is largely inspired by the paper [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache]
(https://arxiv.org/abs/2402.02750) and currently supports ["~QuantoQuantizedCache"] and ["~HQQQuantizedCache"] classes. For more information on the inner workings see the paper.

To enable quantization of the key-value cache, one needs to indicate `cache_implementation="quantized"` in the `generation_config`.
Quantization related arguments should be passed to the `generation_config` either as a `dict` or an instance of a [`~QuantizedCacheConfig`] class.
One has to indicate which quantization backend to use in the [`~QuantizedCacheConfig`], the default is `quanto`.

<Tip warning={true}>

Cache quantization can be detrimental in terms of latency if the context length is short and there is enough GPU VRAM available to run without cache quantization. It is recommended to seek balance between memory efficiency and latency.
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

Since the ["~DynamicCache"] dynamically grows with each generation step, it prevents you from taking advantage of JIT optimizations. The ["~StaticCache"] pre-allocates 
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

### Sink Cache

Sink Cache was introduced in ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453). It allows you to generate long sequences of text ("infinite length" according to the paper) without any fine-tuning. That is achieved by smart handling of previous keys and values, specifically it retains a few initial tokens from the sequence, called "sink tokens". This is based on the observation that these initial tokens attract a significant portion of attention scores during the generation process. Tokens that come after "sink tokens" are discarded on a sliding windowed basis, keeping only the latest `window_size` tokens. By keeping these initial tokens as "attention sinks," the model maintains stable performance even when dealing with very long texts, thus discarding most of the previous knowledge.

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

## Model-specific Cache Classes

Some models require storing previous keys, values, or states in a specific way, and the above cache classes cannot be used. For such cases, we have several specialized cache classes that are designed for specific models. These models only accept their own dedicated cache classes and do not support using any other cache types.

Below, we provide detailed descriptions and usage guidelines for each specialized cache class:


### Hybrid Cache

Hybrid Cache is used in specific models only. It is a combination of SlidingWindow and Static Cache, where every layer in the model operates with one of these cache types. Currently only Gemma2 supports it and uses it be default whenever you load the model. Using Hybrid Cache with other models can results in performance degradation and unexpected generation results. It can be used by passing `cache_implementation="hybrid"`, similar to the other cache classes.

<Tip warning={true}>

In case you want to reuse an already filled HybridCache by calling `forward()`, you have to pass in a valid `cacge_position` which will indicate the positions of inputs in the sequence. Note that `cache_position` is not affected by padding, and always adds one more position for each token.

</Tip>

Here's an example of how to use the Gemma2 model with its dedicated cache:

```python
from transformers import AutoTokenizer, GemmaForCausalLM, HybridCache

model = GemmaForCausalLM.from_pretrained("google/gemma-2-9b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

prompt = "What is your favorite condiment?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_out = model.generate(inputs.input_ids, max_length=30, return_dict_in_generate=True)
text = tokenizer.batch_decode(generate_out.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Use `forward()` call with a pre-filled cache
# We have to prepare a cache position and leave unprocessed tokens (i.e. the last token only)
past_key_values = generate_out.past_key_values
cache_position = torch.arange(generate_out.sequences.shape[1], dtype=torch.int64, device=model.device)
new_input_ids = generate_out.sequences[:, -1:]

out = model(new_input_ids, past_key_values=past_key_values, cache_position=cache_position)
logits = out.logits
```


### Mamba Cache

The ["~MambaCache"] is specifically designed for (Mamba model)["./model_doc/mamba.md"], which features a unique architecture integrating Structured State Space sequence to manage long-context sequences. Since Mamba architecture is drastically different from a Transformer architecture, it needd its own cache class to store previous State Space and Convolutional states. Unlike conventional key/value cache classes that concatenate keys and values of previous tokens, Mamba Cache maintains a compact and efficient memory footprint, even during long-context generation.

The cache is automatically initialized when generating with Mamba model or doing a `forward()` call. Keep in mind that in case of passing an already initialized non-empty cache into the model,  you will have to manually initialize `cache_position`. 

```python
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
text = tokenizer.batch_decode(out)
```

### HybridMambaAttentionDynamicCache

The ["~HybridMambaAttentionDynamicCache"] is uniquely designed for (Jamba models)["./model_doc/jamba.md"], which combines elements of both Mamba and transformer-based architectures. The Jamba Cache manages states in a way that leverages the model's hybrid structure, providing key/value cache in Attention layers and mamba-like cache in State Space layers.

Jamba Cache, unlike Mamba Cache, is not initialized automatically when using the model's `forward()` call and you have to pass a valid `HybridMambaAttentionDynamicCache` yourself. For generation the cache will be initialized automatically, and you don't need to do anything. 

Here's an example of how to use the Jamba model with its dedicated cache:

```python
from transformers import AutoTokenizer, JambaForCausalLM, JambaModel
from transformers.models.jamba.modeling_jamba import HybridMambaAttentionDynamicCache

model = JambaForCausalLM.from_pretrained("ai21labs/Jamba-v0.1")
tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# Use `forward()` call and init cache manually
cache = HybridMambaAttentionDynamicCacheconfig(model.config, batch_size=inputs.input_ids.shape[0], dtype=model.dtype, device=model.device)
model = JambaModelfrom_pretrained("ai21labs/Jamba-v0.1")

out = model(inputs.input_ids, past_key_values=cache)
logits = out.logits
```

### Encoder-Decoder Cache

The ["~EncoderDecoderCache"] is a wrapper designed to handle the caching needs of encoder-decoder models. This cache type is specifically built to manage both self-attention and cross-attention caches, ensuring storage and retrieval of past key/values required for these complex models. Cool thing about Encoder-Decoder Cache is that you can set different cache types for the encoder and for the decoder, depending on your use case. Currently this cache is only supported in (Whisper)["./model_doc/whisper.md"] models but we will be adding more models soon. 

In terms of usage, there is nothing special to be done and calling `generate()` or `forward()` will handle everything for you.


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


## Re-use Cache to continue generation


