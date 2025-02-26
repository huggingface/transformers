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
# 使用缓存进行生成的最佳实践

Efficient caching is crucial for optimizing the performance of models in various generative tasks,
including text generation, translation, summarization and other transformer-based applications.
Effective caching helps reduce computation time and improve response rates, especially in real-time or resource-intensive applications.

在各种生成任务中，高效缓存对于优化的模型性能至关重要，这些任务包括文本生成、翻译、摘要以及其他基于Transformer的应用。有效的缓存有助于减少计算时间，并提高响应速度，尤其是在实时或资源密集型的应用。

Transformers support various caching methods, leveraging "Cache" classes to abstract and manage the caching logic.
This document outlines best practices for using these classes to maximize performance and efficiency.
Check out all the available `Cache` classes in the [API documentation](./internal/generation_utils).

Transformers支持多种缓存方法，通过利用“Cache”类来抽象和管理缓存逻辑。
本文档概述了使用这些类以实现最佳性能和效率的最佳实践。
有关所有可用的`Cache`类，请查看[API文档](./internal/generation_utils)。

## What is Cache and why we should care?
## 什么是缓存，以及我们为何需要关注它？

Imagine you’re having a conversation with someone, and instead of remembering what was said previously, you have to start from scratch every time you respond. This would be slow and inefficient, right? In the world of Transformer models, a similar concept applies, and that's where Caching keys and values come into play. From now on, I'll refer to the concept as KV Cache.

想象一下，你正在与某人进行对话，而每次回应时，你不得不从头开始回忆之前说过的话。这种方式既缓慢又低效，对吧？在Transformer模型的世界中，也存在类似的概念，这就引入了缓存 keys 和 values 的重要性。从现在起，我将这一概念称为KV缓存（KV Cache、key-value cache）。

KV cache is needed to optimize the generation in autoregressive models, where the model predicts text token by token. This process can be slow since the model can generate only one token at a time, and each new prediction is dependent on the previous context. That means, to predict token number 1000 in the generation, you need information from the previous 999 tokens, which comes in the form of some matrix multiplications across the representations of those tokens. But to predict token number 1001, you also need the same information from the first 999 tokens, plus additional information from token number 1000. That is where key-value cache is used to optimize the sequential generation process by storing previous calculations to reuse in subsequent tokens, so they don't need to be computed again.

KV 缓存的存在是为了优化自回归模型中的文本生成的过程，这些模型逐个预测文本的token。由于模型一次只能生成一个token，且每个新的预测都依赖于之前的上下文，这一过程可能相当缓慢。这意味着，要预测生成过程中的第1000个token，你需要前999个token的信息，这些信息以矩阵乘法的方式通过那些token的表示传递过来。然而，要预测第1001个token时，你同样需要前999个token的信息，再加上第1000个token的额外信息。正是在这一环节，key-value 缓存 派上了用场，它通过存储之前的计算结果，以便在后继token中复用，从而避免了重复计算，优化了序列生成过程。

More concretely, key-value cache acts as a memory bank for these generative models, where the model stores key-value pairs derived from self-attention layers for previously processed tokens. By storing this information, the model can avoid redundant computations and instead retrieve keys and values of previous tokens from the cache. Note that caching can be used only in inference and should be disabled when training, otherwise it might cause unexpected errors.

更为具体地说，key-value 缓存 扮演了生成模型记忆库的角色，模型在此存储了从前处理过的token通过自注意力层提取的 key-value 对。通过保存这些信息，模型能够规避冗余计算，直接从缓存中检索先前token的keys 和 values 即可。需要注意的是，缓存机制仅适用于推理阶段，在训练时应将其关闭，否则可能导致意想不到的错误。

<details>
  <summary><em>对于那些热衷于深入探究的好奇心朋友们 For the Curious Minds Who Like to Dive Deep</em></summary>

  ### 揭秘：缓存对象在注意力机制中的运作原理 Under the Hood: How Cache Object Works in Attention Mechanism

  When utilizing a cache object in the input, the Attention module performs several critical steps to integrate past and present information seamlessly.
  在使用输入参数中的缓存对象时，注意力模块会执行几个关键步骤，以实现过去与当前信息的无缝整合。

  The Attention module concatenates the current key-values with the past key-values stored in the cache. This results in attention weights of shape `(new_tokens_length, past_kv_length + new_tokens_length)`. Essentially, the past and current key-values are combined to compute attention scores, ensuring that the model considers both previous context and new input. The concatenated key-values are used to compute the attention scores resulting in attention weights of shape `(new_tokens_length, past_kv_length + new_tokens_length)`.
  注意力模块将当前的 key-values 与缓存中存储的过去的 key-values 进行连接。这会产生形状为 `(new_tokens_length, past_kv_length + new_tokens_length)` 的注意力权重。本质上，过去和当前的 key-values 会被结合用来计算注意力分数，确保模型同时考虑先前的上下文和新输入。连接后的 key-values 用于计算注意力分数，从而得到形状为 `(new_tokens_length, past_kv_length + new_tokens_length)` 的注意力权重。

  Therefore, when iteratively calling `forward()` instead of the `generate()` method, it’s crucial to ensure that the attention mask shape matches the combined length of past and current key-values. The attention mask should have the shape `(batch_size, past_kv_length + new_tokens_length)`. This is usually handled internally when you call `generate()` method. If you want to implement your own generation loop with Cache classes, take this into consideration and prepare the attention mask to hold values to current and past tokens.
  因此，逐次调用 `forward()` 方法而不是 `generate()` 方法时，确保注意力 mask 的形状与过去和当前 key-values 的总长度匹配至关重要。注意力 mask 应具有形状 `(batch_size, past_kv_length + new_tokens_length)`。通常在调用 `generate()` 方法时会由内部处理。如果你想自己实现生成循环并使用 Cache 类，需要考虑这一点，并准备好注意力 mask 以容纳当前和过去 token 的值。

  <Tip warning={true}>

  One important concept you need to know when writing your own generation loop, is `cache_position`. In case you want to reuse an already filled Cache object by calling `forward()`, you have to pass in a valid `cache_position` which will indicate the positions of inputs in the sequence. Note that `cache_position` is not affected by padding, and always adds one more position for each token. For example, if key/value cache contains 10 tokens (no matter how many of it is a pad token), the cache position for the next token should be `torch.tensor([10])`.
  当你自己实现生成循环时，需要了解一个重要的概念——`cache_position`。如果你希望通过调用 `forward()` 方法重用已经填充的 Cache 对象，需要传递一个有效的 `cache_position`，这将指示序列中输入的位置。请注意，`cache_position` 不受填充影响，并且每新增一个 token 总会增加一个位置。例如，如果key-values 缓存包含 10 个 token（无论其中有多少个是填充 token），下一个 token 的缓存位置应为 `torch.tensor([10])`。

  </Tip>


  See an example below for how to implement your own generation loop.
  下面是一个如何实现自己生成循环的示例。

  ```python
  >>> import torch
  >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

  >>> model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  >>> model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
  >>> tokenizer = AutoTokenizer.from_pretrained(model_id)

  >>> past_key_values = DynamicCache()
  >>> messages = [{"role": "user", "content": "Hello, what's your name."}]
  >>> inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

  >>> generated_ids = inputs.input_ids
  >>> cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=model.device)
  >>> max_new_tokens = 10

  >>> for _ in range(max_new_tokens):
  ...     outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
  ...     # Greedily sample one next token
  ...     next_token_ids = outputs.logits[:, -1:].argmax(-1)
  ...     generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
  ...
  ...     # Prepare inputs for the next generation step by leaaving unprocessed tokens, in our case we have only one new token
  ...     # and expanding attn mask for the new token, as explained above
  ...     attention_mask = inputs["attention_mask"]
  ...     attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
  ...     inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
  ...     cache_position = cache_position[-1:] + 1 # add one more position for the next token

  >>> print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
  ```
  ```txt
  <|user|>
  Hello, what's your name. 
  <|assistant|>
  My name is Sarah. 
  <|
  ```

</details>



## Generate with Cache
## 使用 Cache 生成

In 🤗 Transformers, we support various Cache types to optimize the performance across different models and tasks. By default, all models generate with caching,
在 🤗 Transformers 中，我们支持各种 Cache 类型以在不同模型和任务中优化性能。默认情况下，所有模型都会使用缓存进行生成，
with the [`~DynamicCache`] class being the default cache for most models. It allows us to dynamically grow cache size, by saving more and more keys and values as we generate. If for some reason you don't want to use caches, you can pass `use_cache=False` into the `generate()` method.
其中 [`~DynamicCache`] 类是大多数模型的默认缓存。它允许我们在生成过程中动态增加缓存大小，通过保存更多的键和值。如果你出于某些原因不想使用缓存，可以将 `use_cache=False` 传递给 `generate()` 方法。

Refer to the table below to see the difference between cache types and choose the one that suits best for your use-case. Models for which initialization is recommended should be initialized before calling the model and passed to model as a kwarg. In all other cases you can simply define desired `cache_implementation` and we take care of the rest for you.
参见下表，了解不同缓存类型之间的差异，并根据你的需求选择最适合的类型。建议在初始化的模型应在调用模型之前进行初始化，并作为关键字参数传递给模型。在其他所有情况下，你可以直接定义所需的 `cache_implementation`，我们会为你处理其余部分。

| Cache 类型             | 是否内存高效 | 是否支持 torch.compile() | 是否需要初始化 | 延迟 | 是否支持长内容生成 |
|------------------------|------------------|--------------------------|----------------------------|---------|-------------------------|
| 动态缓存 Dynamic Cache     | No               | No                       | No                         | Mid     | No                      |
| 静态缓存 Static Cache      | No               | Yes                      | Yes                        | High    | No                      |
| 可卸载的缓存 Offloaded Cache | Yes              | No                       | No                         | Low     | Yes                     |
| 可卸载的静态缓存 Offloaded Static Cache | No               | Yes                      | Yes                        | High    | Yes                     |
| 可量化的缓存 Quantized Cache | Yes              | No                       | No                         | Low     | Yes                     |
| 滑动窗口缓存 Sliding Window Cache | No               | Yes                      | Yes                        | High    | No                      |
| 下沉缓存 Sink Cache        | Yes              | No                       | Yes                        | Mid     | Yes                     |

These cache classes can be set with a `cache_implementation` argument when generating. To learn about the available options for the cache_implementation flag, please refer to the [API Documentation](./main_classes/text_generation#transformers.GenerationConfig). Now, let's explore each cache type in detail and see how to use them. Note that the below examples are for decoder-only Tranformer-based models. We also support ["Model-Specific Cache"] classes for models such as Mamba or Jamba, keep reading for more details.
这些缓存类可以在生成时通过 `cache_implementation` 参数设置。要了解 `cache_implementation` 标记的可用选项，请参阅 [API 文档](./main_classes/text_generation#transformers.GenerationConfig)。现在，让我们详细探讨每种缓存类型，并看看如何使用它们。请注意，以下示例适用于基于decoder-only的 Transformer 模型。我们还支持为 Mamba 或 Jamba 等模型的特定缓存类，继续阅读以获取更多详细信息。

### Quantized Cache
### 可量化的缓存

The key and value cache can occupy a large portion of memory, becoming a [bottleneck for long-context generation](https://huggingface.co/blog/llama31#inference-memory-requirements), especially for Large Language Models.
Quantizing the cache when using `generate()` can significantly reduce memory requirements at the cost of speed.
Key 和 value 的缓存可能会占据大量内存，成为长上下文生成的瓶颈，尤其是对于大规模语言模型。在使用 `generate()` 时量化缓存可以显著减少内存需求，但代价是速度变慢。
在使用 `generate()` 时量化缓存可以显著减少内存需求，但代价是速度变慢。

KV Cache quantization in `transformers` is largely inspired by the paper ["KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"](https://arxiv.org/abs/2402.02750) and currently supports [`~QuantoQuantizedCache`] and [`~HQQQuantizedCache`] classes. For more information on the inner workings see the paper.
`transformers` 中的 KV 缓存量化主要受到了论文 ["KIVI: 一种无需调优的异构 2 位量化方法用于 KV 缓存"](https://arxiv.org/abs/2402.02750) 的启发，并当前支持 `~QuantoQuantizedCache` 和 `~HQQQuantizedCache` 类。有关内部机制的更多信息，请参阅该论文。

To enable quantization of the key-value cache, one needs to indicate `cache_implementation="quantized"` in the `generation_config`.
Quantization related arguments should be passed to the `generation_config` either as a `dict` or an instance of a [`~QuantizedCacheConfig`] class.
One has to indicate which quantization backend to use in the [`~QuantizedCacheConfig`], the default is `quanto`.
要启用 KV 缓存的量化，需要在 `generation_config` 中指定 `cache_implementation="quantized"`。
与量化相关的参数应该通过 `dict` 或者 `~QuantizedCacheConfig` 类的实例传递给 `generation_config`。
需要在 `~QuantizedCacheConfig` 中指定使用的量化后端，默认是 `quanto`。

It is recommended to set `axis-key/axis-value` parameters in the cache config to `0` if you're using the `quanto` backend and to `1` if you're using the `HQQ` backend. For other config values, please use the defaults unless you're running out of memory. In that case, you may consider decreasing the residual length.
如果使用 `quanto` 后端，建议在缓存配置中将 `axis-key` 和 `axis-value` 参数设置为 `0`；如果使用 `HQQ` 后端，则应将这些参数设置为 `1`。对于其他配置参数，请使用默认值，除非运行时出现内存不足的情况。遇到这种情况，您可以考虑减少剩余长度。

<Tip warning={true}>

Cache quantization can be detrimental in terms of latency if the context length is short and there is enough GPU VRAM available to run without cache quantization. It is recommended to seek balance between memory efficiency and latency.
如果上下文长度较短且有足够的 GPU VRAM 可以在无需量化缓存的情况下运行，那么缓存量化可能会因延迟增加而变得不利。建议在内存效率和延迟之间寻求平衡。
</Tip>


```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
>>> model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="quantized", cache_config={"nbits": 4, "backend": "quanto"})
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
I like rock music because it's a great way to express myself. I like the way it makes me feel, the
```

### Offloaded Cache
## 可卸载的缓存

Similarly to KV cache quantization, [`~OffloadedCache`] strategy aims to reduce GPU VRAM usage.
It does so by moving the KV cache for most layers to the CPU.
As the model's `forward()` method iterates over the layers, this strategy maintains the current layer cache on the GPU.
At the same time it asynchronously prefetches the next layer cache as well as sending the previous layer cache back to the CPU.
Unlike KV cache quantization, this strategy always produces the same result as the default KV cache implementation.
Thus, it can serve as a drop-in replacement or a fallback for it.
类似地，`~OffloadedCache` 策略旨在减少 GPU VRAM 的使用量。它通过将大部分层的 KV 缓存移到 CPU 来实现这一目标。随着模型的 `forward()` 方法遍历各层，该策略将在 GPU 上维护当前层的缓存，并异步预取下一层的缓存，同时将上一层的缓存发送回 CPU。与 KV 缓存量化不同，该策略始终会产生与默认 KV 缓存实现相同的结果，因此可以作为默认实现的直接替换或后备选项。

Depending on your model and the characteristics of your generation task (size of context, number of generated tokens, number of beams, etc.)
you may notice a small degradation in generation throughput compared to the default KV cache implementation.
根据您的模型和生成任务的特点（上下文长度、生成的tokens数量、beam 数量等），与默认 KV 缓存实现相比，您可能会注意到生成吞吐量略有下降。

To enable KV cache offloading, pass `cache_implementation="offloaded"` in the `generation_config` or directly to the `generate()` call.
Use `cache_implementation="offloaded_static"` for an offloaded static cache (see also [Offloaded Static Cache](#offloaded-static-cache) below).
要启用 KV 缓存卸载，请在 `generation_config` 中或直接在 `generate()` 调用中传递 `cache_implementation="offloaded"`。要使用可卸载的静态 KV 缓存，请传递 `cache_implementation="offloaded_static"`（参见下方的 [可卸载的静态缓存](#offloaded-static-cache)）。

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> ckpt = "microsoft/Phi-3-mini-4k-instruct"

>>> tokenizer = AutoTokenizer.from_pretrained(ckpt)
>>> model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("Fun fact: The shortest", return_tensors="pt").to(model.device)

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=23, cache_implementation="offloaded")
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
Fun fact: The shortest war in history was between Britain and Zanzibar on August 27, 1896.

>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=23)
>>> print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
Fun fact: The shortest war in history was between Britain and Zanzibar on August 27, 1896.
```

<Tip warning={true}>

Cache offloading requires a CUDA GPU and can be slower than dynamic KV cache. Use it if you are getting CUDA out of memory errors.
缓存卸载需要 CUDA GPU，并且可能比动态 KV 缓存慢。如果您遇到 CUDA 内存不足错误，请使用它。

</Tip>

The example below shows how KV cache offloading can be used as a fallback strategy.
以下示例展示了如何使用KV缓存卸载作为备用策略。

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> def resilient_generate(model, *args, **kwargs):
...     oom = False
...     try:
...         return model.generate(*args, **kwargs)
...     except torch.cuda.OutOfMemoryError as e:
...         print(e)
...         print("retrying with cache_implementation='offloaded'")
...         oom = True
...     if oom:
...         torch.cuda.empty_cache()
...         kwargs["cache_implementation"] = "offloaded"
...         return model.generate(*args, **kwargs)
...
...
>>> ckpt = "microsoft/Phi-3-mini-4k-instruct"
>>> tokenizer = AutoTokenizer.from_pretrained(ckpt)
>>> model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16).to("cuda:0")
>>> prompt = ["okay "*1000 + "Fun fact: The most"]
>>> inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
>>> beams = { "num_beams": 40, "num_beam_groups": 40, "num_return_sequences": 40, "diversity_penalty": 1.0, "max_new_tokens": 23, "early_stopping": True, }
>>> out = resilient_generate(model, **inputs, **beams)
>>> responses = tokenizer.batch_decode(out[:,-28:], skip_special_tokens=True)
```

On a GPU with 50 GB of RAM, running this code will print
在拥有50 GB内存的GPU上，运行此代码，在成功生成40个beams之前将输出

```
CUDA out of memory. Tried to allocate 4.83 GiB. GPU
retrying with cache_implementation='offloaded'
```
### Static Cache
## 静态缓存

Since the "DynamicCache" dynamically grows with each generation step, it prevents you from taking advantage of JIT optimizations. The [`~StaticCache`] pre-allocates
a specific maximum size for the keys and values, allowing you to generate up to the maximum length without having to modify cache size. Check the below usage example.

由于“动态缓存”在每一步生成过程中动态增长，这会妨碍你充分利用即时编译（JIT）优化的优势。而[`~StaticCache`]则预先分配了keys和values的最大空间，使得在生成至最大长度前无需调整缓存大小。请参阅以下使用示例。

For more examples with Static Cache and JIT compilation, take a look at [StaticCache & torchcompile](./llm_optims#static-kv-cache-and-torchcompile)
欲了解更多关于静态缓存与即时编译（JIT）的示例，请参阅[静态缓存与torch.compile](./llm_optims#static-kv-cache-and-torchcompile)。

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
>>> model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

>>> # simply pass the cache implementation="static"
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="static")
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Hello, my name is [Your Name] and I am a [Your Position] at [Your Company]. I am writing"
```


## Offloaded Static Cache
## 可卸载的静态缓存

Like [`~OffloadedCache`] exists for offloading a "DynamicCache", there is also an offloaded static cache. It fully supports
JIT optimizations. Just pass `cache_implementation="offloaded_static"` in the `generation_config` or directly to the `generate()` call.
This will use the [`~OffloadedStaticCache`] implementation instead.
类似于[`~OffloadedCache`]用于卸载“动态缓存”，也存在可卸载的静态缓存。它完全支持JIT优化。只需在`generation_config`或直接传递给`generate()`调用时添加`cache_implementation="offloaded_static"`参数即可。这将使用[`~OffloadedStaticCache`]实现来代替默认方案。

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

>>> # simply pass the cache implementation="offloaded_static"
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=20, cache_implementation="offloaded_static")
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Hello, my name is [Your Name], and I am a [Your Profession] with [Number of Years] of"
```
Cache offloading requires a CUDA GPU.
缓存卸载功能需要配备一块CUDA GPU才能使用。

### Sliding Window Cache
### 滑动窗口缓存

As the name suggests, this cache type implements a sliding window over previous keys and values, retaining only the last `sliding_window` tokens. It should be used with models like Mistral that support sliding window attention. Additionally, similar to Static Cache, this one is JIT-friendly and can be used with the same compile tecniques as Static Cache.

Note that you can use this cache only for models that support sliding window, e.g. Mistral models.

正如其名所示，这种缓存类型实现了对已有的keys和values的滑动窗口机制，仅保留最近的`sliding_window`个标记。它适用于支持滑动窗口注意力机制的模型，如Mistral。此外，与静态缓存相似，这种缓存同样对JIT友好，并能应用与静态缓存相同的编译技术。

需要注意的是，这种缓存仅适用于支持滑动窗口的模型，例如Mistral系列模型。


```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

>>> tokenizer = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
>>> model = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("Yesterday I was on a rock concert and.", return_tensors="pt").to(model.device)

>>> # can be used by passing in cache implementation
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=30, cache_implementation="sliding_window")
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Yesterday I was on a rock concert and. I was so excited to see my favorite band perform live. I was so happy that I could hardly contain myself. I was jumping up and down and"
```

### Sink Cache
### 下沉缓存

Sink Cache was introduced in ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453). It allows you to generate long sequences of text ("infinite length" according to the paper) without any fine-tuning. That is achieved by smart handling of previous keys and values, specifically it retains a few initial tokens from the sequence, called "sink tokens". This is based on the observation that these initial tokens attract a significant portion of attention scores during the generation process. Tokens that come after "sink tokens" are discarded on a sliding windowed basis, keeping only the latest `window_size` tokens. By keeping these initial tokens as "attention sinks," the model maintains stable performance even when dealing with very long texts, thus discarding most of the previous knowledge.

Unlike other cache classes, this one can't be used directly by indicating a `cache_implementation`. You have to initialize the Cache before calling on `generate()` as follows.
下沉缓存是在 ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453)一文中提出的。它能够无需微调即可生成长文本序列（根据论文描述，达到“无限长度”）。对已有的keys和values的智能处理策略使得下沉缓存得以实现，特别地，该机制保留了序列开头的几个标记，称为“下沉标记”。这一设计基于一个观察：在文本生成过程中，这些初始标记会吸引大量的注意力得分。紧接“下沉标记”之后的标记则基于滑动窗口原则进行舍弃，仅保留最新的`window_size`个标记。通过保留这些初始标记作为“注意力下沉点”，模型即使处理超长文本也能保持稳定的性能，尽管这意味着大部分过往的知识被舍弃。

与其他缓存类不同，这种缓存不能直接通过指定`cache_implementation`来使用。你需要在调用`generate()`之前，按照以下方式初始化缓存。

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, SinkCache

>>> tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
>>> model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("This is a long story about unicorns, fairies and magic.", return_tensors="pt").to(model.device)

>>> # get our cache, specify number of sink tokens and window size
>>> # Note that window size already includes sink tokens, so has to be larger
>>> past_key_values = SinkCache(window_length=256, num_sink_tokens=4)
>>> out = model.generate(**inputs, do_sample=False, max_new_tokens=30, past_key_values=past_key_values)
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"This is a long story about unicorns, fairies and magic. It is a story about a young girl named Lily who discovers that she has the power to control the elements. She learns that she can"
```

### Encoder-Decoder Cache
### Encoder-Decoder 缓存

The [`~EncoderDecoderCache`] is a wrapper designed to handle the caching needs of encoder-decoder models. This cache type is specifically built to manage both self-attention and cross-attention caches, ensuring storage and retrieval of past key/values required for these complex models. Cool thing about Encoder-Decoder Cache is that you can set different cache types for the encoder and for the decoder, depending on your use case. Currently this cache is only supported in [Whisper](./model_doc/whisper) models but we will be adding more models soon.

In terms of usage, there is nothing special to be done and calling `generate()` or `forward()` will handle everything for you.

[`~EncoderDecoderCache`]是一种专门设计用于满足Encoder-Decoder 模型缓存需求的封装工具。此缓存类型特别构建用于管理自注意力和交叉注意力缓存，确保复杂模型所需的历史 key/values 的存储与检索。Encoder-Decoder 缓存的一个亮点在于，你可以根据具体应用场景，分别为Encoder和Decoder设置不同的缓存类型。目前，这一缓存仅支持[Whisper](./model_doc/whisper)模型，但我们计划很快增加更多模型的支持。

在用法上，无需特别操作，直接调用`generate()`或`forward()`即可自动处理所有相关事宜。

### Model-specific Cache Classes
### 模型特定的缓存类

Some models require storing previous keys, values, or states in a specific way, and the above cache classes cannot be used. For such cases, we have several specialized cache classes that are designed for specific models. These models only accept their own dedicated cache classes and do not support using any other cache types. Some examples include [`~HybridCache`] for [Gemma2](./model_doc/gemma2) series models or [`~MambaCache`] for [Mamba](./model_doc/mamba) architecture models.

某些模型需要以特定方式存储先前的keys、values或状态，上述通用缓存类无法满足需求。针对这些情形，我们提供了多种专门设计的缓存类，适配特定模型。此类模型仅接受它们专属的缓存类，不支持使用其他缓存类型。例如，[Gemma2](./model_doc/gemma2)系列模型使用[`~HybridCache`]，而[Mamba](./model_doc/mamba)架构模型则采用[`~MambaCache`]。

## Iterative Generation with Cache
## 使用缓存进行交互式生成

We have seen how to use each of the cache types when generating. What if you want to use cache in iterative generation setting, for example in applications like chatbots, where interactions involve multiple turns and continuous back-and-forth exchanges. Iterative generation with cache allows these systems to handle ongoing conversations effectively without reprocessing the entire context at each step. But there are some tips that you should know before you start implementing:

The general format when doing iterative generation is as below. First you have to initialize an empty cache of the type you want, and you can start feeding in new prompts iteratively. Keeping track of dialogues history and formatting can be done with chat templates, read more on that in [chat_templating](./chat_templating)

In case you are using Sink Cache, you have to crop your inputs to that maximum length because Sink Cache can generate text longer than its maximum window size, but it expects the first input to not exceed the maximum cache length.

我们在生成过程中已经了解了各种缓存类型的应用。如果你希望在迭代生成设置中使用缓存，例如在聊天机器人的应用中，其中交互涉及多轮连续的来回对话。使用缓存进行迭代生成可以让这些系统有效地处理持续对话，而无需在每一步都重新处理整个上下文。但在开始实现之前，有一些技巧你需要知道：

进行迭代生成的一般格式如下。首先，你需要初始化一个你想要类型的空缓存，并可以开始逐轮输入新的提示。通过聊天模板可以跟踪对话历史和格式化，更多详细信息请参阅[聊天模板](./chat_templating)。

如果你使用下沉缓存 Sink Cache，必须将输入裁剪为最大长度，因为下沉缓存可以生成超过其最大窗口长度的文本，但它期望第一个输入不要超过最大缓存长度。


```python
>>> import torch
>>> from transformers import AutoTokenizer,AutoModelForCausalLM
>>> from transformers.cache_utils import (
...    DynamicCache,
...    SinkCache,
...    StaticCache,
...    SlidingWindowCache,
...    QuantoQuantizedCache,
...    QuantizedCacheConfig,
... )

>>> model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
>>> model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto')
>>> tokenizer = AutoTokenizer.from_pretrained(model_id)

>>> user_prompts = ["Hello, what's your name?", "Btw, yesterday I was on a rock concert."]

>>> past_key_values = DynamicCache()
>>> max_cache_length = past_key_values.get_max_cache_shape()

>>> messages = []
>>> for prompt in user_prompts:
...     messages.append({"role": "user", "content": prompt})
...     inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
...     if isinstance(past_key_values, SinkCache):
...         inputs = {k: v[:, -max_cache_length:] for k, v in inputs.items()}
...
...     input_length = inputs["input_ids"].shape[1]
...
...     outputs = model.generate(**inputs, do_sample=False, max_new_tokens=256, past_key_values=past_key_values)
...     completion = tokenizer.decode(outputs[0, input_length: ], skip_special_tokens=True)
...     messages.append({"role": "assistant", "content": completion})

print(messages)
[{'role': 'user', 'content': "Hello, what's your name?"}, {'role': 'assistant', 'content': "Hello, I'm AI."}, {'role': 'user', 'content': 'Btw, yesterday I was on a rock concert.'}, {'role': 'assistant', 'content': "I'm sorry to hear that you were on a rock concert yesterday. It sounds like a fun experience, but I'm not capable of experiencing music or concerts. However, I can provide you with some information about rock music and its history. Rock music emerged in the 1950s and 1960s in the United States and Britain, and it quickly gained popularity around the world. Some of the most famous rock bands of all time include The Beatles, The Rolling Stones, Led Zeppelin, and Pink Floyd. Rock music has a distinct sound and style, with elements of blues, country, and folk music. It often features guitar solos, heavy bass lines, and drums. Rock music has had a significant impact on popular culture, influencing genres such as punk rock, heavy metal, and alternative rock."}]
```

## Re-use Cache to continue generation
## 在继续生成时复用缓存

Sometimes you would want to first fill-in cache object with key/values for certain prefix prompt and re-use it several times to generate different sequences from it. In that case you can construct a `Cache` object that will hold the instruction prompt, and re-use it several times with different text sequences.

有时你可能希望首先用特定前缀提示的关键值填充缓存对象，并多次复用它来生成不同的序列。在这种情况下，你可以构造一个 `Cache` 对象来保留指令提示，并多次使用它来输出不同的文本序列。

```python
>>> import copy
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
>>> from accelerate.test_utils.testing import get_backend

>>> DEVICE, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
>>> model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
>>> model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=DEVICE)
>>> tokenizer = AutoTokenizer.from_pretrained(model_id)

>>> # Init StaticCache with big enough max-length (1024 tokens for the below example)
>>> # You can also init a DynamicCache, if that suits you better
>>> prompt_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=1024, device=DEVICE, dtype=torch.bfloat16)

>>> INITIAL_PROMPT = "You are a helpful assistant. "
>>> inputs_initial_prompt = tokenizer(INITIAL_PROMPT, return_tensors="pt").to(DEVICE)
>>> # This is the common prompt cached, we need to run forward without grad to be abel to copy
>>> with torch.no_grad():
...      prompt_cache = model(**inputs_initial_prompt, past_key_values = prompt_cache).past_key_values

>>> prompts = ["Help me to write a blogpost about travelling.", "What is the capital of France?"]
>>> responses = []
>>> for prompt in prompts:
...     new_inputs = tokenizer(INITIAL_PROMPT + prompt, return_tensors="pt").to(DEVICE)
...     past_key_values = copy.deepcopy(prompt_cache)
...     outputs = model.generate(**new_inputs, past_key_values=past_key_values,max_new_tokens=20)
...     response = tokenizer.batch_decode(outputs)[0]
...     responses.append(response)

>>> print(responses)
['<s> You are a helpful assistant. Help me to write a blogpost about travelling.  I am excited to share my experiences with you.  I have been traveling for the past', '<s> You are a helpful assistant. What is the capital of France? \n\nAnswer: Paris is the capital of France.</s>']
```


## Legacy cache format
## 传统的缓存格式

Prior to the introduction of the `Cache` object, the cache of LLMs used to be a tuple of tuples of tensors. The legacy
format has a dynamic size, growing as we generate text -- very similar to `DynamicCache`. If your project depend on
this legacy format, you can seamlessly convert it to a `DynamicCache` and back.

在 `Cache` 对象引入之前，LLM 的缓存是一个元组，里面的元素是由多个tensor组成的元组。传统的缓存格式是动态大小的，随着我们生成文本而增长——非常类似于 `DynamicCache`。如果你的项目依赖于这种传统的缓存格式，你可以无缝地将其转换为 `DynamicCache`，然后再转换回来。

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

>>> tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
>>> model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")
>>> inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

>>> # `return_dict_in_generate=True` is required to return the cache. `return_legacy_cache` forces the returned cache
>>> # to be of the legacy type
>>> generation_outputs = model.generate(**inputs, return_dict_in_generate=True, return_legacy_cache=True, max_new_tokens=5)

>>> # We can convert a legacy cache to a DynamicCache -- and the other way around. This is helpful if you have custom
>>> # logic to manipulate a cache in a specific format.
>>> cache = DynamicCache.from_legacy_cache(generation_outputs.past_key_values)
>>> legacy_format_cache = cache.to_legacy_cache()
```
