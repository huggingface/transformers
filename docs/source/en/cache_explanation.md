<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Caching

Imagine you're having a conversation with someone, and instead of remembering what they previously said, they have to start from scratch every time you respond. This would be slow and inefficient, right?

You can extend this analogy to transformer models. Autoregressive model generation can be slow because it makes a prediction one token at a time. Each new prediction is dependent on all the previous context.

To predict the 1000th token, the model requires information from the previous 999 tokens. The information is represented as matrix multiplications across the token representations.

To predict the 1001th token, you need the same information from the previous 999 tokens in addition to any information from the 1000th token. This is a lot of matrix multiplications a model has to compute over and over for each token!

A key-value (KV) cache eliminates this inefficiency by storing kv pairs derived from the attention layers of previously processed tokens. The stored kv pairs are retrieved from the cache and reused for subsequent tokens, avoiding the need to recompute.

> [!WARNING]
> Caching should only be used for **inference**. It may cause unexpected errors if it's enabled during training.

To better understand how and why caching works, let's take a closer look at the structure of the attention matrices.

## Attention matrices

The **scaled dot-product attention** is calculated as shown below for a batch of size `b`, number of attention heads `h`, sequence length so far `T`, and dimension per attention head `d_head`.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_{\text{head}}}} \times \text{mask} \right) V
$$

The query (`Q`), key (`K`), and value (`V`) matrices are projections from the input embeddings of shape `(b, h, T, d_head)`.

For causal attention, the mask prevents the model from attending to future tokens. Once a token is processed, its representation never changes with respect to future tokens, which means \\( K_{\text{past}} \\) and \\( V_{\text{past}} \\) can be cached and reused to compute the last token's representation.

$$
\text{Attention}(q_t, [\underbrace{k_1, k_2, \dots, k_{t-1}}_{\text{cached}}, k_{t}], [\underbrace{v_1, v_2, \dots, v_{t-1}}_{\text{cached}}, v_{t}])
$$

At inference time, you only need the last token's query to compute the representation \\( x_t \\) that predicts the next token \\( t+1 \\). At each step, the new key and value vectors are **stored** in the cache and **appended** to the past keys and values.

$$
K_{\text{cache}} \leftarrow \text{concat}(K_{\text{past}}, k_t), \quad V_{\text{cache}} \leftarrow \text{concat}(V_{\text{past}}, v_t)
$$

Attention is calculated independently in each layer of the model, and caching is done on a per-layer basis.

Refer to the table below to compare how caching improves efficiency.

| without caching | with caching |
|---|---|
| for each step, recompute all previous `K` and `V`  | for each step, only compute current `K` and `V` 
| attention cost per step is **quadratic** with sequence length | attention cost per step is **linear** with sequence length (memory grows linearly, but compute/token remains low) |



## Cache class

A basic KV cache interface takes a key and value tensor for the current token and returns the updated `K` and `V` tensors. This is internally managed by a model's `forward` method.

```py
new_K, new_V = cache.update(k_t, v_t, layer_idx)
attn_output = attn_layer_idx_fn(q_t, new_K, new_V)
```

When you use Transformers' [`Cache`] class, the self-attention module performs several critical steps to integrate past and present information.

1. The attention module concatenates current kv pairs with past kv pairs stored in the cache. This creates attentions weights with the shape `(new_tokens_length, past_kv_length + new_tokens_length)`. The current and past kv pairs are essentially combined to compute the attention scores, ensuring a model is aware of previous context and the current input.

2. When the `forward` method is called iteratively, it's crucial that the attention mask shape matches the combined length of the past and current kv pairs. The attention mask should have the shape `(batch_size, past_kv_length + new_tokens_length)`. This is typically handled internally in [`~GenerationMixin.generate`], but if you want to implement your own generation loop with [`Cache`], keep this in mind! The attention mask should hold the past and current token values.

3. It is also important to be aware of the `cache_position`. This is important if you want to reuse a prefilled [`Cache`] with the `forward` method because you have to pass a valid `cache_position` value. This indicates the input positions in a sequence. `cache_position` is unaffected by padding, and it always adds one more position for each token. For example, if a kv cache contains 10 tokens - regardless of pad tokens - the cache position for the next token should be `torch.tensor([10])`.

## Cache storage implementation

Caches are structured as a list of layers, where each layer contains a key and value cache. The key and value caches are tensors with the shape `[batch_size, num_heads, seq_len, head_dim]`.

Layers can be of different types (e.g. `DynamicLayer`, `StaticLayer`, `StaticSlidingWindowLayer`), which mostly changes how sequence length is handled and how the cache is updated.

The simplest is a `DynamicLayer` that grows as more tokens are processed. The sequence length dimension (`seq_len`) increases with each new token:

```py
cache.layers[idx].keys = torch.cat([cache.layers[idx].keys, key_states], dim=-2)
cache.layers[idx].values = torch.cat([cache.layers[idx].values, value_states], dim=-2)
```

Other layer types like `StaticLayer` and `StaticSlidingWindowLayer` have a fixed sequence length that is set when the cache is created. This makes them compatible with `torch.compile`. In the case of `StaticSlidingWindowLayer`, existing tokens are shifted out of the cache when a new token is added.

The example below demonstrates how to create a generation loop with [`DynamicCache`]. As discussed, the attention mask is a concatenation of past and current token values and `1` is added to the cache position for the next token.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, infer_device

device = f"{infer_device()}:0"

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

past_key_values = DynamicCache(config=model.config)
messages = [{"role": "user", "content": "Hello, what's your name."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

generated_ids = inputs.input_ids
cache_position = torch.arange(inputs.input_ids.shape[1], dtype=torch.int64, device=model.device)
max_new_tokens = 10

for _ in range(max_new_tokens):
    outputs = model(**inputs, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
    # Greedily sample one next token
    next_token_ids = outputs.logits[:, -1:].argmax(-1)
    generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
    # Prepare inputs for the next generation step by leaving unprocessed tokens, in our case we have only one new token
    # and expanding attn mask for the new token, as explained above
    attention_mask = inputs["attention_mask"]
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}
    cache_position = cache_position[-1:] + 1 # add one more position for the next token

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
"[INST] Hello, what's your name. [/INST]  Hello! My name is LLaMA,"
```

## Cache position

The cache position tracks where to insert new tokens in the attention cache. It represents the *absolute* position of each token in the context, independent of padding or batch structure. Suppose you already cached `N` tokens and are now processing `K` new tokens. The cache position for the new tokens will range from `N` to `N + K - 1`. In other words, you're processing tokens at positions - `[N, N + 1, N + 2, ..., N + K - 1]`.

Cache position is used internally for two purposes:

1. Selecting new tokens to process in the input sequence and ensuring only tokens that haven’t been cached yet are passed to the model's `forward`.
2. Storing key/value pairs at the correct positions in the cache. This is especially important for fixed-size caches, that pre-allocates a specific cache length.

The generation loop usually takes care of the cache position, but if you're writing a custom generation method, it is important that cache positions are accurate since they are used to write and read key/value states into fixed slots.


```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, infer_device

device = f"{infer_device()}:0"

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "You are a helpful assistant."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=10)

```


## Legacy cache format

Before the [`Cache`] class, the cache used to be stored as a tuple of tuples of tensors. This format is dynamic because it grows as text is generated, similar to [`DynamicCache`].

The legacy format is essentially the same data structure but organized differently.
- It's a tuple of tuples, where each inner tuple contains the key and value tensors for a layer.
- The tensors have the same shape `[batch_size, num_heads, seq_len, head_dim]`.
- The format is less flexible and doesn't support features like quantization or offloading.

If your project depends on this legacy format, we recommend to convert to [`DynamicCache`] with [`~DynamicCache.from_legacy_cache`]. Note that legacy cache format is deprecated and not used anymore in `Transformers`. You can convert back to tuple format with [`DynamicCache.to_legacy_cache`] functions, which is helpful if you have custom logic for manipulating a cache in a specific format.

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)

# `return_dict_in_generate=True` is required to return the cache and `return_legacy_cache` forces the returned cache
# in the legacy format
generation_outputs = model.generate(**inputs, return_dict_in_generate=True, return_legacy_cache=True, max_new_tokens=5)

cache = DynamicCache.from_legacy_cache(generation_outputs.past_key_values)
legacy_format_cache = cache.to_legacy_cache()
```
