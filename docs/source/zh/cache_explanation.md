<!--版权2024年HuggingFace团队保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则按“按原样”分发的软件，无论是明示还是暗示的，都没有任何担保或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️ 请注意，本文件虽然使用Markdown编写，但包含了特定的语法，适用于我们的doc-builder（类似于MDX），可能无法在您的Markdown查看器中正常渲染。

-->

# 缓存的工作原理

想象一下，你正在和某人交谈，而对方不记得你们之前说过的任何内容，每当你回应时，他们都得从头开始。这又慢又低效，对吧？

这个类比可以延伸到 transformer 模型上。自回归模型的生成过程可能很慢，因为它一次只预测一个词符（token）。每一个新的预测都依赖于之前的全部上下文。

要预测第 1000 个词符，模型需要前 999 个词符的信息。这些信息以词符表示之间的矩阵乘法形式体现。

要预测第 1001 个词符，你同样需要前 999 个词符的信息，再加上第 1000 个词符的信息。模型每生成一个词符都要重复计算如此多的矩阵乘法！

键值（KV）缓存消除了这种低效：它存储由先前处理过的词符在注意力层中导出的 kv 对。存储的 kv 对会从缓存中取出并复用于后续词符，从而避免重复计算。

> [!WARNING]
> 缓存只应用于**推理**。如果在训练时启用，可能会导致意料之外的错误。

为了更好地理解缓存的工作方式及其原因，让我们仔细看看注意力矩阵的结构。

## 注意力矩阵

**缩放点积注意力**的计算方式如下所示，其中批次大小为 `b`，注意力头数为 `h`，当前序列长度为 `T`，每个注意力头的维度为 `d_head`。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_{\text{head}}}} \times \text{mask} \right) V
$$

查询（`Q`）、键（`K`）、值（`V`）矩阵是输入嵌入的投影，形状为 `(b, h, T, d_head)`。

对于因果注意力，掩码会阻止模型关注未来的词符。一个词符一旦被处理，它的表示就不会再随未来的词符而改变，这意味着 $ K_{\text{past}} $ 和 $ V_{\text{past}} $ 可以被缓存并复用于计算最后一个词符的表示。

$$
\text{Attention}(q_t, [\underbrace{k_1, k_2, \dots, k_{t-1}}_{\text{cached}}, k_{t}], [\underbrace{v_1, v_2, \dots, v_{t-1}}_{\text{cached}}, v_{t}])
$$

在推理时，你只需要最后一个词符的查询来计算用于预测下一个词符 $ t+1 $ 的表示 $ x_t $。每一步，新的键和值向量都会被**存入**缓存，并**追加**到过去的键和值之后。

$$
K_{\text{cache}} \leftarrow \text{concat}(K_{\text{past}}, k_t), \quad V_{\text{cache}} \leftarrow \text{concat}(V_{\text{past}}, v_t)
$$

注意力在模型的每一层中独立计算，缓存也是逐层进行的。

参考下表，比较缓存如何提升效率。

| 不使用缓存 | 使用缓存 |
|---|---|
| 每一步都要重新计算之前所有的 `K` 和 `V` | 每一步只计算当前的 `K` 和 `V` |
| 每一步的注意力开销与序列长度呈**平方**关系 | 每一步的注意力开销与序列长度呈**线性**关系（内存线性增长，但每个词符的计算量保持很低） |

## Cache 类

一个基础的 KV 缓存接口接收当前词符的键和值张量，并返回更新后的 `K` 和 `V` 张量。这由模型的 `forward` 方法在内部管理。

```py
new_K, new_V = cache.update(k_t, v_t, layer_idx)
attn_output = attn_layer_idx_fn(q_t, new_K, new_V)
```

当你使用 Transformers 的 [`Cache`] 类时，自注意力模块会执行几个关键步骤来整合过去与当前的信息。

1. 注意力模块将当前的 kv 对与缓存中存储的过去 kv 对拼接起来。这会生成形状为 `(new_tokens_length, past_kv_length + new_tokens_length)` 的注意力权重。当前与过去的 kv 对实际上被组合起来计算注意力分数，确保模型既能感知先前的上下文，也能感知当前的输入。

2. 当 `forward` 方法被迭代调用时，注意力掩码的形状必须与过去和当前 kv 对的组合长度相匹配，这一点至关重要。注意力掩码的形状应为 `(batch_size, past_kv_length + new_tokens_length)`。这通常在 [`~GenerationMixin.generate`] 内部处理，但如果你想用 [`Cache`] 实现自己的生成循环，请牢记这一点！注意力掩码应当同时覆盖过去和当前的词符。

## 缓存的存储实现

缓存被组织为一个层列表，其中每一层包含一个键缓存和一个值缓存。键缓存和值缓存都是形状为 `[batch_size, num_heads, seq_len, head_dim]` 的张量。

层可以有不同的类型（例如 `DynamicLayer`、`StaticLayer`、`StaticSlidingWindowLayer`），其区别主要在于序列长度的处理方式和缓存的更新方式。

最简单的是 `DynamicLayer`，它随着更多词符被处理而增长。序列长度维度（`seq_len`）随每个新词符增加：

```py
cache.layers[idx].keys = torch.cat([cache.layers[idx].keys, key_states], dim=-2)
cache.layers[idx].values = torch.cat([cache.layers[idx].values, value_states], dim=-2)
```

其他层类型，如 `StaticLayer` 和 `StaticSlidingWindowLayer`，具有在创建缓存时设定的固定序列长度。这使它们与 `torch.compile` 兼容。对于 `StaticSlidingWindowLayer`，当新词符加入时，已有的词符会被移出缓存。

下面的示例演示了如何用 [`DynamicCache`] 创建一个生成循环。如前所述，注意力掩码是过去与当前词符的拼接。

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from accelerate import Accelerator

device = Accelerator().device

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

past_key_values = DynamicCache(config=model.config)
messages = [{"role": "user", "content": "Hello, what's your name."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)

generated_ids = inputs.input_ids
max_new_tokens = 10

for _ in range(max_new_tokens):
    outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    # Greedily sample one next token
    next_token_ids = outputs.logits[:, -1:].argmax(-1)
    generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
    # Prepare inputs for the next generation step by leaving unprocessed tokens, in our case we have only one new token
    # and expanding attn mask for the new token, as explained above
    attention_mask = inputs["attention_mask"]
    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
    inputs = {"input_ids": next_token_ids, "attention_mask": attention_mask}

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
"[INST] Hello, what's your name. [/INST]  Hello! My name is LLaMA,"
```
