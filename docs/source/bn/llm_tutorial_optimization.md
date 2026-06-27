<!--Copyright 2023 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# স্পিড আর মেমোরির জন্য LLM অপ্টিমাইজ করা

[[open-in-colab]]

GPT3/4, [Falcon](https://huggingface.co/tiiuae/falcon-40b), আর [Llama](https://huggingface.co/meta-llama/Llama-2-70b-hf)-এর মতো Large Language Model (LLM) গুলো খুব দ্রুত উন্নতি করছে এবং মানুষের মতো complex task handle করতে পারছে। তাই এখন এগুলো knowledge-based industry-তে খুব গুরুত্বপূর্ণ tool হয়ে উঠেছে।

তবে real-world application-এ এই model deploy করা এখনও challenging।

- মানুষের মতো text বুঝতে আর generate করতে LLM-গুলোর billions of parameters দরকার হয়। ফলে inference-এর সময় memory usage অনেক বেড়ে যায়।
- অনেক task-এ model-কে huge context handle করতে হয়। তাই inference-এর সময় long input sequence manage করতে পারা খুব জরুরি।

মূল challenge হলো computational power আর memory efficiency improve করা, বিশেষ করে যখন input sequence অনেক বড় হয়।

এই গাইডে আমরা efficient LLM deployment-এর জন্য কয়েকটা powerful technique দেখবো:

1. **Lower Precision:** Research দেখিয়েছে যে [8-bit আর 4-bit](./main_classes/quantization) precision ব্যবহার করলে performance প্রায় একই রেখে computation অনেক efficient করা যায়।

2. **Flash Attention:** Flash Attention হলো attention algorithm-এর optimized version যা memory usage কমায় আর GPU memory আরও efficiently ব্যবহার করে speed বাড়ায়।

3. **Architectural Innovations:** LLM সাধারণত autoregressive text generation-এর জন্য ব্যবহার হয় যেখানে long context handle করতে হয়। এজন্য নতুন architecture design এসেছে যেমন [Alibi](https://huggingface.co/papers/2108.12409), [Rotary embeddings](https://huggingface.co/papers/2104.09864), [MQA](https://huggingface.co/papers/1911.02150), আর [GQA](https://huggingface.co/papers/2305.13245)।

এই গাইডে আমরা tensor perspective থেকে autoregressive generation analyse করবো, lower precision-এর pros-cons দেখবো, attention algorithm explore করবো আর modern architecture discuss করবো।

## 1. Lower Precision

LLM-এর memory requirement বুঝতে সবচেয়ে সহজ উপায় হলো model-কে weight matrix আর vector-এর collection হিসেবে দেখা।

আজকের দিনে LLM-এ billions of parameter থাকে। প্রতিটি parameter decimal number হিসেবে store হয়, যেমন `4.5689`। সাধারণত এগুলো `float32`, `bfloat16`, বা `float16` format-এ থাকে।

এখন খুব সহজেই memory estimate করা যায়:

> *X billion parameter-এর model float32 precision-এ load করতে প্রায় 4 × X GB VRAM লাগে*

আজকাল বেশিরভাগ model `bfloat16` বা `float16`-এ train করা হয়। তাই নতুন rule of thumb:

> *X billion parameter-এর model bfloat16/float16 precision-এ load করতে প্রায় 2 × X GB VRAM লাগে*

Short input sequence-এর ক্ষেত্রে inference memory মূলত model weight load করার memory দিয়েই dominated হয়।

উদাহরণ:

- **GPT3** → প্রায় **350 GB** VRAM
- **Bloom** → প্রায় **352 GB**
- **Llama-2-70b** → প্রায় **140 GB**
- **Falcon-40b** → প্রায় **80 GB**
- **MPT-30b** → প্রায় **60 GB**
- **Starcoder** → প্রায় **31 GB**

আজকের সবচেয়ে বড় GPU যেমন A100 বা H100-এও 80GB VRAM থাকে। তাই অনেক model run করতে tensor parallelism বা pipeline parallelism দরকার হয়।

Tensor parallelism এখন 🤗 Transformers-এ support করে।

Naive pipeline parallelism ব্যবহার করতে শুধু `device_map="auto"` দিলেই হবে।

```bash
!pip install transformers accelerate bitsandbytes optimum
```

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", device_map="auto", pad_token_id=0)
```

এখানে attention layer গুলো automatically available GPU-তে distribute হয়ে যাবে।

এই guide-এ আমরা `bigcode/octocoder` ব্যবহার করবো কারণ এটি single 40GB GPU-তেও run করা যায়।

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", dtype=torch.bfloat16, device_map="auto", pad_token_id=0)
tokenizer = AutoTokenizer.from_pretrained("bigcode/octocoder")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

```python
prompt = "Question: Please write a function in Python that transforms bytes to Giga bytes.\n\nAnswer:"

result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
result
```

Output:

```text
Here is a Python function that transforms bytes to Giga bytes:
```

এখন VRAM usage check করি।

```python
def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024
```

```python
bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
```

Output:

```text
29.0260648727417
```

প্রায় 29GB VRAM লাগছে, যা আমাদের estimate-এর কাছাকাছি।

> বেশিরভাগ model এখন bfloat16-এ train করা হয়। তাই GPU support করলে float32 ব্যবহার করার বিশেষ দরকার নেই।

Checkpoint-এর config file দেখে model কোন dtype-এ train হয়েছে তা জানা যায়।

Memory clear করার জন্য:

```python
del pipe
del model

import gc
import torch

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
```

```python
flush()
```

Accelerate library-তে `release_memory()` utility-ও আছে।

```python
from accelerate.utils import release_memory

release_memory(model)
```

## 8-bit Quantization

যদি GPU-তে যথেষ্ট VRAM না থাকে তাহলে model quantization ব্যবহার করা যায়।

Research দেখিয়েছে model weight-কে 8-bit বা 4-bit-এ convert করলেও performance খুব বেশি degrade করে না।

Quantization মূলত weight-এর precision কমিয়ে memory usage কমায়।

```bash
!pip install bitsandbytes
```

8-bit quantization:

```python
model = AutoModelForCausalLM.from_pretrained(
    "bigcode/octocoder",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    pad_token_id=0
)
```

এখন আবার inference run করি।

```python
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
result
```

Memory usage:

```python
bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
```

Output:

```text
15.219234466552734
```

দেখতেই পাচ্ছো VRAM প্রায় অর্ধেক হয়ে গেছে!

তবে inference একটু slow হতে পারে।

```python
del model
del pipe
flush()
```

## 4-bit Quantization

4-bit quantization:

```python
model = AutoModelForCausalLM.from_pretrained(
    "bigcode/octocoder",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    pad_token_id=0
)
```

```python
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
result
```

Memory usage:

```python
bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
```

Output:

```text
9.543574333190918
```

মাত্র 9.5GB VRAM!

এখন RTX3090, V100, T4-এর মতো GPU-তেও model run করা সম্ভব।

তবে 4-bit quantization-এ accuracy কিছুটা degrade হতে পারে।

> Quantization মূলত memory efficiency আর accuracy-এর মধ্যে tradeoff।

যদি GPU memory problem না হয় তাহলে quantization জরুরি না। কিন্তু consumer GPU-তে LLM চালাতে এটি অনেক useful।

## 2. Flash Attention

আজকের প্রায় সব powerful LLM একই ধরনের architecture ব্যবহার করে:

- Feed-forward layer
- Activation layer
- Layer normalization
- Self-attention layer

এর মধ্যে self-attention সবচেয়ে গুরুত্বপূর্ণ কারণ এটি token-গুলোর contextual relationship বুঝতে সাহায্য করে।

কিন্তু self-attention-এর memory complexity quadratic হয়।

Input token সংখ্যা যদি `N` হয় তাহলে memory আর compute complexity প্রায় `N²` হয়।

Short sequence-এর জন্য সমস্যা না হলেও long context-এর জন্য এটি huge bottleneck হয়ে যায়।

Self-attention formula:

:contentReference[oaicite:0]{index=0}

`QKᵀ` matrix-এর size `N²` হওয়ায় memory usage explode করে।

উদাহরণ:

- `N = 1000` → ~50MB
- `N = 16000` → ~19GB
- `N = 100000` → প্রায় 1TB

এটাই long-context handling-এর biggest problem।

Flash Attention এই সমস্যা solve করে।

এটি পুরো `QKᵀ` matrix store না করে chunk-wise computation করে।

মূল সুবিধা:

> Flash Attention mathematically একই output দেয় কিন্তু memory complexity linear করে দেয়।

আরও ভালো বিষয় হলো এটি inference speed-ও বাড়ায় কারণ এটি GPU-এর fast SRAM বেশি ব্যবহার করে আর slow VRAM access কমায়।

> তাই Flash Attention available থাকলে প্রায় সবসময়ই এটি ব্যবহার করা উচিত।

## 3. Architectural Innovations

এখন আমরা architecture-level optimization দেখবো।

বিশেষ করে long-context task-এর জন্য:

- Retrieval QA
- Summarization
- Chat

দুইটা বড় bottleneck:

- Positional embeddings
- Key-value cache

---

## 3.1 Positional Embeddings

Self-attention token-গুলোকে একে অপরের সাথে relate করে।

কিন্তু positional embedding ছাড়া model sentence order বুঝতে পারবে না।

উদাহরণ:

- `"Hello I love you"`
- `"You love I hello"`

দুটো sentence-এর meaning completely different।

এজন্য positional encoding দরকার হয়।

আগে sinusoidal বা learned positional embedding ব্যবহার করা হতো।

কিন্তু এগুলোর সমস্যা:

1. Long input sequence-এ performance degrade হয়
2. Fixed input length-এর বাইরে extrapolate করা কঠিন

এখন relative positional embedding বেশি জনপ্রিয়:

- [RoPE](https://huggingface.co/papers/2104.09864)
- [ALiBi](https://huggingface.co/papers/2108.12409)

RoPE positional information directly attention computation-এ inject করে।

:contentReference[oaicite:1]{index=1}

RoPE ব্যবহার করে:

- Falcon
- Llama
- PaLM

ALiBi আরও simple approach নেয়।

এটি relative distance-কে negative bias হিসেবে attention score-এ যোগ করে।

ALiBi ব্যবহার করে:

- MPT
- BLOOM

দুটোই long-context extrapolation-এর জন্য useful।

---

## 3.2 Key-Value Cache

Auto-regressive generation step-by-step token generate করে।

```python
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

for _ in range(5):
  next_logits = model(input_ids)["logits"][:, -1:]
  next_token_id = torch.argmax(next_logits,dim=-1)

  input_ids = torch.cat([input_ids, next_token_id], dim=-1)
```

কিন্তু প্রতিবার পুরো attention recompute করা inefficient।

তাই key-value cache ব্যবহার করা হয়।

```python
past_key_values = None
generated_tokens = []
next_token_id = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

for _ in range(5):
  next_logits, past_key_values = model(
      next_token_id,
      past_key_values=past_key_values,
      use_cache=True
  ).to_tuple()
```

এখানে previous key-value vector reuse হয়।

ফলে:

- Speed বাড়ে
- Memory usage linear হয়

> তাই সম্ভব হলে সবসময় key-value cache ব্যবহার করা উচিত।

---

## Multi-round Conversation

Chat application-এ key-value cache অনেক useful।

কারণ পুরো chat history বারবার encode করতে হয় না।

```text
User: How many people live in France?
Assistant: Roughly 75 million people live in France

User: And how many are in Germany?
Assistant: Germany has ca. 81 million inhabitants
```

দ্বিতীয় প্রশ্নে প্রথম conversation-এর cache reuse হয়।

---

## Multi-Query Attention (MQA)

MQA-তে সব attention head-এর জন্য same key-value projection ব্যবহার করা হয়।

ফলে:

- KV cache অনেক ছোট হয়
- Memory bandwidth কম লাগে
- Inference speed বাড়ে

MQA ব্যবহার করে:

- Falcon
- PaLM
- MPT
- BLOOM

---

## Grouped Query Attention (GQA)

GQA হলো MQA-এর improved version।

একটি মাত্র KV head ব্যবহার না করে কয়েকটি group ব্যবহার করা হয়।

এতে:

- Memory efficiency প্রায় একই থাকে
- কিন্তু quality loss কম হয়

Llama-2 GQA ব্যবহার করে।

---

# Conclusion

Research community constantly নতুন optimization technique বের করছে।

একটা promising direction হলো speculative decoding যেখানে ছোট model সহজ token generate করে আর বড় model difficult token generate করে।

আজ GPT4, Claude, Llama-2, PaLM এত দ্রুত কাজ করতে পারে কারণ:

- Lower precision
- Flash Attention
- Better architecture
- KV cache
- MQA/GQA

সব optimisation একসাথে ব্যবহার করা হয়।

ভবিষ্যতে GPU আরও powerful হবে, কিন্তু efficient algorithm আর architecture ব্যবহার করা এখনও সবচেয়ে গুরুত্বপূর্ণ 🤗
