<!--版权2024年HuggingFace团队保留所有权利。

根据Apache许可证第2.0版（"许可证"）许可；除非符合许可证，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则按"按原样"分发的软件，无论是明示还是暗示的，都没有任何担保或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️ 请注意，本文件虽然使用Markdown编写，但包含了特定的语法，适用于我们的doc-builder（类似于MDX），可能无法在您的Markdown查看器中正常渲染。

-->

# 缓存策略

键值（KV）向量用于计算注意力分数。对于自回归模型，KV 分数*每次*都要计算，因为模型一次只预测一个词元。每个预测都依赖于之前的词元，这意味着模型每次都在重复相同的计算。

KV *缓存*（cache）把这些计算结果存储起来以便复用，无需重新计算。高效的缓存对于优化模型性能至关重要，因为它能减少计算时间并提高响应速度。关于缓存的工作原理，请参阅[缓存机制详解](./cache_explanation)文档以获得更详细的说明。

Transformers 提供了若干实现不同缓存机制的 [`Cache`] 类。其中一些 [`Cache`] 类为节省显存而优化，另一些则为最大化生成速度而设计。请参考下表比较各种缓存类型，以帮助你为自己的使用场景选择最合适的缓存。

| 缓存类型               | 支持滑动层（sliding layers） | 支持卸载（offloading） | 支持 torch.compile() | 预期显存占用 |
|------------------------|------------------------------|------------------------|----------------------|--------------|
| Dynamic Cache          | 是                           | 是                     | 否                   | 中等         |
| Static Cache           | 是                           | 是                     | 是                   | 高           |
| Quantized Cache        | 否                           | 否                     | 否                   | 低           |

本指南将介绍不同的 [`Cache`] 类，并展示如何在生成中使用它们。

## 默认缓存

[`DynamicCache`] 是所有模型的默认缓存类。它允许缓存大小动态增长，以便在生成过程中存储越来越多的键和值。

请注意，对于使用滑动窗口注意力（Mistral、Gemma2 等）或分块注意力（chunked attention，如 Llama4）的模型，当使用这些注意力类型的层达到其最大容量（滑动窗口或分块大小）时，缓存将停止增长。

可以在 [`~GenerationMixin.generate`] 中配置 `use_cache=False` 来禁用缓存。

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=False)
```

缓存类也可以先初始化，再传给模型的 [`~generation.GenerateDecoderOnlyOutput#past_key_values`] 参数。这在需要更细粒度控制或更高级的用法（例如上下文缓存）时很有用。

在大多数情况下，更简单的方式是在 [`~GenerationConfig#cache_implementation`] 参数中定义缓存策略。

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", dtype=torch.float16, device_map="auto")
inputs = tokenizer("I like rock music because", return_tensors="pt").to(model.device)

past_key_values = DynamicCache(config=model.config)
out = model.generate(**inputs, do_sample=False, max_new_tokens=20, past_key_values=past_key_values)
```

## 固定大小缓存

默认的 [`DynamicCache`] 会让你无法利用大多数即时编译（JIT）优化，因为缓存大小不是固定的。JIT 优化能让你以牺牲显存占用为代价来最小化延迟。以下所有缓存类型都与 [torch.compile](./perf_torch_compile) 等 JIT 优化兼容，可用于加速生成。

固定大小缓存（[`StaticCache`]）会为键值对预分配一个特定的最大缓存容量。在达到最大容量之前，你都可以直接生成而无需修改它。然而，键/值状态采用固定（通常很大）的大小意味着在生成过程中，很多词元实际上会被掩码掉，因为它们不应参与注意力计算。因此这个技巧可以让解码阶段轻松地被 `compile`，但会在注意力计算中浪费一部分词元。与所有事情一样，这是一种权衡：如果你生成的多条序列长度大致相同，它会非常划算；但如果你有一条非常长的序列、其余都是短序列，它可能并非最优（因为固定缓存容量很大，短序列会浪费很多）。使用前请务必理解其影响！

与 [`DynamicCache`] 一样，请注意：对于使用滑动窗口注意力（Mistral、Gemma2 等）或分块注意力（Llama4）的模型，即使指定的最大长度更大，使用这些注意力类型的层上的缓存也永远不会超过滑动窗口/分块大小。

你可以在 [`~GenerationMixin.generate`] 中配置 `cache_implementation="static"` 来启用 [`StaticCache`]。对于贪心（greedy）和采样（sample）解码策略，这还会自动开启解码阶段的`编译`。

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

## 将生成张量保留在 CPU 上

Neuron 和 TPU 等编译器后端会把你的模型追踪（trace）成固定的计算图。生成循环会维护一些每步增长一个词元的张量（输出序列、`attention_mask`、`position_ids`）。每当这些张量在加速器上改变形状时，编译器都会重新追踪计算图，从而拖慢生成速度。

[`~GenerationMixin.generate`] 只会在每次 `forward` 调用之前，把 `forward` 所消费的张量移动到模型设备上。输出则会被移回与输入相同的设备。把你的输入保留在 CPU 上，就能让循环中不断增长的张量簿记工作远离加速器。编译后的计算图保持稳定；又因为输出跟随输入设备，生成的输出也会保留在 CPU 上。

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")

# 让输入保留在 CPU 上，而不是调用 .to(model.device)。
inputs = tokenizer("The French Bread Law states", return_tensors="pt")

# generate 在模型设备上运行 forward，但把输出返回到 input_ids 所在的设备。
output = model.generate(**inputs, do_sample=False, max_new_tokens=20)
print(output.device)
cpu
```

## 缓存卸载

KV 缓存可能占据显存的很大一部分，成为长上下文生成的[瓶颈](https://hf.co/blog/llama31#inference-memory-requirements)。注重显存效率的缓存致力于以降低速度为代价来减少显存占用。对于大语言模型（LLM）以及硬件显存受限的情况，这一点尤为重要。

卸载缓存通过把除一层以外所有模型层的 KV 缓存移到 CPU 来节省 GPU 显存。在模型对各层进行 `forward` 迭代时，只有当前层的缓存保留在 GPU 上。它会异步预取下一层的缓存，并在注意力计算完成后把当前层的缓存送回 CPU。

如果你的 GPU 较小并且遇到显存不足（OOM）错误，可以考虑使用卸载。

> [!WARNING]
> 与完全在设备上的缓存相比，你可能会注意到生成吞吐量有轻微下降，具体取决于你的模型和生成选项（上下文大小、生成词元数、beam 数量等）。这是因为来回搬运键/值状态需要一些额外工作。

[`DynamicCache`] 和 [`StaticCache`] 都支持卸载。你可以在 [`GenerationConfig`] 或 [`~GenerationMixin.generate`] 中配置 `cache_implementation="offloaded"`（动态版本）或 `cache_implementation="offloaded_static"`（静态版本）来启用它。
此外，你也可以使用 `offloading=True` 选项直接实例化自己的 [`DynamicCache`] 或 [`StaticCache`]，并把这个缓存传给 `generate` 或模型的 `forward`（例如，对于动态缓存使用 `past_key_values=DynamicCache(config=model.config, offloading=True)`）。

请注意，上面提到的两个 [`Cache`] 类在直接实例化时还有一个额外选项 `offload_only_non_sliding`。
这个额外参数决定使用滑动窗口/分块注意力的层（如果有）是否也要被卸载。由于这些层通常本来就很短，
最好避免对它们卸载，因为卸载可能带来速度损失。默认情况下，该选项对 [`DynamicCache`] 为 `False`，对 [`StaticCache`] 为 `True`。

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

下面的例子展示了当显存耗尽时如何回退到卸载缓存：

```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

def resilient_generate(model, *args, **kwargs):
    oom = False
    device = Accelerator().device
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

## 量化缓存

[`QuantizedCache`] 通过把 KV 值量化到更低精度来降低显存需求。[`QuantizedCache`] 目前支持两种量化后端：

- `hqq` 支持 int2、int4 和 int8 数据类型。
- `quanto` 支持 int2 和 int4 数据类型。这是默认的量化后端。

> [!WARNING]
> 如果上下文长度较短，且 GPU 显存足够在不启用缓存量化的情况下完成生成，量化缓存可能会损害延迟。请在显存效率和延迟之间找到平衡。

在 [`GenerationConfig`] 中配置 `cache_implementation="quantized"` 即可启用 [`QuantizedCache`]；量化后端以及任何额外的量化相关参数也应以 dict 形式传入。除非你遇到显存不足，否则应使用这些额外参数的默认值。在显存不足的情况下，可以考虑减小残差长度（residual length）。

<hfoptions id="quantized-cache">

对于 `hqq` 后端，我们建议把 `axis-key` 和 `axis-value` 参数设为 `1`。

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

对于 `quanto` 后端，我们建议把 `axis-key` 和 `axis-value` 参数设为 `0`。

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

## 编码器-解码器缓存

[`EncoderDecoderCache`] 是为编码器-解码器模型设计的。它同时管理自注意力和交叉注意力缓存，以确保之前的键值对的存储与读取。可以为编码器和解码器分别单独设置不同的缓存类型。

这种缓存类型不需要任何设置。它只是对上述两个 [`Cache`] 的简单封装，模型会直接独立地使用它们。

## 模型特定缓存

一些模型有独特的过往键值对或状态的存储方式，与任何其他缓存类都不兼容。

Mamba 类模型，例如 [Mamba](./model_doc/mamba)，需要特定的缓存，因为模型没有注意力机制或键值状态。因此，它们与上述 [`Cache`] 类不兼容。

## 迭代式生成

缓存也可以用于迭代式生成场景，即与模型进行来回交互（聊天机器人）。与常规生成一样，带缓存的迭代式生成让模型能够高效地处理持续进行的对话，无需在每一步重新计算整个上下文。

要进行带缓存的迭代式生成，首先初始化一个空的缓存类，然后就可以输入你的新提示。用 [chat template](./chat_templating) 来记录对话历史。

下面的例子演示了 [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)。如果你使用的是其他对话风格的模型，[`~PreTrainedTokenizer.apply_chat_template`] 处理消息的方式可能不同。根据 Jinja 模板的写法，它可能会截掉一些重要的词元。对于多模态聊天模型，请参阅[带缓存的迭代聊天](./tasks/image_text_to_text.md#iterative-chatting-with-cache)指南了解如何处理图像或音频。

例如，一些模型在推理过程中使用特殊的 `<think> ... </think>` 词元。这些词元在重新编码时可能丢失，导致索引问题。你可能需要手动移除或调整补全内容中多余的词元，以保持稳定性。

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

## 裁剪缓存

使用 [`~Cache.crop`] 可以从缓存中回滚词元，例如在迭代式生成中某个候选续写被拒绝时。把要移除的词元数量以负整数形式传入。

```py
# 从缓存中移除最后三个词元。
past_key_values.crop(-3)
```

滑动窗口层和线性注意力层会在前进过程中丢弃过往状态，因此默认情况下没有可回滚的内容。在 forward 前调用 [`~Cache.activate_past_recording`]，可以让它们保留这些状态。否则，一旦滑动窗口层填满了窗口，`crop` 就会抛出 `RuntimeError`；对于线性注意力层则会立即抛出。

```py
# 如果需要回滚词元，请在 forward 之前启用它。
past_key_values.activate_past_recording()
```

`crop(0)` 不会清空缓存。它不会从缓存序列中移除任何词元，对全注意力层来说是空操作。对于滑动窗口层和线性注意力层，它可以丢弃下一次模型调用不会用到的缓存信息，从而在不回滚词元的情况下降低显存占用。

```
                    缓存位置
                    ┌───┬───┬───┬───┬───┬───┐
crop(-3) 之前       │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
                    └───┴───┴───┴───┴───┴───┘
                                ╰─────┬─────╯
                                  回滚部分

crop(-3) 之后
                    ┌───┬───┬───┐
全注意力            │ 0 │ 1 │ 2 │    保留整个前缀
                    └───┴───┴───┘
                            ┌───┐
sliding_window=2            │ 2 │    保留 sliding_window - 1
                            └───┘
```

现在两种层报告的序列长度都是 3，生成将从位置 3 继续进行。

传入 `0` 不会从序列中移除任何词元。它对全注意力层是空操作，但对滑动窗口层和线性注意力层，它仍会丢弃下一次 forward 不会读取的已记录状态，从而在不回滚任何内容的情况下释放显存。

## 预填充缓存（前缀缓存）

在某些情况下，你可能希望为某个前缀提示预先填充一个 [`Cache`] 的键值对，并复用它来生成不同的序列。

下面的例子初始化一个 [`StaticCache`]，然后缓存一个初始提示。之后你就可以从这个预填充的提示出发生成多条序列。

```py
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 用足够大的最大长度初始化 StaticCache（下面的例子需要 1024 个词元）
# 如果更适合你，也可以初始化 DynamicCache
prompt_cache = StaticCache(config=model.config, max_cache_len=1024)

INITIAL_PROMPT = "You are a helpful assistant. "
inputs_initial_prompt = tokenizer(INITIAL_PROMPT, return_tensors="pt").to(model.device.type)
# 这是被缓存的公共提示，我们需要在无梯度模式下运行 forward 以便复制缓存
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
