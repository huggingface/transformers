<!--版权2025年HuggingFace团队保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则按“按原样”分发的软件，无论是明示还是暗示的，都没有任何担保或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️ 请注意，本文件虽然使用Markdown编写，但包含了特定的语法，适用于我们的doc-builder（类似于MDX），可能无法在您的Markdown查看器中正常渲染。

-->

# 注意力后端

所有注意力实现执行的计算都是相同的：每个词符（token）都要与其他所有词符进行比较。区别在于计算*如何*执行。基础注意力会在内存中实例化完整的注意力矩阵，扩展性很差，从而形成拖慢推理的瓶颈。优化的实现会重新组织数学运算以减少内存搬运，从而实现更快、更低成本的推理。

[`AttentionInterface`] 提供了优化的注意力实现。它将注意力实现与模型实现解耦，简化了不同注意力函数之间的实验切换。借助这个统一的接口，可以轻松添加新的后端。

<table>
<tr><th>注意力后端</th><th>说明</th></tr>
<tr><td><code>"flash_attention_3"</code></td><td>在 FlashAttention-2 的基础上进一步重叠各操作，并将前向与反向传播更紧密地融合</td></tr>
<tr><td><code>"flash_attention_2"</code></td><td>将计算切分为更小的块，并使用高速片上内存</td></tr>
<tr><td><code>"flex_attention"</code></td><td>无需手写底层内核即可指定自定义注意力模式（稀疏、块局部、滑动窗口）的框架</td></tr>
<tr><td><code>"sdpa"</code></td><td>PyTorch 内置的<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html">缩放点积注意力</a>实现</td></tr>
<tr><td><code>"paged&#124;flash_attention_3"</code></td><td>FlashAttention-3 的分页版本</td></tr>
<tr><td><code>"paged&#124;flash_attention_2"</code></td><td>FlashAttention-2 的分页版本</td></tr>
<tr><td><code>"paged&#124;sdpa"</code></td><td>SDPA 的分页版本</td></tr>
<tr><td><code>"paged&#124;eager"</code></td><td>eager 的分页版本</td></tr>
</table>

## 设置注意力后端

使用 [`~PreTrainedModel.from_pretrained`] 的 `attn_implementation` 参数，以特定的注意力函数实例化模型。

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", attn_implementation="flash_attention_2"
)
```

使用 [`~PreTrainedModel.set_attn_implementation`] 可以在运行时不重新加载模型即可切换注意力后端。

```py
model.set_attn_implementation("sdpa")
```

### Kernels

借助 [Kernels](https://huggingface.co/docs/kernels/index) 库，可以在运行时直接从 [Hub](https://huggingface.co/models?other=kernels) 下载并加载编译好的计算内核。这避免了 PyTorch 或 CUDA 版本不匹配带来的打包问题。

Kernels 在检测到时会自动注册到 [`AttentionInterface`]。你不需要显式安装 FlashAttention 包。按名称请求 FlashAttention 时也会回退到 Hub 内核，参见 [FlashAttention 回退](./kernel_doc/loading_kernels#flashattention-fallback)。

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", attn_implementation="kernels-community/flash-attn2"
)
```

### SDPA 上下文管理器

PyTorch 的缩放点积注意力（SDPA）会自动为 CUDA 后端选择最快的注意力函数。对于其他后端，它默认使用 PyTorch 的 C++ 实现。

使用 [torch.nn.attention.sdpa_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) 上下文管理器可以强制 SDPA 使用特定的实现。

```py
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", attn_implementation="sdpa"
)

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)
```

## 按骨干网络分别设置注意力

多模态模型为每种模态使用不同的骨干网络。为每个骨干网络分配特定的注意力函数可以优化性能。例如，某些视觉骨干网络在 fp32 下表现更好，而 FlashAttention 并不支持 fp32。

用一个字典将视觉骨干网络映射到不同的注意力函数，同时让文本骨干网络继续使用 FlashAttention。注意力实现字典中的键必须与子配置的名称匹配。

```py
from transformers import AutoModelForImageTextToText

attention_implementation_per_backbone = {"vision_config": "sdpa", "text_config": "flash_attention_2"}

for key in attention_implementation_per_backbone:
    assert key in model.config.sub_configs, f"Invalid key in `attention_implementation`"

model = AutoModelForImageTextToText.from_pretrained(
    "facebook/chameleon-7b", attn_implementation=attention_implementation_per_backbone
)
```

在字典中省略某些骨干网络，它们将使用默认的注意力函数（SDPA）。

```py
model = AutoModelForImageTextToText.from_pretrained(
    "facebook/chameleon-7b", attn_implementation={"text_config": "flash_attention_2"}
)
```

用单个字符串可以为所有骨干网络设置相同的注意力函数。

```py
model = AutoModelForImageTextToText.from_pretrained(
    "facebook/chameleon-7b", attn_implementation="eager"
)
```

用空键可以全局设置注意力函数。

```py
model = AutoModelForImageTextToText.from_pretrained(
    "facebook/chameleon-7b", attn_implementation={"": "eager"}
)
```

## 创建新的注意力函数

通过 [`AttentionInterface.register`] 将自定义或新建的注意力函数添加到注意力注册表。模型通过 `attn_implementation` 参数使用这些函数。

> [!WARNING]  
> 注册自定义注意力函数时，请同时注册匹配的注意力掩码函数。如果自定义的 `attn_implementation` 名称没有在 [`AttentionMaskInterface`] 中注册，Transformers 会跳过掩码创建，并向注意力层传入 `attention_mask=None`。此时你的注意力函数必须自行处理因果、填充、打包或滑动窗口等约束，否则这些约束可能被静默丢弃。

下面的示例自定义了一个注意力函数，为每一层打印一条语句。它通过注册 `masking_utils.sdpa_mask` 作为注意力掩码函数，保留了原实现中的掩码。

```python
import torch
from transformers import AutoModelForCausalLM, AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

def my_new_sdpa(*args, **kwargs):
    print("I just entered the attention computation")
    return sdpa_attention_forward(*args, **kwargs)

AttentionInterface.register("my_new_sdpa", my_new_sdpa)
AttentionMaskInterface.register("my_new_sdpa", sdpa_mask)  # must have the same name as the registered attention function

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation="my_new_sdpa")
model(torch.ones(1, 5, dtype=int))
```

你还可以为注意力函数添加新的参数。支持 [`AttentionInterface`] 的模型会将 kwargs 传递到注意力层和注意力函数。在模型的 forward 函数中以 kwargs 形式传参即可。自定义注意力函数必须遵循以下签名和返回格式。

```python
import torch
from transformers import AutoModelForCausalLM, AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

def custom_attention(
    module: torch.nn.Module,  # required arg
    query: torch.Tensor,  # required arg
    key: torch.Tensor,  # required arg
    value: torch.Tensor,  # required arg
    attention_mask: Optional[torch.Tensor],  # required arg
    a_new_kwargs = None,  # You can now add as many kwargs as you need
    another_new_kwargs = None,  # You can now add as many kwargs as you need
    **kwargs,  # You need to accept **kwargs as models will pass other args
) -> tuple[torch.Tensor, Optional[torch.Tensor]]
    ...  # do your magic!
    return attn_output, attn_weights  # attn_weights are optional here

AttentionInterface.register("custom", custom_attention)
AttentionMaskInterface.register("custom", sdpa_mask)  # to leave the existing mask untouched

model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="custom")
model(torch.ones(1, 5, dtype=int), a_new_kwargs=..., another_new_kwargs=...)
```

请查阅模型的 [modeling 代码](https://github.com/huggingface/transformers/tree/main/src/transformers/models)，确认它会向注意力函数发送哪些参数和 kwargs。

## AttentionMaskInterface

[`AttentionMaskInterface`] 是一个注册表，[`create_*_mask`](#构建注意力掩码) 系列函数通过它把掩码转换为当前注意力后端所期望的格式。FlexAttention 需要 [BlockMask](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask)，SDPA 需要 4D 张量，FlashAttention 需要基础的 2D 填充掩码。使用 [`AttentionMaskInterface.register`] 可以注册自定义后端，或覆盖既有后端的格式转换器。

```python
import torch
from transformers import AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

def my_new_sdpa_mask(*args, **kwargs):
    print("I just entered the attention mask computation")
    return sdpa_mask(*args, **kwargs)

AttentionMaskInterface.register("my_new_sdpa_mask", my_new_sdpa_mask)
```

如果没有为当前 `attn_implementation` 注册格式转换器，掩码创建将被跳过，并向注意力层传入 `attention_mask=None`。

注册的函数必须符合以下签名。

```python
def custom_attention_mask(
    batch_size: int,  # required arg
    q_length: int,  # required arg
    kv_length: int,  # required arg
    q_offset: int = 0,  # required arg
    kv_offset: int = 0,  # required arg
    mask_function: Callable = causal_mask_function,  # required arg
    attention_mask: Optional[torch.Tensor] = None,  # required arg
    **kwargs,  # a few additional args may be passed as kwargs, especially the model's config is always passed
) -> Optional[torch.Tensor]:
```

`mask_function` 参数是一个 `Callable`，模仿 PyTorch 的 [mask_mod](https://pytorch.org/blog/flexattention/) 函数。它接收 4 个索引 `(batch_idx, head_idx, q_idx, kv_idx)`，返回一个布尔值，表示该位置是否参与注意力计算。这与[构建注意力掩码](#构建注意力掩码)中 `or_mask_function` 和 `and_mask_function` 使用的基本形式相同。

> [!TIP]
> 如果 `mask_function` 无法创建掩码，可以对 torch.export 使用这个[变通方法](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/executorch.py)。

## 构建注意力掩码

使用 [transformers.masking_utils](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py#L894) 中的 `create_*_mask` 系列函数构建注意力掩码。每个函数都会从模型配置中读取当前的注意力后端，在 [`AttentionMaskInterface`] 中查找该后端的掩码格式转换器，并返回该后端期望的格式。你不需要自己对掩码取反、扩展或转换类型。

选择与注意力模式匹配的函数。

| 函数 | 适用场景 |
|---|---|
| [`create_causal_mask`] | 仅解码器模型，每个词符关注自身及之前的词符 |
| [`create_bidirectional_mask`] | 编码器模型，或解码器对编码器状态的交叉注意力 |
| [`create_sliding_window_causal_mask`] | 采用滑动窗口注意力模式的解码器模型 |
| [`create_chunked_causal_mask`] | 将序列切分为固定大小块的分块因果注意力解码器模型 |
| [`create_bidirectional_sliding_window_mask`] | 采用滑动窗口注意力模式的编码器模型 |

> [!WARNING]
> 旧版的可调用掩码辅助函数——`get_extended_attention_mask`、`create_extended_attention_mask_for_decoder`、`invert_attention_mask`——会发出弃用警告，并将在未来版本中移除。请改用 `create_*_mask` 系列函数。

<hfoptions id="build-mask">
<hfoption id="causal attention">

在解码器的 forward 中调用 [`create_causal_mask`]。传入配置、输入嵌入、用户提供的 2D `attention_mask` 以及缓存。该函数通过嵌入读取批次大小、查询长度、dtype 和设备，并通过缓存计算键长度。

```py
from transformers.masking_utils import create_causal_mask

attention_mask = create_causal_mask(
    config=self.config,
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
)
```

</hfoption>
<hfoption id="encoder self-attention">

编码器自注意力调用 [`create_bidirectional_mask`]。编码器没有缓存，因此不需要 `past_key_values`。

```py
from transformers.masking_utils import create_bidirectional_mask

attention_mask = create_bidirectional_mask(
    config=self.config,
    inputs_embeds=embedding_output,
    attention_mask=attention_mask,
)
```

</hfoption>
<hfoption id="cross-attention">

对于交叉注意力，将编码器状态作为 `encoder_hidden_states` 传入，这样掩码会使用编码器的键和值长度，而不是解码器的查询长度。

```py
encoder_attention_mask = create_bidirectional_mask(
    config=self.config,
    inputs_embeds=embedding_output,
    attention_mask=encoder_attention_mask,
    encoder_hidden_states=encoder_hidden_states,
)
```

</hfoption>
</hfoptions>

使用 `or_mask_function` 和 `and_mask_function` 参数可以在基础掩码之上叠加额外的约束。`or_mask_function` 用于允许更多位置参与注意力，`and_mask_function` 用于进一步收紧基础模式。两者都遵循 [AttentionMaskInterface](#attentionmaskinterface) 中描述的 4 索引 `mask_function` 签名，接收 `(batch_idx, head_idx, q_idx, kv_idx)` 并返回布尔值。

> [!WARNING]
> `or_mask_function` 和 `and_mask_function` 可以表达任意注意力模式，但它们比内置模式更慢，且与 ExecuTorch 不兼容。这种开销在较小的模型（约 2 亿参数）上最明显，因为掩码创建在前向传播耗时中占比更大。只有当标准的 `create_*_mask` 函数无法表达你的需求时才使用它们。

例如，在因果掩码上叠加一个处处返回 `True` 的函数，可以将其变成完全双向的掩码。与因果模式取并集后，每个词符都可以关注其他所有词符。

```py
mask_kwargs = {
    "config": self.config,
    "inputs_embeds": inputs_embeds,
    "attention_mask": attention_mask,
    "past_key_values": past_key_values,
    "position_ids": position_ids,
    "or_mask_function": lambda *args: torch.tensor(True, dtype=torch.bool),
}

attention_mask = create_causal_mask(**mask_kwargs)
```

在生成过程中，[`~GenerationMixin.generate`] 通过 [`create_masks_for_generate`] 构建掩码，它会根据模型配置分发到相应的 `create_*_mask`。在模型类上覆盖它，即可为生成过程接入自定义的掩码策略。

## 传入自定义 4D 注意力掩码

当你需要的注意力模式无法由 `create_*_mask` 函数表达时，可以传入自己的 4D 掩码。4D 掩码的形状为 `(batch_size, 1, query_length, kv_length)`，其中 `1` 会将同一个掩码广播到所有注意力头。Transformers 会检测并原样使用该掩码，跳过 `create_*_mask`。

4D 掩码使用两种取值约定之一。

| dtype | 参与注意力 | 被掩蔽 |
|---|---|---|
| 布尔型 | `True` | `False` |
| 浮点型 | `0.0` | `-inf` |

浮点约定是在 softmax 之前把掩码加到注意力分数上。分数加上 `0.0` 保持不变，因此该位置参与注意力；分数加上 `-inf` 在 softmax 后变为零，因此该位置被排除。

> [!IMPORTANT]
> 可接受的约定取决于注意力后端。`sdpa` 接受布尔型或浮点型掩码。`eager` 将掩码加到分数上，因此只接受浮点型掩码。`flash_attention_2` 和 `flex_attention` 使用各自的格式（2D 填充掩码和 [BlockMask](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask)），不接受原始的 4D 掩码。

一个常见的错误是在浮点 4D 掩码中沿用 2D 填充掩码的 `1`/`0` 约定。由于掩码是被加到分数上的，`0.0` 会保留该位置，而 `1.0` 只会增加一点偏置。

下面的示例对比了错误掩码与正确掩码。两者都基于同一个 `1`/`0` 因果模式。

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

input_ids = tokenizer("my favorite condiment on a", return_tensors="pt").input_ids
seq_len = input_ids.shape[1]

# 1 attends, 0 masks
causal = torch.tril(torch.ones(seq_len, seq_len))

# wrong: 1.0/0.0 floats are added to the scores, so 0.0 keeps a token and 1.0 barely changes it
wrong_mask = causal[None, None]

# correct: 0.0 attends, -inf masks
correct_mask = torch.where(causal.bool(), 0.0, float("-inf"))[None, None]
```

错误的掩码保留了所有位置，因为 `0.0` 才是起掩蔽作用的值，所以它永远不会排除任何位置。

```text
        wrong_mask                          correct_mask
   (1 attends, 0 masks)              (0 attends, -inf masks)

      k0 k1 k2 k3 k4                     k0   k1   k2   k3   k4
   q0  1  0  0  0  0                  q0  0  -inf -inf -inf -inf
   q1  1  1  0  0  0                  q1  0   0   -inf -inf -inf
   q2  1  1  1  0  0                  q2  0   0    0   -inf -inf
   q3  1  1  1  1  0                  q3  0   0    0    0   -inf
   q4  1  1  1  1  1                  q4  0   0    0    0    0
```

## 双向注意力

仅解码器模型默认使用因果（单向）注意力，即每个词符只关注自身及之前的词符。设置 `is_causal=False` 可切换为双向注意力，此时每个词符都关注其他所有词符。这让你可以把仅解码器模型用作文本编码器，例如用于生成嵌入。

> [!NOTE]
> 这只适用于因果（解码器）模型。它不会把编码器模型变成解码器模型。

在模型配置中设置 `is_causal=False`，可以让每次前向传播都默认使用双向注意力。

```py
from transformers import AutoModel, AutoConfig

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
config.is_causal = False

model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B", config=config)

# all forward passes now use bidirectional attention
outputs = model(**inputs)
```

在 forward 调用中传入 `is_causal` 而不是修改模型配置，可以在因果与双向注意力之间切换而不必加载两次模型。该 kwarg 会临时覆盖配置，并在调用结束后恢复。

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")

# run with bidirectional attention
outputs = model(**inputs, is_causal=False)

# run with default causal attention
outputs = model(**inputs)
```
