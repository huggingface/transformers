<!-- 版权所有2022年HuggingFace团队保留所有权利。
根据 Apache 许可证 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样” BASIS 分发的，没有任何明示或暗示的担保或条件。请参阅许可证中的特定语言的权限和限制。⚠️ 请注意，此文件是 Markdown 格式，但包含我们文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确显示。


-->

## 概述

我们介绍 GPT-NeoX-Japanese，这是一个用于日语的自回归语言模型，训练基于 [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox)。

日语是一种具有大量词汇和平假名、片假名和汉字书写系统的独特语言。为了解决日语的特殊结构，我们使用了 [特殊的子词标记器](https://github.com/tanreinama/Japanese-BPEEncoder_V2)。我们非常感谢 *tanreinama* 开源这个非常有帮助的标记器。

根据 Google 的 [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) 研究建议，我们从变压器块中去除了偏置参数，从而实现了更好的模型性能。详细信息请参考 [此文章](https://medium.com/ml-abeja/training-a-better-gpt-2-93b157662ae4)。

该模型的开发由 [Shinya Otani](https://github.com/SO0529)、[Takayoshi Makabe](https://github.com/spider-man-tm)、[Anuj Arora](https://github.com/Anuj040) 和 [Kyo Hattori](https://github.com/go5paopao) 领导，来自 [ABEJA, Inc.](https://www.abejainc.com/)。有关此模型构建活动的更多信息，请参阅 [此处（ja）](https://tech-blog.abeja.asia/entry/abeja-gpt-project-202207)。

### 生成

可以使用 `generate()` 方法使用 GPT NeoX 日语模型生成文本。

```python
>>> from transformers import GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseTokenizer

>>> model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")

>>> prompt = "人とAIが協調するためには、"

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

>>> print(gen_text)
人とAIが協調するためには、AIと人が共存し、AIを正しく理解する必要があります。
```

## 文档资源

- [因果语言建模任务指南](../tasks/language_modeling)

## GPTNeoXJapaneseConfig

[[autodoc]] GPTNeoXJapaneseConfig

## GPTNeoXJapaneseTokenizer

[[autodoc]] GPTNeoXJapaneseTokenizer

## GPTNeoXJapaneseModel

[[autodoc]] GPTNeoXJapaneseModel
    - forward

## GPTNeoXJapaneseForCausalLM

[[autodoc]] GPTNeoXJapaneseForCausalLM
    - forward