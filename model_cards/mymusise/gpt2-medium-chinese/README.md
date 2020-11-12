---
language: zh
---

# gpt2-medium-chinese


# Overview

- **Language model**: GPT2-Medium
- **Model size**: 1.2GiB 
- **Language**: Chinese
- **Training data**: [wiki2019zh_corpus](https://github.com/brightmart/nlp_chinese_corpus)
- **Source code**: [gpt2-quickly](https://github.com/mymusise/gpt2-quickly)

# Example

```python
from transformers import BertTokenizer, TFGPT2LMHeadModel
from transformers import TextGenerationPipeline

tokenizer = BertTokenizer.from_pretrained("mymusise/EasternFantasyNoval")
model = TFGPT2LMHeadModel.from_pretrained("mymusise/EasternFantasyNoval")

text_generator = TextGenerationPipeline(model, tokenizer)
print(text_generator("今日", max_length=64, do_sample=True, top_k=10))
print(text_generator("跨越山丘", max_length=64, do_sample=True, top_k=10))
```
输出
```text
[{'generated_text': '今日 ， 他 的 作 品 也 在 各 种 报 刊 发 表 。 201 1 年 ， 他 开 设 了 他 的 网 页 版 《 the dear 》 。 此 外 ， 他 还 在 各 种 电 视 节 目 中 出 现 过 。 2017 年 1 月 ， 他 被 任'}]
[{'generated_text': '跨越山丘 ， 其 中 有 三 分 之 二 的 地 区 被 划 入 山 区 。 最 高 峰 是 位 于 山 脚 上 的 大 岩 （ ） 。 其 中 的 山 脚 下 有 一 处 有 名 为 的 河 谷 ， 因 其 高 度 在 其 中 ， 而 得 名 。'}]
```

[Try it on colab](https://colab.research.google.com/github/mymusise/gpt2-quickly/blob/main/examples/gpt2_medium_chinese.ipynb)
