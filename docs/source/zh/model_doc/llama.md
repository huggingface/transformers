<!--版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发，不附带任何形式的担保或条件。请参阅许可证以了解具体的语言权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->
# LLaMA

## 概述

LLaMA 模型是由 Hugo Touvron，Thibaut Lavril，Gautier Izacard，Xavier Martinet，Marie-Anne Lachaux，Timoth é e Lacroix，Baptiste Rozi è re，Naman Goyal，Eric Hambro，Faisal Azhar，Aurelien Rodriguez，Armand Joulin，Edouard Grave，Guillaume Lample 在 [LLaMA: 开放高效的基础语言模型](https://arxiv.org/abs/2302.13971) 中提出的。它是一系列参数从 7B 到 65B 不等的基础语言模型。

来自论文的摘要如下：

*我们引入了 LLaMA，这是一系列参数从 7B 到 65B 不等的基础语言模型。我们使用公开可用的数据集进行模型训练，并展示了可以在不使用专有和无法访问的数据集的情况下训练最先进的模型的可能性。特别是，LLaMA-13B 在大多数基准测试中优于 GPT-3（175B），LLaMA-65B 与最佳模型 Chinchilla-70B 和 PaLM-540B 竞争。我们将所有模型发布给研究社区。*

提示：

- LLaMA 模型的权重可以通过填写 [此表单](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form) 来获取- 下载权重后，需要使用 [转换脚本](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) 将其转换为 Hugging Face Transformers 格式。

可以使用以下（示例）命令调用脚本：

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- 转换后，可以通过以下方式加载模型和分词器 (Tokenizer)：

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")
```

请注意，执行该脚本需要足够的 CPU RAM 来承载整个模型的 float16 精度（即使最大版本分为几个检查点，它们各自包含模型的每个权重的一部分，因此我们需要将它们全部加载到 RAM 中）。对于 65B 模型，需要 130GB 的 RAM。

- LLaMA 分词器 (Tokenizer)是基于 [sentencepiece](https://github.com/google/sentencepiece) 的 BPE 模型。sentencepiece 的一个特殊之处是，当解码序列时，如果第一个标记是单词的开头（例如“Banana”），分词器 (Tokenizer)不会在字符串前添加前缀空格。
此模型由 [zphang](https://huggingface.co/zphang) 贡献，[BlackSamorez](https://huggingface.co/BlackSamorez) 也做出了贡献。Hugging Face 中的实现代码基于 GPT-NeoX [这里](https://github.com/EleutherAI/gpt-neox)。原始作者的代码可以在 [这里](https://github.com/facebookresearch/llama) 找到。
## LlamaConfig

[[autodoc]] LlamaConfig


## LlamaTokenizer

[[autodoc]] LlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LlamaTokenizerFast

[[autodoc]] LlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary

## LlamaModel

[[autodoc]] LlamaModel
    - forward


## LlamaForCausalLM

[[autodoc]] LlamaForCausalLM
    - forward

## LlamaForSequenceClassification

[[autodoc]] LlamaForSequenceClassification
    - forward
