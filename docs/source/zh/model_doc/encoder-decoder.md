<!--版权所有2020年The HuggingFace团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证以了解具体的语言权限和限制。an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
<span> </span>
⚠️请注意，此文件是 Markdown 格式，但包含特定于我们的文档构建器（类似 MDX）的语法，可能无法在您的 Markdown 查看器中正确渲染。<span> </span>
<span> </span>
# 编码器解码器模型
## 概述
[`EncoderDecoderModel`] 可以用于使用任何预训练的自动编码模型作为编码器和任何预训练的自回归模型作为解码器初始化序列到序列模型。在 [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) 中，Sascha Rothe、Shashi Narayan 和 Aliaksei Severyn 展示了使用预训练检查点初始化序列到序列模型对序列生成任务的有效性。
此类架构的一个应用是利用两个预训练的 [`BertModel`] 作为编码器和解码器，用于摘要模型，正如 [Yang Liu 和 Mirella Lapata 在 Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) 中所展示的那样。<span> </span> <span> </span>
在训练/微调了这样一个 [`EncoderDecoderModel`] 之后，它可以像其他模型一样保存/加载（请参阅示例以获取更多信息）。<span> </span>
从预训练的 [`BertModel`] 作为编码器和解码器初始化摘要模型的示例如下：[Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) by Yang Liu and Mirella Lapata.<span> </span>
## 从模型配置随机初始化 `EncoderDecoderModel`
可以使用编码器和解码器配置随机初始化 [`EncoderDecoderModel`]。在下面的示例中，我们展示了如何使用编码器的默认 [`BertModel`] 配置和解码器的默认 [`BertForCausalLM`] 配置来实现这一点。
```python
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = EncoderDecoderModel(config=config)
```

## 从预训练的编码器和预训练的解码器初始化 `EncoderDecoderModel`
可以从预训练的编码器检查点和预训练的解码器检查点初始化 [`EncoderDecoderModel`]。请注意，任何预训练的自动编码模型（例如 BERT）都可以作为编码器，并且预训练的自动编码模型（例如 BERT）、预训练的因果语言模型（例如 GPT2）以及序列到序列模型的预训练解码器部分（例如 BART 的解码器）都可以作为解码器。根据您选择的解码器架构，交叉注意力层可能会被随机初始化。从预训练的编码器和解码器检查点初始化 [`EncoderDecoderModel`] 需要对模型进行下游任务的微调，正如 [“启动编码器-解码器的博客文章”](https://huggingface.co/blog/warm-starting-encoder-decoder) 所示。要这样做，`EncoderDecoderModel` 类提供了一个 [`EncoderDecoderModel.from_encoder_decoder_pretrained`] 方法。
```python
>>> from transformers import EncoderDecoderModel, BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
```

## 加载现有的 `EncoderDecoderModel` 检查点并进行推理
要加载 `EncoderDecoderModel` 类的微调检查点，[`EncoderDecoderModel`] 提供了与 Transformers 中的任何其他模型架构一样的 `from_pretrained(...)` 方法。
要执行推理，可以使用 [`generate`] 方法，该方法允许自动回归生成文本。此方法支持各种解码形式，例如贪婪搜索、束搜索和多项采样。
```python
>>> from transformers import AutoTokenizer, EncoderDecoderModel

>>> # load a fine-tuned seq2seq model and corresponding tokenizer
>>> model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

>>> # let's perform inference on a long piece of text
>>> ARTICLE_TO_SUMMARIZE = (
...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
... )
>>> input_ids = tokenizer(ARTICLE_TO_SUMMARIZE, return_tensors="pt").input_ids

>>> # autoregressively generate summary (uses greedy decoding by default)
>>> generated_ids = model.generate(input_ids)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
nearly 800 thousand customers were affected by the shutoffs. the aim is to reduce the risk of wildfires. nearly 800, 000 customers were expected to be affected by high winds amid dry conditions. pg & e said it scheduled the blackouts to last through at least midday tomorrow.
```

## 将 PyTorch 检查点加载到 `TFEncoderDecoderModel` 中
[`TFEncoderDecoderModel.from_pretrained`] 目前不支持从 pytorch 检查点初始化模型。将 `from_pt=True` 传递给此方法将抛出异常。如果只有特定编码器-解码器模型的 pytorch 检查点，可以使用以下解决方法：pytorch checkpoint. Passing `from_pt=True` to this method will throw an exception. If there are only pytorch
checkpoints for a particular encoder-decoder model, a workaround is:

```python
>>> # a workaround to load from pytorch checkpoint
>>> from transformers import EncoderDecoderModel, TFEncoderDecoderModel

>>> _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> # This is only for copying some specific attributes of this particular model.
>>> model.config = _model.config
```

## 训练
创建模型后，可以像 BART、T5 或任何其他编码器-解码器模型一样对其进行微调。如您所见，模型只需要 2 个输入即可计算损失：`input_ids`（编码输入序列的 `input_ids`）和 `labels`（编码目标序列的 `input_ids`）。<span> </span> <span> </span>
```python
>>> from transformers import BertTokenizer, EncoderDecoderModel

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> input_ids = tokenizer(
...     "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
...     return_tensors="pt",
... ).input_ids

>>> labels = tokenizer(
...     "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
...     return_tensors="pt",
... ).input_ids

>>> # the forward function automatically creates the correct decoder_input_ids
>>> loss = model(input_ids=input_ids, labels=labels).loss
```

详细的 [colab](https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=ZwQIEhKOrJpl) 进行训练。
此模型由 [thomwolf](https://github.com/thomwolf) 贡献。这个模型的 TensorFlow 和 Flax 版本由 [ydshieh](https://github.com/ydshieh) 贡献。were contributed by [ydshieh](https://github.com/ydshieh).


## EncoderDecoderConfig
[[autodoc]] EncoderDecoderConfig
## EncoderDecoderModel
[[autodoc]] EncoderDecoderModel- forward- from_encoder_decoder_pretrained
## TFEncoderDecoderModel
[[autodoc]] TFEncoderDecoderModel- call- from_encoder_decoder_pretrained
## FlaxEncoderDecoderModel
[[autodoc]] FlaxEncoderDecoderModel- __call__- from_encoder_decoder_pretrained