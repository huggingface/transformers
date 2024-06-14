<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 模型输出

所有模型的输出都是 [`~utils.ModelOutput`] 的子类的实例。这些是包含模型返回的所有信息的数据结构，但也可以用作元组或字典。

让我们看一个例子：

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
```

`outputs` 对象是 [`~modeling_outputs.SequenceClassifierOutput`]，如下面该类的文档中所示，它表示它有一个可选的 `loss`，一个 `logits`，一个可选的 `hidden_states` 和一个可选的 `attentions` 属性。在这里，我们有 `loss`，因为我们传递了 `labels`，但我们没有 `hidden_states` 和 `attentions`，因为我们没有传递 `output_hidden_states=True` 或 `output_attentions=True`。

<Tip>

当传递 `output_hidden_states=True` 时，您可能希望 `outputs.hidden_states[-1]` 与 `outputs.last_hidden_states` 完全匹配。然而，这并不总是成立。一些模型在返回最后的 hidden state时对其应用归一化或其他后续处理。

</Tip>


您可以像往常一样访问每个属性，如果模型未返回该属性，您将得到 `None`。在这里，例如，`outputs.loss` 是模型计算的损失，而 `outputs.attentions` 是 `None`。

当将我们的 `outputs` 对象视为元组时，它仅考虑那些没有 `None` 值的属性。例如这里它有两个元素，`loss` 和 `logits`，所以

```python
outputs[:2]
```

将返回元组 `(outputs.loss, outputs.logits)`。

将我们的 `outputs` 对象视为字典时，它仅考虑那些没有 `None` 值的属性。例如在这里它有两个键，分别是 `loss` 和 `logits`。

我们在这里记录了被多个类型模型使用的通用模型输出。特定输出类型在其相应的模型页面上有文档。

## ModelOutput

[[autodoc]] utils.ModelOutput
    - to_tuple

## BaseModelOutput

[[autodoc]] modeling_outputs.BaseModelOutput

## BaseModelOutputWithPooling

[[autodoc]] modeling_outputs.BaseModelOutputWithPooling

## BaseModelOutputWithCrossAttentions

[[autodoc]] modeling_outputs.BaseModelOutputWithCrossAttentions

## BaseModelOutputWithPoolingAndCrossAttentions

[[autodoc]] modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions

## BaseModelOutputWithPast

[[autodoc]] modeling_outputs.BaseModelOutputWithPast

## BaseModelOutputWithPastAndCrossAttentions

[[autodoc]] modeling_outputs.BaseModelOutputWithPastAndCrossAttentions

## Seq2SeqModelOutput

[[autodoc]] modeling_outputs.Seq2SeqModelOutput

## CausalLMOutput

[[autodoc]] modeling_outputs.CausalLMOutput

## CausalLMOutputWithCrossAttentions

[[autodoc]] modeling_outputs.CausalLMOutputWithCrossAttentions

## CausalLMOutputWithPast

[[autodoc]] modeling_outputs.CausalLMOutputWithPast

## MaskedLMOutput

[[autodoc]] modeling_outputs.MaskedLMOutput

## Seq2SeqLMOutput

[[autodoc]] modeling_outputs.Seq2SeqLMOutput

## NextSentencePredictorOutput

[[autodoc]] modeling_outputs.NextSentencePredictorOutput

## SequenceClassifierOutput

[[autodoc]] modeling_outputs.SequenceClassifierOutput

## Seq2SeqSequenceClassifierOutput

[[autodoc]] modeling_outputs.Seq2SeqSequenceClassifierOutput

## MultipleChoiceModelOutput

[[autodoc]] modeling_outputs.MultipleChoiceModelOutput

## TokenClassifierOutput

[[autodoc]] modeling_outputs.TokenClassifierOutput

## QuestionAnsweringModelOutput

[[autodoc]] modeling_outputs.QuestionAnsweringModelOutput

## Seq2SeqQuestionAnsweringModelOutput

[[autodoc]] modeling_outputs.Seq2SeqQuestionAnsweringModelOutput

## Seq2SeqSpectrogramOutput

[[autodoc]] modeling_outputs.Seq2SeqSpectrogramOutput

## SemanticSegmenterOutput

[[autodoc]] modeling_outputs.SemanticSegmenterOutput

## ImageClassifierOutput

[[autodoc]] modeling_outputs.ImageClassifierOutput

## ImageClassifierOutputWithNoAttention

[[autodoc]] modeling_outputs.ImageClassifierOutputWithNoAttention

## DepthEstimatorOutput

[[autodoc]] modeling_outputs.DepthEstimatorOutput

## Wav2Vec2BaseModelOutput

[[autodoc]] modeling_outputs.Wav2Vec2BaseModelOutput

## XVectorOutput

[[autodoc]] modeling_outputs.XVectorOutput

## Seq2SeqTSModelOutput

[[autodoc]] modeling_outputs.Seq2SeqTSModelOutput

## Seq2SeqTSPredictionOutput

[[autodoc]] modeling_outputs.Seq2SeqTSPredictionOutput

## SampleTSPredictionOutput

[[autodoc]] modeling_outputs.SampleTSPredictionOutput

## TFBaseModelOutput

[[autodoc]] modeling_tf_outputs.TFBaseModelOutput

## TFBaseModelOutputWithPooling

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPooling

## TFBaseModelOutputWithPoolingAndCrossAttentions

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions

## TFBaseModelOutputWithPast

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPast

## TFBaseModelOutputWithPastAndCrossAttentions

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPastAndCrossAttentions

## TFSeq2SeqModelOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqModelOutput

## TFCausalLMOutput

[[autodoc]] modeling_tf_outputs.TFCausalLMOutput

## TFCausalLMOutputWithCrossAttentions

[[autodoc]] modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions

## TFCausalLMOutputWithPast

[[autodoc]] modeling_tf_outputs.TFCausalLMOutputWithPast

## TFMaskedLMOutput

[[autodoc]] modeling_tf_outputs.TFMaskedLMOutput

## TFSeq2SeqLMOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqLMOutput

## TFNextSentencePredictorOutput

[[autodoc]] modeling_tf_outputs.TFNextSentencePredictorOutput

## TFSequenceClassifierOutput

[[autodoc]] modeling_tf_outputs.TFSequenceClassifierOutput

## TFSeq2SeqSequenceClassifierOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqSequenceClassifierOutput

## TFMultipleChoiceModelOutput

[[autodoc]] modeling_tf_outputs.TFMultipleChoiceModelOutput

## TFTokenClassifierOutput

[[autodoc]] modeling_tf_outputs.TFTokenClassifierOutput

## TFQuestionAnsweringModelOutput

[[autodoc]] modeling_tf_outputs.TFQuestionAnsweringModelOutput

## TFSeq2SeqQuestionAnsweringModelOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqQuestionAnsweringModelOutput

## FlaxBaseModelOutput

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutput

## FlaxBaseModelOutputWithPast

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutputWithPast

## FlaxBaseModelOutputWithPooling

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutputWithPooling

## FlaxBaseModelOutputWithPastAndCrossAttentions

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions

## FlaxSeq2SeqModelOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqModelOutput

## FlaxCausalLMOutputWithCrossAttentions

[[autodoc]] modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions

## FlaxMaskedLMOutput

[[autodoc]] modeling_flax_outputs.FlaxMaskedLMOutput

## FlaxSeq2SeqLMOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqLMOutput

## FlaxNextSentencePredictorOutput

[[autodoc]] modeling_flax_outputs.FlaxNextSentencePredictorOutput

## FlaxSequenceClassifierOutput

[[autodoc]] modeling_flax_outputs.FlaxSequenceClassifierOutput

## FlaxSeq2SeqSequenceClassifierOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqSequenceClassifierOutput

## FlaxMultipleChoiceModelOutput

[[autodoc]] modeling_flax_outputs.FlaxMultipleChoiceModelOutput

## FlaxTokenClassifierOutput

[[autodoc]] modeling_flax_outputs.FlaxTokenClassifierOutput

## FlaxQuestionAnsweringModelOutput

[[autodoc]] modeling_flax_outputs.FlaxQuestionAnsweringModelOutput

## FlaxSeq2SeqQuestionAnsweringModelOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqQuestionAnsweringModelOutput
