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

# 모델 출력

모든 모델에는 [`~utils.ModelOutput`]의 서브클래스의 인스턴스인 모델 출력이 있습니다. 이들은
모델에서 반환되는 모든 정보를 포함하는 데이터 구조이지만 튜플이나 딕셔너리로도 사용할 수 있습니다.
딕셔너리로도 사용할 수 있습니다.

예제를 통해 살펴보겠습니다:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # 배치 크기 1
outputs = model(**inputs, labels=labels)
```

`outputs` 객체는 아래 해당 클래스의 문서에서 볼 수 있듯이 [`~modeling_outputs.SequenceClassifierOutput`]입니다.
아래 해당 클래스의 문서에서 볼 수 있듯이, `loss`(선택적), `logits`, `hidden_states`(선택적) 및 `attentions`(선택적) 항목이 있습니다. 여기에서는 `labels`를 전달했기 때문에 `loss`가 있지만 `hidden_states`와 `attentions`가 없는데, 이는 `output_hidden_states=True` 또는 `output_attentions=True`를 전달하지 않았기 때문입니다.

<Tip>

`output_hidden_states=True`를 전달할 때 `outputs.hidden_states[-1]`가 `outputs.last_hidden_state`와 정확히 일치할 것으로 예상할 수 있습니다.
하지만 항상 그런 것은 아닙니다. 일부 모델은 마지막 숨겨진 상태가 반환될 때 정규화를 적용하거나 다른 후속 프로세스를 적용합니다.

</Tip>


일반적으로 각 속성들에 접근할 수 있으며, 모델이 해당 속성을 반환하지 않은 경우 `None`이 반환됩니다. 예를 들어 여기서 `outputs.loss`는 모델에서 계산한 손실이고 `outputs.attentions`는 `None`입니다.

`outputs` 객체를 튜플로 간주할 때는 `None` 값이 없는 속성만 고려합니다. 
예를 들어 여기에는 `loss`와 `logits`라는 두 개의 요소가 있습니다. 그러므로, 예를 들어

```python
outputs[:2]
```

는 `(outputs.loss, outputs.logits)` 튜플을 반환합니다.

`outputs` 객체를 딕셔너리로 간주할 때는 `None` 값이 없는 속성만 고려합니다.
예를 들어 여기에는 `loss`와 `logits`라는 두 개의 키가 있습니다.

여기서부터는 두 가지 이상의 모델 유형에서 사용되는 일반 모델 출력을 다룹니다. 구체적인 출력 유형은 해당 모델 페이지에 문서화되어 있습니다.

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
