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

# 用于生成的工具

此页面列出了所有由 [`~generation.GenerationMixin.generate`]。

## 生成输出

[`~generation.GenerationMixin.generate`] 的输出是 [`~utils.ModelOutput`] 的一个子类的实例。这个输出是一种包含 [`~generation.GenerationMixin.generate`] 返回的所有信息数据结构，但也可以作为元组或字典使用。
这里是一个例子：


```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

`generation_output` 的对象是 [`~generation.GenerateDecoderOnlyOutput`] 的一个实例，从该类的文档中我们可以看到，这意味着它具有以下属性：

- `sequences`: 生成的tokens序列
- `scores`（可选）: 每个生成步骤的语言建模头的预测分数
- `hidden_states`（可选）: 每个生成步骤模型的hidden states
- `attentions`（可选）: 每个生成步骤模型的注意力权重

在这里，由于我们传递了 `output_scores=True`，我们具有 `scores` 属性。但我们没有 `hidden_states` 和 `attentions`，因为没有传递 `output_hidden_states=True` 或 `output_attentions=True`。

您可以像通常一样访问每个属性，如果该属性未被模型返回，则将获得 `None`。例如，在这里 `generation_output.scores` 是语言建模头的所有生成预测分数，而 `generation_output.attentions` 为 `None`。

当我们将 `generation_output` 对象用作元组时，它只保留非 `None` 值的属性。例如，在这里它有两个元素，`loss` 然后是 `logits`，所以


```python
generation_output[:2]
```

将返回元组`(generation_output.sequences, generation_output.scores)`。

当我们将`generation_output`对象用作字典时，它只保留非`None`的属性。例如，它有两个键，分别是`sequences`和`scores`。

我们在此记录所有输出类型。


### PyTorch

[[autodoc]] generation.GenerateDecoderOnlyOutput

[[autodoc]] generation.GenerateEncoderDecoderOutput

[[autodoc]] generation.GenerateBeamDecoderOnlyOutput

[[autodoc]] generation.GenerateBeamEncoderDecoderOutput

### TensorFlow

[[autodoc]] generation.TFGreedySearchEncoderDecoderOutput

[[autodoc]] generation.TFGreedySearchDecoderOnlyOutput

[[autodoc]] generation.TFSampleEncoderDecoderOutput

[[autodoc]] generation.TFSampleDecoderOnlyOutput

[[autodoc]] generation.TFBeamSearchEncoderDecoderOutput

[[autodoc]] generation.TFBeamSearchDecoderOnlyOutput

[[autodoc]] generation.TFBeamSampleEncoderDecoderOutput

[[autodoc]] generation.TFBeamSampleDecoderOnlyOutput

[[autodoc]] generation.TFContrastiveSearchEncoderDecoderOutput

[[autodoc]] generation.TFContrastiveSearchDecoderOnlyOutput

### FLAX

[[autodoc]] generation.FlaxSampleOutput

[[autodoc]] generation.FlaxGreedySearchOutput

[[autodoc]] generation.FlaxBeamSearchOutput

## LogitsProcessor

[`LogitsProcessor`] 可以用于修改语言模型头的预测分数以进行生成


### PyTorch

[[autodoc]] AlternatingCodebooksLogitsProcessor
    - __call__

[[autodoc]] ClassifierFreeGuidanceLogitsProcessor
    - __call__

[[autodoc]] EncoderNoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] EncoderRepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] EpsilonLogitsWarper
    - __call__

[[autodoc]] EtaLogitsWarper
    - __call__

[[autodoc]] ExponentialDecayLengthPenalty
    - __call__

[[autodoc]] ForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] ForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] HammingDiversityLogitsProcessor
    - __call__

[[autodoc]] InfNanRemoveLogitsProcessor
    - __call__

[[autodoc]] LogitNormalization
    - __call__

[[autodoc]] LogitsProcessor
    - __call__

[[autodoc]] LogitsProcessorList
    - __call__

[[autodoc]] MinLengthLogitsProcessor
    - __call__

[[autodoc]] MinNewTokensLengthLogitsProcessor
    - __call__

[[autodoc]] NoBadWordsLogitsProcessor
    - __call__

[[autodoc]] NoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] PrefixConstrainedLogitsProcessor
    - __call__

[[autodoc]] RepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] SequenceBiasLogitsProcessor
    - __call__

[[autodoc]] SuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] SuppressTokensLogitsProcessor
    - __call__

[[autodoc]] TemperatureLogitsWarper
    - __call__

[[autodoc]] TopKLogitsWarper
    - __call__

[[autodoc]] TopPLogitsWarper
    - __call__

[[autodoc]] TypicalLogitsWarper
    - __call__

[[autodoc]] UnbatchedClassifierFreeGuidanceLogitsProcessor
    - __call__

[[autodoc]] WhisperTimeStampLogitsProcessor
    - __call__

### TensorFlow

[[autodoc]] TFForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] TFForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] TFForceTokensLogitsProcessor
    - __call__

[[autodoc]] TFLogitsProcessor
    - __call__

[[autodoc]] TFLogitsProcessorList
    - __call__

[[autodoc]] TFLogitsWarper
    - __call__

[[autodoc]] TFMinLengthLogitsProcessor
    - __call__

[[autodoc]] TFNoBadWordsLogitsProcessor
    - __call__

[[autodoc]] TFNoRepeatNGramLogitsProcessor
    - __call__

[[autodoc]] TFRepetitionPenaltyLogitsProcessor
    - __call__

[[autodoc]] TFSuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] TFSuppressTokensLogitsProcessor
    - __call__

[[autodoc]] TFTemperatureLogitsWarper
    - __call__

[[autodoc]] TFTopKLogitsWarper
    - __call__

[[autodoc]] TFTopPLogitsWarper
    - __call__

### FLAX

[[autodoc]] FlaxForcedBOSTokenLogitsProcessor
    - __call__

[[autodoc]] FlaxForcedEOSTokenLogitsProcessor
    - __call__

[[autodoc]] FlaxForceTokensLogitsProcessor
    - __call__

[[autodoc]] FlaxLogitsProcessor
    - __call__

[[autodoc]] FlaxLogitsProcessorList
    - __call__

[[autodoc]] FlaxLogitsWarper
    - __call__

[[autodoc]] FlaxMinLengthLogitsProcessor
    - __call__

[[autodoc]] FlaxSuppressTokensAtBeginLogitsProcessor
    - __call__

[[autodoc]] FlaxSuppressTokensLogitsProcessor
    - __call__

[[autodoc]] FlaxTemperatureLogitsWarper
    - __call__

[[autodoc]] FlaxTopKLogitsWarper
    - __call__

[[autodoc]] FlaxTopPLogitsWarper
    - __call__

[[autodoc]] FlaxWhisperTimeStampLogitsProcessor
    - __call__

## StoppingCriteria

可以使用[`StoppingCriteria`]来更改停止生成的时间（除了EOS token以外的方法）。请注意，这仅适用于我们的PyTorch实现。


[[autodoc]] StoppingCriteria
    - __call__

[[autodoc]] StoppingCriteriaList
    - __call__

[[autodoc]] MaxLengthCriteria
    - __call__

[[autodoc]] MaxTimeCriteria
    - __call__

## Constraints

可以使用[`Constraint`]来强制生成结果包含输出中的特定tokens或序列。请注意，这仅适用于我们的PyTorch实现。

[[autodoc]] Constraint

[[autodoc]] PhrasalConstraint

[[autodoc]] DisjunctiveConstraint

[[autodoc]] ConstraintListState

## BeamSearch

[[autodoc]] BeamScorer
    - process
    - finalize

[[autodoc]] BeamSearchScorer
    - process
    - finalize

[[autodoc]] ConstrainedBeamSearchScorer
    - process
    - finalize

## Streamers

[[autodoc]] TextStreamer

[[autodoc]] TextIteratorStreamer
