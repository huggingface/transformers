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

# 発電用ユーティリティ

このページには、[`~generation.GenerationMixin.generate`] で使用されるすべてのユーティリティ関数がリストされています。

## 出力を生成する

[`~generation.GenerationMixin.generate`] の出力は、次のサブクラスのインスタンスです。
[`~utils.ModelOutput`]。この出力は、返されたすべての情報を含むデータ構造です。
[`~generation.GenerationMixin.generate`] によって作成されますが、タプルまたは辞書としても使用できます。

以下に例を示します。

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

`generation_output` オブジェクトは、できる限り [`~generation.GenerateDecoderOnlyOutput`] です。
以下のそのクラスのドキュメントを参照してください。これは、次の属性があることを意味します。

- `sequences`: 生成されたトークンのシーケンス
- `scores` (オプション): 各生成ステップの言語モデリング ヘッドの予測スコア
- `hidden_​​states` (オプション): 生成ステップごとのモデルの隠れた状態
- `attentions` (オプション): 生成ステップごとのモデルのアテンションの重み

ここでは、`output_scores=True`を渡したので `scores` がありますが、`hidden_​​states` はありません。
`attentions` は、`output_hidden_​​states=True`または`output_attentions=True`を渡さなかったためです。

通常と同じように各属性にアクセスできます。その属性がモデルから返されなかった場合は、
は「なし」を取得します。ここで、たとえば`generation_output.scores`は、生成されたすべての予測スコアです。
言語モデリングのヘッドであり、`generation_output.attentions`は`None`です。

`generation_output` オブジェクトをタプルとして使用する場合、`None` 値を持たない属性のみが保持されます。
たとえば、ここには 2 つの要素、`loss`、次に`logits`があります。

```python
generation_output[:2]
```

たとえば、タプル `(generation_output.sequences,generation_output.scores)` を返します。

`generation_output` オブジェクトを辞書として使用する場合、`None` を持たない属性のみが保持されます。
ここでは、たとえば、`sequences`と`scores`という 2 つのキーがあります。

ここではすべての出力タイプを文書化します。

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

[`LogitsProcessor`] を使用して、言語モデルのヘッドの予測スコアを変更できます。
世代。

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

[`StoppingCriteria`] を使用して、(EOS トークン以外の) 生成を停止するタイミングを変更できます。これは PyTorch 実装でのみ利用可能であることに注意してください。

[[autodoc]] StoppingCriteria
    - __call__

[[autodoc]] StoppingCriteriaList
    - __call__

[[autodoc]] MaxLengthCriteria
    - __call__

[[autodoc]] MaxTimeCriteria
    - __call__

## Constraints

[`Constraint`] を使用すると、生成時に出力に特定のトークンまたはシーケンスが含まれるように強制できます。これは PyTorch 実装でのみ利用可能であることに注意してください。

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
