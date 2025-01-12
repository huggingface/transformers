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

# 생성을 위한 유틸리티 [[utilities-for-generation]]

이 페이지는 [`~generation.GenerationMixin.generate`]에서 사용되는 모든 유틸리티 함수들을 나열합니다.

## 출력을 생성하기 (Generate Outputs) [[generate-outputs]]

[`~generation.GenerationMixin.generate`]의 출력은 [`~utils.ModelOutput`]의 하위 클래스의 인스턴스입니다. 이 출력은 [`~generation.GenerationMixin.generate`]에서 반환되는 모든 정보를 포함하는 데이터 구조체이며, 튜플 또는 딕셔너리로도 사용할 수 있습니다.

다음은 예시입니다:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

`generation_output` 객체는 [`~generation.GenerateDecoderOnlyOutput`]입니다. 아래 문서에서 확인할 수 있듯이, 이 클래스는 다음과 같은 속성을 가지고 있습니다:

- `sequences`: 생성된 토큰 시퀀스
- `scores` (옵션): 각 생성 단계에서 언어 모델링 헤드의 예측 점수
- `hidden_states` (옵션): 각 생성 단계에서 모델의 은닉 상태
- `attentions` (옵션): 각 생성 단계에서 모델의 어텐션 가중치

`output_scores=True`를 전달했기 때문에 `scores`는 포함되어 있지만, `output_hidden_states=True` 또는 `output_attentions=True`를 전달하지 않았으므로 `hidden_states`와 `attentions`는 포함되지 않았습니다.

각 속성은 일반적으로 접근할 수 있으며, 모델이 해당 속성을 반환하지 않았다면 `None`이 반환됩니다. 예를 들어, `generation_output.scores`는 언어 모델링 헤드에서 생성된 모든 예측 점수를 포함하고 있으며, `generation_output.attentions`는 `None`입니다.

`generation_output` 객체를 튜플로 사용할 경우, `None` 값이 아닌 속성만 포함됩니다. 예를 들어, `loss`와 `logits`라는 두 요소가 포함된 경우:

```python
generation_output[:2]
```

위 코드는 `(generation_output.sequences, generation_output.scores)` 튜플을 반환합니다.

`generation_output` 객체를 딕셔너리로 사용할 경우, `None` 값이 아닌 속성만 포함됩니다. 예를 들어, `sequences`와 `scores`라는 두 개의 키를 가질 수 있습니다.

여기서는 모든 출력 유형을 문서화합니다.


### PyTorch [[transformers.generation.GenerateDecoderOnlyOutput]]

[[autodoc]] generation.GenerateDecoderOnlyOutput

[[autodoc]] generation.GenerateEncoderDecoderOutput

[[autodoc]] generation.GenerateBeamDecoderOnlyOutput

[[autodoc]] generation.GenerateBeamEncoderDecoderOutput

### TensorFlow [[transformers.generation.TFGreedySearchEncoderDecoderOutput]]

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

### FLAX [[transformers.generation.FlaxSampleOutput]]

[[autodoc]] generation.FlaxSampleOutput

[[autodoc]] generation.FlaxGreedySearchOutput

[[autodoc]] generation.FlaxBeamSearchOutput

## LogitsProcessor [[logitsprocessor]]

[`LogitsProcessor`]는 생성 중 언어 모델 헤드의 예측 점수를 수정하는 데 사용됩니다.

### PyTorch [[transformers.AlternatingCodebooksLogitsProcessor]]

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

[[autodoc]] MinPLogitsWarper
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

[[autodoc]] WatermarkLogitsProcessor
    - __call__


### TensorFlow [[transformers.TFForcedBOSTokenLogitsProcessor]]

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

### FLAX [[transformers.FlaxForcedBOSTokenLogitsProcessor]]

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

## StoppingCriteria [[transformers.StoppingCriteria]]

[`StoppingCriteria`]는 생성이 언제 멈출지를 결정하는 데 사용됩니다 (EOS 토큰 외). 이 기능은 PyTorch 구현에만 제공됩니다.

[[autodoc]] StoppingCriteria
    - __call__

[[autodoc]] StoppingCriteriaList
    - __call__

[[autodoc]] MaxLengthCriteria
    - __call__

[[autodoc]] MaxTimeCriteria
    - __call__

[[autodoc]] StopStringCriteria
    - __call__

[[autodoc]] EosTokenCriteria
    - __call__

## Constraint [[transformers.Constraint]]

[`Constraint`]는 생성 출력에 특정 토큰이나 시퀀스를 강제로 포함시키는 데 사용됩니다. 이 기능은 PyTorch 구현에만 제공됩니다.

[[autodoc]] Constraint

[[autodoc]] PhrasalConstraint

[[autodoc]] DisjunctiveConstraint

[[autodoc]] ConstraintListState

## 빔 검색 (BeamSearch) [[transformers.BeamScorer]]

[[autodoc]] BeamScorer
    - process
    - finalize

[[autodoc]] BeamSearchScorer
    - process
    - finalize

[[autodoc]] ConstrainedBeamSearchScorer
    - process
    - finalize

## 스트리머 (Streamers) [[transformers.TextStreamer]]

[[autodoc]] TextStreamer

[[autodoc]] TextIteratorStreamer

## 캐시 (Caches) [[transformers.Cache]]

[[autodoc]] Cache
    - update

[[autodoc]] CacheConfig
    - update

[[autodoc]] QuantizedCacheConfig
    - validate

[[autodoc]] DynamicCache
    - update
    - get_seq_length
    - reorder_cache
    - to_legacy_cache
    - from_legacy_cache

[[autodoc]] QuantizedCache
    - update
    - get_seq_length

[[autodoc]] QuantoQuantizedCache

[[autodoc]] HQQQuantizedCache

[[autodoc]] SinkCache
    - update
    - get_seq_length
    - reorder_cache

[[autodoc]] OffloadedCache
    - update
    - prefetch_layer
    - evict_previous_layer

[[autodoc]] StaticCache
    - update
    - get_seq_length
    - reset

[[autodoc]] OffloadedStaticCache
    - update
    - get_seq_length
    - reset

[[autodoc]] HybridCache
    - update
    - get_seq_length
    - reset

[[autodoc]] SlidingWindowCache
    - update
    - reset

[[autodoc]] EncoderDecoderCache
    - get_seq_length
    - to_legacy_cache
    - from_legacy_cache
    - reset
    - reorder_cache

[[autodoc]] MambaCache
    - update_conv_state
    - update_ssm_state
    - reset

## 워터마크 유틸리티 (Watermark Utils) [[transformers.WatermarkDetector]]

[[autodoc]] WatermarkDetector
    - __call__
