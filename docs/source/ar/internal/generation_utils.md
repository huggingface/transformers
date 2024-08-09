# أدوات مساعدة للجيل

تسرد هذه الصفحة جميع دالات الأدوات المساعدة التي يستخدمها [~generation.GenerationMixin.generate].

## مخرجات التوليد

مخرج [~generation.GenerationMixin.generate] هو مثيل لصنف فرعي من [~utils.ModelOutput]. وهذا المخرج هو بنية بيانات تحتوي على جميع المعلومات التي يعيدها [~generation.GenerationMixin.generate]، ولكن يمكن استخدامها أيضًا كمجموعة أو قاموس.

فيما يلي مثال:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

`generation_output` هو كائن [~generation.GenerateDecoderOnlyOutput]، كما يمكننا أن نرى في وثائق تلك الفئة أدناه، وهذا يعني أنه يحتوي على السمات التالية:

- `sequences`: تتابعات الرموز المولدة
- `scores` (اختياري): درجات التنبؤ لرأس النمذجة اللغوية، لكل خطوة من خطوات التوليد
- `hidden_states` (اختياري): الحالات المخفية للنموذج، لكل خطوة من خطوات التوليد
- `attentions` (اختياري): أوزان الانتباه للنموذج، لكل خطوة من خطوات التوليد

هنا لدينا `scores` لأننا مررنا `output_scores=True`، ولكن ليس لدينا `hidden_states` و`attentions` لأننا لم نمرر `output_hidden_states=True` أو `output_attentions=True`.

يمكنك الوصول إلى كل سمة كما تفعل عادةً، وإذا لم تكن السمة قد أعيدت من النموذج، فستحصل على `None`. على سبيل المثال، هنا `generation_output.scores` هي جميع درجات التنبؤ المولدة لرأس النمذجة اللغوية، و`generation_output.attentions` هي `None`.

عند استخدام كائن `generation_output` كمجموعة، فإنه يحتفظ فقط بالسمات التي لا تحتوي على قيم `None`. هنا، على سبيل المثال، يحتوي على عنصرين، `loss` ثم `logits`، لذا

```python
generation_output[:2]
```

ستعيد المجموعة `(generation_output.sequences, generation_output.scores)` على سبيل المثال.

عند استخدام كائن `generation_output` كقاموس، فإنه يحتفظ فقط بالسمات التي لا تحتوي على قيم `None`. هنا، على سبيل المثال، يحتوي على مفتاحين هما `sequences` و`scores`.

نوثق هنا جميع أنواع المخرجات.

### باي تورتش

[[autodoc]] generation.GenerateDecoderOnlyOutput
[[autodoc]] generation.GenerateEncoderDecoderOutput
[[autodoc]] generation.GenerateBeamDecoderOnlyOutput
[[autodoc]] generation.GenerateBeamEncoderDecoderOutput

### تنسورفلو

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

### فلاكس

[[autodoc]] generation.FlaxSampleOutput
[[autodoc]] generation.FlaxGreedySearchOutput
[[autodoc]] generation.FlaxBeamSearchOutput

## معالج الرموز

يمكن استخدام [`LogitsProcessor`] لتعديل درجات التنبؤ لرأس نموذج اللغة من أجل التوليد.

### باي تورتش

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
[[autodoc]] ForceTokensLogitsProcessor
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
[[autodoc]] LogitsWarper
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

### تنسورفلو

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

### فلاكس

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

## معايير التوقف

يمكن استخدام [`StoppingCriteria`] لتغيير وقت إيقاف التوليد (بخلاف رمز EOS). يرجى ملاحظة أن هذا متاح حصريًا لتنفيذات باي تورتش.

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

## القيود

يمكن استخدام [`Constraint`] لإجبار التوليد على تضمين رموز أو تسلسلات محددة في المخرج. يرجى ملاحظة أن هذا متاح حصريًا لتنفيذات باي تورتش.

[[autodoc]] Constraint
[[autodoc]] PhrasalConstraint
[[autodoc]] DisjunctiveConstraint
[[autodoc]] ConstraintListState

## بحث الشعاع

[[autodoc]] BeamScorer
- process
- finalize
[[autodoc]] BeamSearchScorer
- process
- finalize
[[autodoc]] ConstrainedBeamSearchScorer
- process
- finalize

## البث المباشر

[[autodoc]] TextStreamer
[[autodoc]] TextIteratorStreamer

## ذاكرة التخزين المؤقت

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
[[autodoc]] StaticCache
- update
- get_seq_length
- reset

## فائدة علامة مائية

[[autodoc]] WatermarkDetector
- __call__