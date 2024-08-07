# خطوط الأنابيب

تعد خطوط الأنابيب طريقة رائعة وسهلة لاستخدام النماذج من أجل الاستنتاج. هذه خطوط أنابيب هي كائنات تُلخص معظم الرموز المعقدة من المكتبة، وتوفر واجهة برمجة تطبيقات بسيطة مخصصة لعدة مهام، بما في ذلك التعرف على الكيانات المسماة، ونمذجة اللغة المقنعة، وتحليل المشاعر، واستخراج الميزات، والإجابة على الأسئلة. راجع [ملخص المهام](../task_summary) للحصول على أمثلة على الاستخدام.

هناك فئتان من التجريدات الخاصة بخطوط الأنابيب يجب أن تكون على دراية بهما:

- [`pipeline`] وهو أقوى كائن يغلف جميع خطوط الأنابيب الأخرى.
- تتوفر خطوط أنابيب خاصة بمهام معينة للمهام [الصوتية](#audio)، و[الرؤية الحاسوبية](#computer-vision)، و[معالجة اللغات الطبيعية](#natural-language-processing)، و[المتعددة الوسائط](#multimodal).

## تجريد خط الأنابيب

تجريد *خط الأنابيب* هو غلاف حول جميع خطوط الأنابيب الأخرى المتوفرة. يتم إنشاء مثيل له مثل أي خط أنابيب آخر ولكنه يمكن أن يوفر جودة حياة إضافية.

مكالمة بسيطة على عنصر واحد:

```python
>>> pipe = pipeline("text-classification")
>>> pipe("هذا المطعم رائع")
[{'label': 'إيجابي', 'score': 0.9998743534088135}]
```

إذا كنت تريد استخدام نموذج محدد من [المحور](https://huggingface.co)، فيمكنك تجاهل المهمة إذا كان النموذج على المحور يحدد المهمة بالفعل:

```python
>>> pipe = pipeline(model="FacebookAI/roberta-large-mnli")
>>> pipe("هذا المطعم رائع")
[{'label': 'محايد', 'score': 0.7313136458396912}]
```

لاستدعاء خط أنابيب على العديد من العناصر، يمكنك استدعاؤه باستخدام *قائمة*.

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["هذا المطعم رائع", "هذا المطعم فظيع"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
{'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

للحصول على بيانات كاملة، يوصى باستخدام `dataset` مباشرة. وهذا يعني أنك لست بحاجة إلى تخصيص المجموعة الكاملة مرة واحدة، ولا تحتاج إلى إجراء التجميع بنفسك. يجب أن يعمل هذا بنفس سرعة الحلقات المخصصة على GPU. إذا لم يكن الأمر كذلك، فلا تتردد في إنشاء مشكلة.

```python
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

# KeyDataset (only *pt*) will simply return the item in the dict returned by the dataset item
# as we're not interested in the *target* part of the dataset. For sentence pair use KeyPairDataset
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
print(out)
# {"text": "الرقم عشرة نيلي الطازجة تنتظر عليك تصبح على خير زوجي"}
# {"text": ....}
# ....
```

ولتسهيل الاستخدام، يمكن أيضًا استخدام مولد:

```python
from transformers import pipeline

pipe = pipeline("text-classification")


def data():
while True:
# This could come from a dataset, a database, a queue or HTTP request
# in a server
# Caveat: because this is iterative, you cannot use `num_workers > 1` variable
# to use multiple threads to preprocess data. You can still have 1 thread that
# does the preprocessing while the main runs the big inference
yield "هذا اختبار"


for out in pipe(data()):
print(out)
# {"text": "الرقم عشرة نيلي الطازجة تنتظر عليك تصبح على خير زوجي"}
# {"text": ....}
# ....
```

[[autodoc]] pipeline

## تجميع دفعات خط الأنابيب

يمكن لجميع خطوط الأنابيب استخدام الدفعات. سيعمل هذا عندما يستخدم خط الأنابيب قدرته على البث (لذا عند تمرير القوائم أو `Dataset` أو `generator`).

```python
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
print(out)
# [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
# نفس الإخراج تمامًا كما كان من قبل، ولكن يتم تمرير المحتويات
# كدفعات إلى النموذج
```

<Tip warning={true}>
ومع ذلك، فإن هذا ليس فوزًا تلقائيًا للأداء. يمكن أن يكون إما زيادة في السرعة بمقدار 10 مرات أو تباطؤ بمقدار 5 مرات اعتمادًا
على الأجهزة والبيانات والنموذج الفعلي المستخدم.

المثال الذي يكون فيه في الغالب زيادة في السرعة:

</Tip>

```python
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)


class MyDataset(Dataset):
def __len__(self):
return 5000

def __getitem__(self, i):
return "هذا اختبار"


dataset = MyDataset()

for batch_size in [1, 8, 64, 256]:
print("-" * 30)
print(f"Streaming batch_size={batch_size}")
for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
pass
```

```
# على GTX 970
------------------------------
Streaming no batching
100%|██████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
------------------------------
Streaming batch_size=64
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
------------------------------
Streaming batch_size=256
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
(عائدات متناقصة، مشبعة GPU)
```

المثال الذي يكون فيه في الغالب تباطؤ:

```python
class MyDataset(Dataset):
def __len__(self):
return 5000

def __getitem__(self, i):
if i % 64 == 0:
n = 100
else:
n = 1
return "هذا اختبار" * n
```

هذه جملة طويلة جدًا بشكل عرضي مقارنة بالجمل الأخرى. في هذه الحالة، ستحتاج الدفعة بأكملها إلى أن تكون 400
رمزًا طويلاً، لذا ستكون الدفعة بأكملها [64، 400] بدلاً من [64، 4]، مما يؤدي إلى التباطؤ الشديد. والأسوأ من ذلك، أنه في
الدفعات الأكبر، يتعطل البرنامج ببساطة.

```
------------------------------
Streaming no batching
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:05<00:00, 183.69it/s]
------------------------------
Streaming batch_size=8
100%|█████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 265.74it/s]
------------------------------
Streaming batch_size=64
100%|██████████████████████████████████████████████████████████████████████| 1000/1000 [00:26<00:00, 37.80it/s]
------------------------------
Streaming batch_size=256
0%|                                                                                 | 0/1000 [00:00<?, ?it/s]
Traceback (most recent call last):
File "/home/nicolas/src/transformers/test.py", line 42, in <module>
for out in tqdm(pipe(dataset, batch_size=256), total=len(dataset)):
....
q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
RuntimeError: نفاد ذاكرة CUDA. حاول تخصيص 376.00 ميغابايت (GPU 0؛ سعة إجمالية تبلغ 3.95 غيغابايت؛ 1.72 غيغابايت مخصصة بالفعل؛ 354.88 ميغابايت مجانية؛ 2.46 غيغابايت محجوزة في المجموع بواسطة PyTorch)
```

لا توجد حلول جيدة (عامة) لهذه المشكلة، وقد تختلف الأميال الخاصة بك حسب حالات الاستخدام الخاصة بك. القاعدة العامة هي:

بالنسبة للمستخدمين، القاعدة العامة هي:

- **قس الأداء على حملك، بأجهزتك. قس، وقس، واستمر في القياس. الأرقام الحقيقية هي الطريقة الوحيدة للذهاب.**
- إذا كنت مقيدًا بالزمن (منتج مباشر يقوم بالاستدلال)، فلا تقم بالدفعات.
- إذا كنت تستخدم وحدة المعالجة المركزية، فلا تقم بالدفعات.
- إذا كنت تستخدم الإنتاجية (تريد تشغيل نموذجك على مجموعة من البيانات الثابتة)، على GPU، ثم:
- إذا لم يكن لديك أي فكرة عن حجم "sequence_length" (البيانات "الطبيعية")، فلا تقم بالدفعات بشكل افتراضي، وقس و
حاول إضافة دفعات بشكل تجريبي، وأضف فحوصات OOM للتعافي عندما يفشل (وسيفشل في مرحلة ما إذا لم تقم بالتحكم
في "sequence_length".)
- إذا كان "sequence_length" الخاص بك منتظمًا جدًا، فمن المرجح أن تكون الدفعات مثيرة للاهتمام جدًا، فقسها وادفعها حتى تحصل على OOMs.
- كلما كان GPU أكبر، زاد احتمال أن تكون الدفعات أكثر إثارة للاهتمام
- بمجرد تمكين الدفعات، تأكد من أنه يمكنك التعامل مع OOMs بشكل لطيف.

## تجميع دفعات خط أنابيب الجزء

`zero-shot-classification` و`question-answering` هما أمران محددان بعض الشيء، بمعنى أن إدخالًا واحدًا قد يؤدي إلى عدة تمريرات للأمام للنموذج. في ظل الظروف العادية، قد يؤدي ذلك إلى حدوث مشكلات مع وسيط `batch_size`.

للتغلب على هذه المشكلة، فإن كلاً من خطوط الأنابيب هذه محددة بعض الشيء، فهي `ChunkPipeline` بدلاً من
خط أنابيب عادي. باختصار:

```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

الآن يصبح:

```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
model_outputs = pipe.forward(preprocessed)
all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

يجب أن يكون هذا شفافًا جدًا لرمزك لأن خطوط الأنابيب تستخدم بنفس الطريقة.

هذه نظرة مبسطة، نظرًا لأن خط الأنابيب يمكنه التعامل تلقائيًا مع الدفعة إلى! وهذا يعني أنك لست مضطرًا للاهتمام
بعدد تمريرات النموذج التي ستتطلبها الإدخالات بالفعل، ويمكنك تحسين `batch_size`
بصرف النظر عن الإدخالات. لا تزال التحذيرات من القسم السابق سارية.
## خط الأنابيب التعليمات البرمجية المخصصة

إذا كنت ترغب في تجاوز خط أنابيب محدد.

لا تتردد في إنشاء مشكلة لمهمتك الحالية، حيث أن هدف خط الأنابيب هو أن يكون سهل الاستخدام ويدعم معظم الحالات، لذلك قد يدعم "المحولات" حالتك الاستخدامية.

إذا كنت ترغب في التجربة ببساطة، يمكنك:

- إنشاء فئة فرعية لخط الأنابيب الذي تختاره

## تنفيذ خط أنابيب

[تنفيذ خط أنابيب جديد](../add_new_pipeline)

## الصوت

تشمل خطوط الأنابيب المتاحة لمهمة الصوت ما يلي.

### AudioClassificationPipeline

[[autodoc]] AudioClassificationPipeline

- __call__

- الكل

### AutomaticSpeechRecognitionPipeline

[[autodoc]] AutomaticSpeechRecognitionPipeline

- __call__

- الكل

### TextToAudioPipeline

[[autodoc]] TextToAudioPipeline

- __call__

- الكل

### ZeroShotAudioClassificationPipeline

[[autodoc]] ZeroShotAudioClassificationPipeline

- __call__

- الكل

## رؤية الكمبيوتر

تشمل خطوط الأنابيب المتاحة لمهمة رؤية الكمبيوتر ما يلي.

### DepthEstimationPipeline

[[autodoc]] DepthEstimationPipeline

- __call__

- الكل

### ImageClassificationPipeline

[[autodoc]] ImageClassificationPipeline

- __call__

- الكل

### ImageSegmentationPipeline

[[autodoc]] ImageSegmentationPipeline

- __call__

- الكل

### ImageToImagePipeline

[[autodoc]] ImageToImagePipeline

- __call__

- الكل

### ObjectDetectionPipeline

[[autodoc]] ObjectDetectionPipeline

- __call__

- الكل

### VideoClassificationPipeline

[[autodoc]] VideoClassificationPipeline

- __call__

- الكل

### ZeroShotImageClassificationPipeline

[[autodoc]] ZeroShotImageClassificationPipeline

- __call__

- الكل

### ZeroShotObjectDetectionPipeline

[[autodoc]] ZeroShotObjectDetectionPipeline

- __call__

- الكل

## معالجة اللغات الطبيعية

تشمل خطوط الأنابيب المتاحة لمهمة معالجة اللغات الطبيعية ما يلي.

### FillMaskPipeline

[[autodoc]] FillMaskPipeline

- __call__

- الكل

### QuestionAnsweringPipeline

[[autodoc]] QuestionAnsweringPipeline

- __call__

- الكل

### SummarizationPipeline

[[autodoc]] SummarizationPipeline

- __call__

- الكل

### TableQuestionAnsweringPipeline

[[autodoc]] TableQuestionAnsweringPipeline

- __call__

### TextClassificationPipeline

[[autodoc]] TextClassificationPipeline

- __call__

- الكل

### TextGenerationPipeline

[[autodoc]] TextGenerationPipeline

- __call__

- الكل

### Text2TextGenerationPipeline

[[autodoc]] Text2TextGenerationPipeline

- __call__

- الكل

### TokenClassificationPipeline

[[autodoc]] TokenClassificationPipeline

- __call__

- الكل

### TranslationPipeline


[[autodoc]] TranslationPipeline

- __call__

- الكل

### ZeroShotClassificationPipeline

[[autodoc]] ZeroShotClassificationPipeline

- __call__

- الكل

## متعدد الوسائط

تشمل خطوط الأنابيب المتاحة للمهام متعددة الوسائط ما يلي.

### DocumentQuestionAnsweringPipeline

[[autodoc]] DocumentQuestionAnsweringPipeline


- __call__

- الكل

### FeatureExtractionPipeline

[[autodoc]] FeatureExtractionPipeline

- __call__

- الكل

### ImageFeatureExtractionPipeline

[[autodoc]] ImageFeatureExtractionPipeline

- __call__

- الكل

### ImageToTextPipeline

[[autodoc]] ImageToTextPipeline

- __call__

- الكل

### MaskGenerationPipeline

[[autodoc]] MaskGenerationPipeline

- __call__

- الكل

### VisualQuestionAnsweringPipeline

[[autodoc]] VisualQuestionAnsweringPipeline

- __call__

- الكل

## الفئة الأصلية: `Pipeline`

[[autodoc]] Pipeline