# التصدير إلى TorchScript

<Tip>

هذه هي بداية تجاربنا مع TorchScript ولا زلنا نستكشف قدراته مع نماذج المدخلات المتغيرة الحجم. إنه مجال اهتمامنا وسنعمق تحليلنا في الإصدارات القادمة، مع المزيد من أمثلة التعليمات البرمجية، وتنفيذ أكثر مرونة، ومقاييس مقارنة بين التعليمات البرمجية المستندة إلى Python مع TorchScript المترجمة.

</Tip>

وفقًا لـ [وثائق TorchScript](https://pytorch.org/docs/stable/jit.html):

> TorchScript هي طريقة لإنشاء نماذج قابلة للتسلسل والتحسين من تعليمات PyTorch البرمجية.

هناك وحدتان من PyTorch، [JIT and TRACE](https://pytorch.org/docs/stable/jit.html)، تتيحان للمطورين تصدير نماذجهم لإعادة استخدامها في برامج أخرى مثل برامج C++ الموجهة نحو الكفاءة.

نقدم واجهة تتيح لك تصدير نماذج 🤗 Transformers إلى TorchScript بحيث يمكن إعادة استخدامها في بيئة مختلفة عن برامج Python المستندة إلى PyTorch. هنا نشرح كيفية تصدير نماذجنا واستخدامها باستخدام TorchScript.

يتطلب تصدير نموذج أمرين:

- إنشاء مثيل للنموذج باستخدام علم `torchscript`
- تمرير إلى الأمام باستخدام مدخلات وهمية

تنطوي هذه الضرورات على عدة أمور يجب على المطورين توخي الحذر بشأنها كما هو مفصل أدناه.

## علم TorchScript والأوزان المرتبطة

علم `torchscript` ضروري لأن معظم نماذج اللغة 🤗 Transformers لها أوزان مرتبطة بين طبقة `Embedding` وطبقة `Decoding`. لا يسمح لك TorchScript بتصدير النماذج التي تحتوي على أوزان مرتبطة، لذلك من الضروري فصل الأوزان ونسخها مسبقًا.

النماذج التي تم إنشاؤها باستخدام علم `torchscript` لها طبقة `Embedding` و`Decoding` منفصلة، مما يعني أنه لا ينبغي تدريبها لاحقًا. سيؤدي التدريب إلى عدم تزامن الطبقتين، مما يؤدي إلى نتائج غير متوقعة.

هذا لا ينطبق على النماذج التي لا تحتوي على رأس نموذج اللغة، حيث لا تحتوي على أوزان مرتبطة. يمكن تصدير هذه النماذج بأمان دون علم `torchscript`.

## المدخلات الوهمية والأطوال القياسية

تُستخدم المدخلات الوهمية للمرور الأمامي للنموذج. بينما يتم تمرير قيم المدخلات عبر الطبقات، تقوم PyTorch بتتبع العمليات المختلفة التي يتم تنفيذها على كل مصفوفة. ثم يتم استخدام هذه العمليات المسجلة لإنشاء *تتبع* النموذج.

يتم إنشاء التتبع بالنسبة لأبعاد المدخلات. وبالتالي، فهو مقيد بأبعاد المدخلات الوهمية، ولن يعمل لأي طول تسلسل أو حجم دفعة آخر. عند المحاولة بحجم مختلف، يتم رفع الخطأ التالي:

```
`يجب أن يتطابق الحجم الموسع للمصفوفة (3) مع الحجم الموجود (7) في البعد 2 غير المفرد`
```

نوصي بتتبع النموذج باستخدام حجم إدخال وهمي أكبر على الأقل مثل أكبر إدخال سيتم إطعامه للنموذج أثناء الاستدلال. يمكن أن تساعد الحشوة في ملء القيم المفقودة. ومع ذلك، نظرًا لأنه يتم تتبع النموذج باستخدام حجم إدخال أكبر، فإن أبعاد المصفوفة ستكون كبيرة أيضًا، مما يؤدي إلى مزيد من الحسابات.

كن حذرًا من إجمالي عدد العمليات التي تتم على كل إدخال واتبع الأداء عن كثب عند تصدير نماذج طول التسلسل المتغيرة.

## استخدام TorchScript في Python
كن حذرًا من إجمالي عدد العمليات التي تتم على كل إدخال واتبع الأداء عن كثب عند تصدير نماذج طول التسلسل المتغيرة.

## استخدام TorchScript في Python

يوضح هذا القسم كيفية حفظ النماذج وتحميلها، بالإضافة إلى كيفية استخدام التتبع للاستدلال.

### حفظ نموذج

لتصدير `BertModel` مع TorchScript، قم بإنشاء مثيل لـ `BertModel` من فئة `BertConfig` ثم احفظه على القرص تحت اسم الملف `traced_bert.pt`:

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch

enc = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = "[MASK]"
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(
    vocab_size_or_config_json_file=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    torchscript=True,
)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with *from_pretrained* you can also easily set the TorchScript flag
model = BertModel.from_pretrained("google-bert/bert-base-uncased", torchscript=True)

# Creating the trace
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
torch.jit.save(traced_model, "traced_bert.pt")
```

### تحميل نموذج

الآن يمكنك تحميل `BertModel` المحفوظ سابقًا، `traced_bert.pt`، من القرص واستخدامه على `dummy_input` الذي تم تهيئته مسبقًا:

```python
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```

### استخدام نموذج تم تتبعه للاستدلال

استخدم النموذج الذي تم تتبعه للاستدلال باستخدام أسلوب `__call__` الخاص به:

```python
traced_model(tokens_tensor, segments_tensors)
```

## نشر نماذج Hugging Face TorchScript على AWS باستخدام Neuron SDK

قدمت AWS عائلة [Amazon EC2 Inf1](https://aws.amazon.com/ec2/instance-types/inf1/) من مثيلات لخفض التكلفة وأداء التعلم الآلي عالي الأداء في السحابة. تعمل مثيلات Inf1 بواسطة شريحة Inferentia من AWS، وهي مسرع أجهزة مخصص، متخصص في أعباء عمل الاستدلال للتعلم العميق. [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#) هي SDK لـ Inferentia التي تدعم تتبع نماذج المحولات وتحسينها للنشر على Inf1. توفر Neuron SDK ما يلي:

1. واجهة برمجة تطبيقات سهلة الاستخدام مع تغيير سطر واحد من التعليمات البرمجية لتتبع نموذج TorchScript وتحسينه للاستدلال في السحابة.
2. تحسينات الأداء الجاهزة للاستخدام [تحسين التكلفة والأداء](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/benchmark/>).
3. دعم نماذج Hugging Face المحولات المبنية باستخدام إما [بايثون](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html) أو [تنسورفلو](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/huggingface_bert/huggingface_bert.html).

### الآثار المترتبة
### الآثار المترتبة

تعمل نماذج المحولات المستندة إلى بنية [BERT (تمثيلات الترميز ثنائية الاتجاه من المحولات)](https://huggingface.co/docs/transformers/main/model_doc/bert) أو متغيراتها مثل [distilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert) و [roBERTa](https://huggingface.co/docs/transformers/main/model_doc/roberta) بشكل أفضل على Inf1 للمهام غير التوليدية مثل الإجابة على الأسئلة الاستخراجية، وتصنيف التسلسلات، وتصنيف الرموز. ومع ذلك، يمكن تكييف مهام توليد النصوص للعمل على Inf1 وفقًا لهذا [برنامج تعليمي AWS Neuron MarianMT](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html). يمكن العثور على مزيد من المعلومات حول النماذج التي يمكن تحويلها من الصندوق على Inferentia في قسم [ملاءمة بنية النموذج](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/models/models-inferentia.html#models-inferentia) من وثائق Neuron.

### التبعيات

يتطلب استخدام AWS Neuron لتحويل النماذج [بيئة SDK Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html#installation-guide) والتي تأتي مسبقًا على [AMI للتعلم العميق من AWS](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia-launching.html).

### تحويل نموذج لـ AWS Neuron

قم بتحويل نموذج لـ AWS NEURON باستخدام نفس التعليمات البرمجية من [استخدام TorchScript في Python](torchscript#using-torchscript-in-python) لتتبع `BertModel`. قم باستيراد امتداد إطار عمل `torch.neuron` للوصول إلى مكونات Neuron SDK من خلال واجهة برمجة تطبيقات Python:

```python
from transformers import BertModel, BertTokenizer, BertConfig
import torch
import torch.neuron
```

كل ما عليك فعله هو تعديل السطر التالي:

```diff
- torch.jit.trace(model, [tokens_tensor, segments_tensors])
+ torch.neuron.trace(model, [token_tensor, segments_tensors])
```

يتيح ذلك لـ Neuron SDK تتبع النموذج وتحسينه لمثيلات Inf1.

لمعرفة المزيد حول ميزات AWS Neuron SDK والأدوات ودروس البرامج التعليمية والتحديثات الأخيرة، يرجى الاطلاع على [وثائق AWS NeuronSDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/index.html).