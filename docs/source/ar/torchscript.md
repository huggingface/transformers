# التصدير إلى TorchScript

<Tip>

هذه هي بداية تجاربنا مع TorchScript ولا زلنا نستكشف قدراته مع نماذج المدخلات المتغيرة الحجم. إنه مجال اهتمامنا وسنعمق تحليلنا في الإصدارات القادمة، مع المزيد من الأمثلة البرمجية، وتنفيذ أكثر مرونة، ومقاييس مقارنة بين  الأكواد القائمة على Python مع أكواد TorchScript المُجمّعة.

</Tip>

وفقًا لـ [وثائق TorchScript](https://pytorch.org/docs/stable/jit.html):

> TorchScript هي طريقة لإنشاء نماذج قابلة للتسلسل والتحسين من تعليمات PyTorch البرمجية.

هناك وحدتان من PyTorch، [JIT and TRACE](https://pytorch.org/docs/stable/jit.html)، تتيحان للمطورين تصدير نماذجهم لإعادة استخدامها في برامج أخرى مثل برامج C++ المُحسّنة للأداء.

نقدم واجهة تتيح لك تصدير نماذج 🤗 Transformers إلى TorchScript بحيث يمكن إعادة استخدامها في بيئة مختلفة عن برامج Python القائمة إلى PyTorch. هنا نشرح كيفية تصدير نماذجنا واستخدامها باستخدام TorchScript.

يتطلب تصدير نموذج أمرين:

- تهيئة مثيل للنموذج باستخدام علامة `torchscript`
- تمرير مُدخلات وهمية (dummy inputs) خلال النموذج

تنطوي هذه الضرورات على عدة أمور يجب على المطورين توخي الحذر بشأنها كما هو مفصل أدناه.

## علامة TorchScript والأوزان المرتبطة

علامة `torchscript` ضرورية لأن معظم نماذج اللغة 🤗 Transformers لها أوزان مرتبطة بين طبقة `Embedding` وطبقة `Decoding`. لا يسمح لك TorchScript بتصدير النماذج ذات الأوزان المرتبطة، لذلك من الضروري فصل الأوزان ونسخها مسبقًا.

النماذج المُهيأة باستخدام علامة `torchscript` لها طبقة `Embedding` وطبقة`Decoding` منفصلتين، مما يعني أنه لا ينبغي تدريبها لاحقًا. سيؤدي التدريب إلى عدم تزامن الطبقتين، مما يؤدي إلى نتائج غير متوقعة.

هذا لا ينطبق على النماذج التي لا تحتوي على رأس نموذج اللغة، حيث لا تملك أوزانًا مرتبطة. يمكن تصدير هذه النماذج بأمان دون علامة `torchscript`.

## المدخلات الوهمية والأطوال القياسية

تُستخدم المُدخلات الوهمية لتمرير أمامي خلال النموذج. أثناء انتشار قيم المُدخلات عبر الطبقات، يتتبع PyTorch العمليات المختلفة التي يتم تنفيذها على كل مصفوفة(tensor). ثم يتم استخدام هذه العمليات المُسجلة بعد ذلك لإنشاء *أثر* النموذج.

يتم إنشاء التتبع بالنسبة لأبعاد المُدخلات. وبالتالي، فهو مُقيّد بأبعاد المُدخلات الوهمية، ولن يعمل لأي طول تسلسل أو حجم دفعة مختلف. عند المحاولة بحجم مختلف، يتم رفع الخطأ التالي:

```
`The expanded size of the tensor (3) must match the existing size (7) at non-singleton dimension 2`
```

نوصي بتتبع النموذج باستخدام حجم مُدخلات وهمية لا يقل عن أكبر مُدخل سيتم تقديمه للنموذج أثناء الاستدلال. يمكن أن تساعد الحشوة(padding) في ملء القيم المفقودة. ومع ذلك، نظرًا لتتبع النموذج بحجم مُدخل أكبر، ستكون أبعاد المصفوفة ستكون كبيرة أيضًا، مما يؤدي عنه المزيد من الحسابات.

انتبه إلى إجمالي عدد العمليات المُنفذة على كل مُدخل وتابع الأداء عن كثب عند تصدير نماذج متغيرة طول التسلسل.

## استخدام TorchScript في Python

يوضح هذا القسم كيفية حفظ النماذج وتحميلها، بالإضافة إلى كيفية استخدام التتبع للاستدلال.

### حفظ نموذج

لتصدير `BertModel` باستخدام TorchScript، قم بتهيئة ـ `BertModel` من فئة `BertConfig` ثم احفظه على القرص تحت اسم الملف `traced_bert.pt`:

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

يمكنك الآن تحميل `BertModel` المُحفظ سابقًا، `traced_bert.pt`، من القرص واستخدامه على `dummy_input` المُهيأ سابقًا:

```python
loaded_model = torch.jit.load("traced_bert.pt")
loaded_model.eval()

all_encoder_layers, pooled_output = loaded_model(*dummy_input)
```

### استخدام نموذج مُتتبع للاستدلال

استخدم النموذج المُتتبع للاستدلال باستخدام أسلوب `__call__` الخاص به:

```python
traced_model(tokens_tensor, segments_tensors)
```

## نشر نماذج Hugging Face TorchScript على AWS باستخدام Neuron SDK

قدمت AWS عائلة [Amazon EC2 Inf1](https://aws.amazon.com/ec2/instance-types/inf1/) من اﻷجهزة لخفض التكلفة وأداء التعلم الآلي عالي الأداء في البيئة السحابية. تعمل أجهزة Inf1 بواسطة شريحة Inferentia من AWS، وهي مُسرّع أجهزة مُخصص، متخصص في أعباء عمل الاستدلال للتعلم العميق. [AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/#) هي SDK لـ Inferentia التي تدعم تتبع نماذج المحولات وتحسينها للنشر على Inf1. توفر Neuron SDK ما يلي:

1. واجهة برمجة تطبيقات سهلة الاستخدام مع تغيير سطر واحد من التعليمات البرمجية لتتبع نموذج TorchScript وتحسينه للاستدلال في البيئة السحابية.
2. تحسينات الأداء الجاهزة للاستخدام [تحسين التكلفة والأداء](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/benchmark/>).
3. دعم نماذج Hugging Face المحولات المبنية باستخدام إما [PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.html) أو [TensorFlow](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/tensorflow/huggingface_bert/huggingface_bert.html).

### الآثار المترتبة

تعمل نماذج المحولات المستندة إلى بنية [BERT (تمثيلات الترميز ثنائية الاتجاه من المحولات)](https://huggingface.co/docs/transformers/main/model_doc/bert) أو متغيراتها مثل [distilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert) و [roBERTa](https://huggingface.co/docs/transformers/main/model_doc/roberta) بشكل أفضل على Inf1 للمهام غير التوليدية مثل الإجابة على الأسئلة الاستخراجية، وتصنيف التسلسلات، وتصنيف الرموز (tokens). ومع ذلك، يمكن تكييف مهام توليد النصوص للعمل على Inf1 وفقًا لهذا [برنامج تعليمي AWS Neuron MarianMT](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html). يمكن العثور على مزيد من المعلومات حول النماذج التي يمكن تحويلها جاهزة على Inferentia في قسم [ملاءمة بنية النموذج](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/models/models-inferentia.html#models-inferentia) من وثائق Neuron.

### التبعيات (Dependencies)

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
