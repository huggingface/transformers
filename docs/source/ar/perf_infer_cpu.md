# الاستنتاج باستخدام وحدة المعالجة المركزية (CPU Inference)

مع بعض التحسينات، يمكن تشغيل استنتاج النماذج الكبيرة بكفاءة على وحدة المعالجة المركزية (CPU). وتتضمن إحدى تقنيات التحسين هذه تجميع كود PyTorch إلى تنسيق وسيط لبيئات عالية الأداء مثل C++. وتقوم تقنية أخرى بدمج عدة عمليات في نواة واحدة لتقليل النفقات العامة لتشغيل كل عملية على حدة.

ستتعلم كيفية استخدام BetterTransformer للاستنتاج بشكل أسرع، وكيفية تحويل كود PyTorch الخاص بك إلى TorchScript. إذا كنت تستخدم معالج Intel، فيمكنك أيضًا استخدام التحسينات الرسومية من Intel Extension for PyTorch لزيادة سرعة الاستنتاج بشكل أكبر. أخيرًا، تعرف على كيفية استخدام Hugging Face Optimum لتسريع الاستنتاج باستخدام ONNX Runtime أو OpenVINO (إذا كنت تستخدم معالج Intel).

## BetterTransformer

يُسرع BetterTransformer الاستنتاج من خلال تنفيذ fastpath (التنفيذ المتخصص لـ PyTorch الأصلي لوظائف Transformer). ويتمثل التحسينان في تنفيذ fastpath فيما يلي:

1. الدمج (fusion)، والذي يجمع بين عدة عمليات متتالية في "نواة" واحدة لتقليل عدد خطوات الحساب.
2. تخطي ندرة التوكينز (tokens) المبطنة المتأصلة لتجنب الحسابات غير الضرورية مع المصفوفات المُستَعشَقة (nested tensors).

كما يحول BetterTransformer جميع عمليات الانتباه لاستخدام [scaled dot product attention](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) الأكثر كفاءة في الذاكرة.

<Tip>

BetterTransformer غير مدعوم لجميع النماذج. تحقق من هذه [القائمة](https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models) لمعرفة ما إذا كان النموذج يدعم BetterTransformer.

</Tip>

قبل البدء، تأكد من تثبيت Hugging Face Optimum [installed](https://huggingface.co/docs/optimum/installation).

قم بتمكين BetterTransformer باستخدام طريقة [`PreTrainedModel.to_bettertransformer`]:

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder")
model.to_bettertransformer()
```

## TorchScript

TorchScript هو تمثيل وسيط لنموذج PyTorch يمكن تشغيله في بيئات الإنتاج حيث تكون الأداء مهمًا. يمكنك تدريب نموذج في PyTorch ثم تصديره إلى TorchScript لتحرير النموذج من قيود الأداء في Python. تقوم PyTorch [بتتبع](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) نموذج لإرجاع [`ScriptFunction`] يتم تحسينه باستخدام التجميع في الوقت المناسب (JIT). مقارنة بوضع eager الافتراضي، عادةً ما يوفر وضع JIT في PyTorch أداء أفضل للاستنتاج باستخدام تقنيات التحسين مثل دمج المشغل.

للبدء السريع مع TorchScript، راجع البرنامج التعليمي [Introduction to PyTorch TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html).

مع فئة [`Trainer`]، يمكنك تمكين وضع JIT للاستنتاج باستخدام وحدة المعالجة المركزية عن طريق تعيين علامة `--jit_mode_eval`:

```bash
python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
--jit_mode_eval
```

<Tip warning={true}>

بالنسبة لـ PyTorch >= 1.14.0، يمكن أن يستفيد وضع JIT أي نموذج للتنبؤ والتقييم منذ دعم إدخال القاموس في `jit.trace`.

بالنسبة لـ PyTorch < 1.14.0، يمكن أن يفيد وضع JIT نموذجًا إذا تطابق ترتيب معلمات التقديم الخاصة به مع ترتيب إدخال الرباعية في `jit.trace`، مثل نموذج الإجابة على الأسئلة. إذا لم يتطابق ترتيب معلمات التقديم مع ترتيب إدخال الرباعية في `jit.trace`، مثل نموذج تصنيف النص، فسوف يفشل `jit.trace` ونحن نلتقط هذا الاستثناء هنا لجعله يتراجع. يتم استخدام التسجيل لإخطار المستخدمين.

</Tip>

## تحسين الرسم البياني لـ IPEX

يوفر Intel® Extension for PyTorch (IPEX) مزيدًا من التحسينات في وضع JIT لمعالجات Intel، ونوصي بدمجه مع TorchScript للحصول على أداء أسرع. يقوم تحسين الرسم البياني لـ IPEX [graph optimization](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html) بدمج العمليات مثل Multi-head attention، وConcat Linear، وLinear + Add، وLinear + Gelu، وAdd + LayerNorm، والمزيد.

للاستفادة من تحسينات الرسم البياني هذه، تأكد من تثبيت IPEX [installed](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html):

```bash
pip install intel_extension_for_pytorch
```

قم بتعيين علامتي `--use_ipex` و `--jit_mode_eval` في فئة [`Trainer`] لتمكين وضع JIT مع تحسينات الرسم البياني:

```bash
python run_qa.py \
--model_name_or_path csarron/bert-base-uncased-squad-v1 \
--dataset_name squad \
--do_eval \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/ \
--no_cuda \
--use_ipex \
--jit_mode_eval
```

## Hugging Face Optimum

<Tip>

تعرف على المزيد من التفاصيل حول استخدام ORT مع Hugging Face Optimum في دليل [Optimum Inference with ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models). يقدم هذا القسم فقط مثالًا موجزًا وبسيطًا.

</Tip>

ONNX Runtime (ORT) هو مسرع نموذج يقوم بتشغيل الاستنتاج على وحدات المعالجة المركزية بشكل افتراضي. تدعم Hugging Face Optimum استخدام ONNX Runtime، والذي يمكن استخدامه في Hugging Face Transformers، دون إجراء الكثير من التغييرات على كودك. كل ما عليك فعله هو استبدال فئة `AutoClass` من Hugging Face Transformers بما يعادلها [`~optimum.onnxruntime.ORTModel`] للمهمة التي تحاول حلها، وتحميل نقطة تفتيش بتنسيق ONNX.

على سبيل المثال، إذا كنت تقوم بالاستنتاج في مهمة الإجابة على الأسئلة، فقم بتحميل نقطة التفتيش [optimum/roberta-base-squad2](https://huggingface.co/optimum/roberta-base-squad2) التي تحتوي على ملف `model.onnx`:

```py
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

model = ORTModelForQuestionAnswering.from_pretrained("optimum/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

onnx_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

question = "What's my name?"
context = "My name is Philipp and I live in Nuremberg."
pred = onnx_qa(question, context)
```

إذا كان لديك معالج Intel، فراجع Hugging Face [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) الذي يدعم مجموعة متنوعة من تقنيات الضغط (الكمية، التشذيب، تقطير المعرفة) وأدوات لتحويل النماذج إلى تنسيق [OpenVINO](https://huggingface.co/docs/optimum/intel/inference) للاستدلال عالي الأداء.