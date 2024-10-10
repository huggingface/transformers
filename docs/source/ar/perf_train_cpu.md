# التدريب الفعال على وحدة المعالجة المركزية

يركز هذا الدليل على التدريب الفعال للنماذج الكبيرة على وحدة المعالجة المركزية.

## الدقة المختلطة مع IPEX
تستخدم الدقة المختلطة أنواع البيانات ذات الدقة الفردية (fp32) ونصف الدقة (bf16/fp16) في نموذج لتسريع التدريب أو الاستدلال مع الحفاظ على الكثير من دقة الدقة الفردية. تدعم وحدات المعالجة المركزية الحديثة مثل الجيل الثالث والرابع من معالجات Intel® Xeon® Scalable الدقة bf16 بشكلٍ أصلي، لذلك يجب أن تحصل على أداء أفضل بشكلٍ افتراضي من خلال تمكين التدريب على الدقة المختلطة باستخدام bf16.

لزيادة تحسين أداء التدريب، يمكنك استخدام Intel® Extension لـ PyTorch (IPEX)، وهي مكتبة مبنية على PyTorch وتضيف دعمًا إضافيًا على مستوى بنية تعليمات CPU مثل Intel® Advanced Vector Extensions 512 Vector Neural Network Instructions (Intel® AVX512-VNNI)، وIntel® Advanced Matrix Extensions (Intel® AMX) لتعزيز الأداء على معالجات Intel. ومع ذلك، لا يُضمن أن يكون للأجهزة التي تحتوي على AVX2 فقط (مثل AMD أو معالجات Intel الأقدم) أداء أفضل في ظل IPEX.

تم تمكين الدقة المختلطة التلقائية (AMP) لواجهات برمجة التطبيقات الخلفية لوحدة المعالجة المركزية منذ PyTorch 1.10. كما يتم دعم AMP لـ bf16 على وحدات المعالجة المركزية وتحسين مشغل bf16 في IPEX وتم نقله جزئيًا إلى الفرع الرئيسي لـ PyTorch. يمكنك الحصول على أداء أفضل وتجربة مستخدم محسّنة مع AMP IPEX.

تحقق من مزيد من المعلومات التفصيلية حول [الدقة المختلطة التلقائية](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/amp.html).

### تثبيت IPEX:

يتبع إصدار IPEX إصدار PyTorch، لتثبيته عبر pip:

| إصدار PyTorch | إصدار IPEX |
| :---------------: | :----------: |
| 2.1.x | 2.1.100+cpu |
| 2.0.x | 2.0.100+cpu |
| 1.13 | 1.13.0+cpu |
| 1.12 | 1.12.300+cpu |

يرجى تشغيل `pip list | grep torch` للحصول على إصدار PyTorch، بحيث يمكنك الحصول على اسم إصدار IPEX.
```bash
pip install intel_extension_for_pytorch==<version_name> -f https://developer.intel.com/ipex-whl-stable-cpu
```
يمكنك التحقق من أحدث الإصدارات في [ipex-whl-stable-cpu](https://developer.intel.com/ipex-whl-stable-cpu) إذا لزم الأمر.

تحقق من المزيد من الطرق لتثبيت [IPEX](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/installation.html).

### الاستخدام في Trainer
لتمكين الدقة المختلطة التلقائية مع IPEX في Trainer، يجب على المستخدمين إضافة `use_ipex` و`bf16` و`no_cuda` في وسائط الأوامر التدريبية.

خذ مثالاً على حالات الاستخدام في [تحويلات الأسئلة والأجوبة](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)

- التدريب باستخدام IPEX مع الدقة المختلطة التلقائية BF16 على وحدة المعالجة المركزية:
<pre> python run_qa.py \
--model_name_or_path google-bert/bert-base-uncased \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 12 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir /tmp/debug_squad/ \
<b>--use_ipex</b> \
<b>--bf16</b> \
<b>--use_cpu</b></pre> 

إذا كنت تريد تمكين `use_ipex` و`bf16` في نصك البرمجي، فقم بإضافة هذه المعلمات إلى `TrainingArguments` على النحو التالي:
```diff
training_args = TrainingArguments(
    output_dir=args.output_path,
+   bf16=True,
+   use_ipex=True,
+   use_cpu=True,
    **kwargs
)
```

### مثال عملي

مقال المدونة: [تسريع تحويلات PyTorch باستخدام Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids)