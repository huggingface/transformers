# تدريب PyTorch على Apple silicon

سابقًا، كان تدريب النماذج على جهاز Mac يقتصر على وحدة المعالجة المركزية فقط. مع إصدار PyTorch v1.12، يمكنك الاستفادة من تدريب النماذج باستخدام معالجات الرسوميات (GPUs) من Apple silicon للحصول على أداء وتدريب أسرع بشكل ملحوظ. يتم تشغيل هذه الميزة في PyTorch من خلال دمج معالجات الرسوميات من Apple Metal Performance Shaders (MPS) كخلفية. تقوم خلفية [MPS](https://pytorch.org/docs/stable/notes/mps.html) بتنفيذ عمليات PyTorch على شكل معالجات رسومات مخصصة وتضع هذه الوحدات على جهاز `mps`.

<Tip warning={true}>

لم يتم بعد تنفيذ بعض عمليات PyTorch في MPS وقد تتسبب في حدوث خطأ. لتجنب ذلك، يجب عليك تعيين متغير البيئة `PYTORCH_ENABLE_MPS_FALLBACK=1` لاستخدام نوى وحدة المعالجة المركزية بدلاً من ذلك (ستظل ترى `UserWarning`).

<br>

إذا واجهتك أي أخطاء أخرى، يرجى فتح قضية في مستودع [PyTorch](https://github.com/pytorch/pytorch/issues) لأن [`Trainer`] يدعم فقط خلفية MPS.

</Tip>

مع تعيين جهاز `mps`، يمكنك:

* تدريب شبكات أو أحجام دفعات أكبر محليًا
* تقليل زمن استرداد البيانات لأن بنية الذاكرة الموحدة لمعالج الرسومات تسمح بالوصول المباشر إلى مخزن الذاكرة الكامل
* تقليل التكاليف لأنك لست بحاجة إلى التدريب على معالجات رسومات (GPUs) قائمة على السحابة أو إضافة معالجات رسومات (GPUs) محلية إضافية

ابدأ بالتأكد من تثبيت PyTorch. يتم دعم تسريع MPS على macOS 12.3+.

```bash
pip install torch torchvision torchaudio
```

يستخدم [`TrainingArguments`] جهاز `mps` بشكل افتراضي إذا كان متوفرًا، مما يعني أنك لست بحاجة إلى تعيين الجهاز بشكل صريح. على سبيل المثال، يمكنك تشغيل نص البرنامج النصي [run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py) مع تمكين خلفية MPS تلقائيًا دون إجراء أي تغييرات.

```diff
export TASK_NAME=mrpc

python examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
- --use_mps_device \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
```

لا تدعم خلفية `mps` الخلفيات الخاصة بالتكوينات الموزعة (distributed setups) مثل `gloo` و`nccl`، مما يعني أنه يمكنك التدريب على معالج رسومات (GPU) واحد فقط مع خلفية MPS.

يمكنك معرفة المزيد عن خلفية MPS في منشور المدونة [تقديم تدريب PyTorch المعجل على Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/).