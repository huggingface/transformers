# التحسين

يوفر نموذج `.optimization` ما يلي:

- محسن مع وزن ثابت للاضمحلال يمكن استخدامه لضبط دقيق للنماذج، و
- العديد من الجداول الزمنية على شكل كائنات جدول زمني ترث من `_LRSchedule`:
- فئة تراكم التدرجات لتراكم تدرجات الدفعات المتعددة

## AdamW (PyTorch)

[[autodoc]] AdamW

## AdaFactor (PyTorch)

[[autodoc]] Adafactor

## AdamWeightDecay (TensorFlow)

[[autodoc]] AdamWeightDecay

[[autodoc]] create_optimizer

## الجداول الزمنية

### جداول معدلات التعلم (Pytorch)

[[autodoc]] SchedulerType

[[autodoc]] get_scheduler

[[autodoc]] get_constant_schedule

[[autodoc]] get_constant_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_constant_schedule.png"/>

[[autodoc]] get_cosine_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_schedule.png"/>

[[autodoc]] get_cosine_with_hard_restarts_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_hard_restarts_schedule.png"/>

[[autodoc]] get_linear_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_linear_schedule.png"/>

[[autodoc]] get_polynomial_decay_schedule_with_warmup

[[autodoc]] get_inverse_sqrt_schedule

[[autodoc]] get_wsd_schedule

### Warmup (TensorFlow)

[[autodoc]] WarmUp

## استراتيجيات التدرج

### GradientAccumulator (TensorFlow)

[[autodoc]] GradientAccumulator