# المرافق لتدريب النماذج 

هذه الصفحة تسرد جميع دالات المساعدة التي يستخدمها [مدرب].
معظم هذه الدالات مفيدة فقط إذا كنت تدرس كود المدرب في المكتبة.

## المرافق

[[autodoc]] EvalPrediction

[[autodoc]] IntervalStrategy

[[autodoc]] enable_full_determinism

[[autodoc]] set_seed

[[autodoc]] torch_distributed_zero_first

## تفاصيل الرد الاتصالي

[[autodoc]] trainer_callback.CallbackHandler

## التقييم الموزع

[[autodoc]] trainer_pt_utils.DistributedTensorGatherer

## محلل حجج المدرب

[[autodoc]] HfArgumentParser

## أدوات التصحيح

[[autodoc]] debug_utils.DebugUnderflowOverflow