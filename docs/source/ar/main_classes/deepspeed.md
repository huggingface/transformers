# DeepSpeed

تعد DeepSpeed، المدعومة من Zero Redundancy Optimizer (ZeRO)، مكتبة تحسين لتدريب وتناسب النماذج الكبيرة جدًا على وحدة معالجة الرسومات (GPU). وهي متوفرة في عدة مراحل Zero، حيث تقوم كل مرحلة بتوفير ذاكرة GPU بشكل تدريجي من خلال تقسيم حالة المحسن والتدرجات والمؤشرات، وتمكين النقل إلى وحدة المعالجة المركزية (CPU) أو NVMe. تم دمج DeepSpeed مع فئة 'Trainer'، ويتم التعامل مع معظم الإعدادات تلقائيًا.

ومع ذلك، إذا كنت تريد استخدام DeepSpeed دون 'Trainer'، يوفر Transformers فئة 'HfDeepSpeedConfig'.

## HfDeepSpeedConfig

- [integrations.HfDeepSpeedConfig]