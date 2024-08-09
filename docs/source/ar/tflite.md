# التصدير إلى TFLite

[TensorFlow Lite](https://www.tensorflow.org/lite/guide) هو إطار عمل خفيف الوزن لنشر نماذج التعلم الآلي على الأجهزة المحدودة الموارد، مثل الهواتف المحمولة، والأنظمة المدمجة، وأجهزة إنترنت الأشياء (IoT). تم تصميم TFLite لتشغيل النماذج وتحسينها بكفاءة على هذه الأجهزة ذات الطاقة الحاسوبية والذاكرة واستهلاك الطاقة المحدودة.

يتم تمثيل نموذج TensorFlow Lite بتنسيق محمول فعال خاص يتم تحديده بواسطة امتداد الملف `.tflite`.

🤗 Optimum يقدم وظيفة لتصدير نماذج 🤗 Transformers إلى TFLite من خلال الوحدة النمطية `exporters.tflite`. بالنسبة لقائمة هندسات النماذج المدعومة، يرجى الرجوع إلى [وثائق 🤗 Optimum](https://huggingface.co/docs/optimum/exporters/tflite/overview).

لتصدير نموذج إلى TFLite، قم بتثبيت التبعيات المطلوبة:

```bash
pip install optimum[exporters-tf]
```

للاطلاع على جميع الحجج المتاحة، راجع [وثائق 🤗 Optimum](https://huggingface.co/docs/optimum/main/en/exporters/tflite/usage_guides/export_a_model)، أو عرض المساعدة في سطر الأوامر:

```bash
optimum-cli export tflite --help
```

لتصدير نقطة تفتيش نموذج من 🤗 Hub، على سبيل المثال، `google-bert/bert-base-uncased`، قم بتشغيل الأمر التالي:

```bash
optimum-cli export tflite --model google-bert/bert-base-uncased --sequence_length 128 bert_tflite/
```

يجب أن تشاهد السجلات التي تشير إلى التقدم وتعرض المكان الذي تم فيه حفظ `model.tflite` الناتج، مثل هذا:

```bash
Validating TFLite model...
	-[✓] TFLite model output names match reference model (logits)
	- Validating TFLite Model output "logits":
		-[✓] (1, 128, 30522) matches (1, 128, 30522)
		-[x] values not close enough, max diff: 5.817413330078125e-05 (atol: 1e-05)
The TensorFlow Lite export succeeded with the warning: The maximum absolute difference between the output of the reference model and the TFLite exported model is not within the set tolerance 1e-05:
- logits: max diff = 5.817413330078125e-05.
 The exported model was saved at: bert_tflite
```

يوضح المثال أعلاه تصدير نقطة تفتيش من 🤗 Hub. عند تصدير نموذج محلي، تأكد أولاً من حفظ ملفات أوزان النموذج ومحول الرموز في نفس الدليل (`local_path`). عند استخدام CLI، قم بتمرير `local_path` إلى حجة `model` بدلاً من اسم نقطة التفتيش على 🤗 Hub.