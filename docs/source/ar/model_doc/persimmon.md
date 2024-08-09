# Persimmon

## نظرة عامة

قام فريق [ADEPT](https://www.adept.ai/blog/persimmon-8b) بإنشاء نموذج Persimmon، وكتبه إريك إلسن، وأغسطس أودينا، وماكسويل ناي، وساغناك تاسيرلار، وتري داو، وكورتيس هاوثورن، وديباك موبارثي، وأروش سوماني.

قدم المؤلفون Persimmon-8B، وهو نموذج فك تشفير يعتمد على تصميم محولات كلاسيكي، مع تطبيع الاستعلام والمفتاح. Persimmon-8B هو نموذج مرخص بالكامل بحوالي 8 مليارات معامل، تم إصداره بموجب ترخيص Apache. بعض السمات الرئيسية لـ Persimmon-8B هي حجم السياق الطويل (16K)، والأداء، والقدرات على الامتدادات متعددة الوسائط.

يعرض المؤلفون نهجهم في تقييم النماذج، مع التركيز على التوليد العملي للنص، مما يعكس كيفية تفاعل المستخدمين مع نماذج اللغة. يتضمن العمل أيضًا تحليلًا مقارنًا، حيث يتنافس Persimmon-8B مع النماذج البارزة الأخرى (MPT 7B Instruct وLlama 2 Base 7B 1-Shot) عبر مهام تقييم مختلفة. وتُظهر النتائج الأداء التنافسي لـ Persimmon-8B، حتى مع محدودية بيانات التدريب.

من حيث تفاصيل النموذج، يحدد العمل تصميم Persimmon-8B ومنهجية التدريب، مما يوفر رؤى حول خيارات التصميم وطول التسلسل وتكوين مجموعة البيانات. ويقدم المؤلفون كود استدلال سريع يتفوق على التطبيقات التقليدية من خلال دمج مشغل CUDA واستخدام رسم CUDA مع الحفاظ على تماسك الشفرة. ويعربون عن تطلعهم إلى كيفية استفادة المجتمع من هذا الإسهام لدفع الابتكار، مما يشير إلى إصدارات أخرى قادمة كجزء من سلسلة مستمرة من التطورات.

تمت المساهمة بهذا النموذج من قبل [ArthurZ](https://huggingface.co/ArthurZ). يمكن العثور على الكود الأصلي [هنا](https://github.com/persimmon-ai-labs/adept-inference).

## نصائح الاستخدام

<Tip warning={true}>
تم تدريب نماذج "Persimmon" باستخدام "bfloat16"، ولكن الاستدلال الأصلي يستخدم "float16". تستخدم نقاط التفتيش المُحمّلة على المركز "torch_dtype = 'float16'"، والتي سيتم استخدامها بواسطة واجهة برمجة التطبيقات "AutoModel" لتحويل نقاط التفتيش من "torch.float32" إلى "torch.float16".

نوع بيانات الأوزان عبر الإنترنت غير ذي صلة في الغالب، ما لم تكن تستخدم "torch_dtype='auto'" عند تهيئة نموذج باستخدام "model = AutoModelForCausalLM.from_pretrained("path"، torch_dtype="auto")". والسبب هو أنه سيتم أولاً تنزيل النموذج (باستخدام نوع بيانات نقاط التفتيش عبر الإنترنت) ثم تحويله إلى نوع بيانات افتراضي لـ "torch" (يصبح "torch.float32"). يجب على المستخدمين تحديد "torch_dtype" الذي يريدونه، وإذا لم يفعلوا ذلك، فسيكون "torch.float32".

لا يُنصح بالتدريب الدقيق للنموذج في "float16" وهو معروف بإنتاج "nan"، وبالتالي يجب ضبط دقة النموذج باستخدام "bfloat16".
</Tip>

نصائح:

- لتحويل النموذج، تحتاج إلى استنساخ مستودع الأصل باستخدام "git clone https://github.com/persimmon-ai-labs/adept-inference"، ثم الحصول على نقاط التفتيش:

```bash
git clone https://github.com/persimmon-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
tar -xvf 8b_base_model_release.tar
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py --input_dir /path/to/downloaded/persimmon/weights/ --output_dir /output/path \
--pt_model_path /path/to/8b_chat_model_release/iter_0001251/mp_rank_00/model_optim_rng.pt
--ada_lib_path /path/to/adept-inference
```

بالنسبة لنموذج الدردشة:

```bash
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
tar -xvf 8b_base_model_release.tar
```

بعد ذلك، يمكن تحميل النماذج عبر:

```py
from transformers import PersimmonForCausalLM, PersimmonTokenizer

model = PersimmonForCausalLM.from_pretrained("/output/path")
tokenizer = PersimmonTokenizer.from_pretrained("/output/path")
```

- يستخدم Persimmon محول رموز يعتمد على "sentencepiece"، مع نموذج "Unigram". يدعم "bytefallback"، وهو متاح فقط في "tokenizers==0.14.0" لمحول الرموز السريع.

يتم استخدام "LlamaTokenizer" لأنه عبارة عن غلاف قياسي حول "sentencepiece". سيتم تحديث قالب "chat" بوظائف القوالب في طلب سحب لاحق!

- يقترح المؤلفون استخدام تنسيق المحث التالي لوضع الدردشة: `f"human: {prompt}\n\nadept:"`

## PersimmonConfig

[[autodoc]] PersimmonConfig

## PersimmonModel

[[autodoc]] PersimmonModel

- forward

## PersimmonForCausalLM

[[autodoc]] PersimmonForCausalLM

- forward

## PersimmonForSequenceClassification

[[autodoc]] PersimmonForSequenceClassification

- forward

## PersimmonForTokenClassification

[[autodoc]] PersimmonForTokenClassification

- forward