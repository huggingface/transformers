# Fuyu

## نظرة عامة

تم إنشاء نموذج Fuyu بواسطة [ADEPT](https://www.adept.ai/blog/fuyu-8b)، وكتبه روهان بافيشي، وإريش إلسن، وكيرتيس هاوثورن، وماكسويل ناي، وأوغستوس أودينا، وأروش سوماني، وساجناك تاسيلار.

قدم المؤلفون Fuyu-8B، وهو نموذج متعدد الوسائط يعتمد على فك التشفير فقط ويستند إلى بنية المحولات الكلاسيكية، مع تطبيع الاستعلام والمفتاح. تمت إضافة مشفر خطي لإنشاء تضمينات متعددة الوسائط من إدخالات الصور.

من خلال معاملة رموز الصور مثل رموز النص واستخدام حرف خاص لفاصل الصور، يعرف النموذج متى ينتهي سطر الصورة. تتم إزالة تضمينات الموضع الصوري. يتجنب هذا الحاجة إلى مراحل تدريب مختلفة لقرارات الصور المختلفة. مع 8 مليار معلمة ومرخص بموجب CC-BY-NC، يُعرف Fuyu-8B بقدرته على التعامل مع كل من النص والصور، وحجم السياق المثيرة للإعجاب البالغ 16K، وأدائه العام.

<Tip warning={true}>

تم تدريب نماذج "فيويو" باستخدام "bfloat16"، ولكن الاستدلال الأصلي يستخدم "float16" تستخدم نقاط التحقق المرفوعة على المحور `torch_dtype = 'float16'` والتي سيتم

يتم استخدامها بواسطة `AutoModel` API لتحويل نقاط التحقق من `torch.float32` إلى `torch.float16`.

إن نوع بيانات الأوزان عبر الإنترنت غير ذي صلة إلى حد كبير، ما لم تكن تستخدم `torch_dtype="auto"` عند تهيئة نموذج باستخدام `model = AutoModelForCausalLM.from_pretrained("path"، torch_dtype = "auto")`. والسبب هو أنه سيتم أولاً تنزيل النموذج (باستخدام `dtype` من نقاط التحقق عبر الإنترنت) ثم تحويله إلى `dtype` الافتراضي لـ `torch` (يصبح `torch.float32`). يجب على المستخدمين تحديد `torch_dtype` الذي يريدونه، وإذا لم يفعلوا ذلك، فسيكون `torch.float32`.

لا يُنصح بالتدريب الدقيق للنموذج في `float16` ومن المعروف أنه ينتج عنه `نان`، لذلك يجب ضبط دقة النموذج باستخدام `bfloat16`.

</Tip>

نصائح:

- لتحويل النموذج، تحتاج إلى استنساخ المستودع الأصلي باستخدام `git clone https://github.com/persimmon-ai-labs/adept-inference`، ثم الحصول على نقاط التحقق:

```bash
git clone https://github.com/persimmon-ai-labs/adept-inference
wget path/to/fuyu-8b-model-weights.tar
tar -xvf fuyu-8b-model-weights.tar
بايثون src/transformers/models/fuyu/convert_fuyu_weights_to_hf.py --input_dir /path/to/downloaded/fuyu/weights/ --output_dir /output/path \
--pt_model_path /path/to/fuyu_8b_release/iter_0001251/mp_rank_00/model_optim_rng.pt
--ada_lib_path /path/to/adept-inference
```

لنموذج الدردشة:

```bash
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
tar -xvf 8b_base_model_release.tar
```

بعد ذلك، يمكن تحميل النموذج عبر:

```py
from transformers import FuyuConfig, FuyuForCausalLM
model_config = FuyuConfig()
model = FuyuForCausalLM(model_config).from_pretrained('/output/path')
```

يجب تمرير الإدخالات عبر معالج محدد للحصول على التنسيقات الصحيحة.

يتطلب المعالج معالج صور ومعالج توكن. وبالتالي، يمكن تحميل الإدخالات عبر:

```py
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.fuyu.processing_fuyu import FuyuProcessor
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor


tokenizer = AutoTokenizer.from_pretrained('adept-hf-collab/fuyu-8b')
image_processor = FuyuImageProcessor()


processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
text_prompt = "Generate a coco-style caption.\\n"

bus_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
bus_image_pil = Image.open(io.BytesIO(requests.get(bus_image_url).content))
inputs_to_model = processor(text=text_prompt, images=bus_image_pil)
```

تمت المساهمة بهذا النموذج بواسطة [Molbap](https://huggingface.co/Molbap).

يمكن العثور على الكود الأصلي [هنا](https://github.com/persimmon-ai-labs/adept-inference).

- يستخدم Fuyu معالج رموز يعتمد على `sentencepiece`، مع نموذج "Unigram". يدعم bytefallback، وهو متاح فقط في `tokenizers==0.14.0` لمعالج الرموز السريع.

يتم استخدام `LlamaTokenizer` لأنه عبارة عن غلاف قياسي حول sentencepiece.

- يقترح المؤلفون استخدام المطالبة التالية لتعليق الصور: `f"Generate a coco-style caption.\\n"`

## FuyuConfig

[[autodoc]] FuyuConfig

## FuyuForCausalLM

[[autodoc]] FuyuForCausalLM

- forword

## FuyuImageProcessor

[[autodoc]] FuyuImageProcessor

- __call__

## FuyuProcessor

[[autodoc]] FuyuProcessor

- __call__