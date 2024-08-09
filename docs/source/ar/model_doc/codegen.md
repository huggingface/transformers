# CodeGen

## نظرة عامة

اقترح نموذج CodeGen في [نموذج محادثي للتركيب البرمجي](https://arxiv.org/abs/2203.13474) بواسطة Erik Nijkamp و Bo Pang و Hiroaki Hayashi و Lifu Tu و Huan Wang و Yingbo Zhou و Silvio Savarese و Caiming Xiong.

CodeGen هو نموذج لغة مولدة للتركيب البرمجي تم تدريبه بشكل متسلسل على [The Pile](https://pile.eleuther.ai/) و BigQuery و BigPython.

الملخص من الورقة هو ما يلي:

> يسعى تركيب البرنامج إلى توليد برنامج كمبيوتر كحل لمشكلة مواصفات معينة. نقترح نهجًا محادثيًا لتركيب البرنامج من خلال نماذج اللغة الكبيرة، والتي تتناول التحديات المتمثلة في البحث في مساحة برنامج ضخمة وتحديد نية المستخدم التي واجهتها في الأساليب السابقة. يلقي نهجنا الجديد عملية كتابة المواصفات والبرنامج كمحادثة متعددة الأدوار بين المستخدم والنظام. إنه يعامل تركيب البرنامج كمشكلة تنبؤ تسلسلي، يتم فيها التعبير عن المواصفات بلغة طبيعية ويتم أخذ عينات من البرنامج المطلوب بشكل شرطي. نقوم بتدريب عائلة من نماذج اللغة الكبيرة، تسمى CodeGen، على بيانات اللغة الطبيعية وبيانات لغة البرمجة. مع الإشراف الضعيف في البيانات وتوسيع حجم البيانات وحجم النموذج، تظهر القدرات المحادثية من نمذجة اللغة الذاتية البسيطة. لدراسة سلوك النموذج على التركيب البرمجي المحادثي، نقوم بتطوير معيار برمجة متعدد الأدوار (MTPB)، حيث يتطلب حل كل مشكلة تركيبًا متعدد الخطوات من خلال محادثة متعددة الأدوار بين المستخدم والنموذج. توضح نتائجنا ظهور القدرات المحادثية وفعالية نموذج التركيب البرمجي المحادثي المقترح. بالإضافة إلى ذلك، يتفوق نموذجنا CodeGen (بحد أقصى 16 بليون معلمة مدربة على TPU-v4) على OpenAI's Codex على معيار HumanEval. نجعل مكتبة التدريب JaxFormer بما في ذلك نقاط المراقبة متاحة كمساهمة مفتوحة المصدر: [هذا عنوان URL https](https://github.com/salesforce/codegen).

تمت المساهمة بهذا النموذج من قبل [Hiroaki Hayashi](https://huggingface.co/rooa).

يمكن العثور على الكود الأصلي [هنا](https://github.com/salesforce/codegen).

## تسمية نقطة التفتيش

- تتوفر نقاط تفتيش نموذج CodeGen [checkpoints](https://huggingface.co/models?other=codegen) على بيانات تدريب مسبق مختلفة بأحجام متغيرة.
- التنسيق هو: `Salesforce/codegen-{size}-{data}`، حيث
- `size`: `350M`، `2B`، `6B`، `16B`
- `data`:
  - `nl`: مدرب مسبقًا على Pile
  - `multi`: تم التهيئة الأولية باستخدام `nl`، ثم تم التدريب المسبق بشكل أكبر على بيانات لغات برمجة متعددة
  - `mono`: تم التهيئة الأولية باستخدام `multi`، ثم تم التدريب المسبق بشكل أكبر على بيانات Python

على سبيل المثال، يوفر `Salesforce/codegen-350M-mono` نقطة تفتيش بحجم 350 مليون معلمة تم تدريبها مسبقًا بشكل متسلسل على Pile ولغات برمجة متعددة وPython.

## مثال على الاستخدام

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> checkpoint = "Salesforce/codegen-350M-mono"
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)

>>> text = "def hello_world():"

>>> completion = model.generate(**tokenizer(text, return_tensors="pt"))

>>> print(tokenizer.decode(completion[0]))
def hello_world():
  print("Hello World")

hello_world()
```

## الموارد

- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)

## CodeGenConfig

[[autodoc]] CodeGenConfig

- all

## CodeGenTokenizer

[[autodoc]] CodeGenTokenizer

- create_token_type_ids_from_sequences
- save_vocabulary

## CodeGenTokenizerFast

[[autodoc]] CodeGenTokenizerFast

## CodeGenModel

[[autodoc]] CodeGenModel

- forward

## CodeGenForCausalLM

[[autodoc]] CodeGenForCausalLM

- forward