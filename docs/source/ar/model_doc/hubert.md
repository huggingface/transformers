# Hubert

## نظرة عامة

اقترح Hubert في [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) بواسطة Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.

مقدمة الورقة البحثية هي التالية:

> تواجه الأساليب ذاتية الإشراف لتعلم تمثيل الكلام ثلاث مشكلات فريدة: (1) هناك وحدات صوت متعددة في كل عبارة مدخلة، (2) لا توجد قائمة مصطلحات لوحدات الصوت المدخلة أثناء مرحلة التدريب المسبق، (3) لوحدات الصوت أطوال متغيرة بدون تجزئة صريحة. وللتعامل مع هذه المشكلات الثلاث، نقترح نهج BERT المخفي (HuBERT) لتعلم تمثيل الكلام ذاتي الإشراف، والذي يستخدم خطوة تجميع غير متصلة لتوفير تسميات مستهدفة محاذاة لخسارة التنبؤ على غرار BERT. المكون الرئيسي لنهجنا هو تطبيق خسارة التنبؤ على المناطق المُقنَّعة فقط، مما يجبر النموذج على تعلم نموذج صوتي ولغوي مجمع على المدخلات المستمرة. يعتمد HuBERT بشكل أساسي على اتساق خطوة التجميع غير الخاضعة للإشراف بدلاً من الجودة الجوهرية لعلامات التصنيف المعينة. بدءًا من معلم k-means البسيط لـ 100 مجموعة، واستخدام تكرارين من التجميع، فإن نموذج HuBERT إما يتطابق مع أداء wav2vec 2.0 المميز أو يحسنه في معايير Librispeech (960h) و Libri-light (60,000h) مع مجموعات ضبط دقيق فرعية لمدة 10 دقائق و 1 ساعة و 10 ساعات و 100 ساعة و 960 ساعة. باستخدام نموذج معلمات 1B، يُظهر HuBERT انخفاضًا نسبيًا في WER يصل إلى 19% و13% على مجموعات التقييم الأكثر تحديًا dev-other وtest-other.

تمت المساهمة بهذا النموذج من قبل [patrickvonplaten](https://huggingface.co/patrickvonplaten).

# نصائح الاستخدام

- Hubert هو نموذج كلام يقبل مصفوفة عائمة تتوافق مع شكل الموجة الخام لإشارة الكلام.
- تم ضبط نموذج Hubert الدقيق باستخدام التصنيف الزمني للاتصال (CTC)، لذلك يجب فك تشفير إخراج النموذج باستخدام [`Wav2Vec2CTCTokenizer`].

## استخدام Flash Attention 2

Flash Attention 2 هو إصدار أسرع وأكثر تحسينًا من النموذج.

### التثبيت

أولاً، تحقق مما إذا كان الأجهزة الخاصة بك متوافقة مع Flash Attention 2. يمكن العثور على أحدث قائمة بالأجهزة المتوافقة في [الوثائق الرسمية](https://github.com/Dao-AILab/flash-attention#installation-and-features). إذا لم يكن الأجهزة الخاص بك متوافقًا مع Flash Attention 2، فيمكنك الاستفادة من تحسينات نواة الاهتمام من خلال دعم Transformer الأفضل المشمولة [أعلاه](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

بعد ذلك، قم بتثبيت أحدث إصدار من Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

### الاستخدام

فيما يلي مخطط تسريع متوقع يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات من `facebook/hubert-large-ls960-ft`، وflash-attention-2، وإصدار sdpa (scale-dot-product-attention). نعرض متوسط التسريع الذي تم الحصول عليه على تقسيم التحقق `clean` من `librispeech_asr`:

```python
>>> from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/hubert-large-ls960-ft", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
...
```

### التسريع المتوقع

فيما يلي مخطط تسريع متوقع يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات لنموذج `facebook/hubert-large-ls960-ft` وإصدارات flash-attention-2 وsdpa (scale-dot-product-attention). نعرض متوسط التسريع الذي تم الحصول عليه على تقسيم التحقق `clean` من `librispeech_asr`:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/kamilakesbi/transformers_image_doc/resolve/main/data/Hubert_speedup.png">
</div>

## الموارد

- [دليل مهام تصنيف الصوت](../tasks/audio_classification)
- [دليل مهام التعرف التلقائي على الكلام](../tasks/asr)

## HubertConfig

[[autodoc]] HubertConfig

<frameworkcontent>
<pt>

## HubertModel

[[autodoc]] HubertModel

- forward

## HubertForCTC

[[autodoc]] HubertForCTC

- forward

## HubertForSequenceClassification

[[autodoc]] HubertForSequenceClassification

- forward

</pt>
<tf>

## TFHubertModel

[[autodoc]] TFHubertModel

- call

## TFHubertForCTC

[[autodoc]] TFHubertForCTC

- call

</tf>
</frameworkcontent>