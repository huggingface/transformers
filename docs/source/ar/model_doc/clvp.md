# CLVP  

## نظرة عامة  
 اقترح جيمس بيتكر نموذج CLVP (Contrastive Language-Voice Pretrained Transformer) في ورقته البحثية [Better speech synthesis through scaling](https://arxiv.org/abs/2305.07243).  
 فيما يلي الملخص المستخرج من الورقة البحثية:  
 *"في السنوات الأخيرة، أحدث تطبيق المحولات التلقائية والنماذج التنافسية ثورة في مجال توليد الصور. تقوم هذه الأساليب بوضع عملية توليد الصور كعمليات احتمالية متدرجة وتستفيد من كميات كبيرة من البيانات والحوسبة لتعلم توزيع الصور. لا تحتاج منهجية تحسين الأداء هذه إلى أن تقتصر على الصور. تصف هذه الورقة طريقة لتطبيق التقدمات في مجال توليد الصور على تركيب الكلام. والنتيجة هي Tortoise - نظام نص إلى كلام تعبيري ومتعدد الأصوات."*  
 ساهم في هذا النموذج [سوسناتو دار](https://huggingface.co/susnato).  
 يمكن العثور على الكود الأصلي [هنا](https://github.com/neonbjb/tortoise-tts).  
 ## نصائح الاستخدام  
1. CLVP هو جزء لا يتجزأ من نموذج Tortoise TTS.  
2. يمكن استخدام CLVP لمقارنة مرشحي الكلام المولَّد المختلفين مع النص المقدم، ويتم تمرير أفضل رموز الكلام إلى نموذج الانتشار.  
3. يوصى بشدة باستخدام طريقة [`ClvpModelForConditionalGeneration.generate()`] للاستخدام في السلحفاة.  
4. لاحظ أن نموذج CLVP يتوقع أخذ عينات من الصوت عند 22.05 كيلو هرتز على عكس نماذج الصوت الأخرى التي تتوقع 16 كيلو هرتز.  
 ## شرح موجز:  
 - [`ClvpTokenizer`] يقسم نص الإدخال، و [`ClvpFeatureExtractor`] يستخرج مخطط ميلوغرام من الصوت المطلوب.  
 - [`ClvpConditioningEncoder`] يأخذ رموز النص وتمثيلات الصوت ويحولها إلى تضمينات مشروطة بالنص والصوت.  
 - يستخدم [`ClvpForCausalLM`] هذه التضمينات لتوليد عدة مرشحين للكلام.  
 - يتم تمرير كل مرشح كلامي عبر مشفر الكلام ([`ClvpEncoder`]) الذي يحوله إلى تمثيل متجهي، ويحول مشفر النص ([`ClvpEncoder`]) رموز النص إلى نفس الفراغ الكامن.  
 - في النهاية، نقارن كل متجه كلامي مع متجه النص لمعرفة أي متجه كلامي يشبه متجه النص أكثر.  
 - [`ClvpModelForConditionalGeneration.generate()`] يضغط كل المنطق الموضح أعلاه في طريقة واحدة.  
 مثال:  
 ```python
>>> import datasets
>>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

>>> # Define the Text and Load the Audio (We are taking an audio example from Hugging Face Hub using `datasets` library).
>>> text = "This is an example text."

>>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
>>> sample = ds[0]["audio"]

>>> # Define processor and model.
>>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
>>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

>>> # Generate processor output and model output.
>>> processor_output = processor(raw_speech=sample["array"], sampling_rate=sample["sampling_rate"], text=text, return_tensors="pt")
>>> generated_output = model.generate(**processor_output)
```
 ## ClvpConfig  
 [[autodoc]] ClvpConfig
 - from_sub_model_configs
 ## ClvpEncoderConfig  
 [[autodoc]] ClvpEncoderConfig
 ## ClvpDecoderConfig  
 [[autodoc]] ClvpDecoderConfig
 ## ClvpTokenizer  
 [[autodoc]] ClvpTokenizer
 - save_vocabulary
 ## ClvpFeatureExtractor  
 [[autodoc]] ClvpFeatureExtractor
 - __call__
 ## ClvpProcessor  
 [[autodoc]] ClvpProcessor
 - __call__
 - decode
 - batch_decode
 ## ClvpModelForConditionalGeneration  
 [[autodoc]] ClvpModelForConditionalGeneration
 - forward
 - generate
 - get_text_features
 - get_speech_features
 ## ClvpForCausalLM  
 [[autodoc]] ClvpForCausalLM
 ## ClvpModel  
 [[autodoc]] ClvpModel
 ## ClvpEncoder  
 [[autodoc]] ClvpEncoder
 ## ClvpDecoder  
 [[autodoc]] ClvpDecoder