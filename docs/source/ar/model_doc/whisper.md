# Whisper

## نظرة عامة

اقترح نموذج Whisper في [الاعتراف الكلامي القوي عبر الإشراف الضعيف واسع النطاق](https://cdn.openai.com/papers/whisper.pdf) بواسطة Alec Radford و Jong Wook Kim و Tao Xu و Greg Brockman و Christine McLeavey و Ilya Sutskever.

الملخص من الورقة هو ما يلي:

> *ندرس قدرات أنظمة معالجة الكلام التي يتم تدريبها ببساطة للتنبؤ بكميات كبيرة من نصوص الصوت على الإنترنت. عندما يتم توسيع نطاقها إلى 680000 ساعة من الإشراف متعدد اللغات والمهام، تعمم النماذج الناتجة جيدًا على المعايير القياسية وغالباً ما تكون قادرة على المنافسة مع النتائج الخاضعة للإشراف الكامل ولكن في إعداد نقل التعلم بدون الإشراف دون الحاجة إلى أي ضبط دقيق. عند مقارنتها بالبشر، تقترب النماذج من دقتهم ومتانتهم. نحن نقوم بإطلاق نماذج وشفرة الاستدلال لتخدم كأساس لمزيد من العمل في معالجة الكلام القوي.*

تمت المساهمة بهذا النموذج من قبل [Arthur Zucker](https://huggingface.co/ArthurZ). تمت المساهمة في إصدار Tensorflow من هذا النموذج بواسطة [amyeroberts](https://huggingface.co/amyeroberts).

يمكن العثور على الكود الأصلي [هنا](https://github.com/openai/whisper).

## نصائح الاستخدام

- عادة ما يؤدي النموذج جيدًا دون الحاجة إلى أي ضبط دقيق.
- يتبع التصميم الهندسي بنية الترميز الكلاسيكي فك الترميز، مما يعني أنه يعتمد على وظيفة [`~generation.GenerationMixin.generate`] للاستدلال.
- يمكن للمرء استخدام [`WhisperProcessor`] لتحضير الصوت للنموذج، وفك ترميز معرفات ID المتوقعة مرة أخرى إلى نص.
- لتحويل النموذج والمعالج، نوصي باستخدام ما يلي:

```bash
python src/transformers/models/whisper/convert_openai_to_hf.py --checkpoint_path "" --pytorch_dump_folder_path "Arthur/whisper-3" --convert_preprocessor True
```

سيحدد البرنامج النصي تلقائيًا جميع المعلمات اللازمة من نقطة تفتيش OpenAI. يلزم تثبيت مكتبة "tiktoken"
لتحويل برنامج تحليل OpenAI إلى إصدار "tokenizers".

## الاستدلال

فيما يلي دليل خطوة بخطوة لنسخ نص عينة صوتية باستخدام نموذج Whisper مُدرب مسبقًا:

```python
>>> from datasets import load_dataset
>>> from transformers import WhisperProcessor, WhisperForConditionalGeneration

>>> # حدد ملف صوتي واقرأه:
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> audio_sample = ds[0]["audio"]
>>> waveform = audio_sample["array"]
>>> sampling_rate = audio_sample["sampling_rate"]

>>> # تحميل نموذج Whisper بتنسيق Hugging Face:
>>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

>>> # استخدم النموذج والمعالج لنسخ الصوت:
>>> input_features = processor(
...     waveform, sampling_rate=sampling_rate, return_tensors="pt"
... ).input_features

>>> # إنشاء معرفات tokens
>>> predicted_ids = model.generate(input_features)

>>> # فك ترميز معرفات tokens إلى نص
>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

>>> transcription[0]
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام Whisper. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب Pull Request وسنراجعه! يجب أن يُظهر المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- [قم بتدريب Whisper](https://huggingface.co/blog/fine-tune-whisper) على مجموعة البيانات الخاصة بك للحصول على أداء أفضل لأسفل البث.
- [Distil-Whisper](https://huggingface.co/distil-whisper): ما يصل إلى 6 مرات أسرع، و2 مرة أصغر نماذج Whisper المقطرة للغة الإنجليزية. نقوم بإطلاق [نقاط تفتيش النموذج](https://huggingface.co/distil-whisper)، و [كود التقطير](https://github.com/huggingface/distil-whisper).
- نسخة متشعبة مع نص برمجي لتحويل [نموذج Whisper بتنسيق Hugging Face إلى تنسيق OpenAI](https://github.com/zuazo-forks/transformers/blob/convert_hf_to_openai/src/transformers/models/whisper/convert_hf_to_openai.py). 🌎

مثال على الاستخدام:

```bash
pip install -U openai-whisper
python convert_hf_to_openai.py \
--checkpoint openai/whisper-tiny \
--whisper_dump_path whisper-tiny-openai.pt
```

## WhisperConfig

[[autodoc]] WhisperConfig

## WhisperTokenizer

[[autodoc]] WhisperTokenizer

- set_prefix_tokens
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary
- batch_decode
- decode
- basic_normalize
- normalize

## WhisperTokenizerFast

[[autodoc]] WhisperTokenizerFast

- set_prefix_tokens
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary
- batch_decode
- decode
- basic_normalize
- normalize

## WhisperFeatureExtractor

[[autodoc]] WhisperFeatureExtractor

- __call__

## WhisperProcessor

[[autodoc]] WhisperProcessor

- __call__
- from_pretrained
- save_pretrained
- batch_decode
- decode

<frameworkcontent>

<pt>

## WhisperModel

[[autodoc]] WhisperModel

- forward
- _mask_input_features

## WhisperForConditionalGeneration

[[autodoc]] WhisperForConditionalGeneration

- forward
- generate

## WhisperForCausalLM

[[autodoc]] WhisperForCausalLM

- forward

## WhisperForAudioClassification

[[autodoc]] WhisperForAudioClassification

- forward

</pt>

<tf>

## TFWhisperModel

[[autodoc]] TFWhisperModel

- call

## TFWhisperForConditionalGeneration

[[autodoc]] TFWhisperForConditionalGeneration

- call

</tf>

<jax>

## FlaxWhisperModel

[[autodoc]] FlaxWhisperModel

- __call__

## FlaxWhisperForConditionalGeneration

[[autodoc]] FlaxWhisperForConditionalGeneration

- __call__

## FlaxWhisperForAudioClassification

[[autodoc]] FlaxWhisperForAudioClassification

- __call__

</jax>

</frameworkcontent>