# DPR

## نظرة عامة

Dense Passage Retrieval (DPR) هي مجموعة من الأدوات والنماذج لبحوث أسئلة وأجوبة المجال المفتوح المتقدمة. تم تقديمه في "استرجاع المرور الكثيف لأسئلة المجال المفتوح وأجوبتها" بواسطة فلاديمير كاربوخين، وبارلاس أوجوز، وسيون مين، وباتريك لويس، وليديل وو، وسيرجي إدونوف، ودانقي تشن، وون-تاو ييه.

ملخص الورقة هو ما يلي:

*يعتمد نظام الأسئلة والأجوبة في المجال المفتوح على استرجاع المرور الكفء لاختيار سياقات المرشحين، حيث تعد نماذج الفضاء المتفرق، مثل TF-IDF أو BM25، هي الطريقة الفعلية. في هذا العمل، نُظهر أن الاسترجاع يمكن تنفيذه عمليًا باستخدام التمثيلات الكثيفة بمفردها، حيث يتم تعلم التضمينات من عدد صغير من الأسئلة والمرور بواسطة إطار عمل مشفر مزدوج بسيط. عندما تم تقييمه على مجموعة واسعة من مجموعات بيانات QA المجال المفتوح، تفوق نظام الاسترجاع الكثيف لدينا على نظام Lucene-BM25 القوي إلى حد كبير بنسبة 9٪ -19٪ مطلقًا من حيث دقة استرجاع المرور العلوي 20، وساعد نظام QA من البداية إلى النهاية في إنشاء حالة جديدة من الفن في نقاط مرجعية QA المجال المفتوح متعددة.*

تمت المساهمة بهذا النموذج من قبل [lhoestq](https://huggingface.co/lhoestq). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/DPR).

## نصائح الاستخدام

يتكون DPR من ثلاثة نماذج:

* مشفر السؤال: قم بتشفير الأسئلة على شكل متجهات
* مشفر السياق: قم بتشفير السياقات على شكل متجهات
* القارئ: استخراج إجابة الأسئلة داخل السياقات المستردة، إلى جانب درجة الملاءمة (عالية إذا كانت الفترة المستدلة تجيب بالفعل على السؤال).

## تكوين DPR

[[autodoc]] DPRConfig

## معالج مشفر سياق DPR

[[autodoc]] DPRContextEncoderTokenizer

## معالج مشفر سياق DPR السريع

[[autodoc]] DPRContextEncoderTokenizerFast

## معالج مشفر سؤال DPR

[[autodoc]] DPRQuestionEncoderTokenizer

## معالج مشفر سؤال DPR السريع

[[autodoc]] DPRQuestionEncoderTokenizerFast

## معالج قارئ DPR

[[autodoc]] DPRReaderTokenizer

## معالج قارئ DPR السريع

[[autodoc]] DPRReaderTokenizerFast

## المخرجات المحددة لـ DPR

[[autodoc]] models.dpr.modeling_dpr.DPRContextEncoderOutput

[[autodoc]] models.dpr.modeling_dpr.DPRQuestionEncoderOutput

[[autodoc]] models.dpr.modeling_dpr.DPRReaderOutput

<frameworkcontent>
<pt>

## مشفر السياق DPR

[[autodoc]] DPRContextEncoder

- forword

## مشفر السؤال DPR

[[autodoc]] DPRQuestionEncoder

- forword

## قارئ DPR

[[autodoc]] DPRReader

- forword

</pt>
<tf>

## TFDPRContextEncoder

[[autodoc]] TFDPRContextEncoder

- call

## TFDPRQuestionEncoder

[[autodoc]] TFDPRQuestionEncoder

- call

## TFDPRReader

[[autodoc]] TFDPRReader

- call

</tf>
</frameworkcontent>