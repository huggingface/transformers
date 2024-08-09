# GroupViT

## نظرة عامة

اقترح نموذج GroupViT في [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094) بواسطة Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang.

استوحى من [CLIP](clip)، GroupViT هو نموذج للرؤية اللغوية يمكنه أداء تجزئة دلالية بدون الإشراف على أي فئات معجمية معينة.

المستخلص من الورقة هو ما يلي:

*يعد التجميع والتعرف على المكونات المهمة لفهم المشهد المرئي، على سبيل المثال، للكشف عن الأشياء والتجزئة الدلالية. مع أنظمة التعلم العميق من البداية إلى النهاية، يحدث تجميع مناطق الصورة عادة بشكل ضمني عبر الإشراف من أعلى إلى أسفل من علامات التعرف على مستوى البكسل. بدلاً من ذلك، في هذه الورقة، نقترح إعادة آلية التجميع إلى الشبكات العصبية العميقة، والتي تسمح للشرائح الدلالية بالظهور تلقائيًا مع الإشراف النصي فقط. نقترح مجموعة هرمية من محول الرؤية (GroupViT)، والتي تتجاوز تمثيل هيكل الشبكة المنتظمة وتتعلم تجميع مناطق الصورة في شرائح ذات أشكال تعسفية أكبر تدريجيًا. نقوم بتدريب GroupViT بشكل مشترك مع مشفر نصي على مجموعة بيانات نصية للصور واسعة النطاق عبر الخسائر التباينية. بدون أي ملاحظات على مستوى البكسل، يتعلم GroupViT تجميع المناطق الدلالية معًا وينتقل بنجاح إلى مهمة التجزئة الدلالية بطريقة بدون إشراف، أي بدون أي ضبط دقيق إضافي. يحقق دقة بدون إشراف تبلغ 52.3% mIoU على مجموعة بيانات PASCAL VOC 2012 و22.4% mIoU على مجموعة بيانات PASCAL Context، وينافس طرق التعلم عن طريق النقل التي تتطلب مستويات أعلى من الإشراف.*

ساهم بهذا النموذج [xvjiarui](https://huggingface.co/xvjiarui). ساهم في إصدار TensorFlow بواسطة [ariG23498](https://huggingface.co/ariG23498) بمساعدة [Yih-Dar SHIEH](https://huggingface.co/ydshieh) و[Amy Roberts](https://huggingface.co/amyeroberts) و[Joao Gante](https://huggingface.co/joaogante).

يمكن العثور على الكود الأصلي [هنا](https://github.com/NVlabs/GroupViT).

## نصائح الاستخدام

- يمكنك تحديد `output_segmentation=True` في الأمام من `GroupViTModel` للحصول على logits التجزئة للنصوص المدخلة.

## الموارد

قائمة بموارد Hugging Face الرسمية والمجتمعية (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام GroupViT.

- أسرع طريقة للبدء في استخدام GroupViT هي التحقق من [دفاتر الملاحظات التوضيحية](https://github.com/xvjiarui/GroupViT/blob/main/demo/GroupViT_hf_inference_notebook.ipynb) (والتي تعرض الاستدلال على التجزئة بدون إشراف).

- يمكنك أيضًا الاطلاع على [تجربة HuggingFace Spaces](https://huggingface.co/spaces/xvjiarui/GroupViT) للعب مع GroupViT.

## GroupViTConfig

[[autodoc]] GroupViTConfig

- from_text_vision_configs

## GroupViTTextConfig

[[autodoc]] GroupViTTextConfig

## GroupViTVisionConfig

[[autodoc]] GroupViTVisionConfig

<frameworkcontent>

<pt>

## GroupViTModel

[[autodoc]] GroupViTModel

- forward

- get_text_features

- get_image_features

## GroupViTTextModel

[[autodoc]] GroupViTTextModel

- forward

## GroupViTVisionModel

[[autodoc]] GroupViTVisionModel

- forward

</pt>

<tf>

## TFGroupViTModel

[[autodoc]] TFGroupViTModel

- call

- get_text_features

- get_image_features

## TFGroupViTTextModel


[[autodoc]] TFGroupViTTextModel

- call

## TFGroupViTVisionModel

[[autodoc]] TFGroupViTVisionModel

- call

</tf>

</frameworkcontent>