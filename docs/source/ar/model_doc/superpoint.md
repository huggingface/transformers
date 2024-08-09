# SuperPoint

## نظرة عامة

اقتُرح نموذج SuperPoint في الورقة البحثية بعنوان "SuperPoint: Self-Supervised Interest Point Detection and Description" من قبل دانييل ديتون، وتوماش ماليسيويتش، وأندرو رابينوفيتش.

هذا النموذج هو نتيجة لتدريب ذاتي الإشراف لشبكة متكاملة تمامًا للكشف عن نقاط الاهتمام ووصفها. ويمكن للنموذج اكتشاف نقاط الاهتمام التي تتكرر في تحويلات التشابه وتوفير وصف لكل نقطة. استخدام النموذج في حد ذاته محدود، ولكنه يمكن أن يستخدم كمستخلص للخصائص لمهمات أخرى مثل تقدير التشابه، ومطابقة الصور، وما إلى ذلك.

وفيما يلي الملخص المستخرج من الورقة البحثية:

"تقدم هذه الورقة إطارًا ذاتي الإشراف لتدريب كاشفات نقاط الاهتمام والواصفات المناسبة لعدد كبير من مشكلات الهندسة متعددة المشاهدات في رؤية الكمبيوتر. وعلى عكس الشبكات العصبية المستندة إلى الرقع، فإن نموذجنا المتكامل تمامًا يعمل على الصور بالحجم الكامل ويحسب بشكل مشترك مواقع نقاط الاهتمام على مستوى البكسل والواصفات المرتبطة بها في تمرير أمامي واحد. نقدم التكيّف التشابهي، وهو نهج متعدد المقاييس ومتعدد التشابهات لتعزيز قابلية تكرار كشف نقاط الاهتمام وإجراء التكيّف عبر المجالات (على سبيل المثال، من المجال التخيلي إلى الواقعي). وعندما يتم تدريب نموذجنا على مجموعة بيانات الصور العامة MS-COCO باستخدام التكيّف التشابهي، فإنه يستطيع اكتشاف مجموعة أكثر ثراءً من نقاط الاهتمام بشكل متكرر مقارنة بالنموذج العميق المُكيّف مسبقًا وأي كاشف زوايا تقليدي آخر. ويحقق النظام النهائي نتائج رائدة في مجال تقدير التشابه على مجموعة بيانات HPatches عند مقارنته بنماذج LIFT وSIFT وORB."

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/superpoint_architecture.png" alt="drawing" width="500"/>

<small>نظرة عامة على SuperPoint. مأخوذة من <a href="https://arxiv.org/abs/1712.07629v4">الورقة البحثية الأصلية.</a></small>

## نصائح الاستخدام

فيما يلي مثال سريع على استخدام النموذج للكشف عن نقاط الاهتمام في صورة:

```python
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
```

تحتوي الإخراجات على قائمة إحداثيات نقاط الاهتمام مع درجاتها ووصفها (متجه بطول 256).

يمكنك أيضًا إدخال صور متعددة إلى النموذج. وبسبب طبيعة SuperPoint، لإخراج عدد متغير من نقاط الاهتمام، ستحتاج إلى استخدام خاصية القناع (mask) لاسترجاع المعلومات المقابلة:

```python
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import requests

url_image_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_1 = Image.open(requests.get(url_image_1, stream=True).raw)
url_image_2 = "http://images.cocodataset.org/test-stuff2017/000000000568.jpg"
image_2 = Image.open(requests.get(url_image_2, stream=True).raw)

images = [image_1, image_2]

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(images, return_tensors="pt")
outputs = model(**inputs)

for i in range(len(images)):
    image_mask = outputs.mask[i]
    image_indices = torch.nonzero(image_mask).squeeze()
    image_keypoints = outputs.keypoints[i][image_indices]
    image_scores = outputs.scores[i][image_indices]
    image_descriptors = outputs.descriptors[i][image_indices]
```

بعد ذلك، يمكنك طباعة نقاط الاهتمام على الصورة لتصور النتيجة:

```python
import cv2
for keypoint, score in zip(image_keypoints, image_scores):
    keypoint_x, keypoint_y = int(keypoint[0].item()), int(keypoint[1].item())
    color = tuple([score.item() * 255] * 3)
    image = cv2.circle(image, (keypoint_x, keypoint_y), 2, color)
cv2.imwrite("output_image.png", image)
```

تمت المساهمة بهذا النموذج من قبل [stevenbucaille](https://huggingface.co/stevenbucaille). ويمكن العثور على الكود الأصلي [هنا](https://github.com/magicleap/SuperPointPretrainedNetwork).

## الموارد

فيما يلي قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها برمز 🌎) لمساعدتك في البدء باستخدام SuperPoint. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب (Pull Request) وسنقوم بمراجعته! ويفضل أن يُظهر المورد شيئًا جديدًا بدلاً من تكرار مورد موجود.

- يمكن العثور على دفتر ملاحظات يُظهر الاستنتاج والتصور باستخدام SuperPoint [هنا](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SuperPoint/Inference_with_SuperPoint_to_detect_interest_points_in_an_image.ipynb). 🌎

## SuperPointConfig

[[autodoc]] SuperPointConfig

## SuperPointImageProcessor

[[autodoc]] SuperPointImageProcessor

- preprocess

## SuperPointForKeypointDetection

[[autodoc]] SuperPointForKeypointDetection

- forward