<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# تصنيف الفيديو (Video classification)

[[open-in-colab]]

تصنيف الفيديو هو مهمة تهدف إلى إسناد تصنيف أو فئة واحدة إلى فيديو كامل. يُتوقّع أن يحتوي كل فيديو على فئة واحدة فقط. تأخذ نماذج تصنيف الفيديو فيديوًا كمدخل وتُعيد تنبؤًا حول الفئة التي ينتمي إليها الفيديو. يمكن استخدام هذه النماذج لتصنيف طبيعة الفيديو. من التطبيقات الواقعية لتصنيف الفيديو التعرف على الأفعال/الأنشطة (action/activity recognition)، وهو مفيد لتطبيقات اللياقة. كما أنه مساعد للأشخاص ذوي الإعاقة البصرية، خاصة أثناء التنقل.

يوضّح هذا الدليل كيفية:

1. ضبط (fine-tune) نموذج [VideoMAE](https://huggingface.co/docs/transformers/main/en/model_doc/videomae) على مجموعة فرعية من بيانات [UCF101](https://www.crcv.ucf.edu/data/UCF101.php).
2. استخدام نموذجك المضبوط للاستدلال.

<Tip>

للاطّلاع على جميع البُنى ونقاط التحقق المتوافقة مع هذه المهمة، ننصح بزيارة [صفحة المهمة](https://huggingface.co/tasks/video-classification).

</Tip>

قبل البدء، تأكّد من تثبيت جميع المكتبات اللازمة:

```bash
pip install -q pytorchvideo transformers evaluate
```

ستستخدم [PyTorchVideo](https://pytorchvideo.org/) (المشار إليه باسم `pytorchvideo`) لمعالجة الفيديوهات وتجهيزها.

نشجّعك على تسجيل الدخول إلى حسابك على Hugging Face لرفع ومشاركة نموذجك مع المجتمع. عند المطالبة، أدخل رمز الوصول (token) لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات UCF101

ابدأ بتحميل مجموعة فرعية من [مجموعة بيانات UCF-101](https://www.crcv.ucf.edu/data/UCF101.php). سيمنحك هذا فرصة للتجربة والتأكد من أن كل شيء يعمل قبل استثمار وقت أطول في التدريب على المجموعة الكاملة.

```py
>>> from huggingface_hub import hf_hub_download

>>> hf_dataset_identifier = "sayakpaul/ucf101-subset"
>>> filename = "UCF101_subset.tar.gz"
>>> file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
```

بعد تنزيل المجموعة الفرعية، تحتاج إلى استخراج الأرشيف المضغوط:

```py
>>> import tarfile

>>> with tarfile.open(file_path) as t:
...      t.extractall(".")
```

على مستوى عالٍ، تُنظّم مجموعة البيانات كما يلي:

```bash
UCF101_subset/
    train/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    val/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
    test/
        BandMarching/
            video_1.mp4
            video_2.mp4
            ...
        Archery
            video_1.mp4
            video_2.mp4
            ...
        ...
```

يمكنك بعدها عدّ عدد الفيديوهات الإجمالي.

```py
>>> import pathlib
>>> dataset_root_path = "UCF101_subset"
>>> dataset_root_path = pathlib.Path(dataset_root_path)
```

```py
>>> video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
>>> video_count_val = len(list(dataset_root_path.glob("val/*/*.avi")))
>>> video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
>>> video_total = video_count_train + video_count_val + video_count_test
>>> print(f"Total videos: {video_total}")
```

```py
>>> all_video_file_paths = (
...     list(dataset_root_path.glob("train/*/*.avi"))
...     + list(dataset_root_path.glob("val/*/*.avi"))
...     + list(dataset_root_path.glob("test/*/*.avi"))
...  )
>>> all_video_file_paths[:5]
```

تظهر مسارات الفيديو (بعد الفرز) على النحو التالي:

```bash
...
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c04.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c06.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c02.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c06.avi'
...
```

لتعريف تسميات الفئات، أنشئ خرائط بين أسماء الفئات والأعداد الصحيحة:

* `label2id`: يربط أسماء الفئات بأعداد صحيحة.
* `id2label`: يربط الأعداد الصحيحة بأسماء الفئات.

```py
>>> class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
>>> label2id = {label: i for i, label in enumerate(class_labels)}
>>> id2label = {i: label for label, i in label2id.items()}

>>> print(f"Unique classes: {list(label2id.keys())}.")
```

لنحمّل نموذج VideoMAE المصمم للتصنيف على الفيديو.

```py
>>> import torch
>>> from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

>>> model_ckpt = "MCG-NJU/videomae-base"
>>> image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
>>> model = VideoMAEForVideoClassification.from_pretrained(
...     model_ckpt,
...     label2id=label2id,
...     id2label=id2label,
...     ignore_mismatched_sizes=True,  # استخدمه إذا كنت تخطط لضبط نقطة تحقق مضبوطة مسبقًا
... )
```

أثناء تحميل النموذج، قد تلاحظ التحذير التالي:

```bash
Some weights of the model checkpoint at MCG-NJU/videomae-base were not used when initializing VideoMAEForVideoClassification: [..., 'decoder.decoder_layers.1.attention.output.dense.bias', 'decoder.decoder_layers.2.attention.attention.key.weight']
- This IS expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model trained on another task OR with another architecture
- This IS NOT expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model that you expect to be exactly identical (initializing a model trained on the same task in a similar architecture)
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

يعود هذا إلى أن VideoMAE نموذج مُشفِّر (encoder) قد تمت تهيئته مسبقًا من أجل مهمة إعادة بناء الفيديو ذاتيًا (MAE)، وأنت الآن تستخدم رأس تصنيف متوافق مع UCF101.

## تحضير المجموعات للتدريب

لاستخدام المعالجة المسبقة للفيديوهات، ستستفيد من [مكتبة PyTorchVideo](https://pytorchvideo.org/). ابدأ باستيراد التبعيات اللازمة.

```py
>>> import pytorchvideo.data

>>> from pytorchvideo.transforms import (
...     ApplyTransformToKey,
...     Normalize,
...     RandomShortSideScale,
...     RemoveKey,
...     ShortSideScale,
...     UniformTemporalSubsample,
... )

>>> from torchvision.transforms import (
...     Compose,
...     Lambda,
...     RandomCrop,
...     RandomHorizontalFlip,
...     Resize,
... )
```

بالنسبة لتحويلات مجموعة التدريب، استخدم مزيجًا من العيّنة الزمنية الموحدة، وتطبيع البكسلات، والقص العشوائي، والانقلاب الأفقي العشوائي. بالنسبة لتحويلات مجموعتي التحقق والتقييم، احتفظ بسلسلة التحويلات نفسها باستثناء القص العشوائي والانقلاب الأفقي. لمعرفة المزيد عن تفاصيل هذه التحويلات راجع [التوثيق الرسمي لـ PyTorchVideo](https://pytorchvideo.org).

استخدم `image_processor` المرتبط بالنموذج المُدرّب مسبقًا للحصول على المعلومات التالية:

- متوسط الصورة والانحراف المعياري لتطبيع بكسلات الإطارات.
- الدقة المكانية التي ستُعاد إليها أبعاد الإطارات.

ابدأ بتعريف بعض الثوابت.

```py
>>> mean = image_processor.image_mean
>>> std = image_processor.image_std
>>> if "shortest_edge" in image_processor.size:
...     height = width = image_processor.size["shortest_edge"]
>>> else:
...     height = image_processor.size["height"]
...     width = image_processor.size["width"]
>>> resize_to = (height, width)

>>> num_frames_to_sample = model.config.num_frames
>>> sample_rate = 4
>>> fps = 30
>>> clip_duration = num_frames_to_sample * sample_rate / fps
```

الآن، عرّف تحويلات كل مجموعة والبيانات نفسها. بدءًا من مجموعة التدريب:

```py
>>> train_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     RandomShortSideScale(min_size=256, max_size=320),
...                     RandomCrop(resize_to),
...                     RandomHorizontalFlip(p=0.5),
...                 ]
...             ),
...         ),
...     ]
... )

>>> train_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "train"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
...     decode_audio=False,
...     transform=train_transform,
... )
```

يمكن تطبيق سلسلة العمل نفسها على مجموعتي التحقق والتقييم:

```py
>>> val_transform = Compose(
...     [
...         ApplyTransformToKey(
...             key="video",
...             transform=Compose(
...                 [
...                     UniformTemporalSubsample(num_frames_to_sample),
...                     Lambda(lambda x: x / 255.0),
...                     Normalize(mean, std),
...                     Resize(resize_to),
...                 ]
...             ),
...         ),
...     ]
... )

>>> val_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "val"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )

>>> test_dataset = pytorchvideo.data.Ucf101(
...     data_path=os.path.join(dataset_root_path, "test"),
...     clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
...     decode_audio=False,
...     transform=val_transform,
... )
```

يمكنك الوصول إلى الوسيط `num_videos` لمعرفة عدد الفيديوهات في كل مجموعة.

```py
>>> print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)
# (300, 30, 75)
```

## تصور الفيديو المُعالج مسبقًا بهدف تتبّع الأعطال

```py
>>> import imageio
>>> import numpy as np
>>> from IPython.display import Image

>>> def unnormalize_img(img):
...     """إلغاء تطبيع بكسلات الصورة."""
...     img = (img * std) + mean
...     img = (img * 255).astype("uint8")
...     return img.clip(0, 255)

>>> def create_gif(video_tensor, filename="sample.gif"):
...     """إعداد GIF من موتر فيديو.
...
...     يُتوقع أن يكون شكل موتر الفيديو:
...     (num_frames, num_channels, height, width).
...     """
...     frames = []
...     for video_frame in video_tensor:
...         frame_unnormalized = unnormalize_img(video_frame.permute(1, 2, 0).numpy())
...         frames.append(frame_unnormalized)
...     kargs = {"duration": 0.25}
...     imageio.mimsave(filename, frames, "GIF", **kargs)
...     return filename

>>> def display_gif(video_tensor, gif_name="sample.gif"):
...     """تحضير وعرض GIF من موتر فيديو."""
...     video_tensor = video_tensor.permute(1, 0, 2, 3)
...     gif_filename = create_gif(video_tensor, gif_name)
...     return Image(filename=gif_filename)

>>> sample_video = next(iter(train_dataset))
>>> video_tensor = sample_video["video"]
>>> display_gif(video_tensor)
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif.gif" alt="Person playing basketball"/>
</div>

## تدريب النموذج

استخدمنا `Trainer` لتبسيط حلقة التدريب والتقييم وحفظ نقاط التحقق.

```py
>>> from transformers import TrainingArguments, Trainer

>>> model_name = model_ckpt.split("/")[-1]
>>> new_model_name = f"{model_name}-finetuned-ucf101-subset"
>>> num_epochs = 4

>>> args = TrainingArguments(
...     new_model_name,
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=batch_size,
...     per_device_eval_batch_size=batch_size,
...     warmup_ratio=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
...     max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
... )
```

للتقييم سنستخدم الدقة (accuracy).

```py
>>> import evaluate
>>> import numpy as np

>>> metric = evaluate.load("accuracy")

>>> def compute_metrics(eval_pred):
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

ملاحظة حول التقييم: في ورقة [VideoMAE](https://huggingface.co/papers/2203.12602)، يستخدم المؤلفون استراتيجية تقييم متعددة المقاطع والمحاصيل. لتبسيط الدليل، لا نعتمدها هنا.

عرّف دالة `collate_fn` لتجميع الأمثلة في دفعات. تتكوّن كل دفعة من مفتاحين: `pixel_values` و`labels`.

```py
>>> def collate_fn(examples):
...     # إعادة ترتيب إلى (num_frames, num_channels, height, width)
...     pixel_values = torch.stack(
...         [example["video"].permute(1, 0, 2, 3) for example in examples]
...     )
...     labels = torch.tensor([example["label"] for example in examples])
...     return {"pixel_values": pixel_values, "labels": labels}
```

ثم مرّر كل ذلك مع مجموعات البيانات إلى `Trainer`:

```py
>>> trainer = Trainer(
...     model,
...     args,
...     train_dataset=train_dataset,
...     eval_dataset=val_dataset,
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
...     data_collator=collate_fn,
... )
```

قد تتساءل لماذا مرّرت `image_processor` كـ "tokenizer" رغم أنك جهّزت البيانات مسبقًا. هذا فقط لضمان رفع ملف إعدادات معالج الصور (بصيغة JSON) إلى المستودع على الـ Hub أيضًا.

الآن اضبط النموذج باستدعاء `train`:

```py
>>> train_results = trainer.train()
```

بعد اكتمال التدريب، شارك نموذجك على الـ Hub باستخدام الطريقة [`~transformers.Trainer.push_to_hub`]:

```py
>>> trainer.push_to_hub()
```

## الاستدلال

رائع! الآن بعد أن قمت بضبط نموذج، يمكنك استخدامه للاستدلال.

حمّل فيديوًا للاختبار:

```py
>>> sample_test_video = next(iter(test_dataset))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif_two.gif" alt="Teams playing basketball"/>
</div>

أسهل طريقة لتجربة نموذجك المضبوط في الاستدلال هي استخدام [`pipeline`](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.VideoClassificationPipeline). أنشئ `pipeline` لتصنيف الفيديو باستخدام نموذجك، ومرّر الفيديو إليه:

```py
>>> from transformers import pipeline

>>> video_cls = pipeline(model="my_awesome_video_cls_model")
>>> video_cls("https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/v_BasketballDunk_g14_c06.avi")
[{'score': 0.9272987842559814, 'label': 'BasketballDunk'},
 {'score': 0.017777055501937866, 'label': 'BabyCrawling'},
 {'score': 0.01663011871278286, 'label': 'BalanceBeam'},
 {'score': 0.009560945443809032, 'label': 'BandMarching'},
 {'score': 0.0068979403004050255, 'label': 'BaseballPitch'}]
```

يمكنك أيضًا تكرار نتائج الـ `pipeline` يدويًا إن رغبت.

```py
>>> def run_inference(model, video):
...     # (num_frames, num_channels, height, width)
...     perumuted_sample_test_video = video.permute(1, 0, 2, 3)
...     inputs = {
...         "pixel_values": perumuted_sample_test_video.unsqueeze(0),
...         "labels": torch.tensor(
...             [sample_test_video["label"]]
...         ),  # يمكن تخطي هذا إن لم تتوفر تسميات
...     }

...     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
...     inputs = {k: v.to(device) for k, v in inputs.items()}
...     model = model.to(device)

...     # تمرير أمامي
...     with torch.no_grad():
...         outputs = model(**inputs)
...         logits = outputs.logits

...     return logits
```

مرّر المدخلات إلى النموذج وأعد `logits`:

```py
>>> logits = run_inference(trained_model, sample_test_video["video"])
```

فك ترميز `logits`:

```py
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
# Predicted class: BasketballDunk
```
