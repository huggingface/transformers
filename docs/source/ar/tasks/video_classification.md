# تصنيف الفيديو

[[open-in-colab]]

تصنيف الفيديو هو مهمة تعيين تسمية أو فئة لفيديو كامل. من المتوقع أن يكون لكل فيديو فئة واحدة فقط. تتلقى نماذج تصنيف الفيديو فيديو كمدخلات وتعيد تنبؤًا بالفئة التي ينتمي إليها الفيديو. يمكن استخدام هذه النماذج لتصنيف محتوى الفيديو. أحد التطبيقات الواقعية لتصنيف الفيديو هو التعرف على الإجراءات/الأنشطة، وهو مفيد لتطبيقات اللياقة البدنية. كما أنه يساعد الأشخاص ضعاف البصر، خاصة عند التنقل.

سيوضح هذا الدليل كيفية:

1. ضبط نموذج [VideoMAE](https://huggingface.co/docs/transformers/main/en/model_doc/videomae) على مجموعة فرعية من مجموعة بيانات [UCF101](https://www.crcv.ucf.edu/data/UCF101.php).
2. استخدام نموذجك المضبوط للتنبؤ.

<Tip>

لمعرفة جميع البنى ونقاط المراقبة المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/video-classification).

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install -q pytorchvideo transformers evaluate
```

ستستخدم [PyTorchVideo](https://pytorchvideo.org/) (المسماة `pytorchvideo`) لمعالجة الفيديوهات وإعدادها.

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات UCF101

ابدأ بتحميل مجموعة فرعية من [مجموعة بيانات UCF-101](https://www.crcv.ucf.edu/data/UCF101.php). سيعطيك هذا فرصة للتجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

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
...     t.extractall(".")
```

بشكل عام، يتم تنظيم مجموعة البيانات على النحو التالي:

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

بعد ذلك، يمكنك حساب عدد مقاطع الفيديو الإجمالية.

```py
>>> import pathlib
>>> dataset_root_path = "UCF101_subset"
>>> dataset_root_path = pathlib.Path(dataset_root_path)
```

```py
>>> video_count_train = len(list(dataset_root_path.glob("train/*/*.avi")))
>>> video_count_val = len(list(dataset_root_
path.glob("val/*/*.avi")))
>>> video_count_test = len(list(dataset_root_path.glob("test/*/*.avi")))
>>> video_total = video_count_train + video_count_val + video_count_test
>>> print(f"Total videos: {video_total}")
```

```py
>>> all_video_file_paths = (
...     list(dataset_root_path.glob("train/*/*.avi"))
...     + list(dataset_root_path.glob("val/*/*.avi"))
...     + list(dataset_root_path.glob("test/*/*.avi"))
... )
>>> all_video_file_paths[:5]
```

تظهر مسارات الفيديو (المُرتبة) على النحو التالي:

```bash
...
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c04.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g07_c06.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c02.avi',
'UCF101_subset/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g09_c06.avi'
...
```

ستلاحظ أن هناك مقاطع فيديو تنتمي إلى نفس المجموعة/المشهد حيث تُشير المجموعة إلى "g" في مسارات ملفات الفيديو. على سبيل المثال، `v_ApplyEyeMakeup_g07_c04.avi`  و  `v_ApplyEyeMakeup_g07_c06.avi` .

بالنسبة لعمليات التقسيم والتحقق من الصحة، لا تريد أن تكون لديك مقاطع فيديو من نفس المجموعة/المشهد لمنع [تسرب البيانات](https://www.kaggle.com/code/alexisbcook/data-leakage). تأخذ المجموعة الفرعية التي تستخدمها في هذا البرنامج التعليمي هذه المعلومات في الاعتبار.

بعد ذلك، ستقوم باستنتاج مجموعة العلامات الموجودة في مجموعة البيانات. كما ستقوم بإنشاء قاموسين سيكونان مفيدين عند تهيئة النموذج:

* `label2id`: يقوم بتعيين أسماء الفئات إلى أعداد صحيحة.
* `id2label`: يقوم بتعيين الأعداد الصحيحة إلى أسماء الفئات.

```py
>>> class_labels = sorted({str(path).split("/")[2] for path in all_video_file_paths})
>>> label2id = {label: i for i, label in enumerate(class_labels)}
>>> id2label = {i: label for label, i in label2id.items()}

>>> print(f"Unique classes: {list(label2id.keys())}.")

# Unique classesة: ['ApplyEyeMakeup'، 'ApplyLipstick'، 'Archery'، 'BabyCrawling'، 'BalanceBeam'، 'BandMarching'، 'BaseballPitch'، 'Basketball'، 'BasketballDunk'، 'BenchPress'].
```

هناك 10 فئات فريدة. لكل فئة، هناك 30 مقطع فيديو في مجموعة التدريب.

## تحميل نموذج لضبط دقيق

قم بتهيئة نموذج تصنيف فيديو من نقطة تفتيش مُدربة مسبقًا ومعالج الصور المرتبط بها. يحتوي مشفر النموذج على معلمات مُدربة مسبقًا، ورأس التصنيف مُهيأ بشكل عشوائي. سيكون معالج الصور مفيدًا عند كتابة خط أنابيب المعالجة المسبقة لمجموعة البيانات الخاصة بنا.

```py
>>> from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

>>> model_ckpt = "MCG-NJU/videomae-base"
>>> image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
>>> model = VideoMAEForVideoClassification.from_pretrained(
...     model_ckpt,
...     label2id=label2id,
...     id2label=id2label,
...     ignore_mismatched_sizes=True,  # قم بتوفير هذا في حالة كنت تخطط لضبط دقيق لنقطة تفتيش مُدربة بالفعل
... )
```

بينما يتم تحميل النموذج، قد تلاحظ التحذير التالي:

```bash
Some weights of the model checkpoint at MCG-NJU/videomae-base were not used when initializing VideoMAEForVideoClassification: [..., 'decoder.decoder_layers.1.attention.output.dense.bias', 'decoder.decoder_layers.2.attention.attention.key.weight']
- This IS expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing VideoMAEForVideoClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of VideoMAEForVideoClassification were not initialized from the model checkpoint at MCG-NJU/videomae-base and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

يُخبرنا التحذير أننا نقوم بحذف بعض الأوزان (مثل أوزان وانحياز طبقة "المصنف") وتهيئة بعض الأوزان الأخرى بشكل عشوائي (أوزان وانحياز طبقة "مصنف" جديدة). هذا متوقع في هذه الحالة، لأننا نقوم بإضافة رأس جديد لا تتوفر له أوزان مُدربة مسبقًا، لذا يحذرنا البرنامج من أنه يجب علينا ضبط النموذج دقيقًا قبل استخدامه للتنبؤ، وهو ما سنقوم به بالضبط.

**ملاحظة**: أن [هذه النقطة](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) تؤدي إلى أداء أفضل في هذه المهمة لأن نقطة التفتيش تم الحصول عليها عن طريق الضبط الدقيق على مهمة أسفل مجرى مماثلة ذات تداخل كبير في النطاق. يمكنك التحقق من [هذه النقطة](https://huggingface.co/sayakpaul/videomae-base-finetuned-kinetics-finetuned-ucf101-subset) والتي تم الحصول عليها عن طريق الضبط الدقيق لـ `MCG-NJU/videomae-base-finetuned-kinetics`.

## إعداد مجموعات البيانات للتدريب

لمعالجة مقاطع الفيديو مسبقًا، ستستخدم مكتبة [PyTorchVideo](https://pytorchvideo.org/). ابدأ باستيراد التبعيات التي نحتاجها.

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

بالنسبة لتحويلات مجموعة بيانات التدريب، استخدم مزيجًا من الاستعيان الزمني الموحد، وتطبيع البكسل، والتقطيع العشوائي، والانعكاس الأفقي العشوائي. بالنسبة لتحويلات مجموعة بيانات التحقق من الصحة والتقييم، احتفظ بنفس تسلسل التحولات باستثناء التقطيع العشوائي والانعكاس الأفقي. لمزيد من المعلومات حول تفاصيل هذه التحولات، راجع [الوثائق الرسمية لـ PyTorchVideo](https://pytorchvideo.org).

استخدم `image_processor` المرتبط بالنموذج المُدرب مسبقًا للحصول على المعلومات التالية:

* متوسط الانحراف المعياري للصورة التي سيتم تطبيعها معها بكسل إطار الفيديو.
* الدقة المكانية التي سيتم تغيير حجم إطارات الفيديو إليها.

ابدأ بتحديد بعض الثوابت.

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

الآن، قم بتعريف تحويلات مجموعة البيانات المحددة ومجموعات البيانات على التوالي. بدءًا من مجموعة التدريب:

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

يمكن تطبيق نفس تسلسل سير العمل على مجموعات التحقق من الصحة والتقييم:

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
...     data_path=os.path.min(dataset_root_path, "val"),
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

**ملاحظة**: تم أخذ خطوط أنابيب مجموعة البيانات أعلاه من [مثال PyTorchVideo الرسمي](https://pytorchvideo.org/docs/tutorial_classification#dataset). نحن نستخدم الدالة [`pytorchvideo.data.Ucf101()`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.Ucf101) لأنها مصممة خصيصًا لمجموعة بيانات UCF-101. في الأساس، فإنه يعيد كائن [`pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset`](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html#pytorchvideo.data.LabeledVideoDataset). تعد فئة `LabeledVideoDataset` الفئة الأساسية لجميع مقاطع الفيديو في مجموعة بيانات PyTorchVideo. لذلك، إذا كنت تريد استخدام مجموعة بيانات مخصصة غير مدعومة افتراضيًا بواسطة PyTorchVideo، فيمكنك توسيع فئة `LabeledVideoDataset` وفقًا لذلك. راجع وثائق [API للبيانات](https://pytorchvideo.readthedocs.io/en/latest/api/data/data.html) لمعرفة المزيد. أيضًا، إذا كانت مجموعة البيانات الخاصة بك تتبع بنية مماثلة (كما هو موضح أعلاه)، فإن استخدام `pytorchvideo.data.Ucf101()` يجب أن يعمل بشكل جيد.

يمكنك الوصول إلى وسيط `num_videos` لمعرفة عدد مقاطع الفيديو في مجموعة البيانات.

```py
>>> print(train_dataset.num_videos, val_dataset.num_videos, test_dataset.num_videos)
# (300، 30، 75)
```

## تصور الفيديو المعالج مسبقًا للتصحيح الأفضل

```py
>>> import imageio
>>> import numpy as np
>>> from IPython.display import Image

>>> def unnormalize_img(img):
...     """Un-normalizes the image pixels."""
...     img = (img * std) + mean
...     img = (img * 255).astype("uint8")
...     return img.clip(0, 255)

>>> def create_gif(video_tensor, filename="sample.gif"):
...     """Prepares a GIF from a video tensor.
...     
...     The video tensor is expected to have the following shape:
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
...     """Prepares and displays a GIF from a video tensor."""
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

استفد من [`Trainer`](https://huggingface.co/docs/transformers/main_classes/trainer) من  🤗 Transformers لتدريب النموذج. لتهيئة مثيل لـ `Trainer`، تحتاج إلى تحديد تكوين التدريب ومقاييس التقييم. والأهم من ذلك هو [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments)، وهي فئة تحتوي على جميع السمات لتكوين التدريب. يتطلب اسم مجلد الإخراج، والذي سيتم استخدامه لحفظ نقاط التفتيش للنموذج. كما يساعد في مزامنة جميع المعلومات في مستودع النموذج على 🤗 Hub.

معظم الحجج التدريبية واضحة، ولكن هناك واحدة مهمة جدًا هنا هي `remove_unused_columns=False`. سيؤدي هذا إلى إسقاط أي ميزات لا يستخدمها الدالة call الخاصة بالنموذج. بشكل افتراضي، يكون True لأنه من المثالي عادةً إسقاط أعمدة الميزات غير المستخدمة، مما يجعل من السهل تفكيك الإدخالات في دالة الاستدعاء الخاصة بالنموذج. ولكن، في هذه الحالة، فأنت بحاجة إلى الميزات غير المستخدمة ('video' على وجه الخصوص) من أجل إنشاء 'pixel_values' (وهو مفتاح إلزامي يتوقعه نموذجنا في إدخالاته).


```py 
>>> from transformers import TrainingArguments، Trainer

>>> model_name = model_ckpt.split("/")[-1]
>>> new_model_name = f"{model_name}-finetuned-ucf101-subset"
>>> num_epochs = 4

>>> args = TrainingArguments(
...     new_model_name،
...     remove_unused_columns=False،
...     eval_strategy="epoch"،
...     save_strategy="epoch"،
...     learning_rate=5e-5،
...     per_device_train_batch_size=batch_size،
...     per_device_eval_batch_size=batch_size،
...     warmup_ratio=0.1،
...     logging_steps=10،
...     load_best_model_at_end=True،
...     metric_for_best_model="accuracy"،
...     push_to_hub=True،
...     max_steps=(train_dataset.num_videos // batch_size) * num_epochs،
... )
```

مجموعة البيانات التي تم إرجاعها بواسطة `pytorchvideo.data.Ucf101()` لا تنفذ طريقة `__len__`. لذلك، يجب علينا تحديد `max_steps` عند إنشاء مثيل لـ `TrainingArguments`. 

بعد ذلك، تحتاج إلى تحديد دالة لحساب المقاييس من التوقعات، والتي ستستخدم `metric` التي ستقوم بتحميلها الآن. المعالجة المسبقة الوحيدة التي يجب عليك القيام بها هي أخذ argmax من logits المتوقعة:

```py
import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
```

**ملاحظة حول التقييم**:

في [ورقة VideoMAE](https://arxiv.org/abs/2203.12602)، يستخدم المؤلفون استراتيجية التقييم التالية. حيث يقومون بتقييم النموذج على عدة مقاطع من مقاطع الفيديو الاختبارية وتطبيق تقطيعات مختلفة على تلك المقاطع والإبلاغ عن النتيجة الإجمالية. ومع ذلك، حرصًا على البساطة والإيجاز، لا نأخذ ذلك في الاعتبار في هذا البرنامج التعليمي.

قم أيضًا بتعريف `collate_fn`، والتي ستُستخدم لدمج الأمثلة في مجموعات. تتكون كل مجموعة من مفتاحين، وهما `pixel_values` و`labels`.

```py
>>> def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```

بعد ذلك، قم ببساطة بتمرير كل هذا بالإضافة إلى مجموعات البيانات إلى `Trainer`:

```py
>>> trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
```

قد تتساءل عن سبب تمرير `image_processor` كـ tokenizer على الرغم من أنك قمت بمعالجة البيانات بالفعل. هذا فقط للتأكد من أن ملف تكوين معالج الصور (المخزن بتنسيق JSON) سيتم تحميله أيضًا إلى المستودع على Hub.

الآن، نقوم بضبط نموذجنا عن طريق استدعاء طريقة `train`:

```py
>>> train_results = trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

قم بتحميل مقطع فيديو للاستنتاج:

```py
>>> sample_test_video = next(iter(test_dataset))
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/sample_gif_two.gif" alt="فرق تلعب كرة السلة"/>
</div>

أبسط طريقة لتجربة نموذجك المضبوط للاستنتاج هي استخدامه في [`pipeline`](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.VideoClassificationPipeline). قم بتنفيذ عملية أنابيب لتصنيف الفيديو باستخدام نموذجك، ومرر الفيديو إليه:

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

يمكنك أيضًا محاكاة نتائج الأنابيب يدويًا إذا أردت.

```py
>>> def run_inference(model, video):
    # (num_frames, num_channels, height, width)
    perumuted_sample_test_video = video.permute(1, 0, 2, 3)
    inputs = {
        "pixel_values": perumuted_sample_test_video.unsqueeze(0),
        "labels": torch.tensor(
            [sample_test_video["label"]]
        ),  # يمكن تخطي هذا إذا لم تكن لديك تسميات متاحة.
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits
```

الآن، قم بتمرير إدخالك إلى النموذج وإرجاع `logits`:

```py
>>> logits = run_inference(trained_model, sample_test_video["video"])
```

بعد فك تشفير `logits`، نحصل على:

```py
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
# Predicted class: BasketballDunk
```