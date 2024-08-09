# TVP

## نظرة عامة

تم اقتراح إطار عمل Text-Visual Prompting (TVP) في الورقة البحثية [Text-Visual Prompting for Efficient 2D Temporal Video Grounding](https://arxiv.org/abs/2303.04995) بواسطة Yimeng Zhang, Xin Chen, Jinghan Jia, Sijia Liu, Ke Ding.

ملخص الورقة البحثية هو كما يلي:

> *في هذا البحث، قمنا بدراسة مشكلة التأريض الزمني للفيديو (TVG)، والتي تهدف إلى التنبؤ بنقاط وقت البداية/الانتهاء للحظات الموصوفة بجملة نصية ضمن فيديو طويل غير مقطوع. ومن خلال الاستفادة من الميزات المرئية ثلاثية الأبعاد الدقيقة، حققت تقنيات TVG تقدمًا ملحوظًا في السنوات الأخيرة. ومع ذلك، فإن التعقيد العالي للشبكات العصبية التلافيفية ثلاثية الأبعاد (CNNs) يجعل استخراج الميزات المرئية ثلاثية الأبعاد الكثيفة يستغرق وقتًا طويلاً، مما يتطلب ذاكرة مكثفة وموارد حوسبة. نحو تحقيق TVG بكفاءة، نقترح إطارًا جديدًا للتحفيز النصي والمرئي (TVP)، والذي يتضمن أنماط الاضطراب المُحسّنة (التي نسميها "المطالبات") في كل من المدخلات المرئية والميزات النصية لنموذج TVG. في تناقض حاد مع شبكات CNN ثلاثية الأبعاد، نظهر أن TVP يسمح لنا بالتدريب الفعال على برنامج تشفير الرؤية وترميز اللغة في نموذج TVG ثنائي الأبعاد وتحسين أداء دمج الميزات عبر الوسائط باستخدام ميزات مرئية ثنائية الأبعاد متفرقة منخفضة التعقيد فقط. علاوة على ذلك، نقترح خسارة IoU (TDIoU) للمسافة الزمنية من أجل التعلم الفعال لـ TVG. تُظهر التجارب التي أجريت على مجموعتي بيانات مرجعية، وهما مجموعات بيانات Charades-STA وActivityNet Captions، بشكل تجريبي أن TVP المقترح يعزز بشكل كبير أداء TVG ثنائي الأبعاد (على سبيل المثال، تحسين بنسبة 9.79% على Charades-STA وتحسين بنسبة 30.77% على ActivityNet Captions) ويحقق استدلال 5 × تسريع عبر TVG باستخدام ميزات مرئية ثلاثية الأبعاد.*

تتناول هذه الورقة البحثية مشكلة التحديد الزمني للفيديو (TVG)، والتي تتمثل في تحديد بدايات ونهايات أحداث محددة في فيديو طويل، كما هو موصوف في جملة نصية. ويتم اقتراح تقنية Text-Visual Prompting (TVP) لتحسين TVG. وتنطوي TVP على دمج أنماط مصممة خصيصًا، يُطلق عليها "prompts"، في كل من المكونات المرئية (القائمة على الصور) والنصية (القائمة على الكلمات) لنموذج TVG. وتوفر هذه الـ "prompts" سياقًا مكانيًا زمنيًا إضافيًا، مما يحسن قدرة النموذج على تحديد توقيتات الأحداث في الفيديو بدقة. ويستخدم هذا النهج مدخلات بصرية ثنائية الأبعاد بدلاً من ثلاثية الأبعاد. وعلى الرغم من أن المدخلات ثلاثية الأبعاد تقدم المزيد من التفاصيل المكانية الزمنية، إلا أنها تستغرق أيضًا وقتًا أطول في المعالجة. ويهدف استخدام المدخلات ثنائية الأبعاد مع طريقة الـ "prompting" إلى توفير مستويات مماثلة من السياق والدقة بكفاءة أكبر.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/tvp_architecture.png"
alt="drawing" width="600"/>

<small> تصميم TVP. مأخوذة من <a href="https://arxiv.org/abs/2303.04995">الورقة البحثية الأصلية.</a> </small>

تمت المساهمة بهذا النموذج بواسطة [Jiqing Feng](https://huggingface.co/Jiqing). ويمكن العثور على الكود الأصلي [هنا](https://github.com/intel/TVP).

## نصائح وأمثلة الاستخدام

"Prompts" هي أنماط اضطراب محسنة، سيتم إضافتها إلى إطارات الفيديو أو الميزات النصية. تشير "المجموعة الشاملة" إلى استخدام نفس المجموعة المحددة من الـ "prompts" لأي مدخلات، وهذا يعني أن هذه الـ "prompts" يتم إضافتها باستمرار إلى جميع إطارات الفيديو والميزات النصية، بغض النظر عن محتوى الإدخال.

يتكون TVP من مشفر بصري ومشفر متعدد الوسائط. يتم دمج مجموعة شاملة من الـ "prompts" البصرية والنصية في إطارات الفيديو والعناصر النصية التي تم أخذ عينات منها، على التوالي. وبشكل خاص، يتم تطبيق مجموعة من الـ "prompts" البصرية المختلفة على إطارات الفيديو التي تم أخذ عينات منها بشكل موحد من فيديو واحد غير محدد الطول بالترتيب.

هدف هذا النموذج هو دمج الـ "prompts" القابلة للتدريب في كل من المدخلات المرئية والميزات النصية لمشاكل التحديد الزمني للفيديو (TVG).

من حيث المبدأ، يمكن تطبيق أي مشفر بصري أو متعدد الوسائط في تصميم TVP المقترح.

يقوم [`TvpProcessor`] بتغليف [`BertTokenizer`] و [`TvpImageProcessor`] في مثيل واحد لتشفير النص وإعداد الصور على التوالي.

يوضح المثال التالي كيفية تشغيل التحديد الزمني للفيديو باستخدام [`TvpProcessor`] و [`TvpForVideoGrounding`].

```python
import av
import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, TvpForVideoGrounding


def pyav_decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps):
    '''
    Convert the video from its original fps to the target_fps and decode the video with PyAV decoder.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling.
            If clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the given video.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
    '''
    video = container.streams.video[0]
    fps = float(video.average_rate)
    clip_size = sampling_rate * num_frames / target_fps * fps
    delta = max(num_frames - clip_size, 0)
    start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    timebase = video.duration / num_frames
    video_start_pts = int(start_idx * timebase)
    video_end_pts = int(end_idx * timebase)
    seek_offset = max(video_start_pts - 1024, 0)
    container.seek(seek_offset, any_frame=False, backward=True, stream=video)
    frames = {}
    for frame in container.decode(video=0):
        if frame.pts < video_start_pts:
            continue
        frames[frame.pts] = frame
        if frame.pts > video_end_pts:
            break
    frames = [frames[pts] for pts in sorted(frames)]
    return frames, fps


def decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps):
    '''
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling.
            If clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the given video.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video.
    '''
    assert clip_idx >= -2, "Not a valied clip_idx {}".format(clip_idx)
    frames, fps = pyav_decode(container, sampling_rate, num_frames, clip_idx, num_clips, target_fps)
    clip_size = sampling_rate * num_frames / target_fps * fps
    index = np.linspace(0, clip_size - 1, num_frames)
    index = np.clip(index, 0, len(frames) - 1).astype(np.int64)
    frames = np.array([frames[idx].to_rgb().to_ndarray() for idx in index])
    frames = frames.transpose(0, 3, 1, 2)
    return frames


file = hf_hub_download(repo_id="Intel/tvp_demo", filename="AK2KG.mp4", repo_type="dataset")
model = TvpForVideoGrounding.from_pretrained("Intel/tvp-base")

decoder_kwargs = dict(
    container=av.open(file, metadata_errors="ignore"),
    sampling_rate=1,
    num_frames=model.config.num_frames,
    clip_idx=0,
    num_clips=1,
    target_fps=3,
)
raw_sampled_frms = decode(**decoder_kwargs)

text = "a person is sitting on a bed."
processor = AutoProcessor.from_pretrained("Intel/tvp-base")
model_inputs = processor(
    text=[text], videos=list(raw_sampled_frms), return_tensors="pt", max_text_length=100#, size=size
)

model_inputs["pixel_values"] = model_inputs["pixel_values"].to(model.dtype)
output = model(**model_inputs)

def get_video_duration(filename):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num/rate
        return duration
    return -1

duration = get_video_duration(file)
start, end = processor.post_process_video_grounding(output.logits, duration)

print(f"The time slot of the video corresponding to the text \"{text}\" is from {start}s to {end}s")
```

## نصائح:

- يستخدم هذا التنفيذ لـ TVP [`BertTokenizer`] لتوليد تضمين نصي ونموذج Resnet-50 لحساب التضمين المرئي.
- تم إصدار نقاط مرجعية للنموذج المُدرب مسبقًا [tvp-base](https://huggingface.co/Intel/tvp-base).
- يرجى الرجوع إلى [الجدول 2](https://arxiv.org/pdf/2303.04995.pdf) لأداء TVP على مهمة التحديد الزمني للفيديو.

## TvpConfig

[[autodoc]] TvpConfig

## TvpImageProcessor

[[autodoc]] TvpImageProcessor

- preprocess

## TvpProcessor

[[autodoc]] TvpProcessor

- __call__

## TvpModel

[[autodoc]] TvpModel

- forward

## TvpForVideoGrounding

[[autodoc]] TvpForVideoGrounding

- forward