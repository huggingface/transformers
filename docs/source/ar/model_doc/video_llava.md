# Video-LLaVA

## نظرة عامة
Video-LLaVa هو نموذج لغة مفتوح المصدر متعدد الوسائط تم تدريبه عن طريق الضبط الدقيق لـ LlamA/Vicuna على بيانات اتباع التعليمات متعددة الوسائط التي تم إنشاؤها بواسطة Llava1.5 وVideChat. وهو نموذج لغة تنازلي، يعتمد على بنية المحول (transformer architecture). يوحد Video-LLaVa التمثيلات المرئية في مساحة ميزات اللغة، ويمكّن نموذج لغة اللغة من أداء قدرات الاستدلال المرئي على الصور ومقاطع الفيديو في نفس الوقت.

تم اقتراح نموذج Video-LLaVA في ورقة "Video-LLaVA: Learning United Visual Representation by Alignment Before Projection" من قبل Bin Lin وYang Ye وBin Zhu وJiaxi Cui وMunang Ning وPeng Jin وLi Yuan.

ملخص الورقة هو كما يلي:

*"عزز نموذج اللغة والرؤية الكبير (LVLM) أداء مختلف مهام التدفق السفلي في فهم اللغة والرؤية. تقوم معظم الطرق الحالية بتشفير الصور ومقاطع الفيديو في مساحات ميزات منفصلة، والتي يتم تغذيتها بعد ذلك كمدخلات لنماذج اللغة الكبيرة. ومع ذلك، بسبب عدم وجود توحيد لعملية تمثيل الصور ومقاطع الفيديو، أي عدم الاتساق قبل الإسقاط، يصبح من الصعب على نموذج لغة اللغة الكبيرة (LLM) تعلم التفاعلات متعددة الوسائط من عدة طبقات إسقاط رديئة. في هذا العمل، نوحد التمثيل المرئي في مساحة ميزات اللغة للانتقال بنموذج LLM الأساسي نحو LVLM موحد. ونتيجة لذلك، أنشأنا خط أساس بسيطًا ولكنه قويًا لـ LVLM، وهو Video-LLaVA، والذي يتعلم من مجموعة بيانات مختلطة من الصور ومقاطع الفيديو، مما يعزز كل منهما الآخر. يحقق Video-LLaVA أداءً متفوقًا في مجموعة واسعة من معايير الصور التي تغطي 5 مجموعات بيانات لأسئلة الصور و4 مجموعات أدوات معايير الصور. بالإضافة إلى ذلك، يتفوق نموذجنا Video-LLaVA أيضًا على Video-ChatGPT بنسبة 5.8٪ و9.9٪ و18.6٪ و10.1٪ على مجموعات بيانات MSRVTT وMSVD وTGIF وActivityNet على التوالي. وتجدر الإشارة إلى أن التجارب الواسعة توضح أن Video-LLaVA يفيد الصور ومقاطع الفيديو بشكل متبادل في إطار تمثيل مرئي موحد، متفوقًا على النماذج المصممة خصيصًا للصور أو مقاطع الفيديو. نهدف من خلال هذا العمل إلى تقديم رؤى متواضعة حول المدخلات متعددة الوسائط لنموذج اللغة."*

## نصائح الاستخدام:

- نوصي المستخدمين باستخدام padding_side="left" عند حساب التوليد الدفعي لأنه يؤدي إلى نتائج أكثر دقة. فقط تأكد من استدعاء processor.tokenizer.padding_side = "left" قبل التوليد.

- لاحظ أن النموذج لم يتم تدريبه صراحةً لمعالجة عدة صور/مقاطع فيديو في نفس المطالبة، على الرغم من أن هذا ممكن من الناحية الفنية، فقد تواجه نتائج غير دقيقة.

- لاحظ أن مدخلات الفيديو يجب أن تحتوي على 8 إطارات بالضبط، حيث تم تدريب النماذج في هذا الإعداد.

تمت المساهمة بهذا النموذج من قبل [RaushanTurganbay](https://huggingface.co/RaushanTurganbay). يمكن العثور على الكود الأصلي [هنا](https://github.com/PKU-YuanGroup/Video-LLaVA).

## مثال الاستخدام

### وضع الوسائط الفردية

يمكن للنموذج قبول كل من الصور ومقاطع الفيديو كمدخلات. فيما يلي مثال على التعليمات البرمجية للاستنتاج في الدقة النصفية (`torch.float16`):

```python
import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Load the model in half-precision
model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map="auto")
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# Load the video as an np.arrau, sampling uniformly 8 frames
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
video = read_video_pyav(container, indices)

# For better results, we recommend to prompt the model in the following format
prompt = "USER: <video>Why is this funny? ASSISTANT:"
inputs = processor(text=prompt, videos=video, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=60)
processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
```

بالنسبة لمحادثات الدورات المتعددة، قم بتغيير تنسيق المطالبة إلى ما يلي:

```bash
"USER: <video>What do you see in this video? ASSISTANT: A baby reading a book. USER: Why is the it funny? ASSISTANT:"
```

### وضع الوسائط المختلطة

يمكن للنموذج أيضًا التوليد من إدخالات الصور ومقاطع الفيديو المتشابكة. ومع ذلك، لاحظ أنه لم يتم تدريبه في إعداد الصور ومقاطع الفيديو المتشابكة والتي قد تؤثر على الأداء. فيما يلي مثال على الاستخدام لوسائط الإدخال المختلطة، أضف السطور التالية إلى مقطع التعليمات البرمجية أعلاه:

```python
from PIL import Image
import requests

# التوليد من إدخالات الصور ومقاطع الفيديو المختلطة
# تحميل صورة وكتابة مطالبة جديدة
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "USER: <image> How many cats are there in the image? ASSISTANT: There are two cats. USER: <video>Why is this video funny? ASSISTANT:"

inputs = processor(text=prompt, images=image, videos=clip, padding=True, return_tensors="pt")

# التوليد
generate_ids = model.generate(**inputs, max_length=50)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

```

## تحسين النموذج

### التكميم باستخدام Bitsandbytes لكفاءة الذاكرة

يمكن تحميل النموذج في عدد أقل من البتات، مما يقلل بشكل كبير من عبء الذاكرة مع الحفاظ على أداء النموذج الأصلي. يسمح هذا بالنشر الفعال في الحالات التي تكون فيها الموارد محدودة.

أولاً، تأكد من تثبيت bitsandbytes عن طريق تشغيل `pip install bitsandbytes` وامتلاك حق الوصول إلى جهاز GPU متوافق مع CUDA. قم بتحميل النموذج الكمي ببساطة عن طريق إضافة [`BitsAndBytesConfig`](../main_classes/quantization#transformers.BitsAndBytesConfig) كما هو موضح أدناه:

```python
from transformers import VideoLlavaForConditionalGeneration, BitsAndBytesConfig

# حدد كيفية تكميم النموذج
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", quantization_config=quantization_config, device_map="auto")
```

### Flash-Attention 2 لتسريع التوليد

بالإضافة إلى ذلك، يمكننا تسريع استنتاج النموذج بشكل كبير باستخدام [Flash Attention](../perf_train_gpu_one.md#flash-attention-2)، وهو تنفيذ أسرع لآلية الانتباه المستخدمة داخل النموذج.

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

يجب أن يكون لديك أيضًا أجهزة متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في وثائق مستودع [flash attention](https://github.com/Dao-AILab/flash-attention) الرسمي. يمكن استخدام FlashAttention-2 فقط عندما يتم تحميل النموذج في `torch.float16` أو `torch.bfloat16`.

لتحميل وتشغيل نموذج باستخدام Flash Attention-2، قم ببساطة بإضافة `attn_implementation="flash_attention_2"` عند تحميل النموذج كما يلي:

```python
from transformers import VideoLlavaForConditionalGeneration

model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
).to(0)
```

## VideoLlavaConfig

[[autodoc]] VideoLlavaConfig

## VideoLlavaImageProcessor

[[autodoc]] VideoLlavaImageProcessor

## VideoLlavaProcessor

[[autodoc]] VideoLlavaProcessor

## VideoLlavaForConditionalGeneration

[[autodoc]] VideoLlavaForConditionalGeneration

- forward