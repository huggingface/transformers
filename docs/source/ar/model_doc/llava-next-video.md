# LLaVa-NeXT-Video  

## نظرة عامة  
 اقترح نموذج LLaVa-NeXT-Video في [LLaVA-NeXT: A Strong Zero-shot Video Understanding Model](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/) بواسطة يوانهان جانج، وبو لي، وهاوتيان ليو، ويونغ جاي لي، وليانجكي جوي، ودي فو، وجياشي فينج، وزييوي ليو، وتشونيوان لي. ويحسن LLaVa-NeXT-Video من [LLaVa-NeXT](llava_next) عن طريق الضبط الدقيق على مزيج من مجموعات بيانات الفيديو والصور، مما يزيد من أداء النموذج على مقاطع الفيديو.  

لدى [LLaVA-NeXT](llava_next) أداء قوي ومفاجئ في فهم محتوى الفيديو بطريقة الصفر باستخدام تقنية AnyRes التي يستخدمها. وتمثل تقنية AnyRes بشكل طبيعي صورة عالية الدقة في صور متعددة. وهذه التقنية قابلة للتعميم بطبيعتها لتمثيل مقاطع الفيديو لأن مقاطع الفيديو يمكن اعتبارها مجموعة من الإطارات (مماثلة لمجموعة من الصور في LLaVa-NeXT). ويستخدم الإصدار الحالي من LLaVA-NeXT تقنية AnyRes ويتم تدريبه باستخدام الضبط الدقيق الخاضع للإشراف (SFT) أعلى LLaVA-Next على بيانات الفيديو لتحقيق فهم أفضل للفيديو. ويعد النموذج حاليًا الأفضل ضمن النماذج مفتوحة المصدر على [VideoMME bench](https://arxiv.org/abs/2405.21075).  

مقدمة المدونة هي كما يلي:  

في 30 يناير 2024، أصدرنا LLaVA-NeXT، وهو نموذج متعدد الوسائط كبير مفتوح المصدر (LMM) تم تدريبه حصريًا على بيانات الصور النصية. وبفضل تقنية AnyRes المقترحة، فإنه يعزز القدرات في الاستدلال، وOCR، ومعرفة العالم، مما يظهر أداءً ملحوظًا عبر طيف من مهام الفهم متعددة الوسائط القائمة على الصور، بل ويتفوق على Gemini-Pro في العديد من معايير الصور، مثل MMMU وMathVista.  

**في استكشاف اليوم، نتعمق في أداء LLaVA-NeXT في مجال مهام فهم الفيديو. ونكشف أن LLaVA-NeXT لديه أداء قوي ومفاجئ في فهم محتوى الفيديو. ويتضمن الإصدار الحالي من LLaVA-NeXT لتحسينات الفيديو ما يلي:  

- قدرات تمثيل الفيديو بدون تصوير باستخدام AnyRes: تمثل تقنية AnyRes بشكل طبيعي صورة عالية الدقة في صور متعددة يمكن لشبكة VIT مسبقة التدريب معالجتها، وتشكلها في تسلسل متجاور. وهذه التقنية قابلة للتعميم بطبيعتها لتمثيل مقاطع الفيديو (التي تتكون من إطارات متعددة)، مما يسمح لنموذج LLaVA-Next الذي تم تدريبه على الصور فقط بأداء جيد جدًا في مهام الفيديو. وتجدر الإشارة إلى أن هذه هي المرة الأولى التي تظهر فيها نماذج LMM قدرة قوية على نقل الوضع بدون تصوير.  

- الاستدلال باستخدام تعميم الطول يحسن مقاطع الفيديو الأطول. تمكن تقنية التدرج الخطي من تعميم الطول، مما يسمح لـ LLaVA-NeXT بمعالجة الفيديوهات الطويلة بشكل فعال بما يتجاوز حد "max_token_length" لشبكة اللغة.  

- قدرة قوية على فهم الفيديو. (1) يمنح LLaVa-Next-Image، الذي يجمع بين التقنيتين المذكورتين أعلاه، أداءً أفضل بدون تصوير من النماذج متعددة الوسائط الكبيرة مفتوحة المصدر التي تم ضبطها على مقاطع الفيديو. (2) يحقق LLaVa-Next-Video، الذي يقوم بضبط دقيق خاضع للإشراف إضافي (SFT) لـ LLaVa-Next-Image على بيانات الفيديو، فهمًا أفضل للفيديو مقارنة بـ LLaVa-Next-Image. (3) يظهر LLaVa-Next-Video-DPO، الذي يحاذي استجابة النموذج مع ملاحظات الذكاء الاصطناعي باستخدام التحسين المباشر للأفضليات (DPO)، زيادة كبيرة في الأداء.  

- النشر والاستدلال الفعالان باستخدام SGLang. يسمح بخمسة أضعاف سرعة الاستدلال في مهام الفيديو، مما يسمح بخدمة أكثر قابلية للتطوير مثل إعادة كتابة نص مليون فيديو. راجع التعليمات في مستودعنا.**  

تمت المساهمة بهذا النموذج بواسطة [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).  

يمكن العثور على الكود الأصلي [هنا](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/inference).  

## نصائح الاستخدام  

- ننصح المستخدمين باستخدام `padding_side="left"` عند حساب التوليد المجمع حيث يؤدي إلى نتائج أكثر دقة. ما عليك سوى التأكد من استدعاء `processor.tokenizer.padding_side = "left"` قبل التوليد.  

- لاحظ أن كل نقطة تفتيش تم تدريبها بتنسيق موجه محدد، اعتمادًا على نموذج اللغة الضخم (LLM) الذي تم استخدامه. يمكنك استخدام `apply_chat_template` من المحلل اللغوي لتنسيق موجهاتك بشكل صحيح. فيما يلي مثال يوضح كيفية القيام بذلك.  

سنستخدم [LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) وسجل محادثة يتضمن مقاطع فيديو وصور. يجب أن يكون كل حقل محتوى قائمة من القواميس، كما يلي:  

```python
from transformers import LlavaNextVideoProcessor

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."},
            ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s shown in this image?"},
            {"type": "image"},
            ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This image shows a red stop sign."},]
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video"},
            ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Note that the template simply formats your prompt, you still have to tokenize it and obtain pixel values for your visuals
print(text_prompt)
```

## مثال الاستخدام  

### وضع الوسائط الفردية  

يمكن للنموذج قبول كل من الصور ومقاطع الفيديو كإدخال. فيما يلي مثال على التعليمات البرمجية للاستدلال في الدقة النصفية (`torch.float16`):  

```python
import av
import torch
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

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
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16, device_map="auto")
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

# Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
video = read_video_pyav(container, indices)

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video"},
            ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=prompt, videos=video, return_tensors="pt")

out = model.generate(**inputs, max_new_tokens=60)
processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
```  

### وضع الوسائط المختلطة  

يمكن للنموذج أيضًا التوليد من إدخالات الصور ومقاطع الفيديو المتشابكة. ومع ذلك، لاحظ أنه لم يتم تدريبه في إعداد متشابك للصور ومقاطع الفيديو، والذي قد يؤثر على الأداء. فيما يلي مثال على الاستخدام لوسائط الإدخال المختلطة، أضف السطور التالية إلى مقطع التعليمات البرمجية أعلاه:  

```python
from PIL import Image
import requests

# Generate from image and video mixed inputs
# Load and image and write a new prompt
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "How many cats are there in the image?"},
            {"type": "image"},
            ],
    },
    {

        "role": "assistant",
        "content": [{"type": "text", "text": "There are two cats"}],
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Why is this video funny?"},
            {"type": "video"},
            ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=prompt, images=image, videos=clip, padding=True, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=50)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


```  

## تحسين النموذج  

### التكميم باستخدام Bitsandbytes لكفاءة الذاكرة  

يمكن تحميل النموذج في عدد أقل من البتات، مما يقلل بشكل كبير من عبء الذاكرة مع الحفاظ على أداء النموذج الأصلي. يسمح ذلك بالنشر الفعال في الحالات المقيدة بالموارد.  

أولاً، تأكد من تثبيت bitsandbytes عن طريق تشغيل `pip install bitsandbytes` وامتلاك حق الوصول إلى جهاز GPU متوافق مع CUDA. قم بتحميل النموذج الكمي ببساطة عن طريق إضافة [`BitsAndBytesConfig`](../main_classes/quantization#transformers.BitsAndBytesConfig) كما هو موضح أدناه:  

```python
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"،
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf"، quantization_config=quantization_config، device_map="auto")
```  

### Flash-Attention 2 لتسريع التوليد  

بالإضافة إلى ذلك، يمكننا تسريع استدلال النموذج بشكل كبير باستخدام [Flash Attention](../perf_train_gpu_one.md#flash-attention-2)، وهو تنفيذ أسرع لآلية الانتباه المستخدمة داخل النموذج.  

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2:  

```bash
pip install -U flash-attn --no-build-isolation
```  

يجب أن يكون لديك أيضًا أجهزة متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في وثائق مستودع [flash attention](https://github.com/Dao-AILab/flash-attention) الرسمي. يمكن استخدام FlashAttention-2 فقط عندما يتم تحميل النموذج في `torch.float16` أو `torch.bfloat16`.  

لتحميل وتشغيل نموذج باستخدام Flash Attention-2، أضف ببساطة `attn_implementation="flash_attention_2"` عند تحميل النموذج كما يلي:  

```python
from transformers import LlavaNextVideoForConditionalGeneration

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf"،
    torch_dtype=torch.float16،
    attn_implementation="flash_attention_2"،
).to(0)
```  

## LlavaNextVideoConfig  

[[autodoc]] LlavaNextVideoConfig  

## LlavaNextVideoProcessor  

[[autodoc]] LlavaNextVideoProcessor  

## LlavaNextVideoImageProcessor  

[[autodoc]] LlavaNextVideoImageProcessor  

## LlavaNextVideoForConditionalGeneration  

[[autodoc]] LlavaNextVideoForConditionalGeneration  

- forward