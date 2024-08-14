# الاستنتاج باستخدام وحدة معالجة الرسومات (GPU)

تعد وحدات معالجة الرسومات (GPU) الخيار القياسي لأجهزة التعلم الآلي، على عكس وحدات المعالجة المركزية (CPU)، لأنها مُحَسَّنة لعرض نطاق الذاكرة والتوازي. ولمواكبة الأحجام الأكبر للنماذج الحديثة أو لتشغيل هذه النماذج الكبيرة على الأجهزة الموجودة والقديمة، هناك العديد من التحسينات التي يمكنك استخدامها لتسريع الاستنتاج باستخدام وحدة معالجة الرسومات. في هذا الدليل، ستتعلم كيفية استخدام FlashAttention-2 (آلية اهتمام أكثر كفاءة في استخدام الذاكرة)، وBetterTransformer (مسار سريع للتنفيذ الأصلي في PyTorch)، وbitsandbytes لضبط نموذجك إلى دقة أقل. وأخيرًا، تعلم كيفية استخدام 🤗 Optimum لتسريع الاستدلال باستخدام ONNX Runtime على وحدات معالجة الرسومات Nvidia وAMD.

<Tip>

تنطبق معظم التحسينات الموضحة هنا أيضًا على إعدادات وحدات معالجة الرسومات المتعددة!

</Tip>

## FlashAttention-2

<Tip>

FlashAttention-2 تجريبي وقد يتغير بشكل كبير في الإصدارات المستقبلية.

</Tip>

[FlashAttention-2](https://huggingface.co/papers/2205.14135) هو تنفيذ أسرع وأكثر كفاءة لآلية الاهتمام القياسية التي يمكن أن تسرع الاستدلال بشكل كبير من خلال:

1. موازاة حساب الاهتمام بشكل إضافي عبر طول التسلسل
2. تقسيم العمل بين خيوط وحدة معالجة الرسومات لتقليل التواصل وقراءات/كتابات الذاكرة المشتركة بينها
تدعم FlashAttention-2 حاليًا البنى المعمارية التالية:
* [Bark](https://huggingface.co/docs/transformers/model_doc/bark#transformers.BarkModel)
* [Bart](https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartModel)
* [Chameleon](https://huggingface.co/docs/transformers/model_doc/chameleon#transformers.Chameleon)
* [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel)
* [Cohere](https://huggingface.co/docs/transformers/model_doc/cohere#transformers.CohereModel)
* [Dbrx](https://huggingface.co/docs/transformers/model_doc/dbrx#transformers.DbrxModel)
* [DistilBert](https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertModel)
* [Gemma](https://huggingface.co/docs/transformers/model_doc/gemma#transformers.GemmaModel)
* [Gemma2](https://huggingface.co/docs/transformers/model_doc/gemma2#transformers.Gemma2Model)
* [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
* [GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode#transformers.GPTBigCodeModel)
* [GPTNeo](https://huggingface.co/docs/transformers/model_doc/gpt_neo#transformers.GPTNeoModel)
* [GPTNeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox#transformers.GPTNeoXModel)
* [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj#transformers.GPTJModel)
* [Idefics2](https://huggingface.co/docs/transformers/model_doc/idefics2#transformers.Idefics2Model)
* [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon#transformers.FalconModel)
* [JetMoe](https://huggingface.co/docs/transformers/model_doc/jetmoe#transformers.JetMoeModel)
* [Jamba](https://huggingface.co/docs/transformers/model_doc/jamba#transformers.JambaModel)
* [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel)
* [Llava](https://huggingface.co/docs/transformers/model_doc/llava)
* [Llava-NeXT](https://huggingface.co/docs/transformers/model_doc/llava_next)
* [Llava-NeXT-Video](https://huggingface.co/docs/transformers/model_doc/llava_next_video)
* [VipLlava](https://huggingface.co/docs/transformers/model_doc/vipllava)
* [VideoLlava](https://huggingface.co/docs/transformers/model_doc/video_llava)
* [M2M100](https://huggingface.co/docs/transformers/model_doc/m2m_100)
* [MBart](https://huggingface.co/docs/transformers/model_doc/mbart#transformers.MBartModel)
* [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral#transformers.MistralModel)
* [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralModel)
* [Musicgen](https://huggingface.co/docs/transformers/model_doc/musicgen#transformers.MusicgenModel)
* [MusicGen Melody](https://huggingface.co/docs/transformers/model_doc/musicgen_melody#transformers.MusicgenMelodyModel)
* [NLLB](https://huggingface.co/docs/transformers/model_doc/nllb)
* [OLMo](https://huggingface.co/docs/transformers/model_doc/olmo#transformers.OlmoModel)
* [OPT](https://huggingface.co/docs/transformers/model_doc/opt#transformers.OPTModel)
* [Phi](https://huggingface.co/docs/transformers/model_doc/phi#transformers.PhiModel)
* [Phi3](https://huggingface.co/docs/transformers/model_doc/phi3#transformers.Phi3Model)
* [SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip)
* [StableLm](https://huggingface.co/docs/transformers/model_doc/stablelm#transformers.StableLmModel)
* [Starcoder2](https://huggingface.co/docs/transformers/model_doc/starcoder2#transformers.Starcoder2Model)
* [Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2Model)
* [Qwen2MoE](https://huggingface.co/docs/transformers/model_doc/qwen2_moe#transformers.Qwen2MoeModel)
* [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel)
* [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Model)
* [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel)
* [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Model)
* [Hubert](https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertModel)
* [data2vec_audio](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec#transformers.Data2VecAudioModel)
* [Sew](https://huggingface.co/docs/transformers/main/en/model_doc/sew#transformers.SEWModel)
* [UniSpeech](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech#transformers.UniSpeechModel)
* [unispeech_sat](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel)
يمكنك طلب إضافة دعم FlashAttention-2 لنموذج آخر عن طريق فتح مشكلة أو طلب سحب على GitHub.

قبل البدء، تأكد من تثبيت FlashAttention-2.

<hfoptions id="install">
<hfoption id="NVIDIA">

```bash
pip install flash-attn --no-build-isolation
```

نوصي بشدة بالرجوع إلى [تعليمات التثبيت](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features) المفصلة لمعرفة المزيد حول الأجهزة المدعومة وأنواع البيانات!

</hfoption>
<hfoption id="AMD">

يتم دعم FlashAttention-2 أيضًا على وحدات معالجة الرسومات AMD، ويقتصر الدعم الحالي على **Instinct MI210**، و**Instinct MI250**، و**Instinct MI300**. نوصي بشدة باستخدام هذا [Dockerfile](https://github.com/huggingface/optimum-amd/tree/main/docker/transformers-pytorch-amd-gpu-flash/Dockerfile) لاستخدام FlashAttention-2 على وحدات معالجة الرسومات AMD.

</hfoption>
</hfoptions>

لتمكين FlashAttention-2، مرر وسيط `attn_implementation="flash_attention_2"` إلى [`~AutoModelForCausalLM.from_pretrained`]:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

<Tip>

يمكن استخدام FlashAttention-2 فقط عندما يكون نوع نموذج "fp16" أو "bf16". تأكد من تحويل نموذجك إلى النوع المناسب وتحميله على جهاز مدعوم قبل استخدام FlashAttention-2.

<br>

يمكنك أيضًا تعيين `use_flash_attention_2=True` لتمكين FlashAttention-2 ولكنه مهمل لصالح `attn_implementation="flash_attention_2"`.

</Tip>

يمكن الجمع بين FlashAttention-2 وتقنيات التحسين الأخرى مثل الضبط للحصول على مزيد من تسريع الاستدلال. على سبيل المثال، يمكنك الجمع بين FlashAttention-2 والضبط 8-بت أو 4-بت:

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# تحميل في 8 بت
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    attn_implementation="flash_attention_2",
)

# تحميل في 4 بت
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    attn_ excentricité="flash_attention_2"،
)
```

### تسريع الأداء المتوقع

يمكنك الاستفادة من تسريع الأداء الكبير للاستدلال، خاصة بالنسبة للمدخلات ذات التسلسلات الطويلة. ومع ذلك، نظرًا لأن FlashAttention-2 لا يدعم حساب درجات الاهتمام مع رموز التعبئة، يجب عليك يدويًا تعبئة/إلغاء تعبئة درجات الاهتمام للاستدلال المجمع عندما يحتوي التسلسل على رموز تعبئة. يؤدي هذا إلى تباطؤ كبير في الأجيال المجمعة مع رموز التعبئة.

لتجاوز ذلك، يجب عليك استخدام FlashAttention-2 بدون رموز التعبئة في التسلسل أثناء التدريب (عن طريق تعبئة مجموعة بيانات أو [دمج التسلسلات](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py#L516) حتى الوصول إلى طول التسلسل الأقصى).

بالنسبة لمرور الإرسال الأمامي الفردي على [tiiuae/falcon-7b](https://hf.co/tiiuae/falcon-7b) بطول تسلسل 4096 وأحجام دفعات مختلفة بدون رموز التعبئة، يكون تسريع الأداء المتوقع هو:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/falcon-7b-inference-large-seqlen.png">
</div>

بالنسبة لمرور الإرسال الأمامي الفردي على [meta-llama/Llama-7b-hf](https://hf.co/meta-llama/Llama-7b-hf) بطول تسلسل 4096 وأحجام دفعات مختلفة بدون رموز التعبئة، يكون تسريع الأداء المتوقع هو:

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-7b-inference-large-seqlen.png">
</div>
<div style="text-align-center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-7b-inference-large-seqlen.png">
</div>

بالنسبة للتسلسلات التي تحتوي على رموز التعبئة (توليد باستخدام رموز التعبئة)، يجب عليك إلغاء تعبئة/تعبئة تسلسلات الإدخال لحساب درجات الاهتمام بشكل صحيح. باستخدام طول تسلسل صغير نسبيًا، يؤدي مرور الإرسال الأمامي الفردي إلى زيادة العبء مما يؤدي إلى تسريع الأداء البسيط (في المثال أدناه، يتم ملء 30% من الإدخال برموز التعبئة):

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-small-seqlen-padding.png">
</div>

ولكن بالنسبة لأطوال التسلسل الأكبر، يمكنك توقع فوائد تسريع الأداء حتى أكثر من ذلك:

<Tip>

FlashAttention أكثر كفاءة في استخدام الذاكرة، مما يعني أنه يمكنك التدريب على أطوال تسلسل أكبر دون مواجهة مشكلات نفاد الذاكرة. يمكنك تقليل استخدام الذاكرة بنسبة تصل إلى 20 مرة لأطوال التسلسل الأكبر. الق نظرة على مستودع [flash-attention](https://github.com/Dao-AILab/flash-attention) لمزيد من التفاصيل.

</Tip>

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-large-seqlen-padding.png">
</div>

## PyTorch scaled dot product attention

يمكن لـ PyTorch's [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA) أيضًا استدعاء FlashAttention ونواة الاهتمام الكفؤة في استخدام الذاكرة تحت الغطاء. يجري حاليًا إضافة دعم SDPA بشكل أصلي في Transformers ويتم استخدامه بشكل افتراضي لـ `torch>=2.1.1` عندما يكون التنفيذ متاحًا. يمكنك أيضًا تعيين `attn_implementation="sdpa"` في `from_pretrained()` لطلب استخدام SDPA بشكل صريح.
في الوقت الحالي، يدعم Transformers الاستدلال والتدريب SDPA للبنى المعمارية التالية:
* [Audio Spectrogram Transformer](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer#transformers.ASTModel)
* [Bart](https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartModel)
* [Bert](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel)
* [Chameleon](https://huggingface.co/docs/transformers/model_doc/chameleon#transformers.Chameleon)
* [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel)
* [Cohere](https://huggingface.co/docs/transformers/model_doc/cohere#transformers.CohereModel)
* [Dbrx](https://huggingface.co/docs/transformers/model_doc/dbrx#transformers.DbrxModel)
* [DeiT](https://huggingface.co/docs/transformers/model_doc/deit#transformers.DeiTModel)
* [Dpr](https://huggingface.co/docs/transformers/model_doc/dpr#transformers.DprReader)
* [Falcon](https://huggingface.co/docs/transformers/model_doc/falcon#transformers.FalconModel)
* [Gemma](https://huggingface.co/docs/transformers/model_doc/gemma#transformers.GemmaModel)
* [Gemma2](https://huggingface.co/docs/transformers/model_doc/gemma2#transformers.Gemma2Model)
* [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)
* [GPTBigCode](https://huggingface.co/docs/transformers/model_doc/gpt_bigcode#transformers.GPTBigCodeModel)
* [GPTNeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox#transformers.GPTNeoXModel)
* [JetMoe](https://huggingface.co/docs/transformers/model_doc/jetmoe#transformers.JetMoeModel)
* [Jamba](https://huggingface.co/docs/transformers/model_doc/jamba#transformers.JambaModel)
* [Llama](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaModel)
* [OLMo](https://huggingface.co/docs/transformers/model_doc/olmo#transformers.OlmoModel)
* [PaliGemma](https://huggingface.co/docs/transformers/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration)
* [Phi](https://huggingface.co/docs/transformers/model_doc/phi#transformers.PhiModel)
* [Idefics](https://huggingface.co/docs/transformers/model_doc/idefics#transformers.IdeficsModel)
* [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperModel)
* [Mistral](https://huggingface.co/docs/transformers/model_doc/mistral#transformers.MistralModel)
* [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralModel)
* [StableLm](https://huggingface.co/docs/transformers/model_doc/stablelm#transformers.StableLmModel)
* [Starcoder2](https://huggingface.co/docs/transformers/model_doc/starcoder2#transformers.Starcoder2Model)
* [Qwen2](https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2Model)
* [Qwen2MoE](https://huggingface.co/docs/transformers/model_doc/qwen2_moe#transformers.Qwen2MoeModel)
* [Musicgen](https://huggingface.co/docs/transformers/model_doc/musicgen#transformers.MusicgenModel)
* [MusicGen Melody](https://huggingface.co/docs/transformers/model_doc/musicgen_melody#transformers.MusicgenMelodyModel)
* [ViT](https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTModel)
* [ViTHybrid](https://huggingface.co/docs/transformers/model_doc/vit_hybrid#transformers.ViTHybridModel)
* [ViTMAE](https://huggingface.co/docs/transformers/model_doc/vit_mae#transformers.ViTMAEModel)
* [ViTMSN](https://huggingface.co/docs/transformers/model_doc/vit_msn#transformers.ViTMSNModel)
* [VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae#transformers.VideoMAEModell)
* [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2Model)
* [Hubert](https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertModel)
* [data2vec_audio](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec#transformers.Data2VecAudioModel)
* [SigLIP](https://huggingface.co/docs/transformers/model_doc/siglip)
* [Sew](https://huggingface.co/docs/transformers/main/en/model_doc/sew#transformers.SEWModel)
* [UniSpeech](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech#transformers.UniSpeechModel)
* [unispeech_sat](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/unispeech-sat#transformers.UniSpeechSatModel)
* [YOLOS](https://huggingface.co/docs/transformers/model_doc/yolos#transformers.YolosModel)



<Tip>

يمكن استخدام FlashAttention فقط للنماذج ذات النوع "fp16" أو "bf16" torch، لذا تأكد من تحويل نموذجك إلى النوع المناسب أولاً. يمكن لخلفية الاهتمام الفعالة للذاكرة التعامل مع نماذج "fp32".

</Tip>

<Tip>

لا تدعم SDPA مجموعات معينة من معلمات الاهتمام، مثل "head_mask" و "output_attentions=True".
في هذه الحالة، يجب أن تشاهد رسالة تحذير وسنقوم بالرجوع إلى التنفيذ (الأبطأ).

</Tip>

بشكل افتراضي، تقوم SDPA بتحديد نواة الأداء الأكثر كفاءة المتاحة، ولكن يمكنك التحقق مما إذا كانت الخلفية متاحة في إعداد معين (الأجهزة، حجم المشكلة) باستخدام ["torch.backends.cuda.sdp_kernel"](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel) كمدير سياق:

```diff
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16).to("cuda")

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

+ with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

إذا رأيت خطأ مع تتبع المكدس أدناه، فحاول استخدام الإصدار الليلي من PyTorch الذي قد يكون له تغطية أوسع لـ FlashAttention:

```bash
RuntimeError: No available kernel. Aborting execution.

# install PyTorch nightly
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

## BetterTransformer

<Tip warning={true}>

يتم نقل بعض ميزات BetterTransformer إلى أعلى مستوى في المحولات مع الدعم الافتراضي لـ native `torch.nn.scaled_dot_product_attention`. لا يزال لدى BetterTransformer تغطية أوسع من تكامل SDPA في المحولات، ولكن يمكنك توقع المزيد والمزيد من الهندسات المعمارية التي تدعم SDPA بشكل أصلي في المحولات.

</Tip>

<Tip>

تحقق من معاييرنا مع BetterTransformer وscaled dot product attention في [التسريع والوفورات في الذاكرة خارج الصندوق لنماذج فك تشفير 🤗 مع PyTorch 2.0](https://pytorch.org/blog/out-of-the-box-acceleration/) وتعرف على المزيد حول تنفيذ fastpath في [BetterTransformer](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2) منشور المدونة.

</Tip>

BetterTransformer يسرع الاستدلال باستخدام تنفيذه المتخصص في PyTorch لمهام المحول. هناك تحسينان في تنفيذ fastpath:

1. الانصهار، الذي يجمع بين عدة عمليات متتالية في "kernel" واحد لتقليل عدد خطوات الحساب
2. تخطي التفرقة الطبيعية لرموز التعبئة لتجنب الحساب غير الضروري مع المصفوفات المضمنة

BetterTransformer يحول أيضًا جميع عمليات الاهتمام لاستخدام [scaled dot product attention (SDPA)](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention)، الأكثر كفاءة في الذاكرة، ويستدعي نوى محسنة مثل [FlashAttention](https://huggingface.co/papers/2205.14135) تحت غطاء المحرك.

قبل أن تبدأ، تأكد من أن لديك 🤗 Optimum [مثبت](https://huggingface.co/docs/optimum/installation).

بعد ذلك، يمكنك تمكين BetterTransformer باستخدام طريقة [`PreTrainedModel.to_bettertransformer`]():

```python
model = model.to_bettertransformer()
```

يمكنك إرجاع نموذج المحولات الأصلي باستخدام طريقة [`~PreTrainedModel.reverse_bettertransformer`](): يجب عليك استخدام هذا قبل حفظ نموذجك لاستخدام نمذجة المحولات الأساسية:

```py
model = model.reverse_bettertransformer()
model.save_pretrained ("model_saved")
```

## bitsandbytes

bitsandbytes هي مكتبة تكميم تتضمن دعمًا لـ 4 بت و8 بت. يقلل التكميم حجم نموذجك مقارنة بإصدار الدقة الكاملة الأصلي، مما يجعله أسهل في وضع نماذج كبيرة على وحدات معالجة الرسومات (GPUs) ذات الذاكرة المحدودة.

تأكد من أن لديك bitsandbytes و 🤗 Accelerate مثبت:

```bash
# هذه الإصدارات تدعم 8 بت و4 بت
pip install bitsandbytes>=0.39.0 accelerate>=0.20.0

# تثبيت المحولات
pip install transformers
```

### 4 بت

لتحميل نموذج في 4 بت للاستدلال، استخدم معلمة "load_in_4bit". وسيط "device_map" اختياري، ولكن يُنصح بتعيينه على "auto" للسماح لـ 🤗 Accelerate بتخصيص النموذج تلقائيًا وبكفاءة بالنظر إلى الموارد المتاحة في البيئة.

```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name، device_map="auto"، load_in_4bit=True)
```

لتحميل نموذج في 4 بت للاستدلال باستخدام وحدات معالجة الرسومات (GPUs) متعددة، يمكنك التحكم في مقدار ذاكرة GPU التي تريد تخصيصها لكل GPU. على سبيل المثال، لتوزيع 600 ميجابايت من الذاكرة إلى وحدة معالجة الرسومات الأولى و1 جيجابايت من الذاكرة إلى وحدة معالجة الرسومات الثانية:

```py
max_memory_mapping = {0: "600MB"، 1: "1GB"}
model_name = "bigscience/bloom-3b"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name، device_map="auto"، load_in_4bit=True، max_memory=max_memory_mapping
)
```

### 8 بت

<Tip>

إذا كنت فضوليًا ومهتمًا بمعرفة المزيد عن المفاهيم الأساسية وراء التكميم 8 بت، فاقرأ [Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration) منشور المدونة.

</Tip>

لتحميل نموذج في 8 بت للاستدلال، استخدم معلمة "load_in_8bit". وسيط "device_map" اختياري، ولكن يُنصح بتعيينه على "auto" للسماح لـ 🤗 Accelerate بتخصيص النموذج تلقائيًا وبكفاءة بالنظر إلى الموارد المتاحة في البيئة:

```py
from transformers import AutoModelForCausalLM، BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name، quantization_config=BitsAndBytesConfig (load_in_8bit=True))
```

إذا كنت تقوم بتحميل نموذج في 8 بت لتوليد النص، فيجب عليك استخدام طريقة [`~transformers.GenerationMixin.generate`] بدلاً من دالة ["Pipeline"] التي لا يتم تحسينها لنماذج 8 بت وستكون أبطأ. لا تدعم بعض استراتيجيات أخذ العينات، مثل أخذ العينات النووية، بواسطة ["Pipeline"] لنماذج 8 بت. يجب عليك أيضًا وضع جميع الإدخالات على نفس الجهاز مثل النموذج:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```
لتحميل نموذج في 4 بت للاستنتاج باستخدام عدة وحدات معالجة رسومية (GPUs)، يمكنك التحكم في مقدار ذاكرة الوصول العشوائي (RAM) التي تريد تخصيصها لكل وحدة معالجة رسومية (GPU). على سبيل المثال، لتوزيع 1 جيجابايت من الذاكرة إلى وحدة المعالجة المركزية الأولى و2 جيجابايت من الذاكرة إلى وحدة المعالجة المركزية الثانية:

```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping
)
```

<Tip>

لا تتردد في تجربة تشغيل نموذج T5 الذي يحتوي على 11 مليار معلمة أو نموذج BLOOM الذي يحتوي على 3 مليارات معلمة للاستنتاج على وحدات معالجة الرسومات (GPUs) من المستوى المجاني في Google Colab!

</Tip>

## 🤗 Optimum

<Tip>

لمعرفة المزيد من التفاصيل حول استخدام ORT مع 🤗 Optimum، راجع دليلي [تسريع الاستنتاج على وحدات معالجة الرسومات (GPUs) من NVIDIA](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu#accelerated-inference-on-nvidia-gpus) و[تسريع الاستنتاج على وحدات معالجة الرسومات (GPUs) من AMD](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/amdgpu#accelerated-inference-on-amd-gpus). يقدم هذا القسم فقط مثالًا موجزًا وبسيطًا.

</Tip>

ONNX Runtime (ORT) هو مسرع نموذجي يدعم الاستنتاج المعجل على وحدات معالجة الرسومات (GPUs) من Nvidia، ووحدات معالجة الرسومات (GPUs) من AMD التي تستخدم مجموعة ROCm. يستخدم ORT تقنيات التحسين مثل دمج العمليات الشائعة في عقدة واحدة وطوي الثوابت لخفض عدد الحسابات التي يتم إجراؤها وتسريع الاستنتاج. كما يقوم ORT بوضع العمليات الأكثر كثافة حسابية على وحدة معالجة الرسومات (GPU) والباقي على وحدة المعالجة المركزية (CPU) لتوزيع عبء العمل بين الجهازين بشكل ذكي.

تدعم مكتبة 🤗 Optimum استخدام ONNX Runtime، والتي يمكن استخدامها في مكتبة 🤗 Transformers. ستحتاج إلى استخدام [`~optimum.onnxruntime.ORTModel`] للمهمة التي تحاول حلها، وتحديد معلمة `provider` التي يمكن تعيينها إما إلى [`CUDAExecutionProvider`] أو [`ROCMExecutionProvider`] أو [`TensorrtExecutionProvider`]. إذا كنت تريد تحميل نموذج لم يتم تصديره بعد إلى ONNX، فيمكنك تعيين `export=True` لتحويل نموذجك أثناء التنقل إلى تنسيق ONNX:

```py
from optimum.onnxruntime import ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
  "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  export=True,
  provider="CUDAExecutionProvider",
)
```

الآن يمكنك استخدام النموذج للاستنتاج:

```py
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

pipeline = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer, device="cuda:0")
result = pipeline("Both the music and visual were astounding, not to mention the actors performance.")
```

## الجمع بين التحسينات

غالبًا ما يكون من الممكن الجمع بين العديد من تقنيات التحسين المذكورة أعلاه للحصول على أفضل أداء استنتاج ممكن لنموذجك. على سبيل المثال، يمكنك تحميل نموذج في 4 بت، ثم تمكين BetterTransformer مع FlashAttention:

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# تحميل النموذج في 4 بت
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=quantization_config)

# تمكين BetterTransformer
model = model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# تمكين FlashAttention
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```