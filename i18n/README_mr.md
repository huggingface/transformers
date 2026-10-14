<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<!---
English-Marathi translation guide for Hugging Face documentation:
- Add space around English words and numbers when they appear between Marathi characters.
- Maintain technical terms in standard format with English equivalents where helpful.

Dictionary / शब्दकोश:
- inference: अनुमान (Inference)
- training: प्रशिक्षण (Training)
- pretrained model: पूर्व-प्रशिक्षित मॉडेल (Pretrained model)
- fine-tuning: फाइन-ट्यूनिंग (Fine-tuning)
- pipeline: पाइपलाइन (Pipeline)
- checkpoint: चेकपॉईंट (Checkpoint)
- tokenizer: टोकनायझर (Tokenizer)
- tokenization: टोकनायझेशन (Tokenization)
- natural language processing (NLP): नैसर्गिक भाषा प्रक्रिया (NLP)
- computer vision: संगणकीय दृष्टी (Computer vision)
- multimodal: मल्टिमोडल (Multimodal)
--->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://huggingface.com/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="../README.md">English</a> |
        <a href="README_zh-hans.md">简体中文</a> |
        <a href="README_zh-hant.md">繁體中文</a> |
        <a href="README_ko.md">한국어</a> |
        <a href="README_es.md">Español</a> |
        <a href="README_ja.md">日本語</a> |
        <a href="README_hd.md">हिन्दी</a> |
        <b>मराठी</b> |
        <a href="README_ru.md">Русский</a> |
        <a href="README_pt-br.md">Português</a> |
        <a href="README_te.md">తెలుగు</a> |
        <a href="README_fr.md">Français</a> |
        <a href="README_de.md">Deutsch</a> |
        <a href="README_it.md">Italiano</a> |
        <a href="README_vi.md">Tiếng Việt</a> |
        <a href="README_ar.md">العربية</a> |
        <a href="README_ur.md">اردو</a> |
        <a href="README_bn.md">বাংলা</a> |
        <a href="README_fa.md">فारसी</a> |
        <a href="README_ro.md">Română</a> |
        <a href="README_tr.md">Türkçe</a>
    </p>
</h4>

<h3 align="center">
    <p>अनुमान (Inference) आणि प्रशिक्षणासाठी (Training) अत्याधुनिक पूर्व-प्रशिक्षित मॉडेल्स</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers हे मजकूर (text), संगणकीय दृष्टी (computer vision), ध्वनी (audio), व्हिडिओ (video), आणि मल्टिमोडल मॉडेल्सच्या अत्याधुनिक मशीन लर्निंगसाठी मॉडेल-डेफिनिशन फ्रेमवर्क म्हणून कार्य करते, जे अनुमान (inference) आणि प्रशिक्षण (training) या दोन्हींसाठी उपयुक्त आहे.

हे मॉडेलची व्याख्या (model definition) एकाच ठिकाणी केंद्रित करते, ज्यामुळे संपूर्ण इकोसिस्टममध्ये ही व्याख्या मान्य केली जाते. `transformers` हे सर्व फ्रेमवर्क्समधील मुख्य केंद्र (pivot) आहे: जर एखाद्या मॉडेलच्या व्याख्येस समर्थन मिळाले, तर ते बहुसंख्य प्रशिक्षण फ्रेमवर्क्स (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), इन्फरन्स इंजिन्स (vLLM, SGLang, TGI, ...), आणि संबंधित मॉडेलिंग लायब्ररी (llama.cpp, mlx, ...) यांच्याशी सुसंगत ठरते जे `transformers` मधील मॉडेल व्याख्या वापरतात.

आम्ही नवीन अत्याधुनिक मॉडेल्सना समर्थन देण्याचे आणि त्यांची मॉडेल व्याख्या सोपी, सानुकूल करण्यायोग्य (customizable) आणि कार्यक्षम ठेवून त्यांचा वापर लोकशाहीकृत (democratize) करण्याचे वचन देतो.

[Hugging Face Hub](https://huggingface.co/models) वर १० लाखांपेक्षा अधिक (1M+) Transformers [मॉडेल चेकपॉईंट्स](https://huggingface.co/models?library=transformers&sort=trending) उपलब्ध आहेत, जे तुम्ही वापरू शकता.

एखादे मॉडेल शोधण्यासाठी आजच [Hub](https://huggingface.co/) एक्सप्लोर करा आणि त्वरित सुरुवात करण्यासाठी Transformers चा वापर करा.

## इन्स्टॉलेशन (Installation)

Transformers हे Python 3.10+, आणि [PyTorch](https://pytorch.org/get-started/locally/) 2.5+ सह कार्य करते.

[venv](https://docs.python.org/3/library/venv.html) किंवा [uv](https://docs.astral.sh/uv/) (एक वेगवान Rust-आधारित Python पॅकेज आणि प्रोजेक्ट मॅनेजर) वापरून व्हर्च्युअल एन्व्हायर्नमेंट (virtual environment) तयार आणि सक्रिय करा.

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

तुमच्या व्हर्च्युअल एन्व्हायर्नमेंटमध्ये Transformers इन्स्टॉल करा.

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

जर तुम्हाला लायब्ररीमधील नवीनतम बदल हवे असतील किंवा तुम्ही योगदान देण्यास इच्छुक असाल तर सोअर्स (source) वरून Transformers इन्स्टॉल करा. तथापि, *लेटेस्ट* आवृत्ती कदाचित पूर्णपणे स्थिर (stable) नसू शकते. जर तुम्हाला कोणतीही अडचण आली तर मोकळेपणाने [issue](https://github.com/huggingface/transformers/issues) ओपन करा.

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
```

## क्विकस्टार्ट (Quickstart)

[Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API सह Transformers वापरण्यास त्वरित सुरुवात करा. `Pipeline` हा एक उच्च-स्तरीय इन्फरन्स क्लास आहे जो मजकूर, ध्वनी, व्हिजन आणि मल्टिमोडल कामांना समर्थन देतो. हे इनपुटचे प्री-प्रोसेसिंग हाताळते आणि योग्य आउटपुट परत करते.

एक पाइपलाइन तयार करा आणि मजकूर निर्मितीसाठी (text generation) वापरण्याचे मॉडेल निर्दिष्ट करा. मॉडेल आपोआप डाउनलोड होऊन कॅश केले जाते जेणेकरून तुम्ही ते पुन्हा सहज वापरू शकाल. शेवटी, मॉडेलला प्रॉम्प्ट करण्यासाठी काही मजकूर द्या.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

एखाद्या मॉडेलशी चॅट करण्यासाठी (chat with a model), वापराची पद्धत सारखीच आहे. फरक एवढाच आहे की तुम्हाला तुमच्या आणि सिस्टिममधील चॅटचा इतिहास (`Pipeline` साठी इनपुट) तयार करावा लागेल.

> [!TIP]
> जोपर्यंत [`transformers serve` चालू आहे](https://huggingface.co/docs/transformers/main/en/serving), तोपर्यंत तुम्ही थेट कमांड लाइनवरूनही मॉडेलशी चॅट करू शकता.
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```py
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

विविध पद्धती (modalities) आणि कामांसाठी `Pipeline` कसे कार्य करते हे पाहण्यासाठी खालील उदाहरणे विस्तृत करा.

<details>
<summary>ऑटोमॅटिक स्पीच रेकग्निशन (Automatic speech recognition)</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>प्रतिमा वर्गीकरण (Image classification)</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'label': 'macaw', 'score': 0.997848391532898},
 {'label': 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
  'score': 0.0016551691805943847},
 {'label': 'lorikeet', 'score': 0.00018523589824326336},
 {'label': 'African grey, African gray, Psittacus erithacus',
  'score': 7.85409429227002e-05},
 {'label': 'quail', 'score': 5.502637941390276e-05}]
```

</details>

<details>
<summary>व्हिज्युअल प्रश्नोत्तरे (Visual question answering)</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</details>

## मी Transformers का वापरावे? (Why should I use Transformers?)

1. वापरण्यास सोपी अत्याधुनिक मॉडेल्स:
    - नैसर्गिक भाषा समज आणि निर्मिती (NLU & NLG), संगणकीय दृष्टी, ध्वनी, व्हिडिओ आणि मल्टिमोडल कामांवर उत्कृष्ट कामगिरी.
    - संशोधक, अभियंते आणि विकासकांसाठी शिकण्यास अत्यंत सोपे.
    - शिकण्यासाठी केवळ तीन मूलभूत वर्गांसह (classes) वापरकर्ता-सुलभ ॲब्स्ट्रॅक्शन्स.
    - आमच्या सर्व पूर्व-प्रशिक्षित मॉडेल्ससाठी एक युनिफाइड (एकीकृत) API.

1. कमी संगणकीय खर्च, कमी कार्बन फूटप्रिंट:
    - नव्याने सुरवातीपासून प्रशिक्षण देण्याऐवजी पूर्व-प्रशिक्षित मॉडेल्स शेअर करा.
    - संगणकीय वेळ आणि उत्पादन खर्च कमी करा.
    - सर्व प्रकारच्या डेटासाठी १० लाखांहून अधिक (1M+) चेकपॉईंट्ससह शेकडो मॉडेल आर्किटेक्चर्स.

1. मॉडेलच्या जीवनचक्रातील प्रत्येक टप्प्यासाठी योग्य फ्रेमवर्क निवडा:
    - कोडच्या फक्त ३ ओळींमध्ये अत्याधुनिक मॉडेल्सना प्रशिक्षित करा.
    - PyTorch/JAX/TF2.0 फ्रेमवर्क्सदरम्यान एकाच मॉडेलला इच्छेनुसार हलवा.
    - प्रशिक्षण, मूल्यांकन आणि उत्पादनासाठी योग्य फ्रेमवर्क निवडा.

1. तुमच्या गरजेनुसार मॉडेल किंवा उदाहरण सहजपणे सानुकूलित (Customize) करा:
    - मूळ लेखकांनी प्रकाशित केलेले परिणाम पुनरुत्पादित करण्यासाठी आम्ही प्रत्येक आर्किटेक्चरसाठी उदाहरणे देतो.
    - मॉडेलचे अंतर्गत घटक शक्य तितके पारदर्शक आणि सुसंगत ठेवले आहेत.
    - जलद प्रयोगांसाठी मॉडेल फाइल्स लायब्ररीपासून स्वतंत्रपणे वापरल्या जाऊ शकतात.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## मी Transformers कधी वापरू नये? (When shouldn't I use Transformers?)

- ही लायब्ररी न्यूरल नेटवर्कच्या बिल्डिंग ब्लॉक्सचे मॉड्युलर टूलबॉक्स नाही. मॉडेल फाइल्समधील कोडमध्ये हेतुपुरस्सर अतिरिक्त ॲब्स्ट्रॅक्शन्स जोडलेले नाहीत, जेणेकरून संशोधक नवीन फाइल्सच्या चक्रव्यूहात न अडकता प्रत्येक मॉडेलवर जलद गतीने काम करू शकतील.
- ट्रेनिंग API हे Transformers द्वारे प्रदान केलेल्या PyTorch मॉडेल्ससह काम करण्यासाठी अनुकूलित (optimized) आहे. सामान्य मशीन लर्निंग लूपसाठी, तुम्ही [Accelerate](https://huggingface.co/docs/accelerate) सारखी दुसरी लायब्ररी वापरली पाहिजे.
- [उदाहरणातील स्क्रिप्ट्स](https://github.com/huggingface/transformers/tree/main/examples) ही केवळ *उदाहरणे* आहेत. ती तुमच्या विशिष्ट समस्येसाठी जशीच्या तशी चालतीलच असे नाही, आणि तुम्हाला तुमच्या गरजेनुसार कोड अनुकूलित करावा लागू शकतो.

## Transformers वापरणारे १०० प्रोजेक्ट्स (100 projects using Transformers)

Transformers हे केवळ पूर्व-प्रशिक्षित मॉडेल्स वापरण्याचे साधन नाही, तर ते आणि Hugging Face Hub भोवती तयार झालेला एक समृद्ध समुदाय आहे. विकसक, संशोधक, विद्यार्थी, प्राध्यापक, अभियंते आणि इतर कोणालाही त्यांच्या स्वप्नातील प्रकल्प साकार करण्यास सक्षम बनवणे हे Transformers चे उद्दिष्ट आहे.

Transformers च्या १००,००० (100k) स्टार्सचा आनंद साजरा करण्यासाठी, आम्ही [awesome-transformers](../awesome-transformers.md) पृष्ठाद्वारे समुदायावर प्रकाश टाकला आहे, ज्यामध्ये Transformers सह तयार केलेल्या १०० अविश्वसनीय प्रकल्पांची यादी आहे.

जर तुमच्याकडे असा एखादा प्रकल्प असेल जो या यादीचा भाग असावा असे तुम्हाला वाटत असेल, तर कृपया तो जोडण्यासाठी एक PR उघडा!

## मॉडेल उदाहरणे (Example models)

तुम्ही आमची बहुतांश मॉडेल्स थेट त्यांच्या [Hub मॉडेल पृष्ठांवर](https://huggingface.co/models) तपासू शकता.

विविध वापरांसाठी काही उदाहरण मॉडेल्स पाहण्यासाठी खालील प्रत्येक मोडॅलिटी विस्तृत करा.

<details>
<summary>ऑडिओ / ध्वनी (Audio)</summary>

- [CLAP](https://huggingface.co/laion/clap-htsat-fused) सह ऑडिओ वर्गीकरण (Audio classification)
- [Parakeet](https://huggingface.co/nvidia/parakeet-ctc-1.1b#transcribing-using-transformers-%F0%9F%A4%97), [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) आणि [Moonshine-Streaming](https://huggingface.co/UsefulSensors/moonshine-streaming-medium) सह स्वयंचलित भाषण ओळख (ASR)
- [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks) सह कीवर्ड स्पॉटिंग
- [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) सह स्पीच-टू-स्पीच निर्मिती
- [MusicGen](https://huggingface.co/facebook/musicgen-large) सह मजकूरावरून ऑडिओ निर्मिती (Text to audio)
- [CSM](https://huggingface.co/sesame/csm-1b) सह मजकूरावरून भाषण (Text to speech)

</details>

<details>
<summary>संगणकीय दृष्टी (Computer vision)</summary>

- [SAM](https://huggingface.co/facebook/sam-vit-base) सह स्वयंचलित मास्क निर्मिती
- [DepthPro](https://huggingface.co/apple/DepthPro-hf) सह खोलीचा अंदाज (Depth estimation)
- [DINO v2](https://huggingface.co/facebook/dinov2-base) सह प्रतिमा वर्गीकरण (Image classification)
- [SuperPoint](https://huggingface.co/magic-leap-community/superpoint) सह कीपॉइंट डिटेक्शन
- [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor) सह कीपॉइंट मॅचिंग
- [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) सह वस्तू शोधणे (Object detection)
- [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple) सह पोझ अंदाज (Pose Estimation)
- [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) सह युनिव्हर्सल सेगमेंटेशन
- [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large) सह व्हिडिओ वर्गीकरण

</details>

<details>
<summary>मल्टिमोडल (Multimodal)</summary>

- [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), [Audio Flamingo](https://huggingface.co/nvidia/audio-flamingo-3-hf) सह ऑडिओ किंवा मजकूरावरून मजकूर निर्मिती
- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) सह दस्तऐवज प्रश्नोत्तरे (Document QA)
- [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) सह प्रतिमा किंवा मजकूरावरून मजकूर
- [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) सह प्रतिमा कॅप्शनिंग (Image captioning)
- [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) सह OCR-आधारित दस्तऐवज आकलन
- [TAPAS](https://huggingface.co/google/tapas-base) सह सारणी प्रश्नोत्तरे (Table QA)
- [Emu3](https://huggingface.co/BAAI/Emu3-Gen) सह एकात्मिक मल्टिमोडल आकलन आणि निर्मिती
- [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) सह व्हिजन-टू-टेक्स्ट
- [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) सह व्हिज्युअल प्रश्नोत्तरे
- [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) सह व्हिज्युअल रेफरिंग एक्स्प्रेशन सेगमेंटेशन

</details>

<details>
<summary>नैसर्गिक भाषा प्रक्रिया (NLP)</summary>

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) सह मास्क केलेल्या शब्दांची पूर्तता (Masked word completion)
- [Gemma](https://huggingface.co/google/gemma-2-2b) सह नामांकित घटक ओळख (NER)
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) सह प्रश्नोत्तरे (Question answering)
- [BART](https://huggingface.co/facebook/bart-large-cnn) सह सारांशीकरण (Summarization)
- [T5](https://huggingface.co/google-t5/t5-base) सह भाषांतर (Translation)
- [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B) सह मजकूर निर्मिती (Text generation)
- [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B) सह मजकूर वर्गीकरण (Text classification)

</details>

## उद्धरण (Citation)

आमच्याकडे 🤗 Transformers लायब्ररीसाठी उद्धृत करण्यासाठी एक [संशोधन निबंध (paper)](https://aclanthology.org/2020.emnlp-demos.6/) उपलब्ध आहे:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-demos.6/",
    pages = "38--45"
}
```
