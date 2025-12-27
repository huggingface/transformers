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
        <a href="https://github.com/huggingface/transformers/blob/main/README.md">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Português</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
        <b>සිංහල</b>
    </p>
</h4>

<h3 align="center">
    <p>උපකල්පන සහ පුහුණුව සඳහා අති නවීන පූර්ව-පුහුණු මාදිලි</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers යනු පෙළ, පරිගණක දැක්ම, ශ්‍රව්‍ය, වීඩියෝ සහ බහුමාධ්‍ය මාදිලි සඳහා අති නවීන යන්ත්‍ර ඉගැන්වීමේ මාදිලි සඳහා මාදිලි-අර්ථ දැක්වීමේ රාමුව ලෙස ක්‍රියා කරයි, උපකල්පන සහ පුහුණුව යන දෙකම සඳහා.

එය පරිසර පද්ධතිය පුරා මෙම අර්ථ දැක්වීම ගැන එකඟ වීම සඳහා මාදිලි අර්ථ දැක්වීම කේන්ද්‍රගත කරයි. `transformers` යනු රාමු හරහා කරකවන ලක්ෂ්‍යයකි: මාදිලි අර්ථ දැක්වීමක් සහය දක්වන්නේ නම්, එය බහුතර පුහුණු රාමු (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), උපකල්පන එන්ජින් (vLLM, SGLang, TGI, ...), සහ `transformers` වෙතින් මාදිලි අර්ථ දැක්වීම භාවිත කරන අයදිරි මාදිලි කිරීමේ පුස්තකාල (llama.cpp, mlx, ...) සමඟ අනුකූල වනු ඇත.

නව අති නවීන මාදිලි සඳහා සහය ලබා දීමට සහ ඒවායේ මාදිලි අර්ථ දැක්වීම සරල, අභිරුචිකරණය කළ හැකි සහ කාර්යක්ෂම වීමෙන් ඒවායේ භාවිතය ප්‍රජාතන්ත්‍රකරණය කිරීමට අපි පොරොන්දු වෙමු.

[Hugging Face Hub](https://huggingface.com/models) හි 1M+ Transformers [මාදිලි චෙක්පොයින්ට්](https://huggingface.co/models?library=transformers&sort=trending) තිබේ ඔබට භාවිත කළ හැක.

අදම [Hub](https://huggingface.com/) ගවේෂණය කර මාදිලියක් සොයා Transformers භාවිතා කර ඔබට වහාම ආරම්භ කිරීමට උපකාර ලබා ගන්න.

## ස්ථාපන

Transformers Python 3.9+, සහ [PyTorch](https://pytorch.org/get-started/locally/) 2.1+ සමඟ ක්‍රියා කරයි.

[venv](https://docs.python.org/3/library/venv.html) හෝ [uv](https://docs.astral.sh/uv/), වේගවත් Rust-පාදක Python පැකේජ සහ ව්‍යාපෘති කළමනාකරුවෙකු සමඟ අතථ්‍ය පරිසරයක් සාදා සක්‍රිය කරන්න.

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

ඔබගේ අතථ්‍ය පරිසරයේ Transformers ස්ථාපනය කරන්න.

```py
# pip
pip install "transformers[torch]"
```

කරුණාකර [ස්ථාපන පිටුව](https://huggingface.co/docs/transformers/installation) වෙත යන්න සවිස්තර නිර්දේශ, වෙනත් ගිණුම් මෙවලම් (Flax, TensorFlow) සඳහා, සහ ම්‍රින්ට් කුකි ස්ථාපනයන් (සංවර්ධන සංවර්ධන)!

## ක්ෂණික සංචාරය

Transformers ලයිබේරියේ භාවිතය කරන්නේ මෙසේ:

```python
>>> from transformers import pipeline

>>> classifier = pipeline('sentiment-analysis')
>>> classifier('Transformers භාවිතා කිරීම සතුටක්!')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

දෙවන පෙළ pipeline තුළ භාවිත වන පුද්ගලික මාදිලිය පූර්ණ කරයි සහ cache කරයි, එබැවින් ඔබට එම කාර්යයට ගමන් කිරීමට කාර්යක්‍රම යනවා භාවිත කරන විට ඔබට පුද්ගලික මාදිලිය යලිකර භාවිත කළ හැක.

සියලු කාර්යයන්ට pipeline ඇත! උදාහරණයක් ලෙස:

```python
>>> from transformers import pipeline

>>> generator = pipeline('text-generation')
>>> generator('මම Transformers ලයිබේරිය ප්‍රිය කරන්නේ මන්ද')
[{'generated_text': 'මම Transformers ලයිබේරිය ප්‍රිය කරන්නේ මන්ද එය ඉතා ප්‍රයෝජනවත් අවශ්‍යතාවන් සපයයි...'}]
```

ඔබට වර්තමානයේ ලභ්‍ය ඕනෑම pipeline භාවිත කළ හැක. [මෙහි](https://huggingface.co/docs/transformers/task_summary) සම්පූර්ණ ලැයිස්තුව බලන්න.

## ක්ෂණික API භාවිතය

ඔබේ කාර්යය සඳහා දත්ත AI මාදිලියක් කිරීමේ අවශ්‍යතාවයක් තිබේ, ඔබට Hugging Face Hub හි පැරණි දොස්තර මාදිලිය භාවිත කිරීමට පහසු ක්‍රමයක් Transformers ලබා දෙයි.

```python
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased-finetuned-mrpc")
>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased-finetuned-mrpc")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'DUPLICATE'
```

## ඔබ Transformers භාවිත කළ යුත්තේ ඇයි?

1. භාවිතයට පහසු අති නවීන මාදිලි:
    - ස්වාභාවික භාෂා අවබෝධ සහ උත්පාදන, පරිගණක දැක්ම, ශ්‍රව්‍ය, වීඩියෝ සහ බහුමාධ්‍ය කාර්යයන් සඳහා ඉහළ ක්‍රියාකාරීත්වය.
    - පර්යේෂකයන්, ඉංජිනේරුවරුන් සහ සංවර්ධකයන් සඳහා අඩු ප්‍රවේශ බාධකයක්.
    - ඉගෙන ගැනීමට පන්ති තුනක් පමණක් සමඟ කිහිපයක් පරිශීලක-මුහුණ වියුක්තකරණ.
    - අපගේ සියලු පූර්ව-පුහුණු මාදිලි භාවිත කිරීම සඳහා ඒකාකාර API.

1. අඩු ගණන විසඳුම් පිරිවැය, කුඩා කාබන් අඩිපාර:
    - කිසිදු ගණනකින් ආරම්භ කරනවා වෙනුවට පුහුණු මාදිලි බෙදා ගැනීම.
    - ගණන කාලය සහ නිෂ්පාදන ප්‍රවිතර අඩු කිරීම.
    - සියලු මාධ්‍යයන් හරහා 1M+ පූර්ව-පුහුණු චෙක්පොයින්ට් සමඟ දර්ශන වුවමනාව වස්තූන් ගණනිකාදර ඉහල.

1. මාදිලියේ ජීවන කාලය සෑම කොටසකටම නිසි රාමුව තෝරන්න:
    - කේත පේළි 3 කින් අති නවීන මාදිලි පුහුණු කරන්න.
    - PyTorch/JAX/TF2.0 රාමු අතර කැමති අයුරින් තනි මාදිලියක් ගමන් කරන්න.
    - පුහුණුව, ඇගයීම සහ නිෂ්පාදනය සඳහා නිසි රාමුව තෝරන්න.

1. ඔබේ අවශ්‍යතාවන්ට මාදිලියක් හෝ උදාහරණයක් පහසුවෙන් අභිරුචිකරණය කරන්න:
    - එක් එක් වස්තු නිර්මාණ ශිල්පය සඳහා අපි උදාහරණ ලබා දෙමු, එහි මුල් කර්තෘවරුන් විසින් ප්‍රකාශිත ප්‍රතිඵල යලි නිපදවන්න.
    - මාදිලියේ අභ්‍යන්තර තාක්‍ෂණයන් හැකි තරම් ස්ථිරපත්‍රව හෙළි කෙරේ.
    - ලිදී අත්හදා බැලීම් සඳහා ලයිබේරිය වෙතින් ස්වායත්තව මාදිලි ගොනු භාවිත කළ හැක.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## ඔබ Transformers භාවිත නොකළ යුත්තේ ඇයි?

- මෙම ලයිබේරිය ස්නායු ජාල සඳහා ගොඩනැගිලි කොටස්වල මොඩ්‍යුලර් අවකාශයක් නොවේ. මාදිලි ගොනුවල කේතය අතිරේක වියුක්තකරණ සමඟ අදහසකින් වැඩිදියුණු නොකෙරේ, එබැවින් පර්යේෂකයන්ට අතිරේක වියුක්තකරණ/ගොනු වෙත කීමකින් නොබැඳී එක් එක් මාදිලිය මත ක්ෂණිකව නැවත එන ගමන් කිරීමට හැකි වේ.
- පුහුණු API Transformers විසින් සපයන PyTorch මාදිලි සමඟ ක්‍රියා කිරීමට ප්‍රශස්ත කර තිබේ. සාමාන්‍ය යන්ත්‍ර ඉගැන්වීම ලූප සඳහා, ඔබ [Accelerate](https://huggingface.co/docs/accelerate) වැනි වෙනත් ලයිබේරියක් භාවිත කළ යුතුයි.
- [උදාහරණ ස්ක්‍රිප්ට්](https://github.com/huggingface/transformers/tree/main/examples) *උදාහරණ* පමණි. ඔවුන් ඔබේ විශේෂ භාවිත නඩුව මත අවශ්‍යයෙන්ම ගමන් ගැරහෙනකම ක්‍රියා නොකළ හැකි අතර එය ක්‍රියා කිරීමට ඔබට කේතය අනුවර්තනය කිරීමට සිදු වනු ඇත.

## Transformers භාවිත කරමින් ව්‍යාපෘති 100 කි

Transformers එකම පූර්ව-පුහුණු මාදිලි භාවිත කිරීමේ මෙවලම් කට්ටලයක් නොව, එය වටා ගොඩනගා ගත් ව්‍යාපෘති ප්‍රජාවක් සහ Hugging Face Hub ය. අපට අවශ්‍ය වන්නේ Transformers සංවර්ධකයන්, පර්යේෂකයන්, සිසුන්, පරාස්ථාපකයන්, ඉංජිනේරුවරුන් සහ වෙනත් ඕනෑම කෙනෙකුට ඔවුන්ගේ සිහින ව්‍යාපෘති ගොඩනගා ගැනීමට හැකි වන ලෙසටය.

Transformers තරු 100,000 සැමරීම සඳහා, අපට අවශ්‍ය වූයේ [awesome-transformers](./awesome-transformers.md) පිටුව සමඟ ප්‍රජාව මත අවධානය යොමු කිරීමය, එය Transformers සමඟ ගොඩනගා ගත් අසිරිමත් ව්‍යාපෘති 100 කි ලැයිස්තුගත කරයි.

ඔබ ලැයිස්තුවේ කොටසක් වී තිබිය යුතු යැයි ඔබ විශ්වාස කරන ව්‍යාපෘතියක් ඔබ සතුව හෝ භාවිත කරන්නේ නම්, කරුණාකර එය එකතු කිරීමට PR එකක් විවෘත කරන්න!

## උදාහරණ මාදිලි

ඔබට අපගේ බොහෝ මාදිලි ඔවුන්ගේ [Hub මාදිලි පිටුවල](https://huggingface.com/models) සෘජුවම පරීක්‍ෂා කළ හැක.

විවිධ භාවිත අවස්ථා සඳහා උදාහරණ මාදිලි කිහිපයක් බැලීමට පහත එක් එක් මාධ්‍යය විශ්කාශ කරන්න.

<details>
<summary>ශ්‍රව්‍ය</summary>

- [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) සමඟ ශ්‍රව්‍ය වර්ගීකරණය
- [Moonshine](https://huggingface.co/UsefulSensors/moonshine) සමඟ ස්වයංක්‍රිය කථන හඳුනාගැනීම
- [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks) සමඟ මූල පද හඳුනාගැනීම
- [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) සමඟ කථනයෙන් කථනයට උත්පාදනය
- [MusicGen](https://huggingface.co/facebook/musicgen-large) සමඟ පෙළෙන් ශ්‍රව්‍ය දක්වා
- [Bark](https://huggingface.co/suno/bark) සමඟ පෙළෙන් කථනය දක්වා

</details>

<details>
<summary>පරිගණක දැක්ම</summary>

- [SAM](https://huggingface.co/facebook/sam-vit-base) සමඟ ස්වයංක්‍රිය ආවරණ උත්පාදනය
- [DepthPro](https://huggingface.co/apple/DepthPro-hf) සමඟ ගැඹුර ඇස්තමේන්තු කිරීම
- [DINO v2](https://huggingface.co/facebook/dinov2-base) සමඟ රූප වර්ගීකරණය
- [SuperPoint](https://huggingface.co/magic-leap-community/superpoint) සමඟ ප්‍රධාන ලක්ෂ්‍ය හඳුනාගැනීම
- [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor) සමඟ ප්‍රධාන ලක්ෂ්‍ය ගැලපීම
- [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) සමඟ වස්තු හඳුනාගැනීම
- [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple) සමඟ ඉරියව් ඇස්තමේන්තු කිරීම
- [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) සමඟ විශ්වීය ඛණ්ඩනය
- [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large) සමඟ වීඩියෝ වර්ගීකරණය

</details>

<details>
<summary>බහුමාධ්‍ය</summary>

- [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B) සමඟ ශ්‍රව්‍ය හෝ පෙළෙන් පෙළ දක්වා
- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) සමඟ ලේඛන ප්‍රශ්න පිළිතුරු
- [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) සමඟ රූප හෝ පෙළෙන් පෙළ දක්වා
- [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) රූප ශීර්ෂක
- [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) සමඟ OCR-පාදක ලේඛන අවබෝධය
- [TAPAS](https://huggingface.co/google/tapas-base) සමඟ වගු ප්‍රශ්න පිළිතුරු
- [Emu3](https://huggingface.co/BAAI/Emu3-Gen) සමඟ ඒකීකෘත බහුමාධ්‍ය අවබෝධය සහ උත්පාදනය
- [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) සමඟ දැක්මෙන් පෙළ දක්වා
- [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) සමඟ දෘශ්‍ය ප්‍රශ්න පිළිතුරු
- [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) සමඟ දෘශ්‍ය සඳහන් ප්‍රකාශන ඛණ්ඩනය

</details>

<details>
<summary>NLP</summary>

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) සමඟ ආවරණය කළ වචන සම්පූර්ණ කිරීම
- [Gemma](https://huggingface.co/google/gemma-2-2b) සමඟ නම් කළ ආයතන හඳුනාගැනීම
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) සමඟ ප්‍රශ්න පිළිතුරු
- [BART](https://huggingface.co/facebook/bart-large-cnn) සමඟ සාරාංශ
- [T5](https://huggingface.co/google-t5/t5-base) සමඟ පරිවර්තනය
- [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B) සමඟ පෙළ උත්පාදනය
- [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B) සමඟ පෙළ වර්ගීකරණය

</details>

## උපුටන ගන්වීම

🤗 Transformers ලයිබේරිය සඳහා ඔබට උපුටන ගත හැකි [පත්‍රිකාවක්](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) දැන් අපකේ ඇත:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```