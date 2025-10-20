<!--- Copyright 2020 The HuggingFace Team. All rights reserved. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. -->

<p align="center"> <picture> <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg"> <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg"> <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;"> </picture> <br/> <br/> </p> <p align="center"> <a href="https://huggingface.com/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a> <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a> <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a> <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a> <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a> <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a> <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a> </p> <h4 align="center"> <p> <a href="https://github.com/huggingface/transformers/blob/main/README.md">English</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Português</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_it.md">Italiano</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> | <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> | <b>தமிழ்</b> </p> </h4> <h3 align="center"> <p>அனுமானம் மற்றும் பயிற்சிக்கான அதிநவீன முன்பயிற்சி மாதிரிகள்</p> </h3> <h3 align="center"> <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/> </h3>
Transformers என்பது உரை, கணினி பார்வை, ஆடியோ, வீடியோ மற்றும் மல்டிமோடல் மாதிரிகளில் உள்ள அதிநவீன இயந்திர கற்றல் மாதிரிகளுக்கான மாதிரி-வரையறை கட்டமைப்பாக செயல்படுகிறது, இது அனுமானம் மற்றும் பயிற்சி இரண்டிற்கும் பயன்படுகிறது.

இது மாதிரி வரையறையை மையப்படுத்துகிறது, இதனால் இந்த வரையறை சூழல் முழுவதும் ஒப்புக் கொள்ளப்படுகிறது. transformers என்பது கட்டமைப்புகள் முழுவதும் உள்ள முக்கிய புள்ளியாகும்: ஒரு மாதிரி வரையறை ஆதரிக்கப்பட்டால், அது பெரும்பாலான பயிற்சி கட்டமைப்புகள் (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), அனுமான இயந்திரங்கள் (vLLM, SGLang, TGI, ...), மற்றும் அருகிலுள்ள மாடலிங் நூலகங்கள் (llama.cpp, mlx, ...) உடன் இணக்கமாக இருக்கும், இவை அனைத்தும் transformers இலிருந்து மாதிரி வரையறையைப் பயன்படுத்துகின்றன.

புதிய அதிநவீன மாதிரிகளை ஆதரிப்பதற்கும், அவற்றின் மாதிரி வரையறையை எளிமையாகவும், தனிப்பயனாக்கக்கூடியதாகவும், திறமையானதாகவும் வைத்திருப்பதன் மூலம் அவற்றின் பயன்பாட்டை ஜனநாயகமாக்குவதற்கும் நாங்கள் உதவுவதாக உறுதியளிக்கிறோம்.

Hugging Face Hub இல் 1M+ க்கும் மேற்பட்ட Transformers மாதிரி சோதனைப்புள்ளிகள் உள்ளன, அவற்றை நீங்கள் பயன்படுத்தலாம்.

இன்றே Hub ஐ ஆராய்ந்து ஒரு மாதிரியைக் கண்டுபிடித்து, Transformers ஐப் பயன்படுத்தி உடனடியாக தொடங்கவும்.

நிறுவல்
Transformers Python 3.9+ மற்றும் PyTorch 2.1+ உடன் செயல்படுகிறது.

venv அல்லது uv, ஒரு வேகமான Rust-அடிப்படையிலான Python தொகுப்பு மற்றும் திட்ட மேலாளர் மூலம் ஒரு மெய்நிகர் சூழலை உருவாக்கி செயல்படுத்தவும்.

py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
உங்கள் மெய்நிகர் சூழலில் Transformers ஐ நிறுவவும்.

py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
நூலகத்தில் சமீபத்திய மாற்றங்களை நீங்கள் விரும்பினால் அல்லது பங்களிப்பதில் ஆர்வமாக இருந்தால், மூலத்திலிருந்து Transformers ஐ நிறுவவும். இருப்பினும், சமீபத்திய பதிப்பு நிலையானதாக இல்லாமல் இருக்கலாம். நீங்கள் பிழையை சந்தித்தால் தயவுசெய்து issue ஒன்றைத் திறக்கவும்.

shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
விரைவான தொடக்கம்
Pipeline API மூலம் Transformers உடன் உடனடியாக தொடங்கவும். Pipeline என்பது உரை, ஆடியோ, பார்வை மற்றும் மல்டிமோடல் பணிகளை ஆதரிக்கும் உயர்நிலை அனுமான வகுப்பாகும். இது உள்ளீட்டை முன்செயலாக்கம் செய்து பொருத்தமான வெளியீட்டை வழங்குகிறது.

உரை உருவாக்கத்திற்குப் பயன்படுத்த வேண்டிய மாதிரியைக் குறிப்பிட்டு ஒரு pipeline ஐ உருவாக்கவும். மாதிரி பதிவிறக்கம் செய்யப்பட்டு தற்காலிக சேமிப்பகத்தில் சேமிக்கப்படுகிறது, எனவே நீங்கள் அதை மீண்டும் எளிதாகப் பயன்படுத்தலாம். இறுதியாக, மாதிரியைத் தூண்டுவதற்கு சில உரையை அனுப்பவும்.

py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
ஒரு மாதிரியுடன் அரட்டை அடிக்க, பயன்பாட்டு முறை அதேதான். ஒரே வித்தியாசம் என்னவென்றால், உங்களுக்கும் அமைப்புக்கும் இடையே ஒரு அரட்டை வரலாற்றை (Pipeline க்கான உள்ளீடு) நீங்கள் உருவாக்க வேண்டும்.

> [!TIP]
கட்டளை வரியிலிருந்து நேரடியாக ஒரு மாதிரியுடன் அரட்டை அடிக்கலாம்.

shell
transformers chat Qwen/Qwen2.5-0.5B-Instruct
py
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
வெவ்வேறு முறைகள் மற்றும் பணிகளுக்கு Pipeline எவ்வாறு செயல்படுகிறது என்பதைக் காண கீழே உள்ள எடுத்துக்காட்டுகளை விரிவாக்கவும்.

<details> <summary>தானியங்கி பேச்சு அங்கீகாரம்</summary> ```py from transformers import pipeline
pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3") pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac") {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}


</details>

<details>
<summary>படம் வகைப்படுத்தல்</summary>

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
</details> <details> <summary>காட்சி கேள்வி பதில்</summary> <h3 align="center"> <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a> </h3> ```py from transformers import pipeline
pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base") pipeline( image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg", question="What is in the image?", ) [{'answer': 'statue of liberty'}]


</details>

## நான் ஏன் Transformers ஐப் பயன்படுத்த வேண்டும்?

1. பயன்படுத்த எளிதான அதிநவீன மாதிரிகள்:
    - இயற்கை மொழி புரிதல் & உருவாக்கம், கணினி பார்வை, ஆடியோ, வீடியோ மற்றும் மல்டிமோடல் பணிகளில் உயர் செயல்திறன்.
    - ஆராய்ச்சியாளர்கள், பொறியாளர்கள் மற்றும் டெவலப்பர்களுக்கு குறைந்த நுழைவு தடை.
    - கற்றுக் கொள்ள வேண்டிய மூன்று வகுப்புகளுடன் சில பயனர்-எதிர்கொள்ளும் சுருக்கங்கள்.
    - எங்களின் அனைத்து முன்பயிற்சி மாதிரிகளையும் பயன்படுத்துவதற்கான ஒருங்கிணைந்த API.

2. குறைந்த கணினி செலவுகள், சிறிய கார்பன் தடம்:
    - புதிதாகப் பயிற்சியளிப்பதற்குப் பதிலாக பயிற்சியளிக்கப்பட்ட மாதிரிகளைப் பகிர்ந்து கொள்ளுங்கள்.
    - கணினி நேரத்தையும் உற்பத்தி செலவுகளையும் குறைக்கவும்.
    - அனைத்து முறைகளிலும் 1M+ முன்பயிற்சி சோதனைப்புள்ளிகளுடன் டஜன் கணக்கான மாதிரி கட்டமைப்புகள்.

3. மாதிரியின் வாழ்நாளின் ஒவ்வொரு பகுதிக்கும் சரியான கட்டமைப்பைத் தேர்வு செய்யவும்:
    - 3 வரிகள் குறியீட்டில் அதிநவீன மாதிரிகளைப் பயிற்றுவிக்கவும்.
    - PyTorch/JAX/TF2.0 கட்டமைப்புகளுக்கு இடையே விருப்பப்படி ஒற்றை மாதிரியை நகர்த்தவும்.
    - பயிற்சி, மதிப்பீடு மற்றும் உற்பத்திக்கு சரியான கட்டமைப்பைத் தேர்ந்தெடுக்கவும்.

4. உங்கள் தேவைகளுக்கு ஏற்ப ஒரு மாதிரி அல்லது உதாரணத்தை எளிதாக தனிப்பயனாக்கவும்:
    - அதன் அசல் ஆசிரியர்களால் வெளியிடப்பட்ட முடிவுகளை மீண்டும் உருவாக்க ஒவ்வொரு கட்டமைப்பிற்கும் எடுத்துக்காட்டுகளை நாங்கள் வழங்குகிறோம்.
    - மாதிரி உள் அமைப்புகள் முடிந்தவரை சீராக வெளிப்படுத்தப்படுகின்றன.
    - விரைவான சோதனைகளுக்காக மாதிரி கோப்புகளை நூலகத்திலிருந்து சுயாதீனமாகப் பயன்படுத்தலாம்.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## நான் ஏன் Transformers ஐப் பயன்படுத்தக் கூடாது?

- இந்த நூலகம் நரம்பியல் வலைகளுக்கான கட்டுமான தொகுதிகளின் மாடுலர் கருவிப்பெட்டி அல்ல. மாதிரி கோப்புகளில் உள்ள குறியீடு கூடுதல் சுருக்கங்களுடன் மறுசீரமைக்கப்படவில்லை, இதன் நோக்கம் ஆராய்ச்சியாளர்கள் கூடுதல் சுருக்கங்கள்/கோப்புகளில் மூழ்காமல் ஒவ்வொரு மாதிரியிலும் விரைவாக மீண்டும் மீண்டும் செயல்பட முடியும்.
- பயிற்சி API ஆனது Transformers வழங்கும் PyTorch மாதிரிகளுடன் வேலை செய்ய உகந்ததாக உள்ளது. பொதுவான இயந்திர கற்றல் சுழற்சிகளுக்கு, [Accelerate](https://huggingface.co/docs/accelerate) போன்ற மற்றொரு நூலகத்தைப் பயன்படுத்த வேண்டும்.
- [உதாரண ஸ்கிரிப்ட்கள்](https://github.com/huggingface/transformers/tree/main/examples) என்பது *உதாரணங்கள்* மட்டுமே. அவை உங்கள் குறிப்பிட்ட பயன்பாட்டு வழக்கில் அவசியம் out-of-the-box வேலை செய்யாமல் இருக்கலாம், மேலும் அது வேலை செய்ய குறியீட்டை மாற்றியமைக்க வேண்டும்.

## Transformers ஐப் பயன்படுத்தும் 100 திட்டங்கள்

Transformers என்பது முன்பயிற்சி மாதிரிகளைப் பயன்படுத்துவதற்கான கருவிப்பெட்டியை விட அதிகம், இது அதைச் சுற்றி கட்டமைக்கப்பட்ட திட்டங்களின் சமூகம் மற்றும் Hugging Face Hub. டெவலப்பர்கள், ஆராய்ச்சியாளர்கள், மாணவர்கள், பேராசிரியர்கள், பொறியாளர்கள் மற்றும் வேறு யாரும் தங்கள் கனவு திட்டங்களைக் கட்டமைக்க Transformers உதவ வேண்டும் என்று நாங்கள் விரும்புகிறோம்.

Transformers 100,000 stars ஐக் கொண்டாடுவதற்காக, Transformers உடன் கட்டமைக்கப்பட்ட 100 நம்பமுடியாத திட்டங்களைப் பட்டியலிடும் [awesome-transformers](./awesome-transformers.md) பக்கத்துடன் சமூகத்தின் மீது கவனம் செலுத்த விரும்பினோம்.

நீங்கள் பட்டியலின் ஒரு பகுதியாக இருக்க வேண்டும் என்று நீங்கள் நம்பும் ஒரு திட்டத்தை நீங்கள் சொந்தமாக வைத்திருந்தால் அல்லது பயன்படுத்தினால், அதைச் சேர்க்க தயவுசெய்து PR ஐத் திறக்கவும்!

## மாதிரி எடுத்துக்காட்டுகள்

நீங்கள் எங்கள் பெரும்பாலான மாதிரிகளை அவற்றின் [Hub மாதிரி பக்கங்களில்](https://huggingface.co/models) நேரடியாக சோதிக்கலாம்.

பல்வேறு பயன்பாட்டு வழக்குகளுக்கான சில மாதிரி எடுத்துக்காட்டுகளைக் காண கீழே உள்ள ஒவ்வொரு முறையையும் விரிவாக்கவும்.

<details>
<summary>ஆடியோ</summary>

- [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) உடன் ஆடியோ வகைப்படுத்தல்
- [Moonshine](https://huggingface.co/UsefulSensors/moonshine) உடன் தானியங்கி பேச்சு அங்கீகாரம்
- [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks) உடன் முக்கிய சொல் கண்டறிதல்
- [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) உடன் பேச்சிலிருந்து பேச்சு உருவாக்கம்
- [MusicGen](https://huggingface.co/facebook/musicgen-large) உடன் உரையிலிருந்து ஆடியோ
- [Bark](https://huggingface.co/suno/bark) உடன் உரையிலிருந்து பேச்சு

</details>

<details>
<summary>கணினி பார்வை</summary>

- [SAM](https://huggingface.co/facebook/sam-vit-base) உடன் தானியங்கி முகமூடி உருவாக்கம்
- [DepthPro](https://huggingface.co/apple/DepthPro-hf) உடன் ஆழம் மதிப்பீடு
- [DINO v2](https://huggingface.co/facebook/dinov2-base) உடன் படம் வகைப்படுத்தல்
- [SuperPoint](https://huggingface.co/magic-leap-community/superpoint) உடன் முக்கிய புள்ளி கண்டறிதல்
- [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor) உடன் முக்கிய புள்ளி பொருத்தம்
- [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) உடன் பொருள் கண்டறிதல்
- [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple) உடன் போஸ் மதிப்பீடு
- [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) உடன் உலகளாவிய பிரிவு
- [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large) உடன் வீடியோ வகைப்படுத்தல்

</details>

<details>
<summary>மல்டிமோடல்</summary>

- [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B) உடன் ஆடியோ அல்லது உரையிலிருந்து உரை
- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) உடன் ஆவண கேள்வி பதில்
- [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) உடன் படம் அல்லது உரையிலிருந்து உரை
- [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) உடன் படம் தலைப்பிடல்
- [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) உடன் OCR-அடிப்படையிலான ஆவண புரிதல்
- [TAPAS](https://huggingface.co/google/tapas-base) உடன் அட்டவணை கேள்வி பதில்
- [Emu3](https://huggingface.co/BAAI/Emu3-Gen) உடன் ஒருங்கிணைந்த மல்டிமோடல் புரிதல் மற்றும் உருவாக்கம்
- [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) உடன் பார்வையிலிருந்து உரை
- [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) உடன் காட்சி கேள்வி பதில்
- [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) உடன் காட்சி குறிப்பு வெளிப்பாடு பிரிவு

</details>

<details>
<summary>NLP</summary>

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) உடன் மறைக்கப்பட்ட சொல் நிறைவு
- [Gemma](https://huggingface.co/google/gemma-2-2b) உடன் பெயரிடப்பட்ட நிறுவன அங்கீகாரம்
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) உடன் கேள்வி பதில்
- [BART](https://huggingface.co/facebook/bart-large-cnn) உடன் சுருக்கம்
- [T5](https://huggingface.co/google-t5/t5-base) உடன் மொழிபெயர்ப்பு
- [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B) உடன் உரை உருவாக்கம்
- [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B) உடன் உரை வகைப்படுத்தல்

</details>

## மேற்கோள்

🤗 Transformers நூலகத்திற்கு நீங்கள் மேற்கோள் காட்டக்கூடிய [காகிதம்](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) இப்போது எங்களிடம் உள்ளது:
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
