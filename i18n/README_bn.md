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
        <b>বাংলা</b> |
    </p>
</h4>

<h3 align="center">
    <p>ইনফারেন্স ও ট্রেনিংয়ের জন্য আধুনিকতম (State-of-the-art) প্রি-ট্রেইন্ড মডেলসমূহ</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>


Transformers হল টেক্সট, কম্পিউটার ভিশন, অডিও, ভিডিও এবং মাল্টিমোডাল মডেলে অত্যাধুনিক মেশিন লার্নিং মডেলের জন্য মডেল-সংজ্ঞা কাঠামো হিসাবে কাজ করে, যা অনুমান এবং প্রশিক্ষণ উভয় ক্ষেত্রেই প্রযোজ্য।

এটি মডেলের সংজ্ঞাগুলিকে কেন্দ্রীভূত করে যাতে এই সংজ্ঞাটি সম্পূর্ণ ইকোসিস্টেমে সম্মত হয়। `transformers` হলো বিভিন্ন ফ্রেমওয়ার্কের মধ্যে একটি কেন্দ্রবিন্দু: যদি একটি মডেল সংজ্ঞা সমর্থিত হয়, তবে এটি বেশিরভাগ প্রশিক্ষণ ফ্রেমওয়ার্ক (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), অনুমান ইঞ্জিন (vLLM, SGLang, TGI, ...), এবং সন্নিহিত মডেলিং লাইব্রেরি (llama.cpp, mlx, ...) যারা `transformers` থেকে মডেলের সংজ্ঞা ব্যবহার করে তাদের সাথে সামঞ্জস্যপূর্ণ হবে।

আমরা নতুন অত্যাধুনিক মডেলগুলিকে সমর্থন করতে এবং তাদের ব্যবহারের সহজলভ্যতা বাড়াতে প্রতিশ্রুতিবদ্ধ, যাতে তাদের মডেলের সংজ্ঞা সহজ, কাস্টমাইজেবল এবং কার্যকরী হয়।

[Hugging Face Hub](https://huggingface.com/models) -এ আপনি ১ মিলিয়নেরও বেশি Transformers [মডেল চেকপয়েন্ট](https://huggingface.co/models?library=transformers&sort=trending) ব্যবহার করতে পারেন।

আজই [Hub](https://huggingface.com/) অন্বেষণ করুন এবং একটি মডেল খুঁজে নিন, আর এখনই Transformers ব্যবহার শুরু করুন।

## ইনস্টলেশন

Transformers Python 3.9+, [PyTorch](https://pytorch.org/get-started/locally/) 2.1+, [TensorFlow](https://www.tensorflow.org/install/pip) 2.6+, এবং [Flax](https://flax.readthedocs.io/en/latest/) 0.4.1+ এর সাথে কাজ করে।

[venv](https://docs.python.org/3/library/venv.html) বা [uv](https://docs.astral.sh/uv/), একটি দ্রুত Rust-ভিত্তিক Python প্যাকেজ এবং প্রজেক্ট ম্যানেজার ব্যবহার করে একটি ভার্চুয়াল পরিবেশ তৈরি ও সক্রিয় করুন।

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```
আপনার ভার্চুয়াল পরিবেশে Transformers ইনস্টল করুন।

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```
যদি আপনি লাইব্রেরির সর্বশেষ পরিবর্তনগুলি চান বা অবদান রাখতে আগ্রহী হন তবে উৎস থেকে Transformers ইনস্টল করুন। তবে, সর্বশেষ সংস্করণটি স্থিতিশীল নাও হতে পারে। যদি আপনি কোনো ত্রুটির সম্মুখীন হন তবে নির্দ্বিধায় একটি [issue](https://github.com/huggingface/transformers/issues) খুলুন।

```Shell
git clone [https://github.com/huggingface/transformers.git](https://github.com/huggingface/transformers.git)
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

## কুইকস্টার্ট

[Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API দিয়ে এখনই Transformers ব্যবহার শুরু করুন। `Pipeline` হল একটি উচ্চ-স্তরের অনুমান শ্রেণী যা টেক্সট, অডিও, ভিশন এবং মাল্টিমোডাল কাজগুলিকে সমর্থন করে। এটি ইনপুট প্রিপ্রসেসিং পরিচালনা করে এবং উপযুক্ত আউটপুট ফিরিয়ে দেয়।

একটি পাইপলাইন ইনস্ট্যান্স তৈরি করুন এবং টেক্সট জেনারেশনের জন্য মডেল নির্দিষ্ট করুন। মডেলটি ডাউনলোড এবং ক্যাশ করা হয় যাতে আপনি সহজেই এটি আবার ব্যবহার করতে পারেন। অবশেষে, মডেলকে প্রম্পট করতে কিছু টেক্সট দিন।

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

একটি মডেলের সাথে চ্যাট করতে, ব্যবহারের ধরণ একই। একমাত্র পার্থক্য হলো আপনাকে আপনার এবং সিস্টেমের মধ্যে একটি চ্যাট হিস্টোরি (যা `Pipeline`-এর ইনপুট) তৈরি করতে হবে।

> [!TIP]
> আপনি সরাসরি কমান্ড লাইন থেকেও একটি মডেলের সাথে চ্যাট করতে পারেন।
> ```Shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```Python
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])

বিভিন্ন মোডালিটি এবং কাজের জন্য Pipeline কিভাবে কাজ করে তা দেখতে নিচের উদাহরণগুলো সম্প্রসারণ করুন।
```

<details>
<summary>স্বয়ংক্রিয় বক্তৃতা স্বীকৃতি (Automatic speech recognition)</summary>

```Python
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("[https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac](https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac)")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>চিত্র শ্রেণীকরণ (Image classification)</summary>

<h3 align="center">
<a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
pipeline("[https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png](https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png)")
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
<summary>ভিজ্যুয়াল প্রশ্ন জিজ্ঞাসা (Visual question answering)</summary>

<h3 align="center">
<a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="[https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg)",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```
</details>

## আমি কেন Transformers ব্যবহার করব?
1. সহজে ব্যবহারযোগ্য অত্যাধুনিক মডেল:
    - প্রাকৃতিক ভাষা বোঝা এবং তৈরি করা, কম্পিউটার ভিশন, অডিও, ভিডিও এবং মাল্টিমোডাল কাজের জন্য উচ্চ কর্মক্ষমতা।
    - গবেষক, প্রকৌশলী এবং ডেভেলপারদের জন্য প্রবেশে কম বাধা।
    - মাত্র তিনটি শ্রেণী শিখতে হয়, যা ব্যবহারকারীর জন্য কম বিমূর্ততা তৈরি করে।
    - আমাদের সকল প্রিট্রেইনড মডেল ব্যবহারের জন্য একটি ইউনিফাইড API।

2. কম কম্পিউট খরচ, ছোট কার্বন পদচিহ্ন:
    - নতুন করে প্রশিক্ষণ না দিয়ে প্রশিক্ষিত মডেলগুলি শেয়ার করুন।
    - কম্পিউট সময় এবং উৎপাদন খরচ হ্রাস করুন।
    - ডজন ডজন মডেল আর্কিটেকচার এবং সকল মোডালিটিতে ১ মিলিয়নেরও বেশি প্রিট্রেইনড চেকপয়েন্ট।

3. একটি মডেলের জীবনের প্রতিটি অংশের জন্য সঠিক ফ্রেমওয়ার্ক বেছে নিন:
    - মাত্র ৩ লাইনের কোডে অত্যাধুনিক মডেল প্রশিক্ষণ দিন।
    - ইচ্ছামতো একটি একক মডেল PyTorch/JAX/TF2.0 ফ্রেমওয়ার্কের মধ্যে স্থানান্তর করুন।
    - প্রশিক্ষণ, মূল্যায়ন এবং উৎপাদনের জন্য সঠিক ফ্রেমওয়ার্ক বেছে নিন।

4. আপনার প্রয়োজন অনুযায়ী একটি মডেল বা উদাহরণ সহজেই কাস্টমাইজ করুন:
    - আমরা প্রতিটি আর্কিটেকচারের জন্য উদাহরণ সরবরাহ করি যাতে এর মূল লেখকদের প্রকাশিত ফলাফলগুলি পুনরুৎপাদন করা যায়।
    - মডেলের অভ্যন্তরীণ অংশগুলি যতটা সম্ভব সামঞ্জস্যপূর্ণভাবে উন্মুক্ত করা হয়।
    - দ্রুত পরীক্ষা-নিরীক্ষার জন্য লাইব্রেরি থেকে স্বাধীনভাবে মডেল ফাইলগুলি ব্যবহার করা যেতে পারে।

<a target="_blank" href="https://huggingface.co/enterprise">
<img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## আমি কেন Transformers ব্যবহার করব না?

- এই লাইব্রেরিটি নিউরাল নেটের বিল্ডিং ব্লকের একটি মডুলার টুলবক্স নয়। মডেল ফাইলগুলির কোড অতিরিক্ত বিমূর্ততা সহ রিফ্যাক্টর করা হয়নি, যাতে গবেষকরা অতিরিক্ত বিমূর্ততা/ফাইলগুলিতে না গিয়ে প্রতিটি মডেলের উপর দ্রুত পুনরাবৃত্তি করতে পারেন।

- প্রশিক্ষণের API টি Transformers দ্বারা সরবরাহকৃত PyTorch মডেলগুলির সাথে কাজ করার জন্য অপটিমাইজ করা হয়েছে। সাধারণ মেশিন লার্নিং লুপগুলির জন্য, আপনার Accelerate এর মতো অন্য একটি লাইব্রেরি ব্যবহার করা উচিত।

- [উদাহরণ স্ক্রিপ্টগুলি](https://github.com/huggingface/transformers/tree/main/examples) কেবল *উদাহরণ*। এগুলি আপনার নির্দিষ্ট ব্যবহারের ক্ষেত্রে সরাসরি কাজ নাও করতে পারে এবং এটি কাজ করার জন্য আপনাকে কোডটি পরিবর্তন করতে হবে।

## Transformers ব্যবহার করে ১০০টি প্রজেক্ট

Transformers শুধুমাত্র প্রিট্রেইনড মডেল ব্যবহারের জন্য একটি টুলকিট নয়, এটি এর চারপাশে এবং Hugging Face Hub-এর একটি কমিউনিটি প্রজেক্ট। আমরা চাই Transformers ডেভেলপার, গবেষক, ছাত্র, অধ্যাপক, প্রকৌশলী এবং অন্য যেকোনো ব্যক্তিকে তাদের স্বপ্নের প্রজেক্ট তৈরি করতে সক্ষম করুক।

Transformers ১,০০,০০০ স্টার উদযাপন করতে, আমরা [awesome-transformers](../awesome-transformers.md) পৃষ্ঠায় কমিউনিটির উপর আলোকপাত করতে চেয়েছিলাম, যেখানে Transformers দিয়ে তৈরি ১০০টি অসাধারণ প্রজেক্টের তালিকা রয়েছে।

যদি আপনার এমন কোনো প্রজেক্ট থাকে যা আপনি মনে করেন এই তালিকার অংশ হওয়া উচিত, তবে দয়া করে এটি যুক্ত করার জন্য একটি PR (পুল রিকোয়েস্ট) খুলুন!

## উদাহরণের মডেলসমূহ

আপনি আমাদের বেশিরভাগ মডেল সরাসরি তাদের [Hub model pages](https://huggingface.co/models) এ পরীক্ষা করতে পারেন।

বিভিন্ন ব্যবহারের ক্ষেত্রে কিছু উদাহরণের মডেল দেখতে প্রতিটি মোডালিটি নিচে সম্প্রসারণ করুন।

<details>
<summary>অডিও</summary>

- [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) দিয়ে অডিও শ্রেণীকরণ

- [Moonshine](https://huggingface.co/UsefulSensors/moonshine) দিয়ে স্বয়ংক্রিয় বক্তৃতা স্বীকৃতি
- [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks) দিয়ে কীওয়ার্ড স্পটিং
- [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) দিয়ে স্পিচ-টু-স্পিচ জেনারেশন
- [MusicGen](https://huggingface.co/facebook/musicgen-large) দিয়ে টেক্সট-টু-অডিও
- [Bark](https://huggingface.co/suno/bark) দিয়ে টেক্সট-টু-স্পিচ

</details>

<details>
<summary>কম্পিউটার ভিশন</summary>

- [SAM](https://huggingface.co/facebook/sam-vit-base) দিয়ে স্বয়ংক্রিয় মাস্ক জেনারেশন
- [DepthPro](https://huggingface.co/apple/DepthPro-hf) দিয়ে গভীরতা অনুমান
- [DINO v2](https://huggingface.co/facebook/dinov2-base) দিয়ে চিত্র শ্রেণীকরণ
- [SuperPoint](https://huggingface.co/magic-leap-community/superpoint) দিয়ে কীপয়েন্ট সনাক্তকরণ
- [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor) দিয়ে কীপয়েন্ট ম্যাচিং
- [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) দিয়ে অবজেক্ট সনাক্তকরণ
- [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple) দিয়ে পোস অনুমান
- [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) দিয়ে ইউনিভার্সাল সেগমেন্টেশন
- [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large) দিয়ে ভিডিও শ্রেণীকরণ

</details>

<details>
<summary>মাল্টিমোডাল</summary>

- [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B) দিয়ে অডিও বা টেক্সট থেকে টেক্সট
- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) দিয়ে ডকুমেন্ট প্রশ্ন জিজ্ঞাসা
- [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) দিয়ে চিত্র বা টেক্সট থেকে টেক্সট
- [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) দিয়ে চিত্র ক্যাপশনিং
- [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) দিয়ে OCR-ভিত্তিক ডকুমেন্ট বোঝা
- [TAPAS](https://huggingface.co/google/tapas-base) দিয়ে টেবিল প্রশ্ন জিজ্ঞাসা
- [Emu3](https://huggingface.co/BAAI/Emu3-Gen) দিয়ে ইউনিফাইড মাল্টিমোডাল বোঝা এবং জেনারেশন
- [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) দিয়ে ভিশন-টু-টেক্সট
- [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) দিয়ে ভিজ্যুয়াল প্রশ্ন জিজ্ঞাসা
- [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) দিয়ে ভিজ্যুয়াল রেফারেন্সিং এক্সপ্রেশন সেগমেন্টেশন

</details>

<details>
<summary>NLP</summary>

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) দিয়ে মাস্কড ওয়ার্ড কমপ্লিশন
- [Gemma](https://huggingface.co/google/gemma-2-2b) দিয়ে নেমড এন্টিটি রিকগনিশন
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) দিয়ে প্রশ্ন জিজ্ঞাসা
- [BART](https://huggingface.co/facebook/bart-large-cnn) দিয়ে সারসংক্ষেপ
- [T5](https://huggingface.co/google-t5/t5-base) দিয়ে অনুবাদ
- [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B) দিয়ে টেক্সট জেনারেশন
- [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B) দিয়ে টেক্সট শ্রেণীকরণ

</details>

## সাইটেশন
এখন আমাদের কাছে 🤗 Transformers লাইব্রেরির জন্য একটি [পেপার](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) আছে যা আপনি উদ্ধৃত করতে পারেন:

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