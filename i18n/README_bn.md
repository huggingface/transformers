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
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_it.md">Italiano</a> |
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


**Transformers** হলো একটা ফ্রেমওয়ার্ক যেটা দিয়ে টেক্সট, কম্পিউটার ভিশন, অডিও, ভিডিও আর মাল্টিমোডাল—সব ধরনের মডেল তৈরি আর চালানো যায়। এটা ট্রেইনিং আর ইনফারেন্স – দুই কাজেই ব্যবহার করা হয়।

Transformers মডেলের ডেফিনিশন এক জায়গায় রাখে। এর মানে হলো, একবার কোনো মডেল `transformers`-এ সাপোর্ট পেলেই সেটা সহজে বিভিন্ন ট্রেইনিং ফ্রেমওয়ার্ক (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning ইত্যাদি), ইনফারেন্স ইঞ্জিন (vLLM, SGLang, TGI ইত্যাদি) আর অন্যান্য লাইব্রেরি (llama.cpp, mlx ইত্যাদি)-তে ব্যবহার করা যায়।

আমরা চাই নতুন আর আধুনিক মডেলগুলো সবাই ব্যবহার করতে পারে। তাই মডেলের ডেফিনিশন রাখা হয়েছে সহজ, কাস্টমাইজযোগ্য আর পারফরম্যান্স-ফ্রেন্ডলি।

এখন পর্যন্ত [Hugging Face Hub](https://huggingface.com/models)-এ ১০ লাখেরও বেশি Transformers [মডেল চেকপয়েন্ট](https://huggingface.co/models?library=transformers&sort=trending) আছে, যেগুলো যেকোনো সময় ব্যবহার করা যায়।

আজই [Hub](https://huggingface.com/) থেকে একটা মডেল বেছে নিন আর Transformers দিয়ে শুরু করুন।


## ইনস্টলেশন

Transformers Python 3.10+ সহ কাজ করে, এবং [PyTorch](https://pytorch.org/get-started/locally/) 2.4+।

[venv](https://docs.python.org/3/library/venv.html) বা [uv](https://docs.astral.sh/uv/) ব্যবহার করে একটি ভার্চুয়াল এনভায়রনমেন্ট তৈরি এবং সক্রিয় করুন।

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

Transformers ব্যবহার শুরু করুন এখনই [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API দিয়ে। `Pipeline` হলো একটি হাই-লেভেল ইনফারেন্স ক্লাস, যা টেক্সট, অডিও, ভিশন এবং মাল্টিমোডাল টাস্ক সাপোর্ট করে। এটি ইনপুট প্রিপ্রসেসিং করে এবং সঠিক আউটপুট রিটার্ন করে।

একটি পাইপলাইন তৈরি করুন এবং টেক্সট জেনারেশনের জন্য কোন মডেল ব্যবহার করবেন তা নির্দিষ্ট করুন। মডেলটি ডাউনলোড হয়ে ক্যাশে রাখা হবে, ফলে পরে সহজেই আবার ব্যবহার করতে পারবেন। সবশেষে, মডেলকে প্রম্পট করার জন্য কিছু টেক্সট দিন।


```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

মডেলের সাথে চ্যাট করতে হলেও ব্যবহার প্যাটার্ন একই। শুধু পার্থক্য হলো, আপনাকে একটি চ্যাট হিস্ট্রি তৈরি করতে হবে (যা `Pipeline`-এ ইনপুট হিসেবে যাবে) আপনার আর সিস্টেমের মধ্যে।

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
<summary>অটোমেটিক স্পিচ রিকগনিশন (ASR)</summary>

```Python
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("[https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac](https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac)")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>ইমেজ ক্লাসিফিকেশন</summary>

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
<summary>ভিজুয়াল কোয়েশ্চন আনসারিং</summary>

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

## কেন Transformers ব্যবহার করবেন?

1. সহজে ব্যবহারযোগ্য সর্বাধুনিক মডেল:

   * ন্যাচারাল ল্যাঙ্গুয়েজ আন্ডারস্ট্যান্ডিং ও জেনারেশন, কম্পিউটার ভিশন, অডিও, ভিডিও এবং মাল্টিমোডাল টাস্কে উচ্চ পারফরম্যান্স।
   * গবেষক, ইঞ্জিনিয়ার এবং ডেভেলপারদের জন্য সহজে শুরু করার সুযোগ।
   * মাত্র তিনটি ক্লাস শিখলেই ব্যবহার করা যায়।
   * সব প্রি-ট্রেইন্ড মডেলের জন্য একটি একীভূত API।

2. কম কম্পিউট খরচ, ছোট কার্বন ফুটপ্রিন্ট:

   * শূন্য থেকে ট্রেইন না করে ট্রেইন্ড মডেল শেয়ার করুন।
   * কম্পিউট টাইম ও প্রোডাকশন খরচ কমান।
   * সব ধরনের মোডালিটির জন্য ১০ লক্ষ+ প্রি-ট্রেইন্ড চেকপয়েন্টসহ ডজনখানেক মডেল আর্কিটেকচার।

3. মডেলের লাইফসাইকেলের প্রতিটি ধাপে সঠিক ফ্রেমওয়ার্ক বেছে নিন:

   * মাত্র ৩ লাইনের কোডে সর্বাধুনিক মডেল ট্রেইন করুন।
   * সহজে PyTorch / JAX / TF2.0 এর মধ্যে মডেল স্থানান্তর করুন।
   * ট্রেইনিং, ইভ্যালুয়েশন ও প্রোডাকশনের জন্য আলাদা ফ্রেমওয়ার্ক ব্যবহার করুন।

4. সহজেই মডেল বা উদাহরণ কাস্টমাইজ করুন:

   * প্রতিটি আর্কিটেকচারের জন্য এমন উদাহরণ দেওয়া আছে যা মূল লেখকদের প্রকাশিত ফলাফল পুনরুত্পাদন করতে সক্ষম।
   * মডেলের অভ্যন্তরীণ অংশগুলো যতটা সম্ভব একভাবে এক্সপোজ করা হয়েছে।
   * দ্রুত এক্সপেরিমেন্টের জন্য লাইব্রেরি ছাড়াও মডেল ফাইল ব্যবহার করা যায়।


<a target="_blank" href="https://huggingface.co/enterprise">
<img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## কেন Transformers ব্যবহার করবেন না?

* এই লাইব্রেরি নিউরাল নেটওয়ার্কের জন্য ব্লক-মডিউল টুলবক্স নয়। মডেল ফাইলের কোডে অতিরিক্ত অ্যাবস্ট্র্যাকশন intentionally করা হয়নি, যাতে গবেষকরা দ্রুত প্রতিটি মডেলের উপর কাজ করতে পারে কোনো অতিরিক্ত ফাইল বা স্তরে না গিয়ে।
* ট্রেইনিং API মূলত Transformers-এর PyTorch মডেলের সাথে কাজ করার জন্য অপটিমাইজ করা হয়েছে। সাধারণ মেশিন লার্নিং লুপের জন্য, [Accelerate](https://huggingface.co/docs/accelerate) এর মতো অন্য লাইব্রেরি ব্যবহার করা উচিত।
* [উদাহরণ স্ক্রিপ্টগুলো](https://github.com/huggingface/transformers/tree/main/examples) শুধু *উদাহরণ*। এগুলো সরাসরি আপনার ব্যবহারের ক্ষেত্রে কাজ নাও করতে পারে, তাই কোড সামঞ্জস্য করতে হতে পারে।

## Transformers দিয়ে ১০০টি প্রজেক্ট

Transformers শুধু প্রি-ট্রেইন্ড মডেল ব্যবহার করার টুলকিট নয়, এটি একটি কমিউনিটি, যা Hugging Face Hub-এর চারপাশে তৈরি। আমরা চাই যে ডেভেলপার, গবেষক, শিক্ষার্থী, অধ্যাপক, ইঞ্জিনিয়ার বা যে কেউ তাদের স্বপ্নের প্রজেক্ট তৈরি করতে পারে।

Transformers 100,000 স্টার উদযাপন করতে আমরা কমিউনিটিকে তুলে ধরতে [awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md) পেজ তৈরি করেছি, যেখানে Transformers দিয়ে তৈরি ১০০টি অসাধারণ প্রজেক্ট তালিকাভুক্ত আছে।

আপনার কোনো প্রজেক্ট আছে যা তালিকায় থাকা উচিত মনে করেন? তাহলে PR খুলে যুক্ত করুন।

## উদাহরণ মডেল

আপনি আমাদের অধিকাংশ মডেল সরাসরি তাদের [Hub মডেল পেজ](https://huggingface.co/models) থেকে পরীক্ষা করতে পারেন।

নিচের প্রতিটি মোডালিটি এক্সপ্যান্ড করে বিভিন্ন ব্যবহার কেসের জন্য কয়েকটি উদাহরণ মডেল দেখুন।


<details>
<summary>অডিও</summary>

* [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) দিয়ে অডিও ক্লাসিফিকেশন
* [Moonshine](https://huggingface.co/UsefulSensors/moonshine) দিয়ে অটোমেটিক স্পিচ রিকগনিশন
* [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks) দিয়ে কীওয়ার্ড স্পটিং
* [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) দিয়ে স্পিচ-টু-স্পিচ জেনারেশন
* [MusicGen](https://huggingface.co/facebook/musicgen-large) দিয়ে টেক্সট-টু-অডিও
* [Bark](https://huggingface.co/suno/bark) দিয়ে টেক্সট-টু-স্পিচ


</details>

<details>
<summary>কম্পিউটার ভিশন</summary>

* [SAM](https://huggingface.co/facebook/sam-vit-base) দিয়ে স্বয়ংক্রিয় মাস্ক জেনারেশন
* [DepthPro](https://huggingface.co/apple/DepthPro-hf) দিয়ে গভীরতা অনুমান
* [DINO v2](https://huggingface.co/facebook/dinov2-base) দিয়ে চিত্র শ্রেণীকরণ
* [SuperPoint](https://huggingface.co/magic-leap-community/superpoint) দিয়ে কীপয়েন্ট সনাক্তকরণ
* [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor) দিয়ে কীপয়েন্ট ম্যাচিং
* [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) দিয়ে অবজেক্ট সনাক্তকরণ
* [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple) দিয়ে পোস অনুমান
* [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) দিয়ে ইউনিভার্সাল সেগমেন্টেশন
* [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large) দিয়ে ভিডিও শ্রেণীকরণ


</details>

<details>
<summary>মাল্টিমোডাল</summary>

* [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B) দিয়ে অডিও বা টেক্সট থেকে টেক্সট জেনারেশন
* [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) দিয়ে ডকুমেন্ট প্রশ্নোত্তর
* [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) দিয়ে ইমেজ বা টেক্সট থেকে টেক্সট জেনারেশন
* [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) দিয়ে ইমেজ ক্যাপশনিং
* [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) দিয়ে OCR-ভিত্তিক ডকুমেন্ট আন্ডারস্ট্যান্ডিং
* [TAPAS](https://huggingface.co/google/tapas-base) দিয়ে টেবিল প্রশ্নোত্তর
* [Emu3](https://huggingface.co/BAAI/Emu3-Gen) দিয়ে ইউনিফাইড মাল্টিমোডাল আন্ডারস্ট্যান্ডিং এবং জেনারেশন
* [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) দিয়ে ভিশন থেকে টেক্সট
* [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) দিয়ে ভিজুয়াল কোয়েশ্চন আনসারিং
* [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) দিয়ে ভিজুয়াল রেফারিং এক্সপ্রেশন সেগমেন্টেশন


</details>

<details>
<summary>NLP</summary>

* [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) দিয়ে মাস্কড ওয়ার্ড কমপ্লিশন
* [Gemma](https://huggingface.co/google/gemma-2-2b) দিয়ে নাম্বড এন্টিটি রিকগনিশন
* [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) দিয়ে প্রশ্নোত্তর
* [BART](https://huggingface.co/facebook/bart-large-cnn) দিয়ে সারসংক্ষেপ (Summarization)
* [T5](https://huggingface.co/google-t5/t5-base) দিয়ে অনুবাদ
* [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B) দিয়ে টেক্সট জেনারেশন
* [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B) দিয়ে টেক্সট ক্লাসিফিকেশন

</details>

## সাইটেশন
আমাদের [একটি পেপার](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) আছে যা আপনি 🤗 Transformers লাইব্রেরির জন্য রেফারেন্স হিসেবে ব্যবহার করতে পারেন।

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