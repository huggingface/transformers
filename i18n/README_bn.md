<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

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
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <b>বাংলা</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p>JAX, PyTorch এবং TensorFlow-এর জন্য অত্যাধুনিক মেশিন লার্নিং</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers হাজারো প্রি-ট্রেইন্ড মডেল প্রদান করে, যা টেক্সট, ইমেজ এবং অডিওর মতো বিভিন্ন modality-তে টাস্ক সম্পাদনের জন্য ব্যবহৃত হয়।

এই মডেলগুলো ব্যবহার করা যায়:

* 📝 টেক্সট — টেক্সট ক্লাসিফিকেশন, তথ্য আহরণ, প্রশ্নোত্তর, স্বয়ংক্রিয় সারাংশ, মেশিন ট্রান্সলেশন এবং ১০০টিরও বেশি ভাষায় টেক্সট জেনারেশনের মতো টাস্কে।
* 🖼️ ছবি — ছবি শ্রেণিবিন্যাস, অবজেক্ট শনাক্তকরণ এবং সেগমেন্টেশনের মতো টাস্কে।
* 🗣️ অডিও — স্পিচ রিকগনিশন ও অডিও ক্লাসিফিকেশনের মতো টাস্কে।

Transformer মডেলগুলো **বহু modality-এর সংমিশ্রণেও** বিভিন্ন কাজ করতে পারে, যেমন: টেবিলভিত্তিক প্রশ্নোত্তর, অপটিক্যাল ক্যারেক্টার রিকগনিশন (OCR), স্ক্যানকৃত ডকুমেন্ট থেকে তথ্য আহরণ, ভিডিও শ্রেণিবিন্যাস ও ভিজ্যুয়াল প্রশ্নোত্তর।

🤗 Transformers খুব দ্রুত প্রি-ট্রেইন্ড মডেল ডাউনলোড ও টেক্সটের জন্য ব্যবহারের API দেয়, নিজের ডেটাসেট-এ ফাইন-টিউন করতে এবং আমাদের [Model Hub](https://huggingface.co/models)-এ কমিউনিটির সাথে শেয়ার করতে সাহায্য করে। একই সময়ে, প্রতিটি Python মডিউল যার মাধ্যমে আর্কিটেকচার সংজ্ঞায়িত, সম্পূর্ণ স্বতন্ত্র ও সম্পাদনযোগ্য, যাতে ঝটপট গবেষণামূলক পরীক্ষা-নিরীক্ষা করা যায়।

🤗 Transformers তিনটি জনপ্রিয় ডিপ লার্নিং লাইব্রেরি—[Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/), এবং [TensorFlow](https://www.tensorflow.org/)-এর সাথে সহজ ইন্টিগ্রেশন সাপোর্ট করে। এক ফ্রেমওয়ার্কে মডেল ট্রেন করুন এবং সহজেই আরেকটিতে inference করুন।

## অনলাইন ডেমো

আপনি আমাদের বেশিরভাগ মডেল সরাসরি [Model Hub](https://huggingface.co/models)-এ তাদের নিজ নিজ পেজে পরীক্ষা করতে পারবেন। আমরা [প্রাইভেট মডেল হোস্টিং, ভার্সনিং, এবং ইনফারেন্স API](https://huggingface.co/pricing) পাবলিক ও প্রাইভেট মডেলের জন্য প্রদান করি।

এখানে কিছু উদাহরণ:

কম্পিউটার লিঙ্গুইস্টিকসে:

- [BERT দিয়ে মাস্কড ওয়ার্ড কমপ্লিশন](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Electra দিয়ে নিজ নাম সনাক্তকরণ](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [GPT-2 দিয়ে টেক্সট জেনারেশন](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [RoBERTa দ্বারা ন্যাচারাল ল্যাংগুয়েজ ইনফারেন্স](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [BART দিয়ে স্বয়ংক্রিয় টেক্সট সারাংশ](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [DistilBERT দিয়ে প্রশ্নোত্তর](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [T5 দিয়ে মেশিন অনুবাদ](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

কম্পিউটার ভিশনে:

- [ViT দিয়ে ইমেজ ক্লাসিফিকেশন](https://huggingface.co/google/vit-base-patch16-224)
- [DETR দিয়ে অবজেক্ট শনাক্তকরণ](https://huggingface.co/facebook/detr-resnet-50)
- [SegFormer দিয়ে সেমান্টিক সেগমেন্টেশন](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [MaskFormer দিয়ে প্যানোপটিক সেগমেন্টেশন](https://huggingface.co/facebook/maskformer-swin-small-coco)
- [DPT দিয়ে ডেপ্থ এস্টিমেশন](https://huggingface.co/docs/transformers/model_doc/dpt)
- [VideoMAE দিয়ে ভিডিও ক্লাসিফিকেশন](https://huggingface.co/docs/transformers/model_doc/videomae)
- [OneFormer দিয়ে ইউনিভার্সাল সেগমেন্টেশন](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

অডিও বিভাগে:

- [Wav2Vec2 দিয়ে স্বয়ংক্রিয় স্পিচ রিকগনিশন](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Wav2Vec2 দিয়ে কীওয়ার্ড শনাক্তকরণ](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Audio Spectrogram Transformer দিয়ে অডিও ক্লাসিফিকেশন](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

মাল্টিমোডাল টাস্কে:

- [TAPAS দিয়ে টেবিলভিত্তিক প্রশ্নোত্তর](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [ViLT দিয়ে ভিজ্যুয়াল প্রশ্নোত্তর](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [CLIP দিয়ে জিরো-শট ইমেজ ক্লাসিফিকেশন](https://huggingface.co/openai/clip-vit-large-patch14)
- [LayoutLM দিয়ে ডকুমেন্ট-ভিত্তিক প্রশ্নোত্তর](https://huggingface.co/impira/layoutlm-document-qa)
- [X-CLIP দিয়ে জিরো-শট ভিডিও ক্লাসিফিকেশন](https://huggingface.co/docs/transformers/model_doc/xclip)

## 🤗 Transformers- ব্যবহারকারী ১০০টি প্রকল্প

🤗 Transformers শুধুমাত্র প্রি-ট্রেইন্ড মডেল ব্যবহারের টুলকিট নয়: এটি প্রজেক্টগুলোর একটি কমিউনিটি, যেগুলো Hugging Face Hub এর চারপাশে গড়ে উঠেছে। আমরা চাই, 🤗 Transformers ডেভেলপার, গবেষক, ছাত্র, শিক্ষক, ইঞ্জিনিয়ার ও সবাইকে তাদের স্বপ্নের প্রকল্প বাস্তবায়নে সহায়তা করুক।

🤗 Transformers-এর ১০০,০০০ স্টার উদযাপনের জন্য, আমরা কমিউনিটিকে সামনে এনে [awesome-transformers](./awesome-transformers.md) পেজটি তৈরি করেছি, যেখানে 🤗 Transformers দিয়ে করা ১০০টি অসাধারণ প্রকল্পের তালিকা দিয়েছি।

যদি আপনার কাছে এমন কোনো প্রকল্প থাকে বা আপনি ব্যবহার করেন, যেটি এই তালিকায় থাকা উচিত বলে মনে করেন, দয়া করে সেটি যোগ করতে একটি PR (Pull Request) খুলুন!


## আপনি যদি Hugging Face টিমের কাছ থেকে ব্যক্তিগত সহায়তা চান

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## দ্রুত শুরু করুন

নির্দিষ্ট কোনো ইনপুট (টেক্সট, ছবি, অডিও ...) নিয়ে দ্রুত কোনো মডেল ব্যবহার করতে চাইলে আমরা `pipeline`-API সরবরাহ করি। পাইপলাইন একটি প্রি-ট্রেইন্ড মডেল ও তার সাথে ব্যবহৃত প্রিপ্রসেসিংকে একত্রিত করে, যা ট্রেনিংয়ের সময় কাজে লাগানো হয়েছিল। নিচে দেখানো হয়েছে, কীভাবে দ্রুত একটি পাইপলাইন ব্যবহার করে ইতিবাচক এবং নেতিবাচক টেক্সট শ্রেণিবিন্যাস করা যায়:



```python
>>> from transformers import pipeline

# সেন্টিমেন্ট বিশ্লেষণের জন্য একটি পাইপলাইন বরাদ্দ করি
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```


দ্বিতীয় কোডলাইনে পাইপলাইন ব্যবহারের জন্য প্রি-ট্রেইন্ড মডেল লোড এবং ক্যাশ করা হয়, এবং তৃতীয় লাইনে দেয়া টেক্সটে সেটা পরীক্ষা করা হয়। এখানে উত্তরটি "ইতিবাচক" ৯৯.৯৭% আত্মবিশ্বাসসহ।

কম্পিউটার ভাষাতত্ত্ব, কম্পিউটার ভিশন এবং স্পিচ প্রসেসিংয়ে অনেক টাস্কের জন্যই প্রস্তুত করা প্রি-ট্রেইন্ড `pipeline` রয়েছে। যেমন, আমরা সহজেই কোনো ছবিতে শনাক্ত হওয়া অবজেক্ট বের করতে পারি:



``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# সুন্দর বিড়ালের ছবি ডাউনলোড করুন
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# অবজেক্ট শনাক্তকরণের জন্য পাইপলাইন
>>> object_detector = pipeline('object-detection')
>>> object_detector(image)
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```


এখানে আমরা ছবিতে শনাক্ত হওয়া অবজেক্টের একটি তালিকা পাই, যেগুলোর সাথে বাউন্ডিং বাক্স ও আত্মবিশ্বাসের মানসহ উপস্থাপিত হয়। নিচে বামে মূল ছবি এবং ডানে পূর্বাভাস দেখানো হয়েছে:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

`pipeline`-API কোন কোন টাস্ক সাপোর্ট করে, তা [এই টিউটোরিয়ালে](https://huggingface.co/docs/transformers/task_summary) জানতে পারবেন।

`pipeline` ছাড়াও মাত্র তিনটি কোডলাইনেই যেকোনো প্রি-ট্রেইন্ড মডেল নামিয়ে ব্যবহার করা যায়। নিচে PyTorch-এর জন্য একটি উদাহরণ:



```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

এবং এটি TensorFlow-এর জন্য একই উদাহরণ:

```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```


Tokenizer-এর কাজ হচ্ছে পরিপ্রেক্ষিত অনুসারে প্রি-প্রসেসিং করা, যা মডেলের জন্য দরকার হয়—এটা একক স্ট্রিং বা একটি লিস্টের ওপর সরাসরি চলতে পারে। এটি একটি ডিকশনারি আউটপুট দেয়, যেটি পরবর্তী কোডে ব্যবহার করা যেতে পারে বা সরাসরি মডেলে পাঠানো যেতে পারে (Python-এর ** অপারেটর দিয়ে)।

মডেল নিজেই PyTorch-এর [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) অথবা TensorFlow-এর [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (ব্যাকএন্ড অনুযায়ী), যেটি চেনা নিয়ম মেনে চালানো যায়। [এই টিউটোরিয়ালটি](https://huggingface.co/docs/transformers/training) দেখুন, কিভাবে ট্রেনিং লুপ বা আমাদের `Trainer`-API দিয়ে অসংখ্য ডেটাসেটে দ্রুত ফাইন-টিউন করা যায়।

## কেন আপনি 🤗 Transformers ব্যবহার করবেন?

1. ব্যবহার-বান্ধব আধুনিক মডেল:
    - ন্যাচারাল ল্যাঙ্গুয়েজ আন্ডারস্ট্যান্ডিং ও জেনারেশন, কম্পিউটার ভিশন ও অডিও টাস্কগুলোর জন্য উচ্চ দক্ষতা।
    - শিক্ষার্থী ও চর্চাকারীদের জন্য সহজ প্রবেশযোগ্যতা।
    - শুধুমাত্র তিনটি মূল ক্লাস শিখলেই হবে।
    - আমাদের সব প্রি-ট্রেইন্ড মডেল ব্যবহারের জন্য একক API।

2. কম কম্পিউটিং খরচ, ছোট CO<sub>2</sub> ফ্লুটপ্রিন্ট:
    - গবেষকরা তাদের প্রশিক্ষিত মডেল শেয়ার করতে পারে, বারবার ট্রেন করতে হয় না।
    - প্র‍্যাক্টিশনাররা কম সময়ে ও কম খরচে কাজ শেষ করতে পারেন।
    - ডজন খানেক আর্কিটেকচার ও ৪ লাখেরও বেশি প্রি-ট্রেইন্ড মডেল সব modality-এর জন্য।

3. মডেল নির্মাণের প্রতিটি ধাপে পছন্দের ফ্রেমওয়ার্ক বেছে নিন:
    - মাত্র ৩টে কোডলাইনে আধুনিক মডেল ট্রেনিং।
    - TF2.0, PyTorch বা JAX-এ মুক্তভাবে একই মডেল ব্যবহার করুন।
    - ট্রেনিং, মূল্যায়ন ও প্রোডাকশনের জন্য সহজেই সঠিক ফ্রেমওয়ার্ক বেছে নিন।

4. সহজেই কাস্টমাইজ করুন:
    - প্রতিটি আর্কিটেকচারের জন্য আমাদের কাছে রেফারেন্স এক্সাম্পল রয়েছে, মূল লেখকদের ফলাফল পুনরুত্পাদনের জন্য।
    - মডেলের আভ্যন্তরীণ গঠন যতটা সম্ভব অভিন্ন রাখা হয়েছে।
    - মডেল ফাইল লাইব্রেরি ছাড়াও স্বাধীনভাবে গবেষণার জন্য ব্যবহার করা যাবে।

## কখন 🤗 Transformers ব্যবহার করবেন না?

- এই লাইব্রেরিটি নিরেট নিউরাল নেটওয়ার্ক বিল্ডিং ব্লক সরবরাহ করে না, বরং নির্দিষ্ট মডেলের দ্রুত উন্নয়ন ও গবেষণার জন্য নকশা করা হয়েছে, যাতে আপনাকে অপ্রয়োজনীয় অ্যাবস্ট্রাকশনে না যেতে হয়।
- Training API সব ধরণের মডেলের জন্য নয়; এটি বিশেষভাবে লাইব্রেরির নিজস্ব মডেলগুলোর জন্য অপ্টিমাইজড। সাধারণ মেশিন লার্নিং ট্রেনিং লুপের জন্য অন্য লাইব্রেরি ব্যবহার করুন (যেমন [Accelerate](https://huggingface.co/docs/accelerate))।
- আমাদের [`examples`](./examples) ফোল্ডার-এ থাকা স্ক্রিপ্টগুলো মূলত নমুনা; এগুলো সরাসরি আপনার প্রজেক্টে চলবে নাও। আপনাকে কিছু কোড পরিবর্তন করতে হতে পারে।

## ইনস্টলেশন

### pip দিয়ে

এই রেপোজিটরিটিতে Python 3.9+, Flax 0.4.1+, PyTorch 2.1+, এবং TensorFlow 2.6+ দিয়ে পরীক্ষা করা হয়েছে।

আপনি [ভার্চুয়াল এনভায়রনমেন্টে](https://docs.python.org/3/library/venv.html) 🤗 Transformers ইনস্টল করার পরামর্শ দেওয়া হয়। ভার্চুয়াল এনভায়রনমেন্টের সাথে পরিচিত না হলে [ব্যবহারকারী গাইড](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) দেখুন।

প্রথমে, আপনি ব্যবহার করতে চাওয়া Python ভার্সন দিয়ে এনভায়রনমেন্ট তৈরি ও অ্যাক্টিভেট করুন।

এরপর Flax, PyTorch বা TensorFlow যেটিই আপনার দরকার সেটি ইনস্টল করুন। [TensorFlow](https://www.tensorflow.org/install/), [PyTorch](https://pytorch.org/get-started/locally/#start-locally), [Flax](https://github.com/google/flax#quick-install) এবং [Jax](https://github.com/google/jax#installation) এর নির্দিষ্ট ইনস্টলেশন গাইড রেফার করুন।

যেকোনো একটি Backend ইনস্টল থাকার পর, 🤗 Transformers এইভাবে pip ব্যবহার করে ইনস্টল করুন:



```bash
pip install transformers
```


আপনি যদি উদাহরণ স্ক্রিপ্ট বা কোডের সর্বশেষ সংস্করণ চান এবং নতুন রিলিজ না আসা পর্যন্ত অপেক্ষা করতে না চান, তাহলে [সোর্স থেকে লাইব্রেরি ইনস্টল](https://huggingface.co/docs/transformers/installation#installing-from-source) করুন।

### conda দিয়ে

conda ব্যবহার করেও 🤗 Transformers ইনস্টল করা যায়:



```shell script
conda install conda-forge::transformers
```


> **_নোট:_** `huggingface`-চ্যানেল থেকে `transformers` ইনস্টল করা পুরনো পদ্ধতি।

Flax, PyTorch, বা TensorFlow-এর ইনস্টলেশন জানতে তাদের অফিসিয়াল গাইড দেখুন।

> **_নোট:_** উইন্ডোজে কেশিং সুবিধা নিতে আপনাকে developers' mode চালু করতে বলা হতে পারে। এটি না পারলে [এই ইস্যুতে](https://github.com/huggingface/huggingface_hub/issues/1062) জানান।

## মডেল আর্কিটেকচার

**[সব মডেল-চেকপয়েন্ট](https://huggingface.co/models)**, যা 🤗 Transformers সরবরাহ করে, huggingface.co [Model Hub](https://huggingface.co/models) থেকে সরাসরি ব্যবহার করা যায়, ব্যবহারকারী ও সংগঠন উভয়ই সেখানে আপলোড করতে পারেন।

বর্তমানে চেকপয়েন্ট সংখ্যা: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers বর্তমানে নিম্নোক্ত আর্কিটেকচার সরবরাহ করে: বিস্তারিত দেখতে [এখানে ক্লিক করুন](https://huggingface.co/docs/transformers/model_summary)।

প্রত্যেক মডেলে Flax, PyTorch বা TensorFlow বাস্তবায়ন আছে কিনা এবং 🤗 Tokenizers দ্বারা সমর্থিত টোকেনাইজার রয়েছে কিনা জানতে, [এই টেবিল](https://huggingface.co/docs/transformers/index#supported-frameworks) দেখুন।

এইসব বাস্তবায়ন বিভিন্ন ডেটাসেটে পরীক্ষা করা হয়েছে (উদাহরণ স্ক্রিপ্ট দেখুন) এবং মূল বাস্তবায়নের ফলাফলের সাথে মিলে যাওয়ার কথা। আরও বিস্তারিত জানতে উদাহরণ সেকশন ও [ডকুমেন্টেশন](https://github.com/huggingface/transformers/tree/main/examples) চেক করুন।

## আরও জানুন

| বিভাগ | বর্ণনা |
|---|---|
| [ডকুমেন্টেশন](https://huggingface.co/docs/transformers/) | সম্পূর্ণ API ডকুমেন্টেশন ও টিউটোরিয়াল |
| [কাজের সংক্ষিপ্ত তালিকা](https://huggingface.co/docs/transformers/task_summary) | 🤗 Transformers দ্বারা সমর্থিত টাস্ক |
| [প্রিপ্রসেসিং টিউটোরিয়াল](https://huggingface.co/docs/transformers/preprocessing) | ডেটা মডেলের জন্য প্রস্তুত করতে `Tokenizer`-ক্লাসের ব্যবহার |
| [ট্রেনিং ও ফাইন-টিউনিং](https://huggingface.co/docs/transformers/training) | PyTorch/TensorFlow লুপ ও `Trainer`-API-র সাথে মডেল ফাইন-টিউনিং |
| [দ্রুত শুরু: ফাইনটিউনিং/এপ্লিকেশন স্ক্রিপ্ট](https://github.com/huggingface/transformers/tree/main/examples) | বহুবিধ টাস্কে মডেল ফাইনটিউনিংয়ের জন্য নমুনা স্ক্রিপ্ট |
| [মডেল আপলোড ও শেয়ার](https://huggingface.co/docs/transformers/model_sharing) | আপনার ফাইন-টিউন মডেল আপলোড করুন ও কমিউনিটিতে শেয়ার করুন |

## রেফারেন্স

আমাদের [একটি পেপার](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) আছে, যা আপনি 🤗 Transformers লাইব্রেরি রেফারেন্স করতে ব্যবহার করতে পারেন।


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
