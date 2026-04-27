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
        <b>English</b> |
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
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fa.md">فارسی</a> |
    </p>
</h4>

<h3 align="center">
    <p>State-of-the-art pretrained models for inference and training</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

کتابخانه Transformers به عنوان بستر اصلی برای تعریف و پیاده‌سازی برترین مدل‌های یادگیری ماشین در حوزه‌های متن، بینایی ماشین، صوت، ویدئو و مدل‌های چندوجهی شناخته می‌شود. این کتابخانه ابزاری جامع است که تمامی مراحل، از آموزش (Training) تا استنتاج (Inference) را به‌خوبی پوشش می‌دهد.

این کتابخانه با یکپارچه‌سازی تعریف مدل‌ها، استانداردی واحد در سراسر اکوسیستم هوش مصنوعی ایجاد کرده است. Transformers نقش یک نقطهٔ اتصال مرکزی را ایفا می‌کند؛ به این معنا که اگر تعریف مدلی در آن پشتیبانی شود، بلافاصله با اکثر چارچوب‌های آموزش (مانند Axolotl، Unsloth، DeepSpeed و PyTorch-Lightning)، موتورهای استنتاج (مانند vLLM، SGLang و TGI) و کتابخانه‌های مدل‌سازی مکمل (مانند llama.cpp و mlx) که همگی از استانداردهای تعریف مدل در Transformers پیروی می‌کنند، سازگار خواهد بود.

ما متعهد می‌شویم که از مدل‌های جدید و پیشرفته پشتیبانی کنیم و استفاده از آن‌ها را همگانی‌تر کنیم؛ با این هدف که تعریف مدل‌هایشان ساده، قابل‌سفارشی‌سازی و کارآمد باشد.

بیش از 1M+ [model checkpoints](https://huggingface.co/models?library=transformers&sort=trending) مربوط به Transformers در [Hugging Face Hub](https://huggingface.com/models) وجود دارد که می‌توانید از آن‌ها استفاده کنید.

امروز [Hub](https://huggingface.com/) را کاوش کنید تا یک مدل پیدا کنید و با کمک Transformers فوراً کار خود را آغاز کنید.

## نصب
یک محیط مجازی (virtual environment) با استفاده از venv یا uv بساز و آن را فعال کن. uv یک مدیر سریع پکیج و پروژهٔ پایتون است که با Rust نوشته شده.  
```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```
برای نصب این کتابخانه، به **Python 3.10+** و **PyTorch 2.4+** نیاز دارید. نصب می‌تواند از طریق `pip` یا `uv` انجام شود:

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```
اگر می‌خواهید جدیدترین تغییرات کتابخانه را داشته باشید یا قصد مشارکت (contribute) در پروژه را دارید، می‌توانید Transformers را از سورس (source) نصب کنید. با این حال، جدیدترین نسخه ممکن است پایدار (stable) نباشد. اگر با خطایی برخورد کردید، با خیال راحت یک [Issue](https://github.com/huggingface/transformers/issues) در گیت‌هاب باز کنید:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
```
## شروع سریع (Quickstart)

با استفاده از [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API می‌توانید خیلی سریع کار با Transformers را شروع کنید.

پایپ لاین (Pipeline) یک کلاس سطح‌بالا برای انجام استنتاج (Inference) است که از کار با متن، صدا، تصویر و وظایف چندوجهی (multimodal) پشتیبانی می‌کند. این ابزار به‌طور خودکار مراحل پیش‌پردازش ورودی را انجام می‌دهد و خروجی مناسب را برمی‌گرداند.

برای شروع، یک pipeline بسازید و مشخص کنید از چه مدلی برای تولید متن استفاده شود. مدل به‌صورت خودکار دانلود و در حافظهٔ کش (cache) ذخیره می‌شود تا در اجراهای بعدی بتوان به‌راحتی دوباره از آن استفاده کرد. 
### تولید متن (Text Generation)

مثالی از تولید متن با استفاده از مدل `Qwen/Qwen2.5-1.5B`:

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

### چت (Chat)

شما می‌توانید از مدل‌های محاوره‌ای مانند `meta-llama/Meta-Llama-3-8B-Instruct` استفاده کنید. این مثال نحوه استفاده از پرامپت‌های سیستم و ساخت تاریخچه چت را نشان می‌دهد:

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

*نکته: همچنین می‌توانید از رابط خط فرمان `transformers chat` برای چت در ترمینال استفاده کنید.*

### سایر مدالیته‌ها (صدا، بینایی کامپیوتر، چندوجهی)

**تشخیص خودکار گفتار (ASR):**

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

**طبقه‌بندی تصویر (Image Classification):**
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
**پاسخگویی بصری به سوالات (Visual Question Answering):**

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

## چرا باید از Transformers استفاده کنیم؟

۱. **استفاده آسان:** دسترسی به پیشرفته‌ترین مدل‌ها با کمترین مانع، انتزاع (abstraction) کم و یک API یکپارچه.  
۲. **کاهش هزینه‌های محاسباتی و ردپای کربن:** به اشتراک‌گذاری مدل‌های از پیش آموزش‌دیده، نیاز به آموزش مدل‌ها از صفر را کاهش می‌دهد.  
۳. **انعطاف‌پذیری در چارچوب‌ها:** پشتیبانی از PyTorch، JAX و TF2.0 و همچنین چارچوب‌های مختلف آموزش و استنتاج.  
۴. **قابلیت شخصی‌سازی:** ارائه مثال‌هایی برای بازتولید نتایج و دسترسی به ساختار داخلی مدل‌ها برای آزمایش و تحقیق.  

## چرا *نباید* از Transformers استفاده کنیم؟

*   این کتابخانه یک جعبه‌ابزار ماژولار برای شبکه‌های عصبی عمومی نیست؛ کدهای مدل‌ها عمداً برای محققان کمتر انتزاعی شده‌اند.
*   رابط آموزش (Training API) بهینه‌سازی شده برای مدل‌های Transformers است. برای حلقه‌های عمومی یادگیری ماشین، بهتر است از کتابخانه‌هایی مانند [Accelerate](Accelerate) استفاده کنید.
*   [اسکریپت‌های نمونه](https://github.com/huggingface/transformers/tree/main/examples) ممکن است برای موارد استفاده خاص شما نیاز به تغییر و سازگاری داشته باشند.

## جامعه کاربری (Community)

تأثیر این کتابخانه بر جامعه کاربری بسیار گسترده است. صفحه [awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md) لیستی از ۱۰۰ پروژه فوق‌العاده را که با استفاده از Transformers ساخته شده‌اند، به نمایش می‌گذارد.

## مدل‌های نمونه

این کتابخانه از مدل‌های متنوعی پشتیبانی می‌کند، از جمله:
*   **صدا (Audio):** Whisper, Wav2Vec2
*   **بینایی کامپیوتر (Computer Vision):** ViT, DINOv2
*   **چندوجهی (Multimodal):** BLIP, LLaVA
*   **پردازش زبان طبیعی (NLP):** LLaMA, Qwen, BERT, GPT-2

*(برای مشاهده لیست کامل مدل‌ها و دسترسی به چک‌پوینت‌های آن‌ها به [Hugging Face Hub](https://huggingface.co/models) مراجعه کنید).*

## استناد (Citation)

اگر از این کتابخانه در تحقیقات خود استفاده می‌کنید، لطفاً با استفاده از فرمت BibTeX زیر به آن استناد کنید:

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
