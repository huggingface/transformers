<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

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
        <b>Русский</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_uk.md">Українська</a> |
    <p>
</h4>

<h3 align="center">
    <p>Сучасне машинне навчання для JAX, PyTorch та TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers надає тисячі попередньо навчених моделей для виконання різноманітних завдань, таких як текст, зображення та аудіо.

Ці моделі можуть бути застосовані до:

* 📝 Тексту для таких завдань, як класифікація текстів, вилучення інформації, відповіді на питання, узагальнення, переклад, генерація текстів на більш ніж 100 мовах.
* 🖼️ Зображень для завдань класифікації зображень, виявлення об'єктів та сегментації.
* 🗣️ Аудіо для завдань розпізнавання мовлення та класифікації аудіо.

Моделі Transformers також можуть виконувати кілька завдань, таких як відповіді на табличні питання, розпізнавання оптичних символів, вилучення інформації з відсканованих документів, класифікація відео та відповіді на візуальні питання.

🤗 Transformers надає API для швидкого завантаження та використання попередньо натренованих моделей, їх тонкої настройки на власних наборах даних і подальшого використання спільнотою на нашому [сайті](https://huggingface.co/models). Водночас кожен модуль Python, що визначає архітектуру, є повністю автономним і може бути модифікований для швидких дослідницьких експериментів.

🤗 Transformers використовує три найпопулярніші бібліотеки глибокого навчання — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) та [TensorFlow](https://www.tensorflow.org/) — і легко інтегрується між ними. Це дозволяє навчати моделі з однією бібліотекою, а потім використовувати іншу для виконання висновків.

## Онлайн демонстрація

Більшість наших моделей можна протестувати безпосередньо на їхніх сторінках на [сайті](https://huggingface.co/models). Ми також пропонуємо [приватний хостинг моделей, контроль версій та API для висновків](https://huggingface.co/pricing) для публічних і приватних моделей.

Ось кілька прикладів:

У сфері NLP (обробка текстів природною мовою):
- [Маскування слів за допомогою BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Розпізнавання сутностей за допомогою Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Генерація тексту за допомогою GPT-2](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [Виведення природної мови за допомогою RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Узагальнення за допомогою BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres...)
- [Відповіді на запитання за допомогою DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used...)
- [Переклад за допомогою T5](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

У сфері комп'ютерного зору:
- [Класифікація зображень за допомогою ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Виявлення об'єктів за допомогою DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Семантична сегментація за допомогою SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Сегментація паноптикуму за допомогою MaskFormer](https://huggingface.co/facebook/maskformer-swin-small-coco)
- [Оцінка глибини за допомогою DPT](https://huggingface.co/docs/transformers/model_doc/dpt)
- [Класифікація відео за допомогою VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)
- [Універсальна сегментація за допомогою OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

У сфері звуку:
- [Автоматичне розпізнавання мовлення за допомогою Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Пошук ключових слів за допомогою Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Класифікація аудіоданих за допомогою аудіоспектрограмного трансформера](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

У мультимодальних завданнях:
- [Запитання-відповіді по таблицях з TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [Візуальні запитання-відповіді з ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [Опис зображень з LLaVa](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [Класифікація зображень без навчання з SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
- [Запитання-відповіді по документам з LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [Класифікація відео без навчання з X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)
- [Об'єктне виявлення без навчання з OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
- [Сегментація зображень без навчання з CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)
- [Автоматична генерація масок з SAM](https://huggingface.co/docs/transformers/model_doc/sam)


## 100 проєктів, що використовують Transformers

Transformers — це не просто набір інструментів для використання попередньо навчених моделей: це спільнота проєктів, створена на його основі, і 
Hugging Face Hub. Ми прагнемо, щоб Transformers дозволив розробникам, дослідникам, студентам, викладачам, інженерам і всім бажаючим 
створювати проєкти своєї мрії.

Щоб відсвяткувати 100 тисяч зірок на Transformers, ми вирішили зробити акцент на спільноті та створили сторінку [awesome-transformers](./awesome-transformers.md), на якій перераховані 100 неймовірних проєктів, створених за допомогою transformers.

Якщо ви є власником або користувачем проєкту, який, на вашу думку, має бути включений до цього списку, будь ласка, відкрийте PR для його додавання!

## Якщо ви хочете отримати індивідуальну підтримку від команди Hugging Face

<a target="_blank" href="https://huggingface.co/support">
    <img alt="Програма експертної підтримки HuggingFace" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## Швидкий гайд

Для використання моделі на заданому вході (текст, зображення, звук тощо) ми надаємо API `pipeline`. Конвеєри поєднують попередньо навчану модель з попередньою обробкою, яка використовувалася під час її навчання. Ось як швидко використовувати конвеєр для класифікації позитивних і негативних текстів:

```python
>>> from transformers import pipeline

# Виділення конвеєра для аналізу настроїв
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('Мы очень рады представить конвейер в transformers.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

Другий рядок коду завантажує і кешує попередньо навчену модель, яку використовує конвеєр, а третій оцінює її на заданому тексті. Тут відповідь "POSITIVE" з упевненістю 99,97%.

У багатьох завданнях, як у НЛП, так і в комп'ютерному зорі та аудіо, вже є готовий `pipeline`. Наприклад, ми можемо легко витягти виявлені об'єкти на зображенні:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Завантажуємо зображення з милими котиками
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# Виділення конвеєра для виявлення об'єктів
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

Тут ми отримуємо список об'єктів, виявлених на зображенні, з рамкою навколо об'єкта та оцінкою достовірності. Ліворуч — вихідне зображення, праворуч — прогнози:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

Детальніше про завдання, які підтримуються API `pipeline`, можна дізнатися в [цьому навчальному посібнику](https://huggingface.co/docs/transformers/task_sum)

На додаток до `pipeline`, для завантаження та використання будь-якої з попередньо навчених моделей у заданому завданні достатньо трьох рядків коду. Ось версія для PyTorch:
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Привет мир!", return_tensors="pt")
>>> outputs = model(**inputs)
```

А ось еквівалентний код для TensorFlow:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Привет мир!", return_tensors="tf")
>>> outputs = model(**inputs)
```

Токенізатор відповідає за всю попередню обробку, яку очікує попередньо навчена модель, і може бути викликаний безпосередньо за допомогою одного рядка (як у наведених вище прикладах) або на списку. У результаті буде отримано словник, який можна використовувати в подальшому коді або просто передати безпосередньо в модель за допомогою оператора розпакування аргументів **.

Сама модель є звичайним [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) або [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (залежно від використовуваного бекенду), яку можна використовувати як зазвичай. [У цьому керівництві](https://huggingface.co/docs/transformers/training) розповідається, як інтегрувати таку модель у класичний цикл навчання PyTorch або TensorFlow, або як використовувати наш API `Trainer` для швидкого налаштування на новому датасеті.

## Чому необхідно використовувати transformers?

1. Прості у використанні сучасні моделі:
    - Висока продуктивність у завданнях розуміння та генерації природної мови, комп'ютерного зору та аудіо.
    - Низький бар'єр для входу для викладачів і практиків.
    - Невелика кількість абстракцій для користувача і всього три класи для вивчення.
    - Єдиний API для використання всіх наших попередньо навчених моделей.

1. Нижчі обчислювальні витрати, менший "вуглецевий слід":
    - Дослідники можуть ділитися навченими моделями замість того, щоб постійно їх перенавчати.
    - Практики можуть скоротити час обчислень і виробничі витрати.
    - Десятки архітектур з більш ніж 60 000 попередньо навчених моделей для всіх модальностей.

1. Вибір відповідного фреймворку для кожного етапу життя моделі:
    - Навчання найсучасніших моделей за 3 рядки коду.
    - Переміщуйте одну модель між фреймворками TF2.0/PyTorch/JAX на свій розсуд.
    - Безперешкодний вибір відповідного фреймворку для навчання, оцінки та виробництва.

1. Легко налаштувати модель або приклад під свої потреби:
    - Ми надаємо приклади для кожної архітектури, щоб відтворити результати, опубліковані їх авторами.
    - Внутрішні компоненти моделі розкриваються максимально послідовно.
    - Файли моделей можна використовувати незалежно від бібліотеки для проведення швидких експериментів.

## Чому я не повинен використовувати transformers?

- Ця бібліотека не є модульним набором будівельних блоків для нейронних мереж. Код у файлах моделей спеціально не рефакториться додатковими абстракціями, щоб дослідники могли швидко ітеративно працювати з кожною моделлю, не заглиблюючись у додаткові абстракції/файли.
- API навчання не призначений для роботи з будь-якою моделлю, а оптимізований для роботи з моделями, наданими бібліотекою. Для роботи з загальними циклами машинного навчання слід використовувати іншу бібліотеку (можливо, [Accelerate](https://huggingface.co/docs/accelerate)).
- Незважаючи на те, що ми прагнемо надати якомога більше прикладів використання, скрипти в нашій папці [прикладів](https://github.com/huggingface/transformers/tree/main/examples) є саме прикладами. Передбачається, що вони не будуть працювати "з коробки" для вирішення вашого конкретного завдання, і вам доведеться змінити кілька рядків коду, щоб адаптувати їх під свої потреби.

## Встановлення

### За допомогою pip

Цей репозиторій протестований на Python 3.8+, Flax 0.4.1+, PyTorch 1.11+ і TensorFlow 2.6+.

Встановлювати 🤗 Transformers слід у [віртуальному середовищі](https://docs.python.org/3/library/venv.html). Якщо ви не знайомі з віртуальними середовищами Python, ознайомтеся з [керівництвом користувача](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Спочатку створіть віртуальне середовище з тією версією Python, яку ви плануєте використовувати, і активуйте його.

Потім необхідно встановити хоча б один бекенд із Flax, PyTorch або TensorFlow.
Будь ласка, зверніться до сторінок [TensorFlow - сторінка встановлення](https://www.tensorflow.org/install/), [PyTorch - сторінка встановлення](https://pytorch.org/get-started/locally/#start-locally) і/або [Flax](https://github.com/google/flax#quick-install) та [Jax](https://github.com/google/jax#installation), де описані команди встановлення для вашої платформи.

Після встановлення одного з цих бекендів 🤗 Transformers можна встановити за допомогою pip наступним чином:

```bash
pip install transformers
```

Якщо ви хочете спробувати приклади або вам потрібен найсучасніший код і ви не можете чекати нового релізу, вам слід [встановити бібліотеку з вихідного коду](https://huggingface.co/docs/transformers/installation#installing-from-source).

### За допомогою conda

Встановити Transformers за допомогою conda можна наступним чином:

```bash
conda install conda-forge::transformers
```

> **_ПРИМІТКА:_** Встановлення `transformers` через канал `huggingface` застаріло.

Як встановити Flax, PyTorch або TensorFlow за допомогою conda, читайте на сторінках, присвячених їх встановленню.

> **_ПРИМІТКА:_** В операційній системі Windows вам може бути запропоновано активувати режим розробника, щоб скористатися перевагами кешування. Якщо це неможливо, повідомте нам про це [тут](https://github.com/huggingface/huggingface_hub/issues/1062).

## Модельні архітектури

**[Усі контрольні точки моделей](https://huggingface.co/models)**, що надаються 🤗 Transformers, безперешкодно інтегруються з huggingface.co [model hub](https://huggingface.co/models), куди вони завантажуються безпосередньо [користувачами](https://huggingface.co/users) і [організаціями](https://huggingface.co/organizations).

Поточна кількість контрольних точок: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 На цей час Transformers підтримує такі архітектури: докладний опис кожної з них дивіться [тут](https://huggingface.co/docs/transformers/model_summary).

Щоб перевірити, чи є у кожної моделі реалізація на Flax, PyTorch або TensorFlow, або пов’язаний із нею токенізатор, що підтримується бібліотекою 🤗 Tokenizers, зверніться до [цієї таблиці](https://huggingface.co/docs/transformers/index#supported-frameworks).

Ці реалізації були протестовані на кількох наборах даних (див. приклади скриптів) і мають відповідати продуктивності оригінальних реалізацій. Детальнішу інформацію про продуктивність можна знайти в розділі "Приклади" [документації](https://github.com/huggingface/transformers/tree/main/examples).

## Вивчи більше

| Розділ | Опис |
|-|-|
| [Документація](https://huggingface.co/docs/transformers/) | Повна документація по API та гіди |
| [Короткі описи задач](https://huggingface.co/docs/transformers/task_summary) | Завдання, які підтримуються 🤗 Transformers |
| [Посібник з попередньої обробки](https://huggingface.co/docs/transformers/preprocessing) | Використання класу `Tokenizer` для підготовки даних для моделей |
| [Навчання та доопрацювання](https://huggingface.co/docs/transformers/training) | Використання моделей, наданих 🤗 Transformers, у циклі навчання PyTorch/TensorFlow та API `Trainer`. |
| [Швидкий тур: Тонке налаштування/скрипти використання](https://github.com/huggingface/transformers/tree/main/examples) | Приклади скриптів для тонкого налаштування моделей на широкому спектрі завдань |
| [Спільне використання та завантаження моделей](https://huggingface.co/docs/transformers/model_sharing) | Завантажуйте та діліться своїми доопрацьованими моделями з громадою |

## Цитування

Тепер у нас є [стаття](https://www.aclweb.org/anthology/2020.emnlp-demos.6/), яку можна цитувати для бібліотеки 🤗 Transformers:
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
