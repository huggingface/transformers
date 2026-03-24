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
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_it.md">Italiano</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
    <p>
</h4>

<h3 align="center">
    <p>Современное машинное обучение для JAX, PyTorch и TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers предоставляет тысячи предварительно обученных моделей для выполнения различных задач, таких как текст, зрение и аудио.

Эти модели могут быть применены к:

* 📝 Тексту для таких задач, как классификация текстов, извлечение информации, ответы на вопросы, обобщение, перевод, генерация текстов на более чем 100 языках.
* 🖼️ Изображениям для задач классификации изображений, обнаружения объектов и сегментации.
* 🗣️ Аудио для задач распознавания речи и классификации аудио.

Модели transformers также могут выполнять несколько задач, такие как ответы на табличные вопросы, распознавание оптических символов, извлечение информации из отсканированных документов, классификация видео и ответы на визуальные вопросы.

🤗 Transformers предоставляет API для быстрой загрузки и использования предварительно обученных моделей, их тонкой настройки на собственных датасетах и последующего взаимодействия ими с сообществом на нашем [сайте](https://huggingface.co/models). В то же время каждый python модуль, определяющий архитектуру, полностью автономен и может быть модифицирован для проведения быстрых исследовательских экспериментов.

🤗 Transformers опирается на три самые популярные библиотеки глубокого обучения - [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) и [TensorFlow](https://www.tensorflow.org/) - и легко интегрируется между ними. Это позволяет легко обучать модели с помощью одной из них, а затем загружать их для выводов с помощью другой.

## Онлайн демонстрация

Большинство наших моделей можно протестировать непосредственно на их страницах с [сайта](https://huggingface.co/models). Мы также предлагаем [приватный хостинг моделей, контроль версий и API для выводов](https://huggingface.co/pricing) для публичных и частных моделей.

Вот несколько примеров:

В области NLP ( Обработка текстов на естественном языке ):
- [Маскированное заполнение слов с помощью BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Распознавание сущностей с помощью Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Генерация текста с помощью GPT-2](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+)
- [Выводы на естественном языке с помощью RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Обобщение с помощью BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Ответы на вопросы с помощью DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Перевод с помощью T5](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

В области компьютерного зрения:
- [Классификация изображений с помощью ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Обнаружение объектов с помощью DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Семантическая сегментация с помощью SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Сегментация паноптикума с помощью MaskFormer](https://huggingface.co/facebook/maskformer-swin-small-coco)
- [Оценка глубины с помощью DPT](https://huggingface.co/docs/transformers/model_doc/dpt)
- [Классификация видео с помощью VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)
- [Универсальная сегментация с помощью OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

В области звука:
- [Автоматическое распознавание речи с помощью Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Поиск ключевых слов с помощью Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Классификация аудиоданных с помощью траснформера аудиоспектрограмм](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

В мультимодальных задачах:
- [Ответы на вопросы по таблице с помощью TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [Визуальные ответы на вопросы с помощью ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [Zero-shot классификация изображений с помощью CLIP](https://huggingface.co/openai/clip-vit-large-patch14)
- [Ответы на вопросы по документам с помощью LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [Zero-shot классификация видео с помощью X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)


## 100 проектов, использующих Transformers

Transformers - это не просто набор инструментов для использования предварительно обученных моделей: это сообщество проектов, созданное на его основе, и
Hugging Face Hub. Мы хотим, чтобы Transformers позволил разработчикам, исследователям, студентам, профессорам, инженерам и всем желающим
создавать проекты своей мечты.

Чтобы отпраздновать 100 тысяч звезд Transformers, мы решили сделать акцент на сообществе, и создали страницу [awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md), на которой перечислены 100
невероятных проектов, созданных с помощью transformers.

Если вы являетесь владельцем или пользователем проекта, который, по вашему мнению, должен быть включен в этот список, пожалуйста, откройте PR для его добавления!

## Если вы хотите получить индивидуальную поддержку от команды Hugging Face

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## Быстрый гайд

Для использования модели на заданном входе (текст, изображение, звук, ...) мы предоставляем API `pipeline`. Конвейеры объединяют предварительно обученную модель с препроцессингом, который использовался при ее обучении. Вот как можно быстро использовать конвейер для классификации положительных и отрицательных текстов:

```python
>>> from transformers import pipeline

# Выделение конвейера для анализа настроений
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('Мы очень рады представить конвейер в transformers.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

Вторая строка кода загружает и кэширует предварительно обученную модель, используемую конвейером, а третья оценивает ее на заданном тексте. Здесь ответ "POSITIVE" с уверенностью 99,97%.

Во многих задачах, как в НЛП, так и в компьютерном зрении и речи, уже есть готовый `pipeline`. Например, мы можем легко извлечь обнаруженные объекты на изображении:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Скачиваем изображение с милыми котиками
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# Выделение конвейера для обнаружения объектов
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

Здесь мы получаем список объектов, обнаруженных на изображении, с рамкой вокруг объекта и оценкой достоверности. Слева - исходное изображение, справа прогнозы:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

Подробнее о задачах, поддерживаемых API `pipeline`, можно узнать в [этом учебном пособии](https://huggingface.co/docs/transformers/task_sum)

В дополнение к `pipeline`, для загрузки и использования любой из предварительно обученных моделей в заданной задаче достаточно трех строк кода. Вот версия для PyTorch:
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Привет мир!", return_tensors="pt")
>>> outputs = model(**inputs)
```

А вот эквивалентный код для TensorFlow:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Привет мир!", return_tensors="tf")
>>> outputs = model(**inputs)
```

Токенизатор отвечает за всю предварительную обработку, которую ожидает предварительно обученная модель, и может быть вызван непосредственно с помощью одной строки (как в приведенных выше примерах) или на списке. В результате будет получен словарь, который можно использовать в последующем коде или просто напрямую передать в модель с помощью оператора распаковки аргументов **.

Сама модель представляет собой обычный [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) или [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (в зависимости от используемого бэкенда), который можно использовать как обычно. [В этом руководстве](https://huggingface.co/docs/transformers/training) рассказывается, как интегрировать такую модель в классический цикл обучения PyTorch или TensorFlow, или как использовать наш API `Trainer` для быстрой тонкой настройки на новом датасете.

## Почему необходимо использовать transformers?

1. Простые в использовании современные модели:
    - Высокая производительность в задачах понимания и генерации естественного языка, компьютерного зрения и аудио.
    - Низкий входной барьер для преподавателей и практиков.
    - Небольшое количество абстракций для пользователя и всего три класса для изучения.
    - Единый API для использования всех наших предварительно обученных моделей.

1. Более низкие вычислительные затраты, меньший "углеродный след":
    - Исследователи могут обмениваться обученными моделями вместо того, чтобы постоянно их переобучать.
    - Практики могут сократить время вычислений и производственные затраты.
    - Десятки архитектур с более чем 60 000 предварительно обученных моделей для всех модальностей.

1. Выбор подходящего фреймворка для каждого этапа жизни модели:
    - Обучение самых современных моделей за 3 строки кода.
    - Перемещайте одну модель между фреймворками TF2.0/PyTorch/JAX по своему усмотрению.
    - Беспрепятственный выбор подходящего фреймворка для обучения, оценки и производства.

1. Легко настроить модель или пример под свои нужды:
    - Мы предоставляем примеры для каждой архитектуры, чтобы воспроизвести результаты, опубликованные их авторами.
    - Внутренние компоненты модели раскрываются максимально последовательно.
    - Файлы моделей можно использовать независимо от библиотеки для проведения быстрых экспериментов.

## Почему я не должен использовать transformers?

- Данная библиотека не является модульным набором строительных блоков для нейронных сетей. Код в файлах моделей специально не рефакторится дополнительными абстракциями, чтобы исследователи могли быстро итеративно работать с каждой из моделей, не погружаясь в дополнительные абстракции/файлы.
- API обучения не предназначен для работы с любой моделью, а оптимизирован для работы с моделями, предоставляемыми библиотекой. Для работы с общими циклами машинного обучения следует использовать другую библиотеку (возможно, [Accelerate](https://huggingface.co/docs/accelerate)).
- Несмотря на то, что мы стремимся представить как можно больше примеров использования, скрипты в нашей папке [примеров](https://github.com/huggingface/transformers/tree/main/examples) являются именно примерами. Предполагается, что они не будут работать "из коробки" для решения вашей конкретной задачи, и вам придется изменить несколько строк кода, чтобы адаптировать их под свои нужды.

## Установка

### С помощью pip

Данный репозиторий протестирован на Python 3.10+ и PyTorch 2.4+.

Устанавливать 🤗 Transformers следует в [виртуальной среде](https://docs.python.org/3/library/venv.html). Если вы не знакомы с виртуальными средами Python, ознакомьтесь с [руководством пользователя](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Сначала создайте виртуальную среду с той версией Python, которую вы собираетесь использовать, и активируйте ее.

Затем необходимо установить хотя бы один бекенд из Flax, PyTorch или TensorFlow.
Пожалуйста, обратитесь к страницам [TensorFlow установочная страница](https://www.tensorflow.org/install/), [PyTorch установочная страница](https://pytorch.org/get-started/locally/#start-locally) и/или [Flax](https://github.com/google/flax#quick-install) и [Jax](https://github.com/google/jax#installation), где описаны команды установки для вашей платформы.

После установки одного из этих бэкендов 🤗 Transformers может быть установлен с помощью pip следующим образом:

```bash
pip install transformers
```

Если вы хотите поиграть с примерами или вам нужен самый современный код и вы не можете ждать нового релиза, вы должны [установить библиотеку из исходного кода](https://huggingface.co/docs/transformers/installation#installing-from-source).

### С помощью conda

Установить Transformers с помощью conda можно следующим образом:

```bash
conda install conda-forge::transformers
```

> **_ЗАМЕТКА:_** Установка `transformers` через канал `huggingface` устарела.

О том, как установить Flax, PyTorch или TensorFlow с помощью conda, читайте на страницах, посвященных их установке.

> **_ЗАМЕТКА:_** В операционной системе Windows вам может быть предложено активировать режим разработчика, чтобы воспользоваться преимуществами кэширования. Если для вас это невозможно, сообщите нам об этом [здесь](https://github.com/huggingface/huggingface_hub/issues/1062).

## Модельные архитектуры

**[Все контрольные точки моделей](https://huggingface.co/models)**, предоставляемые 🤗 Transformers, беспрепятственно интегрируются с huggingface.co [model hub](https://huggingface.co/models), куда они загружаются непосредственно [пользователями](https://huggingface.co/users) и [организациями](https://huggingface.co/organizations).

Текущее количество контрольных точек: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 В настоящее время Transformers предоставляет следующие архитектуры: подробное описание каждой из них см. [здесь](https://huggingface.co/docs/transformers/model_summary).

Чтобы проверить, есть ли у каждой модели реализация на Flax, PyTorch или TensorFlow, или связанный с ней токенизатор, поддерживаемый библиотекой 🤗 Tokenizers, обратитесь к [этой таблице](https://huggingface.co/docs/transformers/index#supported-frameworks).

Эти реализации были протестированы на нескольких наборах данных (см. примеры скриптов) и должны соответствовать производительности оригинальных реализаций. Более подробную информацию о производительности можно найти в разделе "Примеры" [документации](https://github.com/huggingface/transformers/tree/main/examples).


## Изучи больше

| Секция | Описание |
|-|-|
| [Документация](https://huggingface.co/docs/transformers/) | Полная документация по API и гайды |
| [Краткие описания задач](https://huggingface.co/docs/transformers/task_summary) | Задачи поддерживаются 🤗 Transformers |
| [Пособие по предварительной обработке](https://huggingface.co/docs/transformers/preprocessing) | Использование класса `Tokenizer` для подготовки данных для моделей |
| [Обучение и доработка](https://huggingface.co/docs/transformers/training) | Использование моделей, предоставляемых 🤗 Transformers, в цикле обучения PyTorch/TensorFlow и API `Trainer`. |
| [Быстрый тур: Тонкая настройка/скрипты использования](https://github.com/huggingface/transformers/tree/main/examples) | Примеры скриптов для тонкой настройки моделей на широком спектре задач |
| [Совместное использование и загрузка моделей](https://huggingface.co/docs/transformers/model_sharing) | Загружайте и делитесь с сообществом своими доработанными моделями |

## Цитирование

Теперь у нас есть [статья](https://www.aclweb.org/anthology/2020.emnlp-demos.6/), которую можно цитировать для библиотеки 🤗 Transformers:
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
