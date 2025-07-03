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
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <b>Українська</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p>Передові попередньо навчені моделі для висновків та навчання</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>


Transformers діє як фреймворк визначення моделей для передових моделей машинного навчання в тексті, комп'ютерному зорі, аудіо, відео та мультимодальних моделях, як для висновків, так і для навчання.

Він централізує визначення моделі так, що це визначення узгоджується в усьому екосистемі. `transformers` є центральною точкою між фреймворками: якщо визначення моделі підтримується, воно буде сумісним з більшістю фреймворків навчання (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), двигунів висновків (vLLM, SGLang, TGI, ...) та суміжних бібліотек моделювання (llama.cpp, mlx, ...), які використовують визначення моделі з `transformers`.

Ми зобов'язуємося допомагати підтримувати нові передові моделі та демократизувати їх використання, роблячи їх визначення простим, налаштовуваним та ефективним.

На [Hugging Face Hub](https://huggingface.com/models) є понад 1M+ [контрольних точок моделей](https://huggingface.co/models?library=transformers&sort=trending) Transformers, які ви можете використовувати.

Дослідіть [Hub](https://huggingface.com/) сьогодні, щоб знайти модель та використати Transformers, щоб допомогти вам почати відразу.

## Встановлення

Transformers працює з Python 3.9+ [PyTorch](https://pytorch.org/get-started/locally/) 2.1+, [TensorFlow](https://www.tensorflow.org/install/pip) 2.6+ та [Flax](https://flax.readthedocs.io/en/latest/) 0.4.1+.

Створіть та активуйте віртуальне середовище з [venv](https://docs.python.org/3/library/venv.html) або [uv](https://docs.astral.sh/uv/), швидким менеджером пакетів та проектів Python на основі Rust.

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

Встановіть Transformers у вашому віртуальному середовищі.

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

Встановіть Transformers з джерела, якщо ви хочете найновіші зміни в бібліотеці або зацікавлені в співпраці. Однак *найновіша* версія може бути нестабільною. Не соромтеся відкрити [issue](https://github.com/huggingface/transformers/issues), якщо ви зіткнетеся з помилкою.

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

## Швидкий старт

Почніть роботу з Transformers відразу за допомогою API [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial). `Pipeline` - це високорівневий клас висновків, який підтримує текстові, аудіо, візуальні та мультимодальні завдання. Він обробляє попередню обробку вхідних даних та повертає відповідний вивід.

Створіть pipeline та вкажіть модель для використання генерації тексту. Модель завантажується та кешується, тому ви можете легко використовувати її знову. Нарешті, передайте деякий текст для підказки моделі.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("секрет випікання дійсно хорошого пирога полягає в ")
[{'generated_text': 'секрет випікання дійсно хорошого пирога полягає в 1) використанні правильних інгредієнтів та 2) точному дотриманні рецепту. рецепт пирога наступний: 1 склянка цукру, 1 склянка борошна, 1 склянка молока, 1 склянка масла, 1 склянка яєць, 1 склянка шоколадних чіпсів. якщо ви хочете зробити 2 пироги, скільки цукру вам потрібно? Щоб зробити 2 пироги, вам знадобиться 2 склянки цукру.'}]
```

Для спілкування з моделлю схема використання така ж. Єдина різниця в тому, що вам потрібно створити історію чату (вхід для `Pipeline`) між вами та системою.

> [!TIP]
> Ви також можете спілкуватися з моделлю безпосередньо з командного рядка.
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```py
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "Ви є зухвалим, дотепним роботом, як його уявляв Голлівуд близько 1986 року."},
    {"role": "user", "content": "Привіт, чи можеш ти розповісти мені про цікаві речі для роботи в Нью-Йорку?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Розгорніть приклади нижче, щоб побачити, як працює `Pipeline` для різних модальностей та завдань.

<details>
<summary>Автоматичне розпізнавання мови</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' У мене є мрія, що одного дня ця нація підніметься і проживе справжній сенс своєї кредо.'}
```

</details>

<details>
<summary>Класифікація зображень</summary>

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
<summary>Візуальні відповіді на питання</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="Що знаходиться на зображенні?",
)
[{'answer': 'статуя свободи'}]
```

</details>

## Чому я повинен використовувати Transformers?

1. Легкі у використанні передові моделі:
    - Висока продуктивність у розумінні та генерації природної мови, комп'ютерному зорі, аудіо, відео та мультимодальних завданнях.
    - Низький бар'єр входу для дослідників, інженерів та розробників.
    - Мало абстракцій для користувача з лише трьома класами для вивчення.
    - Єдиний API для використання всіх наших попередньо навчених моделей.

1. Нижчі обчислювальні витрати, менший вуглецевий слід:
    - Діліться навченими моделями замість навчання з нуля.
    - Зменшуйте час обчислень та виробничі витрати.
    - Десятки архітектур моделей з 1M+ попередньо навченими контрольними точками у всіх модальностях.

1. Виберіть правильний фреймворк для кожної частини життєвого циклу моделі:
    - Навчайте передові моделі в 3 рядках коду.
    - Переміщуйте одну модель між фреймворками PyTorch/JAX/TF2.0 за бажанням.
    - Виберіть правильний фреймворк для навчання, оцінки та виробництва.

1. Легко налаштуйте модель або приклад під ваші потреби:
    - Ми надаємо приклади для кожної архітектури, щоб відтворити результати, опубліковані її оригінальними авторами.
    - Внутрішні частини моделі виставляються якомога послідовніше.
    - Файли моделей можна використовувати незалежно від бібліотеки для швидких експериментів.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## Чому я не повинен використовувати Transformers?

- Ця бібліотека не є модульною інструментальною панеллю будівельних блоків для нейронних мереж. Код у файлах моделей не рефакториться з додатковими абстракціями навмисно, щоб дослідники могли швидко ітераціювати над кожною з моделей без занурення в додаткові абстракції/файли.
- API навчання оптимізовано для роботи з моделями PyTorch, наданими Transformers. Для загальних циклів машинного навчання ви повинні використовувати іншу бібліотеку, як [Accelerate](https://huggingface.co/docs/accelerate).
- [Приклади скриптів](https://github.com/huggingface/transformers/tree/main/examples) є лише *прикладами*. Вони можуть не обов'язково працювати з коробки для вашого конкретного випадку використання, і вам потрібно буде адаптувати код, щоб він працював.

## 100 проектів, що використовують Transformers

Transformers - це більше, ніж набір інструментів для використання попередньо навчених моделей: це спільнота проектів, створених на його основі, та
Hugging Face Hub. Ми хочемо, щоб Transformers дозволив розробникам, дослідникам, студентам, професорам, інженерам та всім бажаючим
створювати проекти своєї мрії.

Щоб відсвяткувати 100 тисяч зірок Transformers, ми вирішили зробити акцент на спільноті та створили сторінку [awesome-transformers](./awesome-transformers.md), на якій перераховані 100
неймовірних проектів, створених за допомогою transformers.

Якщо ви є власником або користувачем проекту, який, на вашу думку, повинен бути включений у цей список, будь ласка, відкрийте PR для його додавання!

## Приклади моделей

Ви можете протестувати більшість наших моделей безпосередньо на їх [сторінках моделей Hub](https://huggingface.co/models).

Розгорніть кожну модальність нижче, щоб побачити кілька прикладів моделей для різних випадків використання.

<details>
<summary>Аудіо</summary>

- Класифікація аудіо з [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
- Автоматичне розпізнавання мови з [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
- Пошук ключових слів з [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- Генерація мови в мову з [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
- Текст в аудіо з [MusicGen](https://huggingface.co/facebook/musicgen-large)
- Текст в мову з [Bark](https://huggingface.co/suno/bark)

</details>

<details>
<summary>Комп'ютерний зір</summary>

- Автоматична генерація масок з [SAM](https://huggingface.co/facebook/sam-vit-base)
- Оцінка глибини з [DepthPro](https://huggingface.co/apple/DepthPro-hf)
- Класифікація зображень з [DINO v2](https://huggingface.co/facebook/dinov2-base)
- Виявлення ключових точок з [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
- Зіставлення ключових точок з [SuperGlue](https://huggingface.co/magic-leap-community/superglue)
- Виявлення об'єктів з [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
- Оцінка поз з [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
- Універсальна сегментація з [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
- Класифікація відео з [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

</details>

<details>
<summary>Мультимодальні</summary>

- Аудіо або текст в текст з [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
- Відповіді на питання по документах з [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
- Зображення або текст в текст з [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Підписи до зображень [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- Розуміння документів на основі OCR з [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- Відповіді на питання по таблицях з [TAPAS](https://huggingface.co/google/tapas-base)
- Уніфіковане мультимодальне розуміння та генерація з [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
- Зір в текст з [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
- Візуальні відповіді на питання з [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- Візуальна сегментація референційних виразів з [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

</details>

<details>
<summary>NLP</summary>

- Заповнення маскованих слів з [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- Розпізнавання іменованих сутностей з [Gemma](https://huggingface.co/google/gemma-2-2b)
- Відповіді на питання з [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- Підсумування з [BART](https://huggingface.co/facebook/bart-large-cnn)
- Переклад з [T5](https://huggingface.co/google-t5/t5-base)
- Генерація тексту з [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
- Класифікація тексту з [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

</details>

## Цитування

Тепер у нас є [стаття](https://www.aclweb.org/anthology/2020.emnlp-demos.6/), яку ви можете цитувати для бібліотеки 🤗 Transformers:
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