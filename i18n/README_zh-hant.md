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
        <b>繁體中文</b> |
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
    </p>
</h4>

<h3 align="center">
    <p>最先進的預訓練模型，專為推理與訓練而生</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers 是一個為最先進的機器學習模型（涵蓋文字、電腦視覺、音訊、影片及多模態）提供推理和訓練支援的模型定義框架。

它將模型定義集中化，使得該定義在整個生態系中能夠達成共識。`transformers` 是貫穿各個框架的樞紐：如果一個模型定義受到支援，它將與大多數訓練框架（如 Axolotl、Unsloth、DeepSpeed、FSDP、PyTorch-Lightning 等）、推理引擎（如 vLLM、SGLang、TGI 等）以及利用 `transformers` 模型定義的周邊建模函式庫（如 llama.cpp、mlx 等）相容。

我們致力於支援最新的頂尖模型，並透過使其模型定義變得簡單、可客製化且高效，來普及它們的應用。

在 [Hugging Face Hub](https://huggingface.com/models) 上，有超過 100 萬個 Transformers [模型檢查點](https://huggingface.co/models?library=transformers&sort=trending) 供您使用。

立即探索 [Hub](https://huggingface.com/)，尋找合適的模型，並使用 Transformers 幫助您快速上手。

## 安裝

Transformers 支援 Python 3.10+ 和 [PyTorch](https://pytorch.org/get-started/locally/) 2.4+。

使用 [venv](https://docs.python.org/3/library/venv.html) 或基於 Rust 的高速 Python 套件及專案管理器 [uv](https://docs.astral.sh/uv/) 來建立並啟用虛擬環境。

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

在您的虛擬環境中安裝 Transformers。

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

如果您想使用函式庫的最新變更或有興趣參與貢獻，可以從原始碼安裝 Transformers。然而，*最新*版本可能不穩定。如果您遇到任何錯誤，歡迎隨時提交一個 [issue](https://github.com/huggingface/transformers/issues)。

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
```

## 快速入門

透過 [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API 快速開始使用 Transformers。`Pipeline` 是一個高階的推理類別，支援文字、音訊、視覺和多模態任務。它負責處理輸入資料的預處理，並回傳適當的輸出。

實例化一個 pipeline 並指定用於文字生成的模型。該模型會被下載並快取，方便您之後輕鬆複用。最後，傳入一些文字來提示模型。

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

與模型進行聊天，使用模式是相同的。唯一的區別是您需要建構一個您與系統之間的聊天歷史（作為 `Pipeline` 的輸入）。

> [!TIP]
> 你也可以直接在命令列中與模型聊天。
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

展開下面的範例，查看 `Pipeline` 如何在不同模態和任務上運作。

<details>
<summary>自動語音辨識</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>影像分類</summary>

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
<summary>視覺問答</summary>

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

## 為什麼我應該使用 Transformers？

1.  易於使用的最先進模型：
    *   在自然語言理解與生成、電腦視覺、音訊、影片和多模態任務上表現卓越。
    *   為研究人員、工程師與開發者提供了低門檻的入門途徑。
    *   面向使用者的抽象層級少，只需學習三個核心類別。
    *   為所有預訓練模型提供了統一的 API 介面。

2.  更低的運算成本，更小的碳足跡：
    *   分享訓練好的模型，而不是從零開始訓練。
    *   減少運算時間和生產成本。
    *   擁有數十種模型架構和超過100萬個橫跨所有模態的預訓練檢查點。

3.  為模型的每個生命週期階段選擇合適的框架：
    *   僅用3行程式碼即可訓練最先進的模型。
    *   在PyTorch/JAX/TF2.0框架之間輕鬆切換單一模型。
    *   為訓練、評估和生產選擇最合適的框架。

4.  輕鬆根據您的需求客製化模型或範例：
    *   我們為每個架構提供了範例，以重現其原作者發表的結果。
    *   模型內部結構盡可能保持一致地暴露給使用者。
    *   模型檔案可以獨立於函式庫使用，便於快速實驗。

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## 為什麼我不應該使用 Transformers？

-   本函式庫並非一個用於建構神經網路的模組化工具箱。模型檔案中的程式碼為了讓研究人員能快速在模型上迭代，而沒有進行過度的抽象重構，避免了深入額外的抽象層/檔案。
-   訓練 API 針對 Transformers 提供的 PyTorch 模型進行了最佳化。對於通用的機器學習迴圈，您應該使用像 [Accelerate](https://huggingface.co/docs/accelerate) 這樣的其他函式庫。
-   [範例指令稿](https://github.com/huggingface/transformers/tree/main/examples)僅僅是*範例*。它們不一定能在您的特定用例上開箱即用，您可能需要修改程式碼才能使其正常運作。

## 100個使用 Transformers 的專案

Transformers 不僅僅是一個使用預訓練模型的工具包，它還是一個圍繞它和 Hugging Face Hub 建構的專案社群。我們希望 Transformers 能夠賦能開發者、研究人員、學生、教授、工程師以及其他任何人，去建構他們夢想中的專案。

為了慶祝 Transformers 獲得 10 萬顆星標，我們希望透過 [awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md) 頁面來聚焦社群，該頁面列出了100個基於 Transformers 建構的精彩專案。

如果您擁有或使用一個您認為應該被列入其中的專案，請隨時提交 PR 將其加入！

## 範例模型

您可以在我們大多數模型的 [Hub 模型頁面](https://huggingface.co/models) 上直接進行測試。

展開下面的每個模態，查看一些用於不同用例的範例模型。

<details>
<summary>音訊</summary>

-   Audio classification with [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
-   Automatic speech recognition with [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
-   Keyword spotting with [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
-   Speech to speech generation with [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
-   Text to audio with [MusicGen](https://huggingface.co/facebook/musicgen-large)
-   Text to speech with [Bark](https://huggingface.co/suno/bark)

</details>

<details>
<summary>電腦視覺</summary>

-   Automatic mask generation with [SAM](https://huggingface.co/facebook/sam-vit-base)
-   Depth estimation with [DepthPro](https://huggingface.co/apple/DepthPro-hf)
-   Image classification with [DINO v2](https://huggingface.co/facebook/dinov2-base)
-   Keypoint detection with [SuperPoint](https://huggingface.co/magic-leap-community/superpoint)
-   Keypoint matching with [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
-   Object detection with [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
-   Pose Estimation with [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
-   Universal segmentation with [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
-   Video classification with [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

</details>

<details>
<summary>多模態</summary>

-   Audio or text to text with [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
-   Document question answering with [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
-   Image or text to text with [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
-   Image captioning [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
-   OCR-based document understanding with [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
-   Table question answering with [TAPAS](https://huggingface.co/google/tapas-base)
-   Unified multimodal understanding and generation with [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
-   Vision to text with [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
-   Visual question answering with [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
-   Visual referring expression segmentation with [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

</details>

<details>
<summary>自然語言處理 (NLP)</summary>

-   Masked word completion with [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
-   Named entity recognition with [Gemma](https://huggingface.co/google/gemma-2-2b)
-   Question answering with [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
-   Summarization with [BART](https://huggingface.co/facebook/bart-large-cnn)
-   Translation with [T5](https://huggingface.co/google-t5/t5-base)
-   Text generation with [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
-   Text classification with [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

</details>

## 引用

現在我們有一篇可供您引用的關於 🤗 Transformers 函式庫的 [論文](https://www.aclweb.org/anthology/2020.emnlp-demos.6/)：
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