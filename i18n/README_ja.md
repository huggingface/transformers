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

<!---
A useful guide for English-Traditional Japanese translation of Hugging Face documentation
- Use square quotes, e.g.,「引用」

Dictionary

API: API(翻訳しない)
add: 追加
checkpoint: チェックポイント
code: コード
community: コミュニティ
confidence: 信頼度
dataset: データセット
documentation: ドキュメント
example: 例
finetune: 微調整
Hugging Face: Hugging Face(翻訳しない)
implementation: 実装
inference: 推論
library: ライブラリ
module: モジュール
NLP/Natural Language Processing: NLPと表示される場合は翻訳されず、Natural Language Processingと表示される場合は翻訳される
online demos: オンラインデモ
pipeline: pipeline(翻訳しない)
pretrained/pretrain: 学習済み
Python data structures (e.g., list, set, dict): リスト、セット、ディクショナリと訳され、括弧内は原文英語
repository: repository(翻訳しない)
summary: 概要
token-: token-(翻訳しない)
Trainer: Trainer(翻訳しない)
transformer: transformer(翻訳しない)
tutorial: チュートリアル
user: ユーザ
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
        <b>日本語</b> |
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
    <p>推論と学習のための最先端の事前学習済みモデル</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformersは、テキスト、コンピュータビジョン、音声、動画、マルチモーダルモデルを用いた最先端の機械学習のためのモデル定義フレームワークとして、推論と学習の両方で機能します。

モデル定義を一元化することで、エコシステム全体でその定義が合意されるようにします。`transformers`はフレームワーク間のピボット（要）となります。モデル定義がサポートされていれば、大部分の学習フレームワーク(Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...)、推論エンジン(vLLM, SGLang, TGI, ...)、および`transformers`のモデル定義を活用する隣接するモデリングライブラリ(llama.cpp, mlx, ...)と互換性があります。

私たちは、モデル定義をシンプル、カスタマイズ可能、かつ効率的なものにすることで、新しい最先端モデルのサポートを支援し、その利用を民主化することを誓います。

[Hugging Face Hub](https://huggingface.com/models)には、100万を超えるTransformersの[モデルチェックポイント](https://huggingface.co/models?library=transformers&sort=trending)があり、すぐに使用できます。

[Hub](https://huggingface.com/)を探索してモデルを見つけ、Transformersを使ってすぐに始めましょう。

## インストール

TransformersはPython 3.10以上、[PyTorch](https://pytorch.org/get-started/locally/) 2.4以上で動作します。

[venv](https://docs.python.org/3/library/venv.html)または、高速なRustベースのPythonパッケージおよびプロジェクトマネージャーである[uv](https://docs.astral.sh/uv/)を使用して、仮想環境を作成し、有効化してください。

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

仮想環境にTransformersをインストールします。

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

ライブラリの最新の変更が必要な場合や、貢献に興味がある場合は、ソースからTransformersをインストールしてください。ただし、*最新*バージョンは安定していない可能性があります。エラーが発生した場合は、お気軽に[issue](https://github.com/huggingface/transformers/issues)を開いてください。

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
```

## クイックスタート

[Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) APIを使用して、すぐにTransformersを始めましょう。`Pipeline`は、テキスト、音声、視覚、およびマルチモーダルタスクをサポートする高レベルの推論クラスです。入力の前処理を行い、適切な出力を返します。

パイプラインをインスタンス化し、テキスト生成に使用するモデルを指定します。モデルはダウンロードされキャッシュされるため、簡単に再利用できます。最後に、モデルにプロンプトとしてテキストを渡します。

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

モデルとチャットする場合も、使用パターンは同じです。唯一の違いは、あなたとシステムの間でチャット履歴（`Pipeline`への入力）を構築する必要があることです。

> [!TIP]
> コマンドラインから直接モデルとチャットすることもできます。
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

以下の例を展開して、さまざまなモダリティやタスクで`Pipeline`がどのように機能するかを確認してください。

<details>
<summary>自動音声認識</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>画像分類</summary>

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
<summary>視覚的質問応答</summary>

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

## なぜtransformersを使う必要があるのでしょうか？

1. 使いやすい最先端のモデル:
    - 自然言語理解・生成、コンピュータビジョン、音声、動画、マルチモーダルタスクで高いパフォーマンスを発揮します。
    - 研究者、エンジニア、開発者にとっての低い参入障壁。
    - 学習するクラスは3つだけで、ユーザが直面する抽象化はほとんどありません。
    - すべての事前学習済みモデルを利用するための統一されたAPI。

1. 低い計算コスト、少ないカーボンフットプリント:
    - ゼロから学習するのではなく、学習済みモデルを共有できます。
    - 計算時間や生産コストを削減できます。
    - すべてのモダリティにおいて、100万以上の事前学習済みチェックポイントを持つ多数のモデルアーキテクチャを提供します。

1. モデルのライフサイクルのあらゆる部分で適切なフレームワークを選択可能:
    - 3行のコードで最先端のモデルを学習。
    - PyTorch/JAX/TF2.0フレームワーク間で1つのモデルを自在に移動可能。
    - 学習、評価、本番環境に適したフレームワークを選択できます。

1. モデルや例をニーズに合わせて簡単にカスタマイズ可能:
    - 原著者が発表した結果を再現するために、各アーキテクチャの例を提供しています。
    - モデル内部は可能な限り一貫して公開されています。
    - モデルファイルはライブラリとは独立して利用することができ、迅速な実験が可能です。

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## なぜtransformersを使ってはいけないのでしょうか？

- このライブラリは、ニューラルネットのためのビルディングブロックのモジュール式ツールボックスではありません。モデルファイルのコードは、研究者が追加の抽象化/ファイルに飛び込むことなく、各モデルを素早く反復できるように、意図的に追加の抽象化でリファクタリングされていません。
- 学習APIはTransformersが提供するPyTorchモデルで動作するように最適化されています。一般的な機械学習のループには、[Accelerate](https://huggingface.co/docs/accelerate)のような別のライブラリを使用する必要があります。
- [example scripts](https://github.com/huggingface/transformers/tree/main/examples)にあるスクリプトはあくまで*例*です。あなたの特定の問題に対してすぐに動作するわけではなく、あなたのニーズに合わせるためにコードを適応させる必要があるでしょう。

## Transformersを使用している100のプロジェクト

Transformersは事前学習済みモデルを使用するためのツールキット以上のものであり、それとHugging Face Hubを中心に構築されたプロジェクトのコミュニティです。私たちは、開発者、研究者、学生、教授、エンジニア、そしてその他の誰もが夢のプロジェクトを構築できるようにTransformersを提供したいと考えています。

Transformersの10万スターを記念して、Transformersで構築された100の素晴らしいプロジェクトをリストアップした[awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md)ページで、コミュニティにスポットライトを当てたいと考えました。

もしあなたがリストに加えるべきだと思うプロジェクトを所有または使用しているなら、ぜひPRを開いて追加してください！

## モデルの例

[Hubのモデルページ](https://huggingface.co/models)で、ほとんどのモデルを直接テストすることができます。

以下の各モダリティを展開して、さまざまなユースケースのモデル例をいくつか確認してください。

<details>
<summary>音声</summary>

- [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)による音声分類
- [Moonshine](https://huggingface.co/UsefulSensors/moonshine)による自動音声認識
- [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)によるキーワードスポッティング
- [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)による音声対音声生成
- [MusicGen](https://huggingface.co/facebook/musicgen-large)によるテキスト対音声
- [Bark](https://huggingface.co/suno/bark)によるテキスト読み上げ

</details>

<details>
<summary>コンピュータビジョン</summary>

- [SAM](https://huggingface.co/facebook/sam-vit-base)による自動マスク生成
- [DepthPro](https://huggingface.co/apple/DepthPro-hf)による深度推定
- [DINO v2](https://huggingface.co/facebook/dinov2-base)による画像分類
- [SuperPoint](https://huggingface.co/magic-leap-community/superpoint)によるキーポイント検出
- [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)によるキーポイントマッチング
- [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)による物体検出
- [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)による姿勢推定
- [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)によるユニバーサルセグメンテーション
- [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)による動画分類

</details>

<details>
<summary>マルチモーダル</summary>

- [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)による音声またはテキスト対テキスト
- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)による文書質問応答
- [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)による画像またはテキスト対テキスト
- [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)による画像キャプション
- [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)によるOCRベースの文書理解
- [TAPAS](https://huggingface.co/google/tapas-base)による表質問応答
- [Emu3](https://huggingface.co/BAAI/Emu3-Gen)による統一マルチモーダル理解と生成
- [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)による視覚対テキスト
- [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)による視覚的質問応答
- [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)による視覚的参照表現セグメンテーション

</details>

<details>
<summary>自然言語処理 (NLP)</summary>

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)によるマスク単語補完
- [Gemma](https://huggingface.co/google/gemma-2-2b)による固有表現認識
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)による質問応答
- [BART](https://huggingface.co/facebook/bart-large-cnn)による要約
- [T5](https://huggingface.co/google-t5/t5-base)による翻訳
- [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)によるテキスト生成
- [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)によるテキスト分類

</details>

## 引用

🤗 Transformersライブラリについて引用できる[論文](https://www.aclweb.org/anthology/2020.emnlp-demos.6/)ができました:
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
