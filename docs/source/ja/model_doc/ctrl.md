<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CTRL

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=Salesforce/ctrl">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-ctrl-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/tiny-ctrl">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

CTRL モデルは、Nitish Shirish Keskar*、Bryan McCann*、Lav R. Varshney、Caiming Xiong, Richard Socher によって [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) で提案されました。
リチャード・ソーチャー。これは、非常に大規模なコーパスの言語モデリングを使用して事前トレーニングされた因果的 (一方向) トランスフォーマーです
最初のトークンが制御コード (リンク、書籍、Wikipedia など) として予約されている、約 140 GB のテキスト データ。

論文の要約は次のとおりです。

*大規模な言語モデルは有望なテキスト生成機能を示していますが、ユーザーは特定の言語モデルを簡単に制御できません
生成されたテキストの側面。 16 億 3,000 万パラメータの条件付きトランスフォーマー言語モデルである CTRL をリリースします。
スタイル、コンテンツ、タスク固有の動作を制御する制御コードを条件付けるように訓練されています。制御コードは
生のテキストと自然に共生する構造から派生し、教師なし学習の利点を維持しながら、
テキスト生成をより明示的に制御できるようになります。これらのコードを使用すると、CTRL でどの部分が予測されるのかを予測することもできます。
トレーニング データにはシーケンスが与えられる可能性が最も高くなります。これにより、大量のデータを分析するための潜在的な方法が提供されます。
モデルベースのソース帰属を介して。*

このモデルは、[keskarnitishr](https://huggingface.co/keskarnitishr) によって提供されました。元のコードが見つかる
[こちら](https://github.com/salesforce/Salesforce/ctrl)。

## Usage tips

- CTRL は制御コードを利用してテキストを生成します。生成を特定の単語や文で開始する必要があります。
  またはリンクして一貫したテキストを生成します。 [元の実装](https://github.com/salesforce/Salesforce/ctrl) を参照してください。
  詳しくは。
- CTRL は絶対位置埋め込みを備えたモデルであるため、通常は入力を右側にパディングすることをお勧めします。
  左。
- CTRL は因果言語モデリング (CLM) の目的でトレーニングされているため、次の予測に強力です。
  シーケンス内のトークン。この機能を利用すると、CTRL は構文的に一貫したテキストを生成できるようになります。
  *run_generation.py* サンプル スクリプトで確認できます。
- PyTorch モデルは、以前に計算されたキーと値のアテンション ペアである`past_key_values`を入力として受け取ることができます。
  TensorFlow モデルは`past`を入力として受け入れます。 `past_key_values`値を使用すると、モデルが再計算されなくなります。
  テキスト生成のコンテキストで事前に計算された値。 [`forward`](model_doc/ctrl#transformers.CTRLModel.forward) を参照してください。
  この引数の使用法の詳細については、メソッドを参照してください。

## Resources

- [テキスト分類タスクガイド](../tasks/sequence_classification)
- [因果言語モデリング タスク ガイド](../tasks/language_modeling)

## CTRLConfig

[[autodoc]] CTRLConfig

## CTRLTokenizer

[[autodoc]] CTRLTokenizer
    - save_vocabulary

<frameworkcontent>
<pt>

## CTRLModel

[[autodoc]] CTRLModel
    - forward

## CTRLLMHeadModel

[[autodoc]] CTRLLMHeadModel
    - forward

## CTRLForSequenceClassification

[[autodoc]] CTRLForSequenceClassification
    - forward

</pt>
<tf>

## TFCTRLModel

[[autodoc]] TFCTRLModel
    - call

## TFCTRLLMHeadModel

[[autodoc]] TFCTRLLMHeadModel
    - call

## TFCTRLForSequenceClassification

[[autodoc]] TFCTRLForSequenceClassification
    - call

</tf>
</frameworkcontent>
