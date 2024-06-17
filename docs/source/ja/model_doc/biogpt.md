<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BioGPT

## Overview

BioGPT モデルは、[BioGPT: generative pre-trained transformer for biomedical text generation and mining](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9) by Renqian Luo、Liai Sun、Yingce Xia、 Tao Qin、Sheng Zhang、Hoifung Poon、Tie-Yan Liu。 BioGPT は、生物医学テキストの生成とマイニングのための、ドメイン固有の生成事前トレーニング済み Transformer 言語モデルです。 BioGPT は、Transformer 言語モデルのバックボーンに従い、1,500 万の PubMed 抄録で最初から事前トレーニングされています。

論文の要約は次のとおりです。

*事前トレーニング済み言語モデルは、一般的な自然言語領域での大きな成功に触発されて、生物医学領域でますます注目を集めています。一般言語ドメインの事前トレーニング済み言語モデルの 2 つの主なブランチ、つまり BERT (およびそのバリアント) と GPT (およびそのバリアント) のうち、1 つ目は BioBERT や PubMedBERT などの生物医学ドメインで広く研究されています。これらはさまざまな下流の生物医学的タスクで大きな成功を収めていますが、生成能力の欠如により応用範囲が制限されています。この論文では、大規模な生物医学文献で事前トレーニングされたドメイン固有の生成 Transformer 言語モデルである BioGPT を提案します。私たちは 6 つの生物医学的自然言語処理タスクで BioGPT を評価し、ほとんどのタスクで私たちのモデルが以前のモデルよりも優れていることを実証しました。特に、BC5CDR、KD-DTI、DDI のエンドツーエンド関係抽出タスクではそれぞれ 44.98%、38.42%、40.76% の F1 スコアを獲得し、PubMedQA では 78.2% の精度を獲得し、新記録を樹立しました。テキスト生成に関する私たちのケーススタディは、生物医学文献における BioGPT の利点をさらに実証し、生物医学用語の流暢な説明を生成します。*

## Usage tips

- BioGPT は絶対位置埋め込みを備えたモデルであるため、通常は入力を左側ではなく右側にパディングすることをお勧めします。
- BioGPT は因果言語モデリング (CLM) 目的でトレーニングされているため、シーケンス内の次のトークンを予測するのに強力です。 run_generation.py サンプル スクリプトで確認できるように、この機能を利用すると、BioGPT は構文的に一貫したテキストを生成できます。
- モデルは、以前に計算されたキーと値のアテンション ペアである`past_key_values`(PyTorch の場合) を入力として受け取ることができます。この (past_key_values または past) 値を使用すると、モデルがテキスト生成のコンテキストで事前に計算された値を再計算できなくなります。 PyTorch の使用法の詳細については、BioGptForCausalLM.forward() メソッドの past_key_values 引数を参照してください。

このモデルは、[kamalkraj](https://huggingface.co/kamalkraj) によって提供されました。元のコードは [ここ](https://github.com/microsoft/BioGPT) にあります。

## Documentation resources

- [因果言語モデリング タスク ガイド](../tasks/language_modeling)

## BioGptConfig

[[autodoc]] BioGptConfig


## BioGptTokenizer

[[autodoc]] BioGptTokenizer
    - save_vocabulary


## BioGptModel

[[autodoc]] BioGptModel
    - forward


## BioGptForCausalLM

[[autodoc]] BioGptForCausalLM
    - forward

    
## BioGptForTokenClassification

[[autodoc]] BioGptForTokenClassification
    - forward


## BioGptForSequenceClassification

[[autodoc]] BioGptForSequenceClassification
    - forward

    