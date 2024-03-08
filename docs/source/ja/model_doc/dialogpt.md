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

# DialoGPT

## Overview

DialoGPT は、[DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536) で Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao,
Jianfeng Gao, Jingjing Liu, Bill Dolan.これは、から抽出された 147M 万の会話のようなやりとりでトレーニングされた GPT2 モデルです。
レディット。

論文の要約は次のとおりです。

*私たちは、大規模で調整可能なニューラル会話応答生成モデル DialoGPT (対話生成事前トレーニング済み) を紹介します。
変成器）。 Reddit のコメント チェーンから抽出された 1 億 4,700 万件の会話のようなやり取りを対象にトレーニングされました。
2005 年から 2017 年にかけて、DialoGPT は人間に近いパフォーマンスを達成するために Hugging Face PyTorch トランスフォーマーを拡張しました。
シングルターンダイアログ設定における自動評価と人間による評価の両方。会話システムが
DialoGPT を活用すると、強力なベースラインよりも関連性が高く、内容が充実し、コンテキストに一貫性のある応答が生成されます。
システム。神経反応の研究を促進するために、事前トレーニングされたモデルとトレーニング パイプラインが公開されています。
よりインテリジェントなオープンドメイン対話システムの生成と開発。*

元のコードは [ここ](https://github.com/microsoft/DialoGPT) にあります。

## Usage tips


- DialoGPT は絶対位置埋め込みを備えたモデルであるため、通常は入力を右側にパディングすることをお勧めします。
  左よりも。
- DialoGPT は、会話データの因果言語モデリング (CLM) 目標に基づいてトレーニングされているため、強力です
  オープンドメイン対話システムにおける応答生成時。
- DialoGPT を使用すると、[DialoGPT's model card](https://huggingface.co/microsoft/DialoGPT-medium) に示されているように、ユーザーはわずか 10 行のコードでチャット ボットを作成できます。

トレーニング：

DialoGPT をトレーニングまたは微調整するには、因果言語モデリング トレーニングを使用できます。公式論文を引用すると： *私たちは
OpenAI GPT-2に従って、マルチターン対話セッションを長いテキストとしてモデル化し、生成タスクを言語としてフレーム化します
モデリング。まず、ダイアログ セッション内のすべてのダイアログ ターンを長いテキスト x_1,..., x_N に連結します (N は
* 詳細については、元の論文を参照してください。

<Tip>

DialoGPT のアーキテクチャは GPT2 モデルに基づいています。API リファレンスと例については、[GPT2 のドキュメント ページ](openai-community/gpt2) を参照してください。

</Tip>
