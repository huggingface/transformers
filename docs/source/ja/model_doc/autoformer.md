<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Autoformer

## 概要

Autoformerモデルは、「[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)」という論文でHaixu Wu、Jiehui Xu、Jianmin Wang、Mingsheng Longによって提案されました。

このモデルは、予測プロセス中にトレンドと季節性成分を逐次的に分解できる深層分解アーキテクチャとしてTransformerを増強します。

論文の要旨は以下の通りです：

*例えば異常気象の早期警告や長期的なエネルギー消費計画といった実応用において、予測時間を延長することは重要な要求です。本論文では、時系列の長期予測問題を研究しています。以前のTransformerベースのモデルは、長距離依存関係を発見するために様々なセルフアテンション機構を採用しています。しかし、長期未来の複雑な時間的パターンによってモデルが信頼できる依存関係を見つけることを妨げられます。また、Transformerは、長い系列の効率化のためにポイントワイズなセルフアテンションのスパースバージョンを採用する必要があり、情報利用のボトルネックとなります。Transformerを超えて、我々は自己相関機構を持つ新しい分解アーキテクチャとしてAutoformerを設計しました。系列分解の事前処理の慣行を破り、それを深層モデルの基本的な内部ブロックとして革新します。この設計は、複雑な時系列に対するAutoformerの進行的な分解能力を強化します。さらに、確率過程理論に触発されて、系列の周期性に基づいた自己相関機構を設計し、サブ系列レベルでの依存関係の発見と表現の集約を行います。自己相関は効率と精度の両方でセルフアテンションを上回ります。長期予測において、Autoformerは、エネルギー、交通、経済、気象、疾病の5つの実用的な応用をカバーする6つのベンチマークで38%の相対的な改善をもたらし、最先端の精度を達成します。*

このモデルは[elisim](https://huggingface.co/elisim)と[kashif](https://huggingface.co/kashif)より提供されました。
オリジナルのコードは[こちら](https://github.com/thuml/Autoformer)で見ることができます。

## 参考資料

Autoformerの使用を開始するのに役立つ公式のHugging Faceおよびコミュニティ（🌎で示されている）の参考資料の一覧です。ここに参考資料を提出したい場合は、気兼ねなくPull Requestを開いてください。私たちはそれをレビューいたします！参考資料は、既存のものを複製するのではなく、何か新しいことを示すことが理想的です。

- HuggingFaceブログでAutoformerに関するブログ記事をチェックしてください：[はい、Transformersは時系列予測に効果的です（+ Autoformer）](https://huggingface.co/blog/autoformer)

## AutoformerConfig

[[autodoc]] AutoformerConfig

## AutoformerModel

[[autodoc]] AutoformerModel
    - forward

## AutoformerForPrediction

[[autodoc]] AutoformerForPrediction
    - forward
