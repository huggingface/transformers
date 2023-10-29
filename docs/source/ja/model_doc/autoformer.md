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

## Overview

Autoformer モデルは、Haixu Wu、Jiehui Xu、Jianmin Wang、Mingsheng Long によって [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008) で提案されました。

このモデルは、Transformer を詳細な分解アーキテクチャとして拡張し、予測プロセス中に傾向と季節成分を段階的に分解できます。

論文の要約は次のとおりです。

*予測時間の延長は、異常気象の早期警報や長期的なエネルギー消費計画などの実際のアプリケーションにとって重要な需要です。この論文は時系列の長期予測問題を研究します。以前の Transformer ベースのモデルは、長距離の依存関係を検出するためにさまざまなセルフアテンション メカニズムを採用していました。ただし、長期的な将来の複雑な時間的パターンにより、モデルは信頼できる依存関係を見つけることができません。また、トランスフォーマーは、長期シリーズの効率性を高めるためにポイント単位のセルフアテンションのスパース バージョンを採用する必要があり、その結果、情報利用のボトルネックが生じます。 Transformers を超えて、私たちは Auto-Correlation メカニズムを備えた新しい分解アーキテクチャとして Autoformer を設計します。級数分解の前処理の慣例を打ち破り、ディープモデルの基本的な内部ブロックとして刷新します。この設計により、Autoformer は複雑な時系列に対する漸進的な分解機能を強化できます。さらに、確率過程理論に触発されて、系列の周期性に基づいた自動相関メカニズムを設計し、サブ系列レベルで依存関係の発見と表現の集約を実行します。自動相関は、効率と精度の両方で自己注意よりも優れています。長期予測では、Autoformer は最先端の精度をもたらし、エネルギー、交通、経済、気象、病気という 5 つの実際のアプリケーションをカバーする 6 つのベンチマークで相対的に 38% 改善しました。

このモデルは、[elisim](https://huggingface.co/elisim) および [kashif](https://huggingface.co/kashif) によって提供されました。
元のコードは [ここ](https://github.com/thuml/Autoformer) にあります。

## Resources

公式 Hugging Face とコミュニティ (🌎 で示されている) リソースのリストは、開始に役立ちます。ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

- HuggingFace ブログの Autoformer ブログ投稿を確認してください: [はい、トランスフォーマーは時系列予測に効果的です (+ Autoformer)](https://huggingface.co/blog/autoformer)

## AutoformerConfig

[[autodoc]] AutoformerConfig


## AutoformerModel

[[autodoc]] AutoformerModel
    - forward


## AutoformerForPrediction

[[autodoc]] AutoformerForPrediction
    - forward
