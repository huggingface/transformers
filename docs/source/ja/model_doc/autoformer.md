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

*Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning. This paper studies the long-term forecasting problem of time series. Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck. Going beyond Transformers, we design Autoformer as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of deep models. This design empowers Autoformer with progressive decomposition capacities for complex time series. Further, inspired by the stochastic process theory, we design the Auto-Correlation mechanism based on the series periodicity, which conducts the dependencies discovery and representation aggregation at the sub-series level. Auto-Correlation outperforms self-attention in both efficiency and accuracy. In long-term forecasting, Autoformer yields state-of-the-art accuracy, with a 38% relative improvement on six benchmarks, covering five practical applications: energy, traffic, economics, weather and disease.*

このモデルは[elisim](https://huggingface.co/elisim)と[kashif](https://huggingface.co/kashif)より提供されました。
オリジナルのコードは[こちら](https://github.com/thuml/Autoformer)で見ることができます。

## 参考資料

Autoformerの使用を開始するのに役立つ公式のHugging Faceおよびコミュニティ（🌎で示されている）の参考資料の一覧です。ここに参考資料を提出したい場合は、気兼ねなくPull Requestを開いてください。私たちはそれをレビューいたします！参考資料は、既存のものを複製するのではなく、何か新しいことを示すことが理想的です。

HuggingFaceのブログでAutoformerに関するブログ記事をチェックしてください：はい、トランスフォーマーは時系列予測に効果的です（+ Autoformer）
- HuggingFaceブログでAutoformerに関するブログ記事をチェックしてください：[はい、Transformersは時系列予測に効果的です（+ Autoformer）](https://huggingface.co/blog/autoformer)

## AutoformerConfig

[[autodoc]] AutoformerConfig

## AutoformerModel

[[autodoc]] AutoformerModel
    - forward

## AutoformerForPrediction

[[autodoc]] AutoformerForPrediction
    - forward
