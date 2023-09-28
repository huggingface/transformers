<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Performance and Scalability

大規模なトランスフォーマーモデルのトレーニングおよび本番環境への展開はさまざまな課題を提起します。
トレーニング中には、モデルが利用可能なGPUメモリよりも多くを必要としたり、トレーニング速度が遅かったりする可能性があります。
デプロイフェーズでは、モデルが本番環境で必要なスループットを処理するのに苦労することがあります。

このドキュメンテーションは、これらの課題を克服し、ユースケースに最適な設定を見つけるのに役立つことを目的としています。
ガイドはトレーニングと推論のセクションに分かれており、それぞれ異なる課題と解決策が存在します。
各セクション内には、トレーニング用のシングルGPU対マルチGPU、推論用のCPU対GPUなど、異なるハードウェア構成用の別々のガイドが用意されています。

このドキュメントを出発点として、シナリオに合った方法に進むための情報源としてご利用ください。

## Training

大規模なトランスフォーマーモデルを効率的にトレーニングするには、GPUやTPUなどのアクセラレータが必要です。
最も一般的なケースは、シングルGPUがある場合です。シングルGPUでのトレーニング効率を最適化するための一般的なアプローチを学ぶには、以下を参照してください。

* [シングルGPUでの効率的なトレーニングのための方法とツール](perf_train_gpu_one): GPUメモリの効果的な利用、トレーニングの高速化などを支援する共通のアプローチを学ぶためにここから始めてください。
* [マルチGPUトレーニングセクション](perf_train_gpu_many): マルチGPU環境に適用されるデータ、テンソル、パイプライン並列性など、さらなる最適化方法について詳細に学びます。
* [CPUトレーニングセクション](perf_train_cpu): CPU上での混合精度トレーニングについて学びます。
* [複数CPUでの効率的なトレーニング](perf_train_cpu_many): 分散CPUトレーニングについて学びます。
* [TensorFlowでTPUを使用したトレーニング](perf_train_tpu_tf): TPUに慣れていない場合は、TPUでのトレーニングとXLAの使用についてのセクションを参照してください。
* [トレーニングのためのカスタムハードウェア](perf_hardware): 独自のディープラーニング環境を構築する際のヒントやトリックを見つけます。
* [Trainer APIを使用したハイパーパラメーター検索](hpo_train)

## Inference

本番環境で大規模なモデルを効率的に推論することは、それらをトレーニングすることと同じくらい難しいことがあります。
以下のセクションでは、CPUおよびシングル/マルチGPU環境で推論を実行する手順について説明します。

* [シングルCPUでの推論](perf_infer_cpu)
* [シングルGPUでの推論](perf_infer_gpu_one)
* [マルチGPU推論](perf_infer_gpu_many)
* [TensorFlowモデルのXLA統合](tf_xla)

## Training and inference

モデルをトレーニングするか、それを使用して推論を実行するかに関係なく適用されるテクニック、ヒント、トリックがここにあります。

* [大規模モデルのインスタンス化](big_models)
* [パフォーマンスの問題のトラブルシューティング](debugging)

## Contribute

このドキュメントはまだ完全ではなく、さらに追加する必要がある項目がたくさんあります。
追加や訂正が必要な場合は、遠慮せずにPRをオープンするか、詳細を議論するためにIssueを開始してください。

AがBよりも優れているという貢献を行う際には、再現可能なベンチマークやその情報の出典へのリンクを含めてみてください（あなた自身の情報である場合を除く）。
