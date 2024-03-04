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

# トレーナー用ユーティリティ

このページには、[`Trainer`] で使用されるすべてのユーティリティ関数がリストされています。

これらのほとんどは、ライブラリ内のトレーナーのコードを学習する場合にのみ役に立ちます。

## Utilities

[[autodoc]] EvalPrediction

[[autodoc]] IntervalStrategy

[[autodoc]] enable_full_determinism

[[autodoc]] set_seed

[[autodoc]] torch_distributed_zero_first

## Callbacks internals

[[autodoc]] trainer_callback.CallbackHandler

## Distributed Evaluation

[[autodoc]] trainer_pt_utils.DistributedTensorGatherer

## Trainer Argument Parser

[[autodoc]] HfArgumentParser

## Debug Utilities

[[autodoc]] debug_utils.DebugUnderflowOverflow
