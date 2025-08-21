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

# Generation

各フレームワークには、それぞれの `GenerationMixin` クラスに実装されたテキスト生成のための Generate メソッドがあります。

- PyTorch [`~generation.GenerationMixin.generate`] は [`~generation.GenerationMixin`] に実装されています。
- TensorFlow [`~generation.TFGenerationMixin.generate`] は [`~generation.TFGenerationMixin`] に実装されています。
- Flax/JAX [`~generation.FlaxGenerationMixin.generate`] は [`~generation.FlaxGenerationMixin`] に実装されています。

選択したフレームワークに関係なく、[`~generation.GenerationConfig`] を使用して生成メソッドをパラメータ化できます。
クラスインスタンス。動作を制御する生成パラメータの完全なリストについては、このクラスを参照してください。
生成方法のこと。

モデルの生成構成を検査する方法、デフォルトとは何か、パラメーターをアドホックに変更する方法を学習するには、
カスタマイズされた生成構成を作成して保存する方法については、「
[テキスト生成戦略ガイド](../generation_strategies)。このガイドでは、関連機能の使用方法についても説明しています。
トークンストリーミングのような。

## GenerationConfig

[[autodoc]] generation.GenerationConfig
	- from_pretrained
	- from_model_config
	- save_pretrained

## GenerationMixin

[[autodoc]] generation.GenerationMixin
	- generate
	- compute_transition_scores

## TFGenerationMixin

[[autodoc]] generation.TFGenerationMixin
	- generate
	- compute_transition_scores

## FlaxGenerationMixin

[[autodoc]] generation.FlaxGenerationMixin
	- generate
