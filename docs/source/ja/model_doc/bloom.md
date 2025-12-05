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

# BLOOM

## Overview

BLOOM モデルは、[BigScience Workshop](https://bigscience.huggingface.co/) を通じてさまざまなバージョンで提案されています。 BigScience は、研究者が時間とリソースをプールして共同でより高い効果を達成する他のオープン サイエンス イニシアチブからインスピレーションを得ています。
BLOOM のアーキテクチャは基本的に GPT3 (次のトークン予測のための自己回帰モデル) に似ていますが、46 の異なる言語と 13 のプログラミング言語でトレーニングされています。
モデルのいくつかの小さいバージョンが同じデータセットでトレーニングされています。 BLOOM は次のバージョンで利用できます。

- [bloom-560m](https://huggingface.co/bigscience/bloom-560m)
- [bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)
- [bloom-3b](https://huggingface.co/bigscience/bloom-3b)
- [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
- [bloom](https://huggingface.co/bigscience/bloom) (176B parameters)

## Resources

BLOOM を使い始めるのに役立つ公式 Hugging Face およびコミュニティ (🌎 で示されている) リソースのリスト。ここに含めるリソースの送信に興味がある場合は、お気軽にプル リクエストを開いてください。審査させていただきます。リソースは、既存のリソースを複製するのではなく、何か新しいものを示すことが理想的です。

<PipelineTag pipeline="text-generation"/>

- [`BloomForCausalLM`] これによってサポートされています [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

以下も参照してください。
- [因果言語モデリング タスク ガイド](../tasks/language_modeling)
- [テキスト分類タスクガイド(英語版)](../../en/tasks/sequence_classification)
- [トークン分類タスクガイド](../tasks/token_classification)
- [質問回答タスク ガイド](../tasks/question_answering)


⚡️ 推論
-  に関するブログ  [最適化の話: ブルーム推論](https://huggingface.co/blog/bloom-inference-optimization)。
- に関するブログ [DeepSpeed と Accelerate を使用した信じられないほど高速な BLOOM 推論](https://huggingface.co/blog/bloom-inference-pytorch-scripts)。

⚙️トレーニング
- に関するブログ [BLOOM トレーニングの背後にあるテクノロジー](https://huggingface.co/blog/bloom-megatron-deepspeed)。

## BloomConfig

[[autodoc]] BloomConfig
    - all


## BloomModel

[[autodoc]] BloomModel
    - forward

## BloomForCausalLM

[[autodoc]] BloomForCausalLM
    - forward

## BloomForSequenceClassification

[[autodoc]] BloomForSequenceClassification
    - forward

## BloomForTokenClassification

[[autodoc]] BloomForTokenClassification
    - forward

## BloomForQuestionAnswering

[[autodoc]] BloomForQuestionAnswering
    - forward

