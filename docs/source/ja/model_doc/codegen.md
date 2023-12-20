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

# CodeGen

## Overview


CodeGen モデルは、[A Conversational Paradigm for Program Synthesis](https://arxiv.org/abs/2203.13474) で Erik Nijkamp、Bo Pang、林宏明、Lifu Tu、Huan Wang、Yingbo Zhou、Silvio Savarese、Caiming Xiong およびカイミン・ションさん。

CodeGen は、[The Pile](https://pile.eleuther.ai/)、BigQuery、BigPython で順次トレーニングされたプログラム合成用の自己回帰言語モデルです。

論文の要約は次のとおりです。

*プログラム合成は、与えられた問題仕様の解決策としてコンピューター プログラムを生成することを目的としています。我々は、大規模な言語モデルを介した会話型プログラム合成アプローチを提案します。これは、従来のアプローチで直面した広大なプログラム空間とユーザーの意図の仕様を検索するという課題に対処します。私たちの新しいアプローチでは、仕様とプログラムを作成するプロセスを、ユーザーとシステムの間の複数回の対話として捉えます。これはプログラム合成をシーケンス予測問題として扱い、仕様が自然言語で表現され、目的のプログラムが条件付きでサンプリングされます。私たちは、自然言語とプログラミング言語のデータに基づいて、CodeGen と呼ばれる大規模な言語モデルのファミリーをトレーニングします。データの監視が弱く、データ サイズとモデル サイズが拡大すると、単純な自己回帰言語モデリングから会話能力が生まれます。会話型プログラム合成におけるモデルの動作を研究するために、マルチターン プログラミング ベンチマーク (MTPB) を開発します。このベンチマークでは、各問題を解決するには、ユーザーとモデル間のマルチターン会話を介したマルチステップ合成が必要です。私たちの調査結果は、会話機能の出現と、提案されている会話プログラム合成パラダイムの有効性を示しています。さらに、私たちのモデル CodeGen (TPU-v4 でトレーニングされた最大 16B パラメーターを含む) は、HumanEval ベンチマークで OpenAI の Codex を上回ります。私たちはチェックポイントを含むトレーニング ライブラリ JaxFormer をオープン ソースのコントリビューションとして利用できるようにしています: [この https URL](https://github.com/salesforce/codegen)*。

このモデルは [林 宏明](https://huggingface.co/rooa) によって寄稿されました。
元のコードは [ここ](https://github.com/salesforce/codegen) にあります。

## Checkpoint Naming

* CodeGen モデル [チェックポイント](https://huggingface.co/models?other=codegen) は、可変サイズのさまざまな事前トレーニング データで利用できます。
* 形式は「Salesforce/codegen-{size}-{data}」です。ここで、
  * `size`: `350M`、`2B`、`6B`、`16B`
  * `data`:
    * `nl`: パイルで事前トレーニング済み
    * `multi`: `nl` で初期化され、複数のプログラミング言語データでさらに事前トレーニングされます。
    * `mono`: `multi` で初期化され、Python データでさらに事前トレーニングされます。
* たとえば、`Salesforce/codegen-350M-mono` は、Pile、複数のプログラミング言語、および Python で順次事前トレーニングされた 3 億 5,000 万のパラメーターのチェックポイントを提供します。

## Usage example

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> checkpoint = "Salesforce/codegen-350M-mono"
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)

>>> text = "def hello_world():"

>>> completion = model.generate(**tokenizer(text, return_tensors="pt"))

>>> print(tokenizer.decode(completion[0]))
def hello_world():
    print("Hello World")

hello_world()
```

## Resources

- [因果言語モデリング タスク ガイド](../tasks/language_modeling)

## CodeGenConfig

[[autodoc]] CodeGenConfig
    - all

## CodeGenTokenizer

[[autodoc]] CodeGenTokenizer
    - save_vocabulary

## CodeGenTokenizerFast

[[autodoc]] CodeGenTokenizerFast

## CodeGenModel

[[autodoc]] CodeGenModel
    - forward

## CodeGenForCausalLM

[[autodoc]] CodeGenForCausalLM
    - forward
