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

# Tokenizer

トークナイザーは、モデルの入力の準備を担当します。ライブラリには、すべてのモデルのトークナイザーが含まれています。ほとんど
トークナイザーの一部は、完全な Python 実装と、
Rust ライブラリ [🤗 Tokenizers](https://github.com/huggingface/tokenizers)。 「高速」実装では次のことが可能になります。

1. 特にバッチトークン化を行う場合の大幅なスピードアップと
2. 元の文字列 (文字と単語) とトークン空間の間でマッピングする追加のメソッド (例:
   特定の文字を含むトークンのインデックス、または特定のトークンに対応する文字の範囲）。

基本クラス [`PreTrainedTokenizer`] および [`PreTrainedTokenizerFast`]
モデル入力の文字列入力をエンコードし (以下を参照)、Python をインスタンス化/保存するための一般的なメソッドを実装します。
ローカル ファイルまたはディレクトリ、またはライブラリによって提供される事前トレーニング済みトークナイザーからの「高速」トークナイザー
(HuggingFace の AWS S3 リポジトリからダウンロード)。二人とも頼りにしているのは、
共通メソッドを含む [`~tokenization_utils_base.PreTrainedTokenizerBase`]
[`~tokenization_utils_base.SpecialTokensMixin`]。

したがって、[`PreTrainedTokenizer`] と [`PreTrainedTokenizerFast`] はメインを実装します。
すべてのトークナイザーを使用するためのメソッド:

- トークン化 (文字列をサブワード トークン文字列に分割)、トークン文字列を ID に変換したり、その逆の変換を行ったりします。
  エンコード/デコード (つまり、トークン化と整数への変換)。
- 基礎となる構造 (BPE、SentencePiece...) から独立した方法で、語彙に新しいトークンを追加します。
- 特別なトークン (マスク、文の始まりなど) の管理: トークンの追加、属性への割り当て。
  トークナイザーにより、簡単にアクセスでき、トークン化中に分割されないようにすることができます。

[`BatchEncoding`] は、
[`~tokenization_utils_base.PreTrainedTokenizerBase`] のエンコード メソッド (`__call__`、
`encode_plus` および `batch_encode_plus`) であり、Python 辞書から派生しています。トークナイザーが純粋な Python の場合
tokenizer の場合、このクラスは標準の Python 辞書と同じように動作し、によって計算されたさまざまなモデル入力を保持します。
これらのメソッド (`input_ids`、`attention_mask`...)。トークナイザーが「高速」トークナイザーである場合 (つまり、
HuggingFace [トークナイザー ライブラリ](https://github.com/huggingface/tokenizers))、このクラスはさらに提供します
元の文字列 (文字と単語) と
トークンスペース (例: 指定された文字または対応する文字の範囲を構成するトークンのインデックスの取得)
与えられたトークンに）。

## PreTrainedTokenizer

[[autodoc]] PreTrainedTokenizer
    - __call__
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## PreTrainedTokenizerFast

[`PreTrainedTokenizerFast`] は [tokenizers](https://huggingface.co/docs/tokenizers) ライブラリに依存します。 🤗 トークナイザー ライブラリから取得したトークナイザーは、
🤗 トランスに非常に簡単にロードされます。これがどのように行われるかを理解するには、[🤗 tokenizers からの tokenizers を使用する](../fast_tokenizers) ページを参照してください。

[[autodoc]] PreTrainedTokenizerFast
    - __call__
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## BatchEncoding

[[autodoc]] BatchEncoding
