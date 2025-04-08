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

# Padding and truncation

バッチ入力はしばしば異なる長さであり、固定サイズのテンソルに変換できないため、変動する長さのバッチから長方形のテンソルを作成するための戦略として、パディングと切り詰めがあります。パディングは、短いシーケンスがバッチ内の最長シーケンスまたはモデルが受け入れる最大長と同じ長さになるように、特別な**パディングトークン**を追加します。切り詰めは、長いシーケンスを切り詰めることで逆方向に機能します。

ほとんどの場合、バッチを最長シーケンスの長さにパディングし、モデルが受け入れる最大長に切り詰めることで、うまく動作します。ただし、APIはそれ以上の戦略もサポートしています。必要な3つの引数は次のとおりです：`padding`、`truncation`、および `max_length`。

`padding`引数はパディングを制御します。ブール値または文字列であることができます：

  - `True`または`'longest'`：バッチ内の最長シーケンスにパディングを追加します（シーケンスが1つしか提供されない場合、パディングは適用されません）。
  - `max_length'`：`max_length`引数で指定された長さまでパディングを追加します。または`max_length`が提供されていない場合はモデルが受け入れる最大長（`max_length=None`）。シーケンスが1つしか提供されている場合でも、パディングは適用されます。
  - `False`または`'do_not_pad'`：パディングは適用されません。これがデフォルトの動作です。

`truncation`引数は切り詰めを制御します。ブール値または文字列であることができます：

  - `True`または`'longest_first'`：最大長を`max_length`引数で指定するか、モデルが受け入れる最大長（`max_length=None`）まで切り詰めます。これはトークンごとに切り詰め、適切な長さに達するまでペア内の最長シーケンスからトークンを削除します。
  - `'only_second'`：最大長を`max_length`引数で指定するか、モデルが受け入れる最大長（`max_length=None`）まで切り詰めます。これはペアの2番目の文だけを切り詰めます（シーケンスのペアまたはシーケンスのバッチのペアが提供された場合）。
  - `'only_first'`：最大長を`max_length`引数で指定するか、モデルが受け入れる最大長（`max_length=None`）まで切り詰めます。これはペアの最初の文だけを切り詰めます（シーケンスのペアまたはシーケンスのバッチのペアが提供された場合）。
  - `False`または`'do_not_truncate'`：切り詰めは適用されません。これがデフォルトの動作です。

`max_length`引数はパディングと切り詰めの長さを制御します。整数または`None`であり、この場合、モデルが受け入れる最大入力長にデフォルトで設定されます。モデルに特定の最大入力長がない場合、`max_length`への切り詰めまたはパディングは無効になります。

以下の表は、パディングと切り詰めを設定する推奨方法を要約しています。以下の例のいずれかで入力シーケンスのペアを使用する場合、`truncation=True`を`['only_first', 'only_second', 'longest_first']`で選択した`STRATEGY`に置き換えることができます。つまり、`truncation='only_second'`または`truncation='longest_first'`を使用して、ペア内の両方のシーケンスを前述のように切り詰める方法を制御できます。



| Truncation                           | Padding                           | Instruction                                                                                 |
|--------------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------|
| no truncation                        | no padding                        | `tokenizer(batch_sentences)`                                                           |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True)` or                                          |
|                                      |                                   | `tokenizer(batch_sentences, padding='longest')`                                        |
|                                      | padding to max model input length | `tokenizer(batch_sentences, padding='max_length')`                                     |
|                                      | padding to specific length        | `tokenizer(batch_sentences, padding='max_length', max_length=42)`                      |
|                                      | padding to a multiple of a value  | `tokenizer(batch_sentences, padding=True, pad_to_multiple_of=8)`                       |
| truncation to max model input length | no padding                        | `tokenizer(batch_sentences, truncation=True)` or                                       |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY)`                                      |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True, truncation=True)` or                         |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY)`                        |
|                                      | padding to max model input length | `tokenizer(batch_sentences, padding='max_length', truncation=True)` or                 |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY)`                |
|                                      | padding to specific length        | Not possible                                                                                |
| truncation to specific length        | no padding                        | `tokenizer(batch_sentences, truncation=True, max_length=42)` or                        |
|                                      |                                   | `tokenizer(batch_sentences, truncation=STRATEGY, max_length=42)`                       |
|                                      | padding to max sequence in batch  | `tokenizer(batch_sentences, padding=True, truncation=True, max_length=42)` or          |
|                                      |                                   | `tokenizer(batch_sentences, padding=True, truncation=STRATEGY, max_length=42)`         |
|                                      | padding to max model input length | Not possible                                                                                |
|                                      | padding to specific length        | `tokenizer(batch_sentences, padding='max_length', truncation=True, max_length=42)` or  |
|                                      |                                   | `tokenizer(batch_sentences, padding='max_length', truncation=STRATEGY, max_length=42)` |
