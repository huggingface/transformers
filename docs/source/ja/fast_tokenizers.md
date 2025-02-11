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

# Use tokenizers from 🤗 Tokenizers

[`PreTrainedTokenizerFast`]は[🤗 Tokenizers](https://huggingface.co/docs/tokenizers)ライブラリに依存しています。🤗 Tokenizersライブラリから取得したトークナイザーは、非常に簡単に🤗 Transformersにロードできます。

具体的な内容に入る前に、まずはいくつかの行でダミーのトークナイザーを作成することから始めましょう：


```python
>>> from tokenizers import Tokenizer
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import BpeTrainer
>>> from tokenizers.pre_tokenizers import Whitespace

>>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
>>> trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

>>> tokenizer.pre_tokenizer = Whitespace()
>>> files = [...]
>>> tokenizer.train(files, trainer)
```

私たちは今、定義したファイルにトレーニングされたトークナイザーを持っています。これをランタイムで引き続き使用するか、
将来の再利用のためにJSONファイルに保存することができます。

## Loading directly from the tokenizer object

🤗 Transformersライブラリでこのトークナイザーオブジェクトをどのように活用できるかを見てみましょう。[`PreTrainedTokenizerFast`]クラスは、
*tokenizer*オブジェクトを引数として受け入れ、簡単にインスタンス化できるようにします。


```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

このオブジェクトは、🤗 Transformers トークナイザーが共有するすべてのメソッドと一緒に使用できます！詳細については、[トークナイザーページ](main_classes/tokenizer)をご覧ください。

## Loading from a JSON file

JSONファイルからトークナイザーを読み込むには、まずトークナイザーを保存することから始めましょう：

```python
>>> tokenizer.save("tokenizer.json")
```

このファイルを保存したパスは、`PreTrainedTokenizerFast` の初期化メソッドに `tokenizer_file` パラメータを使用して渡すことができます：


```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

このオブジェクトは、🤗 Transformers トークナイザーが共有するすべてのメソッドと一緒に使用できるようになりました！詳細については、[トークナイザーページ](main_classes/tokenizer)をご覧ください。

