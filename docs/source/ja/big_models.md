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

# Instantiating a big model

非常に大規模な事前学習済みモデルを使用する場合、RAMの使用量を最小限に抑えることは課題の1つです。通常のPyTorchのワークフローは次のとおりです：

1. ランダムな重みを持つモデルを作成します。
2. 事前学習済みの重みをロードします。
3. これらの事前学習済みの重みをランダムなモデルに配置します。

ステップ1と2の両方がメモリにモデルの完全なバージョンを必要とし、ほとんどの場合は問題ありませんが、モデルのサイズが数ギガバイトになると、これらの2つのコピーをRAMから排除することができなくなる可能性があります。さらに悪いことに、分散トレーニングを実行するために`torch.distributed`を使用している場合、各プロセスは事前学習済みモデルをロードし、これらの2つのコピーをRAMに保存します。

<Tip>

ランダムに作成されたモデルは、メモリ内に「空の」テンソルで初期化されます。これらのランダムな値は、メモリの特定のチャンクにあったものを使用します（したがって、ランダムな値はその時点でのメモリチャンク内の値です）。モデル/パラメータの種類に適した分布（たとえば、正規分布）に従うランダムな初期化は、ステップ3で初期化されていない重みに対して、できるだけ高速に実行されます！

</Tip>

このガイドでは、Transformersがこの問題に対処するために提供するソリューションを探ります。なお、これは現在も開発が進行中の分野であり、将来、ここで説明されているAPIがわずかに変更される可能性があることに注意してください。

## Sharded checkpoints

バージョン4.18.0から、10GBを超えるサイズのモデルチェックポイントは自動的に複数の小さな部分に分割されます。`model.save_pretrained(save_dir)`を実行する際に1つの単一のチェックポイントを持つ代わりに、いくつかの部分的なチェックポイント（それぞれのサイズが<10GB）と、パラメータ名をそれらが格納されているファイルにマップするインデックスが生成されます。

`max_shard_size`パラメータでシャーディング前の最大サイズを制御できるため、例として通常サイズのモデルと小さなシャードサイズを使用します。従来のBERTモデルを使用してみましょう。


```py
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
```

もし[`~PreTrainedModel.save_pretrained`]を使用して保存する場合、新しいフォルダが2つのファイルを含む形で作成されます: モデルの設定情報とその重み情報です。

```py
>>> import os
>>> import tempfile

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir)
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model.bin']
```

最大シャードサイズを200MBに設定します：

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin', 'pytorch_model.bin.index.json']
```

モデルの設定の上に、3つの異なる重みファイルと、`index.json`ファイルが見られます。これは私たちのインデックスです。
このようなチェックポイントは、[`~PreTrainedModel.from_pretrained`]メソッドを使用して完全に再ロードできます：

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     new_model = AutoModel.from_pretrained(tmp_dir)
```

主要な利点は、大規模なモデルの場合、上記のワークフローのステップ2において、各チェックポイントのシャードが前のシャードの後にロードされ、RAMのメモリ使用量をモデルのサイズと最大のシャードのサイズを合わせたものに制限できることです。

内部では、インデックスファイルが使用され、どのキーがチェックポイントに存在し、対応する重みがどこに格納されているかを判断します。このインデックスは通常のJSONファイルのように読み込むことができ、辞書として取得できます。


```py
>>> import json

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     with open(os.path.join(tmp_dir, "pytorch_model.bin.index.json"), "r") as f:
...         index = json.load(f)

>>> print(index.keys())
dict_keys(['metadata', 'weight_map'])
```

メタデータには現時点ではモデルの総サイズのみが含まれています。
将来的には他の情報を追加する予定です：

```py
>>> index["metadata"]
{'total_size': 433245184}
```

重みマップはこのインデックスの主要な部分であり、各パラメータ名（通常はPyTorchモデルの`state_dict`で見つかるもの）をその格納されているファイルにマップします：

```py
>>> index["weight_map"]
{'embeddings.LayerNorm.bias': 'pytorch_model-00001-of-00003.bin',
 'embeddings.LayerNorm.weight': 'pytorch_model-00001-of-00003.bin',
 ...
```

直接モデル内で[`~PreTrainedModel.from_pretrained`]を使用せずに、
シャーディングされたチェックポイントをロードしたい場合（フルチェックポイントの場合に`model.load_state_dict()`を使用するように行う方法）、[`~modeling_utils.load_sharded_checkpoint`]を使用する必要があります：


```py
>>> from transformers.modeling_utils import load_sharded_checkpoint

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     load_sharded_checkpoint(model, tmp_dir)
```


## Low memory loading

シャードされたチェックポイントは、上記のワークフローのステップ2におけるメモリ使用量を削減しますが、
低メモリの環境でそのモデルを使用するために、Accelerateライブラリに基づいた当社のツールを活用することをお勧めします。

詳細については、以下のガイドをご覧ください：[Accelerateを使用した大規模モデルの読み込み](./main_classes/model#large-model-loading)
