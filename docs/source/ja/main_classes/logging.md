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

# Logging

🤗 Transformersには、ライブラリの詳細度を簡単に設定できる中央集中型のロギングシステムがあります。

現在、ライブラリのデフォルトの詳細度は「WARNING」です。

詳細度を変更するには、直接設定メソッドの1つを使用するだけです。例えば、詳細度をINFOレベルに変更する方法は以下の通りです。


```python
import transformers

transformers.logging.set_verbosity_info()
```


環境変数 `TRANSFORMERS_VERBOSITY` を使用して、デフォルトの冗長性をオーバーライドすることもできます。設定できます
`debug`、`info`、`warning`、`error`、`critical` のいずれかに変更します。例えば：

```bash
TRANSFORMERS_VERBOSITY=error ./myprogram.py
```


さらに、一部の「警告」は環境変数を設定することで無効にできます。
`TRANSFORMERS_NO_ADVISORY_WARNINGS` を *1* などの true 値に設定します。これにより、次を使用してログに記録される警告が無効になります。
[`logger.warning_advice`]。例えば：

```bash
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
```


以下は、独自のモジュールまたはスクリプトでライブラリと同じロガーを使用する方法の例です。

```python
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
```

このロギング モジュールのすべてのメソッドは以下に文書化されています。主なメソッドは次のとおりです。
[`logging.get_verbosity`] ロガーの現在の冗長レベルを取得します。
[`logging.set_verbosity`] を使用して、冗長性を選択したレベルに設定します。順番に（少ないものから）
冗長から最も冗長まで)、それらのレベル (括弧内は対応する int 値) は次のとおりです。

- `transformers.logging.CRITICAL` または `transformers.logging.FATAL` (int 値、50): 最も多いもののみをレポートします。
  重大なエラー。
- `transformers.logging.ERROR` (int 値、40): エラーのみを報告します。
- `transformers.logging.WARNING` または `transformers.logging.WARN` (int 値、30): エラーと
  警告。これはライブラリで使用されるデフォルトのレベルです。
- `transformers.logging.INFO` (int 値、20): エラー、警告、および基本情報をレポートします。
- `transformers.logging.DEBUG` (int 値、10): すべての情報をレポートします。

デフォルトでは、モデルのダウンロード中に「tqdm」進行状況バーが表示されます。 [`logging.disable_progress_bar`] および [`logging.enable_progress_bar`] を使用して、この動作を抑制または抑制解除できます。

## `logging` vs `warnings`

Python には、よく組み合わせて使用​​される 2 つのロギング システムがあります。上で説明した `logging` と `warnings` です。
これにより、特定のバケット内の警告をさらに分類できます (例: 機能またはパスの`FutureWarning`)
これはすでに非推奨になっており、`DeprecationWarning`は今後の非推奨を示します。

両方とも`transformers`ライブラリで使用します。 `logging`の`captureWarning`メソッドを活用して適応させて、
これらの警告メッセージは、上記の冗長設定ツールによって管理されます。

それはライブラリの開発者にとって何を意味しますか?次のヒューリスティックを尊重する必要があります。
- `warnings`は、ライブラリおよび`transformers`に依存するライブラリの開発者に優先されるべきです。
- `logging`は、日常のプロジェクトでライブラリを使用するライブラリのエンドユーザーに使用する必要があります。

以下の`captureWarnings`メソッドのリファレンスを参照してください。

[[autodoc]] logging.captureWarnings

## Base setters

[[autodoc]] logging.set_verbosity_error

[[autodoc]] logging.set_verbosity_warning

[[autodoc]] logging.set_verbosity_info

[[autodoc]] logging.set_verbosity_debug

## Other functions

[[autodoc]] logging.get_verbosity

[[autodoc]] logging.set_verbosity

[[autodoc]] logging.get_logger

[[autodoc]] logging.enable_default_handler

[[autodoc]] logging.disable_default_handler

[[autodoc]] logging.enable_explicit_format

[[autodoc]] logging.reset_format

[[autodoc]] logging.enable_progress_bar

[[autodoc]] logging.disable_progress_bar
