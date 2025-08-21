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

# XLA Integration for TensorFlow Models

[[open-in-colab]]

加速線形代数（Accelerated Linear Algebra）、通称XLAは、TensorFlowモデルのランタイムを高速化するためのコンパイラです。[公式ドキュメント](https://www.tensorflow.org/xla)によれば、XLA（Accelerated Linear Algebra）は線形代数のためのドメイン固有のコンパイラで、TensorFlowモデルを潜在的にソースコードの変更なしで高速化できます。

TensorFlowでXLAを使用するのは簡単です。XLAは`tensorflow`ライブラリ内にパッケージ化されており、[`tf.function`](https://www.tensorflow.org/guide/intro_to_graphs)などのグラフを作成する関数内で`jit_compile`引数を使用してトリガーできます。`fit()`や`predict()`などのKerasメソッドを使用する場合、`model.compile()`に`jit_compile`引数を渡すだけでXLAを有効にできます。ただし、XLAはこれらのメソッドに限定されているわけではありません。任意の`tf.function`を高速化するためにも使用できます。

🤗 Transformers内のいくつかのTensorFlowメソッドは、XLAと互換性があるように書き直されています。これには、[GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)、[T5](https://huggingface.co/docs/transformers/model_doc/t5)、[OPT](https://huggingface.co/docs/transformers/model_doc/opt)などのテキスト生成モデルや、[Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)などの音声処理モデルも含まれます。

速度向上の具体的な量はモデルに非常に依存しますが、🤗 Transformers内のTensorFlowテキスト生成モデルでは、約100倍の速度向上を確認しています。このドキュメントでは、これらのモデルにXLAを使用して最大のパフォーマンスを得る方法を説明します。また、ベンチマークとXLA統合のデザイン哲学について詳しく学びたい場合の追加リソースへのリンクも提供します。

## Running TF functions with XLA

以下のTensorFlowモデルを考えてみましょう：


```py
import tensorflow as tf

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, input_shape=(10,), activation="relu"), tf.keras.layers.Dense(5, activation="softmax")]
)
```

上記のモデルは、次元が`(10, )`の入力を受け入れます。このモデルをフォワードパスで実行するには、次のようにします：


```py
# Generate random inputs for the model.
batch_size = 16
input_vector_dim = 10
random_inputs = tf.random.normal((batch_size, input_vector_dim))

# Run a forward pass.
_ = model(random_inputs)
```

XLAでコンパイルされた関数を使用してフォワードパスを実行するには、以下のようにします：


```py
xla_fn = tf.function(model, jit_compile=True)
_ = xla_fn(random_inputs)
```

`model`のデフォルトの `call()` 関数はXLAグラフをコンパイルするために使用されます。ただし、XLAにコンパイルしたい他のモデル関数がある場合、それも可能です。以下はその方法です：


```py
my_xla_fn = tf.function(model.my_xla_fn, jit_compile=True)
```

## Running a TF text generation model with XLA from 🤗 Transformers

🤗 Transformers内でXLAでの高速化された生成を有効にするには、最新バージョンの`transformers`がインストールされている必要があります。次のコマンドを実行してインストールできます：

```bash
pip install transformers --upgrade
```

次に、次のコードを実行できます：


```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# Will error if the minimal version of Transformers is not installed.
from transformers.utils import check_min_version

check_min_version("4.21.0")


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

# One line to create an XLA generation function
xla_generate = tf.function(model.generate, jit_compile=True)

tokenized_input = tokenizer(input_string, return_tensors="tf")
generated_tokens = xla_generate(**tokenized_input, num_beams=2)

decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
# Generated -- TensorFlow is an open-source, open-source, distributed-source application # framework for the
```

`generate()`でXLAを有効にするのは、たった一行のコードです。コードの残り部分は変更されていません。ただし、XLA固有のいくつかの注意点が上記のコードスニペットにあります。これらに注意する必要があり、XLAがもたらす速度向上を実現するためにそれらを把握することが重要です。次のセクションでこれらについて詳しく説明します。


## Gotchas to be aware of

XLAを有効にした関数（上記の`xla_generate()`など）を初めて実行すると、内部で計算グラフを推論しようとしますが、これは時間がかかります。このプロセスは["トレーシング"（tracing）](https://www.tensorflow.org/guide/intro_to_graphs#when_is_a_function_tracing)として知られています。

生成時間が高速ではないことに気付くかもしれません。`xla_generate()`（または他のXLA対応関数）の連続呼び出しでは、関数への入力が最初に計算グラフが構築されたときと同じ形状に従っている場合、計算グラフを推論する必要はありません。これは、入力形状が固定されているモダリティ（例：画像）には問題ありませんが、変数の入力形状モダリティ（例：テキスト）を扱う場合には注意が必要です。

`xla_generate()`が常に同じ入力形状で動作するようにするには、トークナイザを呼び出す際に`padding`引数を指定できます。

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

xla_generate = tf.function(model.generate, jit_compile=True)

# Here, we call the tokenizer with padding options.
tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")

generated_tokens = xla_generate(**tokenized_input, num_beams=2)
decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
```

これにより、`xla_generate()`への入力が常にトレースされた形状の入力を受け取ることを確認し、生成時間の高速化を実現できます。以下のコードでこれを確認できます：

```py
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")

xla_generate = tf.function(model.generate, jit_compile=True)

for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
    tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")
    start = time.time_ns()
    generated_tokens = xla_generate(**tokenized_input, num_beams=2)
    end = time.time_ns()
    print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
```

Tesla T4 GPUを使用すると、次のような出力が期待されます：

```bash
Execution time -- 30819.6 ms

Execution time -- 79.0 ms

Execution time -- 78.9 ms
```

最初の`xla_generate()`呼び出しはトレーシングのために時間がかかりますが、連続する呼び出しは桁違いに高速です。生成オプションのいかなる変更も、再トレーシングを引き起こし、生成時間の遅延を引き起こすことに注意してください。

このドキュメントでは、🤗 Transformersが提供するテキスト生成オプションをすべて網羅していません。高度なユースケースについてはドキュメンテーションを参照することをお勧めします。

## Additional Resources

ここでは、🤗 Transformersと一般的なXLAについてさらに詳しく学びたい場合のいくつかの追加リソースを提供します。

* [このColab Notebook](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb)では、XLA対応のエンコーダーデコーダー（[T5](https://huggingface.co/docs/transformers/model_doc/t5)など）およびデコーダー専用（[GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)など）テキスト生成モデルを試すための対話型デモが提供されています。
* [このブログ記事](https://huggingface.co/blog/tf-xla-generate)では、XLA対応モデルの比較ベンチマークの概要と、TensorFlowでのXLAについての友好的な紹介が提供されています。
* [このブログ記事](https://blog.tensorflow.org/2022/11/how-hugging-face-improved-text-generation-performance-with-xla.html)では、🤗 TransformersのTensorFlowモデルにXLAサポートを追加する際の設計哲学について説明しています。
* 一般的なXLAとTensorFlowグラフについて詳しく学ぶためのおすすめの投稿：
    * [XLA: 機械学習用の最適化コンパイラ](https://www.tensorflow.org/xla)
    * [グラフと`tf.function`の紹介](https://www.tensorflow.org/guide/intro_to_graphs)
    * [`tf.function`を使用したパフォーマンス向上](https://www.tensorflow.org/guide/function)
