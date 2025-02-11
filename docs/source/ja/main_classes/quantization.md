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

# Quantize 🤗 Transformers models

## `AutoGPTQ` Integration


🤗 Transformers には、言語モデルで GPTQ 量子化を実行するための `optimum` API が統合されています。パフォーマンスを大幅に低下させることなく、推論速度を高速化することなく、モデルを 8、4、3、さらには 2 ビットでロードおよび量子化できます。これは、ほとんどの GPU ハードウェアでサポートされています。

量子化モデルの詳細については、以下を確認してください。
- [GPTQ](https://arxiv.org/pdf/2210.17323.pdf) 論文
- GPTQ 量子化に関する `optimum` [ガイド](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
- バックエンドとして使用される [`AutoGPTQ`](https://github.com/PanQiWei/AutoGPTQ) ライブラリ

### Requirements

以下のコードを実行するには、以下の要件がインストールされている必要があります： 

- 最新の `AutoGPTQ` ライブラリをインストールする。
`pip install auto-gptq` をインストールする。

- 最新の `optimum` をソースからインストールする。
`git+https://github.com/huggingface/optimum.git` をインストールする。

- 最新の `transformers` をソースからインストールする。
最新の `transformers` をソースからインストールする `pip install git+https://github.com/huggingface/transformers.git`

- 最新の `accelerate` ライブラリをインストールする。
`pip install --upgrade accelerate` を実行する。

GPTQ統合は今のところテキストモデルのみをサポートしているので、視覚、音声、マルチモーダルモデルでは予期せぬ挙動に遭遇するかもしれないことに注意してください。

### Load and quantize a model

GPTQ は、量子化モデルを使用する前に重みのキャリブレーションを必要とする量子化方法です。トランスフォーマー モデルを最初から量子化する場合は、量子化モデルを作成するまでに時間がかかることがあります (`facebook/opt-350m`モデルの Google colab では約 5 分)。

したがって、GPTQ 量子化モデルを使用するシナリオは 2 つあります。最初の使用例は、ハブで利用可能な他のユーザーによってすでに量子化されたモデルをロードすることです。2 番目の使用例は、モデルを最初から量子化し、保存するかハブにプッシュして、他のユーザーが使用できるようにすることです。それも使ってください。

#### GPTQ Configuration

モデルをロードして量子化するには、[`GPTQConfig`] を作成する必要があります。データセットを準備するには、`bits`の数、量子化を調整するための`dataset`、およびモデルの`Tokenizer`を渡す必要があります。

```python 
model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
```

独自のデータセットを文字列のリストとして渡すことができることに注意してください。ただし、GPTQ 論文のデータセットを使用することを強くお勧めします。

```python
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
quantization = GPTQConfig(bits=4, dataset = dataset, tokenizer=tokenizer)
```

#### Quantization

`from_pretrained` を使用し、`quantization_config` を設定することでモデルを量子化できます。

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=gptq_config)
```

モデルを量子化するには GPU が必要であることに注意してください。モデルを CPU に配置し、量子化するためにモジュールを GPU に前後に移動させます。

CPU オフロードの使用中に GPU の使用量を最大化したい場合は、`device_map = "auto"` を設定できます。

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)
```

ディスク オフロードはサポートされていないことに注意してください。さらに、データセットが原因でメモリが不足している場合は、`from_pretained` で `max_memory` を渡す必要がある場合があります。 `device_map`と`max_memory`の詳細については、この [ガイド](https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map) を参照してください。

<Tip warning={true}>
GPTQ 量子化は、現時点ではテキスト モデルでのみ機能します。さらに、量子化プロセスはハードウェアによっては長時間かかる場合があります (NVIDIA A100 を使用した場合、175B モデル = 4 gpu 時間)。モデルの GPTQ 量子化バージョンが存在しない場合は、ハブで確認してください。そうでない場合は、github で要求を送信できます。
</Tip>

### Push quantized model to 🤗 Hub

他の 🤗 モデルと同様に、`push_to_hub` を使用して量子化モデルをハブにプッシュできます。量子化構成は保存され、モデルに沿ってプッシュされます。

```python
quantized_model.push_to_hub("opt-125m-gptq")
tokenizer.push_to_hub("opt-125m-gptq")
```

量子化されたモデルをローカル マシンに保存したい場合は、`save_pretrained` を使用して行うこともできます。


```python
quantized_model.save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")
```

`device_map` を使用してモデルを量子化した場合は、保存する前にモデル全体を GPU または `cpu` のいずれかに移動してください。

```python
quantized_model.to("cpu")
quantized_model.save_pretrained("opt-125m-gptq")
```

### Load a quantized model from the 🤗 Hub

`from_pretrained`を使用して、量子化されたモデルをハブからロードできます。
属性 `quantization_config` がモデル設定オブジェクトに存在することを確認して、プッシュされた重みが量子化されていることを確認します。

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq")
```

必要以上のメモリを割り当てずにモデルをより速くロードしたい場合は、`device_map` 引数は量子化モデルでも機能します。 `accelerate`ライブラリがインストールされていることを確認してください。

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto")
```

### Exllama kernels for faster inference

4 ビット モデルの場合、推論速度を高めるために exllama カーネルを使用できます。デフォルトで有効になっています。 [`GPTQConfig`] で `disable_exllama` を渡すことで、その動作を変更できます。これにより、設定に保存されている量子化設定が上書きされます。カーネルに関連する属性のみを上書きできることに注意してください。さらに、exllama カーネルを使用したい場合は、モデル全体を GPU 上に置く必要があります。


```py
import torch
gptq_config = GPTQConfig(bits=4, disable_exllama=False)
model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config = gptq_config)
```

現時点では 4 ビット モデルのみがサポートされていることに注意してください。さらに、peft を使用して量子化モデルを微調整している場合は、exllama カーネルを非アクティブ化することをお勧めします。

#### Fine-tune a quantized model 

Hugging Face エコシステムのアダプターの公式サポートにより、GPTQ で量子化されたモデルを微調整できます。
詳細については、[`peft`](https://github.com/huggingface/peft) ライブラリをご覧ください。

### Example demo

GPTQ を使用してモデルを量子化する方法と、peft を使用して量子化されたモデルを微調整する方法については、Google Colab [ノートブック](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) を参照してください。

### GPTQConfig

[[autodoc]] GPTQConfig

## `bitsandbytes` Integration

🤗 Transformers は、`bitsandbytes` で最もよく使用されるモジュールと緊密に統合されています。数行のコードでモデルを 8 ビット精度でロードできます。
これは、`bitsandbytes`の `0.37.0`リリース以降、ほとんどの GPU ハードウェアでサポートされています。

量子化方法の詳細については、[LLM.int8()](https://arxiv.org/abs/2208.07339) 論文、または [ブログ投稿](https://huggingface.co/blog/hf-bitsandbytes-) をご覧ください。統合）コラボレーションについて。

`0.39.0`リリース以降、FP4 データ型を活用し、4 ビット量子化を使用して`device_map`をサポートする任意のモデルをロードできます。

独自の pytorch モデルを量子化したい場合は、🤗 Accelerate ライブラリの [ドキュメント](https://huggingface.co/docs/accelerate/main/en/usage_guides/quantization) をチェックしてください。

`bitsandbytes`統合を使用してできることは次のとおりです

### General usage

モデルが 🤗 Accelerate による読み込みをサポートし、`torch.nn.Linear` レイヤーが含まれている限り、 [`~PreTrainedModel.from_pretrained`] メソッドを呼び出すときに `load_in_8bit` または `load_in_4bit` 引数を使用してモデルを量子化できます。これはどのようなモダリティでも同様に機能するはずです。

```python
from transformers import AutoModelForCausalLM

model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_8bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_4bit=True)
```

デフォルトでは、他のすべてのモジュール (例: `torch.nn.LayerNorm`) は `torch.float16` に変換されますが、その `dtype` を変更したい場合は、`torch_dtype` 引数を上書きできます。

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM

>>> model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_8bit=True, torch_dtype=torch.float32)
>>> model_8bit.model.decoder.layers[-1].final_layer_norm.weight.dtype
torch.float32
```

### FP4 quantization 

#### Requirements

以下のコード スニペットを実行する前に、以下の要件がインストールされていることを確認してください。

- 最新の`bitsandbytes`ライブラリ
`pip install bitsandbytes>=0.39.0`

- 最新の`accelerate`をインストールする
`pip install --upgrade accelerate`

- 最新の `transformers` をインストールする
`pip install --upgrade transformers`

#### Tips and best practices

- **高度な使用法:** 可能なすべてのオプションを使用した 4 ビット量子化の高度な使用法については、[この Google Colab ノートブック](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf) を参照してください。

- **`batch_size=1` による高速推論 :** bitsandbytes の `0.40.0` リリース以降、`batch_size=1` では高速推論の恩恵を受けることができます。 [これらのリリース ノート](https://github.com/TimDettmers/bitsandbytes/releases/tag/0.40.0) を確認し、この機能を活用するには`0.40.0`以降のバージョンを使用していることを確認してください。箱の。

- **トレーニング:** [QLoRA 論文](https://arxiv.org/abs/2305.14314) によると、4 ビット基本モデルをトレーニングする場合 (例: LoRA アダプターを使用)、`bnb_4bit_quant_type='nf4'` を使用する必要があります。 。

- **推論:** 推論の場合、`bnb_4bit_quant_type` はパフォーマンスに大きな影響を与えません。ただし、モデルの重みとの一貫性を保つために、必ず同じ `bnb_4bit_compute_dtype` および `torch_dtype` 引数を使用してください。


#### Load a large model in 4bit

`.from_pretrained` メソッドを呼び出すときに `load_in_4bit=True` を使用すると、メモリ使用量を (おおよそ) 4 で割ることができます。

```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
```

<Tip warning={true}>

モデルが 4 ビットでロードされると、現時点では量子化された重みをハブにプッシュすることはできないことに注意してください。 4 ビットの重みはまだサポートされていないため、トレーニングできないことにも注意してください。ただし、4 ビット モデルを使用して追加のパラメーターをトレーニングすることもできます。これについては次のセクションで説明します。

</Tip>

### Load a large model in 8bit

`.from_pretrained` メソッドを呼び出すときに `load_in_8bit=True` 引数を使用すると、メモリ要件をおよそ半分にしてモデルをロードできます。

```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

次に、通常 [`PreTrainedModel`] を使用するのと同じようにモデルを使用します。

`get_memory_footprint` メソッドを使用して、モデルのメモリ フットプリントを確認できます。

```python
print(model.get_memory_footprint())
```

この統合により、大きなモデルを小さなデバイスにロードし、問題なく実行できるようになりました。

<Tip warning={true}>
モデルが 8 ビットでロードされると、最新の `transformers`と`bitsandbytes`を使用する場合を除き、量子化された重みをハブにプッシュすることは現在不可能であることに注意してください。 8 ビットの重みはまだサポートされていないため、トレーニングできないことにも注意してください。ただし、8 ビット モデルを使用して追加のパラメーターをトレーニングすることもできます。これについては次のセクションで説明します。
また、`device_map` はオプションですが、利用可能なリソース上でモデルを効率的にディスパッチするため、推論には `device_map = 'auto'` を設定することが推奨されます。

</Tip>

#### Advanced use cases

ここでは、FP4 量子化を使用して実行できるいくつかの高度な使用例について説明します。

##### Change the compute dtype

compute dtype は、計算中に使用される dtype を変更するために使用されます。たとえば、隠し状態は`float32`にありますが、高速化のために計算を bf16 に設定できます。デフォルトでは、compute dtype は `float32` に設定されます。

```python
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

##### Using NF4 (Normal Float 4) data type 

NF4 データ型を使用することもできます。これは、正規分布を使用して初期化された重みに適合した新しい 4 ビット データ型です。その実行のために:

```python
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

##### Use nested quantization for more memory efficient inference

また、ネストされた量子化手法を使用することをお勧めします。これにより、パフォーマンスを追加することなく、より多くのメモリが節約されます。経験的な観察から、これにより、NVIDIA-T4 16GB 上でシーケンス長 1024、バッチ サイズ 1、勾配累積ステップ 4 の llama-13b モデルを微調整することが可能になります。

```python
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
```


### Push quantized models on the 🤗 Hub

`push_to_hub`メソッドを単純に使用することで、量子化されたモデルをハブにプッシュできます。これにより、最初に量子化構成ファイルがプッシュされ、次に量子化されたモデルの重みがプッシュされます。
この機能を使用できるようにするには、必ず `bitsandbytes>0.37.2` を使用してください (この記事の執筆時点では、`bitsandbytes==0.38.0.post1` でテストしました)。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", quantization_config=BitsAndBytesConfig(load_in_8bit=True))
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model.push_to_hub("bloom-560m-8bit")
```

<Tip warning={true}>

大規模なモデルでは、ハブ上で 8 ビット モデルをプッシュすることが強く推奨されます。これにより、コミュニティはメモリ フットプリントの削減と、たとえば Google Colab での大規模なモデルの読み込みによる恩恵を受けることができます。

</Tip>

### Load a quantized model from the 🤗 Hub

`from_pretrained`メソッドを使用して、ハブから量子化モデルをロードできます。属性 `quantization_config` がモデル設定オブジェクトに存在することを確認して、プッシュされた重みが量子化されていることを確認します。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```

この場合、引数 `load_in_8bit=True` を指定する必要はありませんが、`bitsandbytes` と `accelerate` がインストールされていることを確認する必要があることに注意してください。
また、`device_map` はオプションですが、利用可能なリソース上でモデルを効率的にディスパッチするため、推論には `device_map = 'auto'` を設定することが推奨されます。

### Advanced use cases

このセクションは、8 ビット モデルのロードと実行以外に何ができるかを探求したい上級ユーザーを対象としています。

#### Offload between `cpu` and `gpu`

この高度な使用例の 1 つは、モデルをロードし、`CPU`と`GPU`の間で重みをディスパッチできることです。 CPU 上でディスパッチされる重みは **8 ビットに変換されない**ため、`float32`に保持されることに注意してください。この機能は、非常に大規模なモデルを適合させ、そのモデルを GPU と CPU の間でディスパッチしたいユーザーを対象としています。

まず、`transformers` から [`BitsAndBytesConfig`] をロードし、属性 `llm_int8_enable_fp32_cpu_offload` を `True` に設定します。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

`bigscience/bloom-1b7`モデルをロードする必要があり、`lm_head`を除くモデル全体に​​適合するのに十分な GPU RAM があるとします。したがって、次のようにカスタム device_map を作成します。

```python
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}
```

そして、次のようにモデルをロードします。
```python
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

以上です！モデルを楽しんでください！

#### Play with `llm_int8_threshold`

`llm_int8_threshold` 引数を操作して、外れ値のしきい値を変更できます。 外れ値 とは、特定のしきい値より大きい隠れた状態の値です。
これは、`LLM.int8()`論文で説明されている外れ値検出の外れ値しきい値に対応します。このしきい値を超える隠し状態の値は外れ値とみなされ、それらの値に対する操作は fp16 で実行されます。通常、値は正規分布します。つまり、ほとんどの値は [-3.5, 3.5] の範囲内にありますが、大規模なモデルでは大きく異なる分布を示す例外的な系統的外れ値がいくつかあります。これらの外れ値は、多くの場合 [-60, -6] または [6, 60] の範囲内にあります。 Int8 量子化は、大きさが 5 程度までの値ではうまく機能しますが、それを超えると、パフォーマンスが大幅に低下します。適切なデフォルトのしきい値は 6 ですが、より不安定なモデル (小規模なモデル、微調整) では、より低いしきい値が必要になる場合があります。
この引数は、モデルの推論速度に影響を与える可能性があります。このパラメータを試してみて、ユースケースに最適なパラメータを見つけることをお勧めします。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_threshold=10,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### Skip the conversion of some modules

一部のモデルには、安定性を確保するために 8 ビットに変換する必要がないモジュールがいくつかあります。たとえば、ジュークボックス モデルには、スキップする必要があるいくつかの `lm_head` モジュールがあります。 `llm_int8_skip_modules` で遊んでみる

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### Fine-tune a model that has been loaded in 8-bit

Hugging Face エコシステムのアダプターの公式サポートにより、8 ビットでロードされたモデルを微調整できます。
これにより、単一の Google Colab で`flan-t5-large`や`facebook/opt-6.7b`などの大規模モデルを微調整することができます。詳細については、[`peft`](https://github.com/huggingface/peft) ライブラリをご覧ください。

トレーニング用のモデルをロードするときに `device_map` を渡す必要がないことに注意してください。モデルが GPU に自動的にロードされます。必要に応じて、デバイス マップを特定のデバイスに設定することもできます (例: `cuda:0`、`0`、`torch.device('cuda:0')`)。 `device_map=auto`は推論のみに使用する必要があることに注意してください。

### BitsAndBytesConfig

[[autodoc]] BitsAndBytesConfig

## Quantization with 🤗 `optimum` 

`optimum`でサポートされている量子化方法の詳細については、[Optimum ドキュメント](https://huggingface.co/docs/optimum/index) を参照し、これらが自分のユースケースに適用できるかどうかを確認してください。
