<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Inference on a Single GPU

このガイドに加えて、[1つのGPUでのトレーニングガイド](perf_train_gpu_one)と[CPUでの推論ガイド](perf_infer_cpu)に関連する情報があります。

## Flash Attention 2

Flash Attention 2は、トランスフォーマーベースのモデルのトレーニングと推論速度を大幅に高速化できます。Flash Attention 2は、Tri Dao氏によって[公式のFlash Attentionリポジトリ](https://github.com/Dao-AILab/flash-attention)で導入されました。Flash Attentionに関する科学論文は[こちら](https://huggingface.co/papers/2205.14135)で見ることができます。

Flash Attention 2を正しくインストールするには、上記のリポジトリに記載されているインストールガイドに従ってください。

以下のモデルに対してFlash Attention 2をネイティブサポートしています：

- Llama
- Falcon

<Tip>

Flash Attention 2は、モデルのdtypeが`fp16`または`bf16`の場合にのみ使用でき、NVIDIA-GPUデバイスでのみ実行されます。この機能を使用する前に、モデルを適切なdtypeにキャストし、サポートされているデバイスにロードしてください。

</Tip>

### Quick usage

モデルでFlash Attention 2を有効にするには、`from_pretrained`の引数に`attn_implementation="flash_attention_2"`を追加します。


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

こちらは、生成または微調整のために使用するテキストです。

### Expected speedups

特に長いシーケンスに対して、微調整と推論の際には、かなりの高速化が期待できます。ただし、Flash Attentionはパディングトークンを使用してアテンションスコアを計算しないため、シーケンスにパディングトークンが含まれる場合、バッチ推論においてアテンションスコアを手動でパッド/アンパッドする必要があり、パディングトークンを含むバッチ生成の大幅な遅延が発生します。

これを克服するために、トレーニング中にシーケンスにパディングトークンを使用せずにFlash Attentionを使用する必要があります（たとえば、データセットをパックすることにより、シーケンスを最大シーケンス長に達するまで連結することなど）。ここに[例](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py#L516)が提供されています。

以下は、パディングトークンのない場合に、シーケンス長が4096の[tiiuae/falcon-7b](https://hf.co/tiiuae/falcon-7b)に対する単純なフォワードパスの予想される高速化です。さまざまなバッチサイズが示されています：

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/falcon-7b-inference-large-seqlen.png">
</div>

以下は、パディングトークンのない場合に、シーケンス長が4096の[`meta-llama/Llama-7b-hf`](https://hf.co/meta-llama/Llama-7b-hf)に対する単純なフォワードパスの予想される高速化です。さまざまなバッチサイズが示されています：

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-7b-inference-large-seqlen.png">
</div>

パディングトークンを含むシーケンス（パディングトークンを使用してトレーニングまたは生成する）の場合、アテンションスコアを正しく計算するために入力シーケンスをアンパッド/パッドする必要があります。比較的小さいシーケンス長の場合、純粋なフォワードパスではパディングトークンが30%未満しか埋められていないため、これはわずかな高速化をもたらします。

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-small-seqlen-padding.png">
</div>

しかし、大きなシーケンス長の場合、純粋な推論（トレーニングも含む）には興味深い高速化が得られます。

Flash Attentionは、アテンション計算をよりメモリ効率の良いものにし、大きなシーケンス長でのCUDA OOMの問題を回避できるようにします。大きなシーケンス長に対して最大20のメモリ削減をもたらすことがあります。詳細については、[公式のFlash Attentionリポジトリ](https://github.com/Dao-AILab/flash-attention)をご覧ください。

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-large-seqlen-padding.png">
</div>


### Advanced usage

この機能をモデルの最適化に多くの既存の機能と組み合わせることができます。以下にいくつかの例を示します：

### Combining Flash Attention 2 and 8-bit models

この機能を8ビットの量子化と組み合わせることができます：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    attn_implementation="flash_attention_2",
)
```

### Combining Flash Attention 2 and 4-bit models

この機能を 4 ビットの量子化と組み合わせることができます：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    attn_implementation="flash_attention_2",
)
```

### Combining Flash Attention 2 and PEFT

この機能を使用して、Flash Attention 2をベースにアダプターをトレーニングする際にPEFTを組み合わせることができます。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM
from peft import LoraConfig

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    attn_implementation="flash_attention_2",
)

lora_config = LoraConfig(
    r=8,
    task_type="CAUSAL_LM"
)

model.add_adapter(lora_config)

... # train your model
```

## `bitsandbytes` integration for FP4 mixed-precision inference

FP4混合精度推論のための`bitsandbytes`統合

You can install `bitsandbytes` and benefit from easy model compression on GPUs. Using FP4 quantization you can expect to reduce up to 8x the model size compared to its native full precision version. Check out below how to get started.

`bitsandbytes`をインストールし、GPUで簡単なモデルの圧縮を利用できます。FP4量子化を使用すると、ネイティブのフルプレシジョンバージョンと比較してモデルサイズを最大8倍削減できることが期待できます。以下を確認して、どのように始めるかをご覧ください。

<Tip>

Note that this feature can also be used in a multi GPU setup.

この機能は、マルチGPUセットアップでも使用できることに注意してください。

</Tip>

### Requirements [[requirements-for-fp4-mixedprecision-inference]]

- Latest `bitsandbytes` library
`pip install bitsandbytes>=0.39.0`

- Install latest `accelerate` from source
`pip install git+https://github.com/huggingface/accelerate.git`

- Install latest `transformers` from source
`pip install git+https://github.com/huggingface/transformers.git`


### Running FP4 models - single GPU setup - Quickstart

以下のコードを実行することで、簡単に単一のGPUでFP4モデルを実行できます:


```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True))
```

注意: `device_map`はオプションですが、推論時に `device_map = 'auto'` を設定することが推奨されています。これにより、利用可能なリソースに効率的にモデルがディスパッチされます。

### Running FP4 models - multi GPU setup

混合4ビットモデルを複数のGPUにロードする方法は、単一GPUセットアップと同じです（単一GPUセットアップと同じコマンドです）：

```py
model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True))
```

しかし、`accelerate`を使用して、各GPUに割り当てるGPU RAMを制御することができます。以下のように、`max_memory`引数を使用します：


```py
max_memory_mapping = {0: "600MB", 1: "1GB"}
model_name = "bigscience/bloom-3b"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True), max_memory=max_memory_mapping
)
```

この例では、最初のGPUは600MBのメモリを使用し、2番目のGPUは1GBを使用します。

### Advanced usage

このメソッドのさらなる高度な使用法については、[量子化](main_classes/quantization)のドキュメンテーションページをご覧ください。

## `bitsandbytes` integration for Int8 mixed-precision matrix decomposition

<Tip>

この機能は、マルチGPU環境でも使用できます。

</Tip>

論文[`LLM.int8()：スケーラブルなTransformer向けの8ビット行列乗算`](https://huggingface.co/papers/2208.07339)によれば、Hugging Face統合がHub内のすべてのモデルでわずか数行のコードでサポートされています。このメソッドは、半精度（`float16`および`bfloat16`）の重みの場合に`nn.Linear`サイズを2倍、単精度（`float32`）の重みの場合は4倍に縮小し、外れ値に対してほとんど影響を与えません。

![HFxbitsandbytes.png](https://cdn-uploads.huggingface.co/production/uploads/1659861207959-62441d1d9fdefb55a0b7d12c.png)

Int8混合精度行列分解は、行列乗算を2つのストリームに分割することによって動作します：(1) システマティックな特徴外れ値ストリームがfp16で行列乗算（0.01%）、(2) int8行列乗算の通常のストリーム（99.9%）。この方法を使用すると、非常に大きなモデルに対して予測の劣化なしにint8推論が可能です。
このメソッドの詳細については、[論文](https://huggingface.co/papers/2208.07339)または[この統合に関するブログ記事](https://huggingface.co/blog/hf-bitsandbytes-integration)をご確認ください。

![MixedInt8.gif](https://cdn-uploads.huggingface.co/production/uploads/1660567469965-62441d1d9fdefb55a0b7d12c.gif)

なお、この機能を使用するにはGPUが必要であり、カーネルはGPU専用にコンパイルされている必要があります。この機能を使用する前に、モデルの1/4（またはハーフ精度の重みの場合は1/2）を保存するのに十分なGPUメモリがあることを確認してください。
このモジュールを使用する際のヘルプに関する詳細は、以下のノートをご覧いただくか、[Google Colabのデモ](#colab-demos)をご覧ください。

### Requirements [[requirements-for-int8-mixedprecision-matrix-decomposition]]

- `bitsandbytes<0.37.0`を使用する場合、NVIDIA GPUを使用していることを確認し、8ビットテンソルコアをサポートしていることを確認してください（Turing、Ampere、またはそれ以降のアーキテクチャー、例：T4、RTX20s RTX30s、A40-A100など）。`bitsandbytes>=0.37.0`の場合、すべてのGPUがサポートされるはずです。
- 正しいバージョンの`bitsandbytes`をインストールするには、次のコマンドを実行してください：
`pip install bitsandbytes>=0.31.5`
- `accelerate`をインストールします：
`pip install accelerate>=0.12.0`


### Running mixed-Int8 models - single GPU setup

必要なライブラリをインストールした後、ミックス 8 ビットモデルを読み込む方法は次の通りです：

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

以下はシンプルな例です：

* `pipeline()` 関数の代わりに、モデルの `generate()` メソッドを使用することをお勧めします。`pipeline()` 関数を使用して推論することは可能ですが、混合8ビットモデルに最適化されておらず、`generate()` メソッドを使用するよりも遅くなります。また、一部のサンプリング戦略（例：ヌクレウスサンプリング）は、`pipeline()` 関数では混合8ビットモデルではサポートされていません。
* すべての入力をモデルと同じデバイスに配置してください。


```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

### Running mixed-int8 models - multi GPU setup

複数のGPUに混合8ビットモデルをロードする方法は、次の通りです（シングルGPUセットアップと同じコマンドです）：

```py
model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

`accelerate`を使用して各GPUに割り当てるGPU RAMを制御する際には、以下のように`max_memory`引数を使用します：


```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping
)
```

In this example, the first GPU will use 1GB of memory and the second 2GB.

### Colab demos

この方法を使用すると、以前のGoogle Colabでは推論できなかったモデルに対して推論を行うことができます。以下は、Google Colabで8ビット量子化を使用してT5-11b（fp32で42GB）を実行するデモのリンクです：

[![Open In Colab: T5-11b demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing)

また、BLOOM-3Bのデモもご覧いただけます：

[![Open In Colab: BLOOM-3b demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing)

