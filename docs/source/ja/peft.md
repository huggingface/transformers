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


# Load adapters with 🤗 PEFT

[[open-in-colab]]

[Parameter-Efficient Fine Tuning (PEFT)](https://huggingface.co/blog/peft) メソッドは、事前学習済みモデルのパラメータをファインチューニング中に凍結し、その上にわずかな訓練可能なパラメータ（アダプター）を追加するアプローチです。アダプターは、タスク固有の情報を学習するために訓練されます。このアプローチは、メモリ使用量が少なく、完全にファインチューニングされたモデルと比較して計算リソースを低く抑えつつ、同等の結果を生成することが示されています。

PEFTで訓練されたアダプターは通常、完全なモデルのサイズよりも1桁小さく、共有、保存、読み込むのが便利です。

<div class="flex flex-col justify-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
  <figcaption class="text-center">Hubに格納されているOPTForCausalLMモデルのアダプター重みは、モデルの全体サイズの約6MBで、モデル重みの全サイズは約700MBです。</figcaption>
</div>

🤗 PEFTライブラリについて詳しく知りたい場合は、[ドキュメンテーション](https://huggingface.co/docs/peft/index)をご覧ください。

## Setup

🤗 PEFTをインストールして始めましょう：


```bash
pip install peft
```

新機能を試してみたい場合、ソースからライブラリをインストールすることに興味があるかもしれません：

```bash
pip install git+https://github.com/huggingface/peft.git
```

## Supported PEFT models

🤗 Transformersは、いくつかのPEFT（Parameter Efficient Fine-Tuning）メソッドをネイティブにサポートしており、ローカルまたはHubに格納されたアダプターウェイトを簡単に読み込んで実行またはトレーニングできます。以下のメソッドがサポートされています：

- [Low Rank Adapters](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [IA3](https://huggingface.co/docs/peft/conceptual_guides/ia3)
- [AdaLoRA](https://arxiv.org/abs/2303.10512)

他のPEFTメソッドを使用したい場合、プロンプト学習やプロンプト調整などについて詳しく知りたい場合、または🤗 PEFTライブラリ全般については、[ドキュメンテーション](https://huggingface.co/docs/peft/index)を参照してください。


## Load a PEFT adapter

🤗 TransformersからPEFTアダプターモデルを読み込んで使用するには、Hubリポジトリまたはローカルディレクトリに `adapter_config.json` ファイルとアダプターウェイトが含まれていることを確認してください。次に、`AutoModelFor` クラスを使用してPEFTアダプターモデルを読み込むことができます。たとえば、因果言語モデリング用のPEFTアダプターモデルを読み込むには：

1. PEFTモデルのIDを指定します。
2. それを[`AutoModelForCausalLM`] クラスに渡します。


```py
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
```

<Tip>

PEFTアダプターを`AutoModelFor`クラスまたは基本モデルクラス（`OPTForCausalLM`または`LlamaForCausalLM`など）で読み込むことができます。

</Tip>

また、`load_adapter`メソッドを呼び出すことで、PEFTアダプターを読み込むこともできます：


```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
peft_model_id = "ybelkada/opt-350m-lora"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
```

## Load in 8bit or 4bit


`bitsandbytes` 統合は、8ビットおよび4ビットの精度データ型をサポートしており、大規模なモデルを読み込む際にメモリを節約するのに役立ちます（詳細については `bitsandbytes` 統合の[ガイド](./quantization#bitsandbytes-integration)を参照してください）。[`~PreTrainedModel.from_pretrained`] に `load_in_8bit` または `load_in_4bit` パラメータを追加し、`device_map="auto"` を設定してモデルを効果的にハードウェアに分散配置できます：

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

## Add a new adapter

既存のアダプターを持つモデルに新しいアダプターを追加するために [`~peft.PeftModel.add_adapter`] を使用できます。ただし、新しいアダプターは現在のアダプターと同じタイプである限り、これを行うことができます。たとえば、モデルに既存の LoRA アダプターがアタッチされている場合：


```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

model.add_adapter(lora_config, adapter_name="adapter_1")
```

新しいアダプタを追加するには:


```py
# attach new adapter with same config
model.add_adapter(lora_config, adapter_name="adapter_2")
```

[`~peft.PeftModel.set_adapter`] を使用して、どのアダプターを使用するかを設定できます：


```py
# use adapter_1
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))

# use adapter_2
model.set_adapter("adapter_2")
output_enabled = model.generate(**inputs)
print(tokenizer.decode(output_enabled[0], skip_special_tokens=True))
```

## Enable and disable adapters

モデルにアダプターを追加したら、アダプターモジュールを有効または無効にすることができます。アダプターモジュールを有効にするには、次の手順を実行します：

```py
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
adapter_model_id = "ybelkada/opt-350m-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)
peft_config = PeftConfig.from_pretrained(adapter_model_id)

# to initiate with random weights
peft_config.init_lora_weights = False

model.add_adapter(peft_config)
model.enable_adapters()
output = model.generate(**inputs)
```

アダプターモジュールを無効にするには：

```py
model.disable_adapters()
output = model.generate(**inputs)
```

## Train a PEFT adapter

PEFTアダプターは[`Trainer`]クラスでサポートされており、特定のユースケースに対してアダプターをトレーニングすることができます。数行のコードを追加するだけで済みます。たとえば、LoRAアダプターをトレーニングする場合:

<Tip>

[`Trainer`]を使用したモデルの微調整に慣れていない場合は、[事前トレーニング済みモデルの微調整](training)チュートリアルをご覧ください。

</Tip>

1. タスクタイプとハイパーパラメータに対するアダプターの構成を定義します（ハイパーパラメータの詳細については[`~peft.LoraConfig`]を参照してください）。


```py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
```

2. モデルにアダプターを追加する。


```py
model.add_adapter(peft_config)
```

3. これで、モデルを [`Trainer`] に渡すことができます！

```py
trainer = Trainer(model=model, ...)
trainer.train()
```

保存するトレーニング済みアダプタとそれを読み込むための手順：
