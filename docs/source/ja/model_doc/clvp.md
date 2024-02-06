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

# CLVP

## Overview

CLVP (Contrastive Language-Voice Pretrained Transformer) モデルは、James Betker によって [Better speech synthesis through scaling](https://arxiv.org/abs/2305.07243) で提案されました。

論文の要約は次のとおりです。

*近年、画像生成の分野は自己回帰変換器と DDPM の応用によって革命を起こしています。これらのアプローチは、画像生成のプロセスを段階的な確率的プロセスとしてモデル化し、大量のコンピューティングとデータを活用して画像の分布を学習します。パフォーマンスを向上させるこの方法論は、画像に限定される必要はありません。この論文では、画像生成ドメインの進歩を音声合成に適用する方法について説明します。その結果、表現力豊かなマルチ音声テキスト読み上げシステムである TorToise が誕生しました。


このモデルは [Susnato Dhar](https://huggingface.co/susnato) によって提供されました。
元のコードは [ここ](https://github.com/neonbjb/tortoise-tts) にあります。

## Usage tips

1. CLVP は Tortoise TTS モデルの不可欠な部分です。
2. CLVP を使用して、生成されたさまざまな音声候補を提供されたテキストと比較することができ、最良の音声トークンが拡散モデルに転送されます。
3. Tortoise の使用には、[`ClvpModelForConditionalGeneration.generate()`] メソッドの使用を強くお勧めします。
4. 16 kHz を期待する他のオーディオ モデルとは対照的に、CLVP モデルはオーディオが 22.05 kHz でサンプリングされることを期待していることに注意してください。

## Brief Explanation:

- [`ClvpTokenizer`] はテキスト入力をトークン化し、[`ClvpFeatureExtractor`] は目的のオーディオからログ メル スペクトログラムを抽出します。
- [`ClvpConditioningEncoder`] は、これらのテキスト トークンとオーディオ表現を取得し、テキストとオーディオに基づいて条件付けされた埋め込みに変換します。
- [`ClvpForCausalLM`] は、これらの埋め込みを使用して複数の音声候補を生成します。
- 各音声候補は音声エンコーダ ([`ClvpEncoder`]) を通過してベクトル表現に変換され、テキスト エンコーダ ([`ClvpEncoder`]) はテキスト トークンを同じ潜在空間に変換します。
- 最後に、各音声ベクトルをテキスト ベクトルと比較して、どの音声ベクトルがテキスト ベクトルに最も類似しているかを確認します。
- [`ClvpModelForConditionalGeneration.generate()`] は、上記のすべてのロジックを 1 つのメソッドに圧縮します。

例 ：

```python
>>> import datasets
>>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

>>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library).
>>> text = "This is an example text."

>>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
>>> sample = ds[0]["audio"]

>>> # Define processor and model.
>>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
>>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

>>> # Generate processor output and model output.
>>> processor_output = processor(raw_speech=sample["array"], sampling_rate=sample["sampling_rate"], text=text, return_tensors="pt")
>>> generated_output = model.generate(**processor_output)
```


## ClvpConfig

[[autodoc]] ClvpConfig
    - from_sub_model_configs

## ClvpEncoderConfig

[[autodoc]] ClvpEncoderConfig

## ClvpDecoderConfig

[[autodoc]] ClvpDecoderConfig

## ClvpTokenizer

[[autodoc]] ClvpTokenizer
    - save_vocabulary

## ClvpFeatureExtractor

[[autodoc]] ClvpFeatureExtractor
    - __call__

## ClvpProcessor

[[autodoc]] ClvpProcessor
    - __call__
    - decode
    - batch_decode

## ClvpModelForConditionalGeneration

[[autodoc]] ClvpModelForConditionalGeneration
    - forward
    - generate
    - get_text_features
    - get_speech_features

## ClvpForCausalLM

[[autodoc]] ClvpForCausalLM

## ClvpModel

[[autodoc]] ClvpModel

## ClvpEncoder

[[autodoc]] ClvpEncoder

## ClvpDecoder

[[autodoc]] ClvpDecoder

