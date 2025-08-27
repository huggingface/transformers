<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CodeLlama

## Overview

Code Llama モデルはによって [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) で提案されました。 Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve.
論文の要約は次のとおりです。

*私たちは Code Llama をリリースします。これは Llama 2 に基づくコードの大規模言語モデル ファミリであり、オープン モデルの中で最先端のパフォーマンス、埋め込み機能、大規模な入力コンテキストのサポート、プログラミング タスクのゼロショット命令追従機能を提供します。 。幅広いアプリケーションをカバーするための複数のフレーバーを提供しています。基盤モデル (Code Llama)、Python 特化 (Code Llama - Python)、およびそれぞれ 7B、13B、および 34B パラメーターを備えた命令追従モデル (Code Llama - Instruct) です。すべてのモデルは 16,000 トークンのシーケンスでトレーニングされ、最大 100,000 トークンの入力で改善が見られます。 7B および 13B コード ラマとコード ラマ - 命令バリアントは、周囲のコンテンツに基づいた埋め込みをサポートします。 Code Llama は、いくつかのコード ベンチマークでオープン モデルの中で最先端のパフォーマンスに達し、HumanEval と MBPP でそれぞれ最大 53% と 55% のスコアを獲得しました。特に、Code Llama - Python 7B は HumanEval および MBPP 上で Llama 2 70B よりも優れたパフォーマンスを示し、すべてのモデルは MultiPL-E 上で公開されている他のすべてのモデルよりも優れています。私たちは、研究と商業利用の両方を許可する寛容なライセンスに基づいて Code Llama をリリースしています。*

すべての Code Llama モデル チェックポイントを [こちら](https://huggingface.co/models?search=code_llama) で確認し、[meta llama org](https://huggingface.co/meta-llama) で正式にリリースされたチェックポイントを確認してください。

このモデルは [ArthurZucker](https://huggingface.co/ArthurZ) によって提供されました。著者のオリジナルのコードは [こちら](https://github.com/facebookresearch/llama) にあります。

## Usage tips and examples

<Tip warning={true}>

Code Llama のベースとなる`Llama2`ファミリー モデルは、`bfloat16`を使用してトレーニングされましたが、元の推論では`float16`を使用します。さまざまな精度を見てみましょう。

* `float32`: モデルの初期化に関する PyTorch の規約では、モデルの重みがどの `dtype` で格納されたかに関係なく、モデルを `float32` にロードします。 「transformers」も、PyTorch との一貫性を保つためにこの規則に従っています。これはデフォルトで選択されます。 `AutoModel` API でストレージの重み付けタイプを使用してチェックポイントのロードをキャストする場合は、`dtype="auto"` を指定する必要があります。 `model = AutoModelForCausalLM.from_pretrained("path", dtype = "auto")`。
* `bfloat16`: コード Llama はこの精度でトレーニングされているため、さらなるトレーニングや微調整に使用することをお勧めします。
* `float16`: この精度を使用して推論を実行することをお勧めします。通常は `bfloat16` より高速であり、評価メトリクスには `bfloat16` と比べて明らかな低下が見られないためです。 bfloat16 を使用して推論を実行することもできます。微調整後、float16 と bfloat16 の両方で推論結果を確認することをお勧めします。

上で述べたように、モデルを初期化するときに `dtype="auto"` を使用しない限り、ストレージの重みの `dtype` はほとんど無関係です。その理由は、モデルが最初にダウンロードされ (オンラインのチェックポイントの `dtype` を使用)、次に `torch` のデフォルトの `dtype` にキャストされるためです (`torch.float32` になります)。指定された `dtype` がある場合は、代わりにそれが使用されます。

</Tip>

チップ：
- 充填タスクはすぐにサポートされます。入力を埋めたい場所には `tokenizer.fill_token` を使用する必要があります。
- モデル変換スクリプトは、`Llama2` ファミリの場合と同じです。

使用例は次のとおりです。

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

スクリプトを実行するには、(最大のバージョンであっても) float16 精度でモデル全体をホストするのに十分な CPU RAM が必要であることに注意してください。
いくつかのチェックポイントがあり、それぞれにモデルの各重みの一部が含まれているため、すべてを RAM にロードする必要があります)。

変換後、モデルとトークナイザーは次の方法でロードできます。

```python
>>> from transformers import LlamaForCausalLM, CodeLlamaTokenizer

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
>>> model = LlamaForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")
>>> PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
>>> input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
>>> generated_ids = model.generate(input_ids, max_new_tokens=128)

>>> filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
>>> print(PROMPT.replace("<FILL_ME>", filling))
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.

    Args:
        s: The string to remove non-ASCII characters from.

    Returns:
        The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result
```

塗りつぶされた部分だけが必要な場合:

```python
>>> from transformers import pipeline
>>> import torch

>>> generator = pipeline("text-generation",model="meta-llama/CodeLlama-7b-hf",dtype=torch.float16, device_map="auto")
>>> generator('def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return result', max_new_tokens = 128)
[{'generated_text': 'def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return resultRemove non-ASCII characters from a string. """\n    result = ""\n    for c in s:\n        if ord(c) < 128:\n            result += c'}]
```

内部では、トークナイザーが [`<FILL_ME>` によって自動的に分割](https://huggingface.co/docs/transformers/main/model_doc/code_llama#transformers.CodeLlamaTokenizer.fill_token) して、[ に続く書式設定された入力文字列を作成します。オリジナルのトレーニング パターン](https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402)。これは、パターンを自分で準備するよりも堅牢です。トークンの接着など、デバッグが非常に難しい落とし穴を回避できます。このモデルまたは他のモデルに必要な CPU および GPU メモリの量を確認するには、その値を決定するのに役立つ [この計算ツール](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) を試してください。

LLaMA トークナイザーは、[sentencepiece](https://github.com/google/sentencepiece) に基づく BPE モデルです。センテンスピースの癖の 1 つは、シーケンスをデコードするときに、最初のトークンが単語の先頭 (例: 「Banana」) である場合、トークナイザーは文字列の先頭にプレフィックス スペースを追加しないことです。

<Tip>

コード Llama は、`Llama2` モデルと同じアーキテクチャを持っています。API リファレンスについては、[Llama2 のドキュメント ページ](llama2) を参照してください。
以下の Code Llama トークナイザーのリファレンスを見つけてください。
</Tip>

## CodeLlamaTokenizer

[[autodoc]] CodeLlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CodeLlamaTokenizerFast

[[autodoc]] CodeLlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary
