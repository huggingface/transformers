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

# Templates for Chat Models

## Introduction

LLM（Language Model）のますます一般的な使用事例の1つは「チャット」です。
チャットのコンテキストでは、通常の言語モデルのように単一のテキストストリングを継続するのではなく、モデルは1つ以上の「メッセージ」からなる会話を継続します。
各メッセージには「ロール」とメッセージテキストが含まれます。

最も一般的に、これらのロールはユーザーからのメッセージには「ユーザー」、モデルからのメッセージには「アシスタント」が割り当てられます。
一部のモデルは「システム」ロールもサポートしています。
システムメッセージは通常会話の開始時に送信され、モデルの動作方法に関する指示が含まれます。

すべての言語モデル、チャット用に微調整されたモデルを含むすべてのモデルは、トークンのリニアシーケンスで動作し、ロールに特有の特別な処理を持ちません。
つまり、ロール情報は通常、メッセージ間に制御トークンを追加して注入され、メッセージの境界と関連するロールを示すことで提供されます。

残念ながら、トークンの使用方法については（まだ！）標準が存在せず、異なるモデルはチャット用のフォーマットや制御トークンが大きく異なる形式でトレーニングされています。
これはユーザーにとって実際の問題になる可能性があります。正しいフォーマットを使用しないと、モデルは入力に混乱し、パフォーマンスが本来よりも遥かに低下します。
これが「チャットテンプレート」が解決しようとする問題です。

チャット会話は通常、各辞書が「ロール」と「コンテンツ」のキーを含み、単一のチャットメッセージを表すリストとして表現されます。
チャットテンプレートは、指定されたモデルの会話を単一のトークン化可能なシーケンスにどのようにフォーマットするかを指定するJinjaテンプレートを含む文字列です。
トークナイザとこの情報を保存することにより、モデルが期待する形式の入力データを取得できるようになります。

さっそく、`BlenderBot` モデルを使用した例を示して具体的にしましょう。`BlenderBot` のデフォルトテンプレートは非常にシンプルで、ほとんどが対話のラウンド間に空白を追加するだけです。


```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
" Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"
```

指定された通り、チャット全体が単一の文字列にまとめられています。デフォルトの設定である「tokenize=True」を使用すると、
その文字列もトークン化されます。しかし、より複雑なテンプレートが実際にどのように機能するかを確認するために、
「meta-llama/Llama-2-7b-chat-hf」モデルを使用してみましょう。ただし、このモデルはゲート付きアクセスを持っており、
このコードを実行する場合は[リポジトリでアクセスをリクエスト](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)する必要があります。

```python
>> from transformers import AutoTokenizer
>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>> tokenizer.use_default_system_prompt = False
>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]"
```

今回、トークナイザは制御トークン [INST] と [/INST] を追加しました。これらはユーザーメッセージの開始と終了を示すためのものです（ただし、アシスタントメッセージには適用されません！）

## How do chat templates work?

モデルのチャットテンプレートは、`tokenizer.chat_template`属性に格納されています。チャットテンプレートが設定されていない場合、そのモデルクラスのデフォルトテンプレートが代わりに使用されます。`BlenderBot`のテンプレートを見てみましょう:

```python

>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> tokenizer.default_chat_template
"{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
```


これは少し抑圧的ですね。可読性を高めるために、新しい行とインデントを追加しましょう。
各ブロックの直前の空白と、ブロックの直後の最初の改行は、デフォルトでJinjaの `trim_blocks` および `lstrip_blocks` フラグを使用して削除します。
これにより、インデントと改行を含むテンプレートを書いても正常に機能することができます。

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ ' ' }}
    {% endif %}
    {{ message['content'] }}
    {% if not loop.last %}
        {{ '  ' }}
    {% endif %}
{% endfor %}
{{ eos_token }}
```

これが初めて見る方へ、これは[Jinjaテンプレート](https://jinja.palletsprojects.com/en/3.1.x/templates/)です。
Jinjaはテキストを生成するためのシンプルなコードを記述できるテンプレート言語です。多くの点で、コードと
構文はPythonに似ています。純粋なPythonでは、このテンプレートは次のようになるでしょう：

```python
for idx, message in enumerate(messages):
    if message['role'] == 'user':
        print(' ')
    print(message['content'])
    if not idx == len(messages) - 1:  # Check for the last message in the conversation
        print('  ')
print(eos_token)
```

実際に、このテンプレートは次の3つのことを行います：
1. 各メッセージに対して、メッセージがユーザーメッセージである場合、それの前に空白を追加し、それ以外の場合は何も表示しません。
2. メッセージの内容を追加します。
3. メッセージが最後のメッセージでない場合、その後に2つのスペースを追加します。最後のメッセージの後にはEOSトークンを表示します。

これは非常にシンプルなテンプレートです。制御トークンを追加しないし、モデルに対する指示を伝える一般的な方法である「システム」メッセージをサポートしていません。
ただし、Jinjaはこれらのことを行うための多くの柔軟性を提供しています！
LLaMAがフォーマットする方法に類似した入力をフォーマットするためのJinjaテンプレートを見てみましょう
（実際のLLaMAテンプレートはデフォルトのシステムメッセージの処理や、一般的なシステムメッセージの処理が若干異なるため、
実際のコードではこのテンプレートを使用しないでください！）


```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
        {{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + message['content'] + ' ' + eos_token }}
    {% endif %}
{% endfor %}
```

願わくば、少し見つめていただければ、このテンプレートが何を行っているかがわかるかもしれません。
このテンプレートは、各メッセージの「役割」に基づいて特定のトークンを追加します。これらのトークンは、メッセージを送信した人を表すものです。
ユーザー、アシスタント、およびシステムメッセージは、それらが含まれるトークンによってモデルによって明確に区別されます。


## How do I create a chat template?

簡単です。単純にJinjaテンプレートを書いて、`tokenizer.chat_template`を設定します。
他のモデルから既存のテンプレートを始点にして、必要に応じて編集すると便利かもしれません！
例えば、上記のLLaMAテンプレートを取って、アシスタントメッセージに"[ASST]"と"[/ASST]"を追加できます。

```
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
        {{ '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {% endif %}
{% endfor %}
```

次に、単に`tokenizer.chat_template`属性を設定してください。
次回、[`~PreTrainedTokenizer.apply_chat_template`]を使用する際に、新しいテンプレートが使用されます！
この属性は`tokenizer_config.json`ファイルに保存されるため、[`~utils.PushToHubMixin.push_to_hub`]を使用して
新しいテンプレートをHubにアップロードし、みんなが正しいテンプレートを使用していることを確認できます！

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # Change the system token
tokenizer.chat_template = template  # Set the new template
tokenizer.push_to_hub("model_name")  # Upload your new template to the Hub!
```

[`~PreTrainedTokenizer.apply_chat_template`] メソッドは、あなたのチャットテンプレートを使用するために [`ConversationalPipeline`] クラスによって呼び出されます。
したがって、正しいチャットテンプレートを設定すると、あなたのモデルは自動的に [`ConversationalPipeline`] と互換性があるようになります。


## What are "default" templates?

チャットテンプレートの導入前に、チャットの処理はモデルクラスレベルでハードコードされていました。
後方互換性のために、このクラス固有の処理をデフォルトテンプレートとして保持し、クラスレベルで設定されています。
モデルにチャットテンプレートが設定されていない場合、ただしモデルクラスのデフォルトテンプレートがある場合、
`ConversationalPipeline`クラスや`apply_chat_template`などのメソッドはクラステンプレートを使用します。
トークナイザのデフォルトのチャットテンプレートを確認するには、`tokenizer.default_chat_template`属性をチェックしてください。

これは、後方互換性のために純粋に行っていることで、既存のワークフローを壊さないようにしています。
モデルにとってクラステンプレートが適切である場合でも、デフォルトテンプレートをオーバーライドして
`chat_template`属性を明示的に設定することを強くお勧めします。これにより、ユーザーにとって
モデルがチャット用に正しく構成されていることが明確になり、デフォルトテンプレートが変更されたり廃止された場合に備えることができます。

## What template should I use?

すでにチャットのトレーニングを受けたモデルのテンプレートを設定する場合、テンプレートがトレーニング中にモデルが見たメッセージのフォーマットとまったく一致することを確認する必要があります。
そうでない場合、性能の低下を経験する可能性が高いです。これはモデルをさらにトレーニングしている場合でも同様です - チャットトークンを一定に保つと、おそらく最高の性能が得られます。
これはトークン化と非常に類似しており、通常はトレーニング中に使用されたトークン化と正確に一致する場合に、推論またはファインチューニングの際に最良の性能が得られます。

一方、ゼロからモデルをトレーニングするか、チャットのためにベース言語モデルをファインチューニングする場合、適切なテンプレートを選択する自由度があります。
LLM（Language Model）はさまざまな入力形式を処理できるほどスマートです。クラス固有のテンプレートがないモデル用のデフォルトテンプレートは、一般的なユースケースに対して良い柔軟な選択肢です。
これは、`ChatMLフォーマット`に従ったもので、多くのユースケースに適しています。次のようになります：

```
{% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
```

If you like this one, here it is in one-liner form, ready to copy into your code:

```python
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
```

このテンプレートは、各メッセージを「``」トークンで囲み、役割を文字列として単純に記述します。
これにより、トレーニングで使用する役割に対する柔軟性が得られます。出力は以下のようになります：


```
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

「ユーザー」、「システム」、および「アシスタント」の役割は、チャットの標準です。
特に、[`ConversationalPipeline`]との連携をスムーズに行う場合には、これらの役割を使用することをお勧めします。ただし、これらの役割に制約はありません。テンプレートは非常に柔軟で、任意の文字列を役割として使用できます。

## I want to use chat templates! How should I get started?

チャットモデルを持っている場合、そのモデルの`tokenizer.chat_template`属性を設定し、[`~PreTrainedTokenizer.apply_chat_template`]を使用してテストする必要があります。
これはモデルの所有者でない場合でも適用されます。モデルのリポジトリが空のチャットテンプレートを使用している場合、またはデフォルトのクラステンプレートを使用している場合でも、
この属性を適切に設定できるように[プルリクエスト](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)を開いてください。

一度属性が設定されれば、それで完了です！ `tokenizer.apply_chat_template`は、そのモデルに対して正しく動作するようになります。これは、
`ConversationalPipeline`などの場所でも自動的にサポートされます。

モデルがこの属性を持つことを確認することで、オープンソースモデルの全コミュニティがそのフルパワーを使用できるようになります。
フォーマットの不一致はこの分野に悩み続け、パフォーマンスに黙って影響を与えてきました。それを終わらせる時が来ました！

