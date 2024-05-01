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

# What 🤗 Transformers can do

🤗 Transformersは、自然言語処理（NLP）、コンピュータビジョン、音声処理などの最新の事前学習済みモデルのライブラリです。このライブラリには、Transformerモデルだけでなく、コンピュータビジョンタスク向けのモダンな畳み込みニューラルネットワークなど、Transformer以外のモデルも含まれています。現代のスマートフォン、アプリ、テレビなど、最も人気のある消費者製品の多くには、深層学習技術が使用されています。スマートフォンで撮影した写真から背景オブジェクトを削除したいですか？これはパノプティック・セグメンテーション（まだ何を意味するかわからなくても心配しないでください、以下のセクションで説明します！）のタスクの一例です。

このページでは、🤗 Transformersライブラリを使用して、たった3行のコードで解決できるさまざまな音声および音声、コンピュータビジョン、NLPタスクの概要を提供します！

## Audio

音声と音声処理のタスクは、他のモダリティとは少し異なります。なぜなら、入力としての生の音声波形は連続的な信号であるからです。テキストとは異なり、生の音声波形は文章を単語に分割できるようにきれいに分割できません。これを解決するために、通常、生の音声信号は定期的な間隔でサンプリングされます。間隔内でより多くのサンプルを取ると、サンプリングレートが高くなり、音声はより元の音声ソースに近づきます。

以前のアプローチでは、音声を前処理してそれから有用な特徴を抽出しました。しかし、現在では、生の音声波形を特徴エンコーダに直接フィードし、音声表現を抽出することから始めることが一般的です。これにより、前処理ステップが簡素化され、モデルは最も重要な特徴を学習できます。

### Audio classification

音声分類は、事前に定義されたクラスのセットから音声データにラベルを付けるタスクです。これは多くの具体的なアプリケーションを含む広範なカテゴリであり、いくつかの例は次のとおりです：

- 音響シーン分類：音声にシーンラベルを付ける（「オフィス」、「ビーチ」、「スタジアム」）
- 音響イベント検出：音声に音声イベントラベルを付ける（「車のクラクション」、「クジラの呼び声」、「ガラスの破裂」）
- タギング：複数の音を含む音声にラベルを付ける（鳥の鳴き声、会議でのスピーカー識別）
- 音楽分類：音楽にジャンルラベルを付ける（「メタル」、「ヒップホップ」、「カントリー」）


```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
>>> preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

### Automatic speech recognition

自動音声認識（ASR）は音声をテキストに変換します。これは、人間のコミュニケーションの自然な形式である音声の一部として、部分的にそのような一般的なオーディオタスクの一つです。今日、ASRシステムはスピーカー、スマートフォン、自動車などの「スマート」テクノロジープロダクトに組み込まれています。私たちは仮想アシスタントに音楽を再生してもらったり、リマインダーを設定してもらったり、天気を教えてもらったりできます。

しかし、Transformerアーキテクチャが助けた主要な課題の一つは、低リソース言語におけるものです。大量の音声データで事前トレーニングし、低リソース言語でラベル付き音声データをわずか1時間だけでファインチューニングすることでも、以前のASRシステムと比較して高品質な結果を得ることができます。以前のASRシステムは100倍以上のラベル付きデータでトレーニングされていましたが、Transformerアーキテクチャはこの問題に貢献しました。


```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```


## Computer vision

最初で初めて成功したコンピュータビジョンのタスクの一つは、[畳み込みニューラルネットワーク（CNN）](glossary#convolution)を使用して郵便番号の画像を認識することでした。画像はピクセルから構成され、各ピクセルには数値があります。これにより、画像をピクセル値の行列として簡単に表現できます。特定のピクセル値の組み合わせは、画像の色を記述します。

コンピュータビジョンのタスクを解決する一般的な方法は次の2つです：

1. 畳み込みを使用して、低レベルの特徴から高レベルの抽象的な情報まで、画像の階層的な特徴を学習する。
2. 画像をパッチに分割し、各画像パッチが画像全体とどのように関連しているかを徐々に学習するためにTransformerを使用する。CNNが好むボトムアップアプローチとは異なり、これはぼんやりとした画像から始めて徐々に焦点を合わせるようなものです。

### Image classification

画像分類は、事前に定義されたクラスのセットから画像全体にラベルを付けます。多くの分類タスクと同様に、画像分類には多くの実用的な用途があり、その一部は次のとおりです：

* 医療：疾患の検出や患者の健康の監視に使用するために医療画像にラベルを付ける
* 環境：森林伐採の監視、野生地の管理情報、または山火事の検出に使用するために衛星画像にラベルを付ける
* 農業：作物の健康を監視するための作物の画像や、土地利用の監視のための衛星画像にラベルを付ける
* 生態学：野生動物の個体数を監視したり、絶滅危惧種を追跡するために動植物の種の画像にラベルを付ける


```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="image-classification")
>>> preds = classifier(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.4335, 'label': 'lynx, catamount'}
{'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
{'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
{'score': 0.0239, 'label': 'Egyptian cat'}
{'score': 0.0229, 'label': 'tiger cat'}
```

### Object detection

画像分類とは異なり、オブジェクト検出は画像内の複数のオブジェクトを識別し、オブジェクトの位置を画像内で定義する境界ボックスによって特定します。オブジェクト検出の例には次のような用途があります：

* 自動運転車：他の車両、歩行者、信号機などの日常の交通オブジェクトを検出
* リモートセンシング：災害モニタリング、都市計画、天候予測
* 欠陥検出：建物のクラックや構造上の損傷、製造上の欠陥を検出


```py
>>> from transformers import pipeline

>>> detector = pipeline(task="object-detection")
>>> preds = detector(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
>>> preds
[{'score': 0.9865,
  'label': 'cat',
  'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
```

### Image segmentation

画像セグメンテーションは、画像内のすべてのピクセルをクラスに割り当てるピクセルレベルのタスクです。これはオブジェクト検出とは異なり、オブジェクトをラベル付けし、予測するために境界ボックスを使用する代わりに、セグメンテーションはより詳細になります。セグメンテーションはピクセルレベルでオブジェクトを検出できます。画像セグメンテーションにはいくつかのタイプがあります：

* インスタンスセグメンテーション：オブジェクトのクラスをラベル付けするだけでなく、オブジェクトの個別のインスタンス（"犬-1"、"犬-2"）もラベル付けします。
* パノプティックセグメンテーション：セマンティックセグメンテーションとインスタンスセグメンテーションの組み合わせ。セマンティッククラスごとに各ピクセルにラベルを付け、オブジェクトの個別のインスタンスもラベル付けします。

セグメンテーションタスクは、自動運転車にとって、周囲の世界のピクセルレベルのマップを作成し、歩行者や他の車両を安全に回避できるようにするのに役立ちます。また、医療画像では、タスクの細かい粒度が異常な細胞や臓器の特徴を識別するのに役立ちます。画像セグメンテーションは、eコマースで衣類を仮想的に試着したり、カメラを通じて実世界にオブジェクトを重ねて拡張現実の体験を作成したりするためにも使用できます。



```py
>>> from transformers import pipeline

>>> segmenter = pipeline(task="image-segmentation")
>>> preds = segmenter(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.9879, 'label': 'LABEL_184'}
{'score': 0.9973, 'label': 'snow'}
{'score': 0.9972, 'label': 'cat'}
```

### Depth estimation

深度推定は、画像内の各ピクセルがカメラからの距離を予測します。このコンピュータビジョンタスクは、特にシーンの理解と再構築に重要です。たとえば、自動運転車では、歩行者、交通標識、他の車などの物体がどれだけ遠いかを理解し、障害物や衝突を回避するために必要です。深度情報はまた、2D画像から3D表現を構築し、生物学的構造や建物の高品質な3D表現を作成するのに役立ちます。

深度推定には次の2つのアプローチがあります：

* ステレオ：深度は、わずかに異なる角度からの同じ画像の2つの画像を比較して推定されます。
* モノキュラー：深度は単一の画像から推定されます。


```py
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation")
>>> preds = depth_estimator(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
```


## Natural language processing

NLPタスクは、テキストが私たちにとって自然なコミュニケーション手段であるため、最も一般的なタスクの一つです。モデルが認識するための形式にテキストを変換するには、トークン化が必要です。これは、テキストのシーケンスを単語やサブワード（トークン）に分割し、それらのトークンを数字に変換することを意味します。その結果、テキストのシーケンスを数字のシーケンスとして表現し、一度数字のシーケンスがあれば、さまざまなNLPタスクを解決するためにモデルに入力できます！

### Text classification

どんなモダリティの分類タスクと同様に、テキスト分類は事前に定義されたクラスのセットからテキストのシーケンス（文レベル、段落、またはドキュメントであることがあります）にラベルを付けます。テキスト分類には多くの実用的な用途があり、その一部は次のとおりです：

* 感情分析：`positive`や`negative`のような極性に従ってテキストにラベルを付け、政治、金融、マーケティングなどの分野での意思決定をサポートします。
* コンテンツ分類：テキストをトピックに従ってラベル付けし、ニュースやソーシャルメディアのフィード内の情報を整理し、フィルタリングするのに役立ちます（`天気`、`スポーツ`、`金融`など）。


```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="sentiment-analysis")
>>> preds = classifier("Hugging Face is the best thing since sliced bread!")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.9991, 'label': 'POSITIVE'}]
```

### Token classification

どんなNLPタスクでも、テキストはテキストのシーケンスを個々の単語やサブワードに分割して前処理されます。これらは[トークン](/glossary#token)として知られています。トークン分類は、事前に定義されたクラスのセットから各トークンにラベルを割り当てます。

トークン分類の一般的なタイプは次の2つです：

* 固有表現認識（NER）：組織、人物、場所、日付などのエンティティのカテゴリに従ってトークンにラベルを付けます。NERは特にバイオメディカル環境で人気であり、遺伝子、タンパク質、薬物名などをラベル付けできます。
* 品詞タグ付け（POS）：名詞、動詞、形容詞などの品詞に従ってトークンにラベルを付けます。POSは、翻訳システムが同じ単語が文法的にどのように異なるかを理解するのに役立ちます（名詞としての「銀行」と動詞としての「銀行」など）。


```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="ner")
>>> preds = classifier("Hugging Face is a French company based in New York City.")
>>> preds = [
...     {
...         "entity": pred["entity"],
...         "score": round(pred["score"], 4),
...         "index": pred["index"],
...         "word": pred["word"],
...         "start": pred["start"],
...         "end": pred["end"],
...     }
...     for pred in preds
... ]
>>> print(*preds, sep="\n")
{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```

### Question answering

質問応答は、コンテキスト（オープンドメイン）を含む場合と含まない場合（クローズドドメイン）がある場合もある、別のトークンレベルのタスクで、質問に対する回答を返します。このタスクは、仮想アシスタントにレストランが営業しているかどうかのような質問をするときに発生します。また、顧客や技術サポートを提供し、検索エンジンがあなたが求めている関連情報を取得するのにも役立ちます。

質問応答の一般的なタイプは次の2つです：

* 抽出型：質問と一部のコンテキストが与えられた場合、モデルがコンテキストから抽出する必要のあるテキストのスパンが回答となります。
* 抽象的：質問と一部のコンテキストが与えられた場合、回答はコンテキストから生成されます。このアプローチは、[`QuestionAnsweringPipeline`]ではなく[`Text2TextGenerationPipeline`]で処理されます。


```py
>>> from transformers import pipeline

>>> question_answerer = pipeline(task="question-answering")
>>> preds = question_answerer(
...     question="What is the name of the repository?",
...     context="The name of the repository is huggingface/transformers",
... )
>>> print(
...     f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
... )
score: 0.9327, start: 30, end: 54, answer: huggingface/transformers
```

### Summarization

要約は、長いテキストから短いバージョンを作成し、元の文書の意味の大部分を保ちながら試みるタスクです。要約はシーケンスからシーケンスへのタスクであり、入力よりも短いテキストシーケンスを出力します。要約を行うことで、読者が主要なポイントを迅速に理解できるようにするのに役立つ長文書がたくさんあります。法案、法的および財務文書、特許、科学論文など、読者の時間を節約し読書の支援となる文書の例があります。

質問応答と同様に、要約には2つのタイプがあります：

* 抽出的要約：元のテキストから最も重要な文を識別して抽出します。
* 抽象的要約：元のテキストからターゲットの要約（入力文書に含まれていない新しい単語を含むことがあります）を生成します。[`SummarizationPipeline`]は抽象的なアプローチを使用しています。


```py
>>> from transformers import pipeline

>>> summarizer = pipeline(task="summarization")
>>> summarizer(
...     "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
... )
[{'summary_text': ' The Transformer is the first sequence transduction model based entirely on attention . It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention . For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers .'}]
```

### Translation

翻訳は、ある言語のテキストシーケンスを別の言語に変換する作業です。これは異なるバックグラウンドを持つ人々がコミュニケーションをとるのに役立ち、広範な観客にコンテンツを翻訳して伝えるのに役立ち、新しい言語を学ぶのを支援する学習ツールにもなります。要約と共に、翻訳はシーケンス間のタスクであり、モデルは入力シーケンスを受け取り、ターゲットの出力シーケンスを返します。

初期の翻訳モデルは主に単一言語でしたが、最近では多言語モデルに対する関心が高まり、多くの言語対で翻訳できるような多言語モデルに注目が集まっています。


```py
>>> from transformers import pipeline

>>> text = "translate English to French: Hugging Face is a community-based open-source platform for machine learning."
>>> translator = pipeline(task="translation", model="google-t5/t5-small")
>>> translator(text)
[{'translation_text': "Hugging Face est une tribune communautaire de l'apprentissage des machines."}]
```

### 言語モデリング

言語モデリングは、テキストのシーケンス内の単語を予測するタスクです。事前学習された言語モデルは、多くの他のダウンストリームタスクに対してファインチューニングできるため、非常に人気のあるNLPタスクとなっています。最近では、ゼロショットまたはフューショット学習を実証する大規模な言語モデル（LLM）に大きな関心が寄せられています。これは、モデルが明示的にトレーニングされていないタスクを解決できることを意味します！言語モデルは、流暢で説得力のあるテキストを生成するために使用できますが、テキストが常に正確であるわけではないため、注意が必要です。

言語モデリングには2つのタイプがあります：

* 因果的：モデルの目標は、シーケンス内の次のトークンを予測することであり、将来のトークンはマスクされます。

    ```py
    >>> from transformers import pipeline

    >>> prompt = "Hugging Face is a community-based open-source platform for machine learning."
    >>> generator = pipeline(task="text-generation")
    >>> generator(prompt)  # doctest: +SKIP
    ```

* マスクされた：モデルの目的は、シーケンス内のトークン全体にアクセスしながら、シーケンス内のマスクされたトークンを予測することです。

    ```py
    >>> text = "Hugging Face is a community-based open-source <mask> for machine learning."
    >>> fill_mask = pipeline(task="fill-mask")
    >>> preds = fill_mask(text, top_k=1)
    >>> preds = [
    ...     {
    ...         "score": round(pred["score"], 4),
    ...         "token": pred["token"],
    ...         "token_str": pred["token_str"],
    ...         "sequence": pred["sequence"],
    ...     }
    ...     for pred in preds
    ... ]
    >>> preds
    [{'score': 0.2236,
      'token': 1761,
      'token_str': ' platform',
      'sequence': 'Hugging Face is a community-based open-source platform for machine learning.'}]
    ```

## Multimodal

マルチモーダルタスクは、特定の問題を解決するために複数のデータモダリティ（テキスト、画像、音声、ビデオ）を処理するためにモデルを必要とします。画像キャプショニングは、モデルが入力として画像を受け取り、画像を説明するテキストのシーケンスまたは画像のいくつかの特性を出力するマルチモーダルタスクの例です。

マルチモーダルモデルは異なるデータタイプまたはモダリティで作業しますが、内部的には前処理ステップがモデルにすべてのデータタイプを埋め込み（データに関する意味のある情報を保持するベクトルまたは数字のリスト）に変換するのを支援します。画像キャプショニングのようなタスクでは、モデルは画像の埋め込みとテキストの埋め込みの間の関係を学習します。

### Document question answering

ドキュメント質問応答は、ドキュメントからの自然言語の質問に答えるタスクです。テキストを入力とするトークンレベルの質問応答タスクとは異なり、ドキュメント質問応答はドキュメントの画像とそのドキュメントに関する質問を受け取り、答えを返します。ドキュメント質問応答は構造化されたドキュメントを解析し、それから重要な情報を抽出するために使用できます。以下の例では、レシートから合計金額とお釣りを抽出することができます。


```py
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/hf-internal-testing/example-documents/resolve/main/jpeg_images/2.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> doc_question_answerer = pipeline("document-question-answering", model="magorshunov/layoutlm-invoices")
>>> preds = doc_question_answerer(
...     question="What is the total amount?",
...     image=image,
... )
>>> preds
[{'score': 0.8531, 'answer': '17,000', 'start': 4, 'end': 4}]
```

このページが各モダリティのタスクの種類とそれぞれの重要性についての追加の背景情報を提供できたことを願っています。次の [セクション](tasks_explained) では、🤗 トランスフォーマーがこれらのタスクを解決するために **どのように** 動作するかを学びます。
