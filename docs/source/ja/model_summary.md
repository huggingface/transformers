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

# The Transformer model family

2017年に導入されて以来、[元のTransformer](https://arxiv.org/abs/1706.03762)モデルは、自然言語処理（NLP）のタスクを超える多くの新しいエキサイティングなモデルをインスパイアしました。[タンパク質の折りたたまれた構造を予測](https://huggingface.co/blog/deep-learning-with-proteins)するモデル、[チーターを走らせるためのトレーニング](https://huggingface.co/blog/train-decision-transformers)するモデル、そして[時系列予測](https://huggingface.co/blog/time-series-transformers)のためのモデルなどがあります。Transformerのさまざまなバリアントが利用可能ですが、大局を見落とすことがあります。これらのすべてのモデルに共通するのは、元のTransformerアーキテクチャに基づいていることです。一部のモデルはエンコーダまたはデコーダのみを使用し、他のモデルは両方を使用します。これは、Transformerファミリー内のモデルの高レベルの違いをカテゴライズし、調査するための有用な分類法を提供し、以前に出会ったことのないTransformerを理解するのに役立ちます。

元のTransformerモデルに慣れていないか、リフレッシュが必要な場合は、Hugging Faceコースの[Transformerの動作原理](https://huggingface.co/course/chapter1/4?fw=pt)章をチェックしてください。

<div align="center">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/H39Z_720T5s" title="YouTubeビデオプレーヤー"
    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
    picture-in-picture" allowfullscreen></iframe>
</div>

## Computer vision

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FacQBpeFBVvrDUlzFlkejoz%2FModelscape-timeline%3Fnode-id%3D0%253A1%26t%3Dm0zJ7m2BQ9oe0WtO-1" allowfullscreen></iframe> 

### Convolutional network

長い間、畳み込みネットワーク（CNN）はコンピュータビジョンのタスクにおいて支配的なパラダイムでしたが、[ビジョンTransformer](https://arxiv.org/abs/2010.11929)はそのスケーラビリティと効率性を示しました。それでも、一部のCNNの最高の特性、特に特定のタスクにとっては非常に強力な翻訳不変性など、一部のTransformerはアーキテクチャに畳み込みを組み込んでいます。[ConvNeXt](model_doc/convnext)は、畳み込みを現代化するためにTransformerから設計の選択肢を取り入れ、例えば、ConvNeXtは画像をパッチに分割するために重なり合わないスライディングウィンドウと、グローバル受容野を増加させるための大きなカーネルを使用します。ConvNeXtは、メモリ効率を向上させ、パフォーマンスを向上させるためにいくつかのレイヤーデザインの選択肢も提供し、Transformerと競合的になります！


### Encoder[[cv-encoder]]

[ビジョン トランスフォーマー（ViT）](model_doc/vit) は、畳み込みを使用しないコンピュータビジョンタスクの扉を開けました。ViT は標準のトランスフォーマーエンコーダーを使用しますが、画像を扱う方法が主要なブレークスルーでした。画像を固定サイズのパッチに分割し、それらをトークンのように使用して埋め込みを作成します。ViT は、当時のCNNと競争力のある結果を示すためにトランスフォーマーの効率的なアーキテクチャを活用しましたが、トレーニングに必要なリソースが少なくて済みました。ViT に続いて、セグメンテーションや検出などの密なビジョンタスクを処理できる他のビジョンモデルも登場しました。

これらのモデルの1つが[Swin](model_doc/swin) トランスフォーマーです。Swin トランスフォーマーは、より小さなサイズのパッチから階層的な特徴マップ（CNNのようで ViT とは異なります）を構築し、深層のパッチと隣接するパッチとマージします。注意はローカルウィンドウ内でのみ計算され、ウィンドウは注意のレイヤー間でシフトされ、モデルがより良く学習するのをサポートする接続を作成します。Swin トランスフォーマーは階層的な特徴マップを生成できるため、セグメンテーションや検出などの密な予測タスクに適しています。[SegFormer](model_doc/segformer) も階層的な特徴マップを構築するためにトランスフォーマーエンコーダーを使用しますが、すべての特徴マップを組み合わせて予測するためにシンプルなマルチレイヤーパーセプトロン（MLP）デコーダーを追加します。

BeIT および ViTMAE などの他のビジョンモデルは、BERTの事前トレーニング目標からインスピレーションを得ました。[BeIT](model_doc/beit) は *masked image modeling (MIM)* によって事前トレーニングされています。画像パッチはランダムにマスクされ、画像も視覚トークンにトークン化されます。BeIT はマスクされたパッチに対応する視覚トークンを予測するようにトレーニングされます。[ViTMAE](model_doc/vitmae) も似たような事前トレーニング目標を持っており、視覚トークンの代わりにピクセルを予測する必要があります。異例なのは画像パッチの75%がマスクされていることです！デコーダーはマスクされたトークンとエンコードされたパッチからピクセルを再構築します。事前トレーニングの後、デコーダーは捨てられ、エンコーダーはダウンストリームのタスクで使用できる状態です。

### Decoder[[cv-decoder]]

デコーダーのみのビジョンモデルは珍しいです。なぜなら、ほとんどのビジョンモデルは画像表現を学ぶためにエンコーダーを使用するからです。しかし、画像生成などのユースケースでは、デコーダーは自然な適応です。これは、GPT-2などのテキスト生成モデルから見てきたように、[ImageGPT](model_doc/imagegpt) でも同様のアーキテクチャを使用しますが、シーケンス内の次のトークンを予測する代わりに、画像内の次のピクセルを予測します。画像生成に加えて、ImageGPT は画像分類のためにもファインチューニングできます。

### Encoder-decoder[[cv-encoder-decoder]]

ビジョンモデルは一般的にエンコーダー（バックボーンとも呼ばれます）を使用して重要な画像特徴を抽出し、それをトランスフォーマーデコーダーに渡すために使用します。[DETR](model_doc/detr) は事前トレーニング済みのバックボーンを持っていますが、オブジェクト検出のために完全なトランスフォーマーエンコーダーデコーダーアーキテクチャも使用しています。エンコーダーは画像表現を学び、デコーダー内のオブジェクトクエリ（各オブジェクトクエリは画像内の領域またはオブジェクトに焦点を当てた学習された埋め込みです）と組み合わせます。DETR は各オブジェクトクエリに対する境界ボックスの座標とクラスラベルを予測します。

## Natural lanaguage processing

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FUhbQAZDlpYW5XEpdFy6GoG%2Fnlp-model-timeline%3Fnode-id%3D0%253A1%26t%3D4mZMr4r1vDEYGJ50-1" allowfullscreen></iframe>

### Encoder[[nlp-encoder]]

[BERT](model_doc/bert) はエンコーダー専用のTransformerで、入力の一部のトークンをランダムにマスクして他のトークンを見ないようにしています。これにより、トークンをマスクした文脈に基づいてマスクされたトークンを予測することが事前トレーニングの目標です。これにより、BERTは入力のより深いかつ豊かな表現を学習するのに左右の文脈を完全に活用できます。しかし、BERTの事前トレーニング戦略にはまだ改善の余地がありました。[RoBERTa](model_doc/roberta) は、トレーニングを長時間行い、より大きなバッチでトレーニングし、事前処理中に一度だけでなく各エポックでトークンをランダムにマスクし、次文予測の目標を削除する新しい事前トレーニングレシピを導入することでこれを改善しました。

性能を向上させる主要な戦略はモデルのサイズを増やすことですが、大規模なモデルのトレーニングは計算コストがかかります。計算コストを削減する方法の1つは、[DistilBERT](model_doc/distilbert) のような小さなモデルを使用することです。DistilBERTは[知識蒸留](https://arxiv.org/abs/1503.02531) - 圧縮技術 - を使用して、BERTのほぼすべての言語理解機能を保持しながら、より小さなバージョンを作成します。

しかし、ほとんどのTransformerモデルは引き続きより多くのパラメータに焦点を当て、トレーニング効率を向上させる新しいモデルが登場しています。[ALBERT](model_doc/albert) は、2つの方法でパラメータの数を減らすことによってメモリ消費量を削減します。大きな語彙埋め込みを2つの小さな行列に分割し、レイヤーがパラメータを共有できるようにします。[DeBERTa](model_doc/deberta) は、単語とその位置を2つのベクトルで別々にエンコードする解かれた注意機構を追加しました。注意はこれらの別々のベクトルから計算されます。単語と位置の埋め込みが含まれる単一のベクトルではなく、[Longformer](model_doc/longformer) は、特に長いシーケンス長のドキュメントを処理するために注意をより効率的にすることに焦点を当てました。固定されたウィンドウサイズの周りの各トークンから計算されるローカルウィンドウ付き注意（特定のタスクトークン（分類のための `[CLS]` など）のみのためのグローバルな注意を含む）の組み合わせを使用して、完全な注意行列ではなく疎な注意行列を作成します。


### Decoder[[nlp-decoder]]

[GPT-2](model_doc/gpt2)は、シーケンス内の次の単語を予測するデコーダー専用のTransformerです。モデルは先を見ることができないようにトークンを右にマスクし、"のぞき見"を防ぎます。大量のテキストを事前トレーニングしたことにより、GPT-2はテキスト生成が非常に得意で、テキストが正確であることがあるにしても、時折正確ではないことがあります。しかし、GPT-2にはBERTの事前トレーニングからの双方向コンテキストが不足しており、特定のタスクには適していませんでした。[XLNET](model_doc/xlnet)は、双方向に学習できる順列言語モデリング目標（PLM）を使用することで、BERTとGPT-2の事前トレーニング目標のベストを組み合わせています。

GPT-2の後、言語モデルはさらに大きく成長し、今では*大規模言語モデル（LLM）*として知られています。大規模なデータセットで事前トレーニングされれば、LLMはほぼゼロショット学習を示すことがあります。[GPT-J](model_doc/gptj)は、6Bのパラメータを持つLLMで、400Bのトークンでトレーニングされています。GPT-Jには[OPT](model_doc/opt)が続き、そのうち最大のモデルは175Bで、180Bのトークンでトレーニングされています。同じ時期に[BLOOM](model_doc/bloom)がリリースされ、このファミリーの最大のモデルは176Bのパラメータを持ち、46の言語と13のプログラミング言語で366Bのトークンでトレーニングされています。

### Encoder-decoder[[nlp-encoder-decoder]]

[BART](model_doc/bart)は、元のTransformerアーキテクチャを保持していますが、事前トレーニング目標を*テキスト補完*の破損に変更しています。一部のテキストスパンは単一の`mask`トークンで置換されます。デコーダーは破損していないトークンを予測し（未来のトークンはマスクされます）、エンコーダーの隠れた状態を使用して予測を補助します。[Pegasus](model_doc/pegasus)はBARTに似ていますが、Pegasusはテキストスパンの代わりに文全体をマスクします。マスクされた言語モデリングに加えて、Pegasusはギャップ文生成（GSG）によって事前トレーニングされています。GSGの目標は、文書に重要な文をマスクし、それらを`mask`トークンで置換することです。デコーダーは残りの文から出力を生成しなければなりません。[T5](model_doc/t5)は、すべてのNLPタスクを特定のプレフィックスを使用してテキスト対テキストの問題に変換するよりユニークなモデルです。たとえば、プレフィックス`Summarize:`は要約タスクを示します。T5は教師ありトレーニング（GLUEとSuperGLUE）と自己教師ありトレーニング（トークンの15％をランダムにサンプルしドロップアウト）によって事前トレーニングされています。


## Audio

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2Fvrchl8jDV9YwNVPWu2W0kK%2Fspeech-and-audio-model-timeline%3Fnode-id%3D0%253A1%26t%3DmM4H8pPMuK23rClL-1" allowfullscreen></iframe>

### Encoder[[audio-encoder]]

[Wav2Vec2](model_doc/wav2vec2) は、生のオーディオ波形から直接音声表現を学習するためのTransformerエンコーダーを使用します。これは、対照的なタスクで事前学習され、一連の偽の表現から真の音声表現を特定します。 [HuBERT](model_doc/hubert) はWav2Vec2に似ていますが、異なるトレーニングプロセスを持っています。ターゲットラベルは、類似したオーディオセグメントがクラスタに割り当てられ、これが隠れユニットになるクラスタリングステップによって作成されます。隠れユニットは埋め込みにマップされ、予測を行います。

### Encoder-decoder[[audio-encoder-decoder]]

[Speech2Text](model_doc/speech_to_text) は、自動音声認識（ASR）および音声翻訳のために設計された音声モデルです。このモデルは、オーディオ波形から抽出されたログメルフィルターバンクフィーチャーを受け入れ、事前トレーニングされた自己回帰的にトランスクリプトまたは翻訳を生成します。 [Whisper](model_doc/whisper) もASRモデルですが、他の多くの音声モデルとは異なり、✨ ラベル付き ✨ オーディオトランスクリプションデータを大量に事前に学習して、ゼロショットパフォーマンスを実現します。データセットの大部分には非英語の言語も含まれており、Whisperは低リソース言語にも使用できます。構造的には、WhisperはSpeech2Textに似ています。オーディオ信号はエンコーダーによってエンコードされたログメルスペクトログラムに変換されます。デコーダーはエンコーダーの隠れ状態と前のトークンからトランスクリプトを自己回帰的に生成します。

## Multimodal

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FcX125FQHXJS2gxeICiY93p%2Fmultimodal%3Fnode-id%3D0%253A1%26t%3DhPQwdx3HFPWJWnVf-1" allowfullscreen></iframe>

### Encoder[[mm-encoder]]

[VisualBERT](model_doc/visual_bert) は、BERTの後にリリースされたビジョン言語タスク向けのマルチモーダルモデルです。これはBERTと事前トレーニングされた物体検出システムを組み合わせ、画像特徴をビジュアル埋め込みに抽出し、テキスト埋め込みと一緒にBERTに渡します。VisualBERTは非マスクテキストを基にしたマスクテキストを予測し、テキストが画像と整合しているかどうかも予測する必要があります。ViTがリリースされた際、[ViLT](model_doc/vilt) は画像埋め込みを取得するためにこの方法を採用しました。画像埋め込みはテキスト埋め込みと共に共同で処理されます。それから、ViLTは画像テキストマッチング、マスク言語モデリング、および全単語マスキングによる事前トレーニングが行われます。

[CLIP](model_doc/clip) は異なるアプローチを取り、(`画像`、`テキスト`) のペア予測を行います。画像エンコーダー（ViT）とテキストエンコーダー（Transformer）は、(`画像`、`テキスト`) ペアデータセット上で共同トレーニングされ、(`画像`、`テキスト`) ペアの画像とテキストの埋め込みの類似性を最大化します。事前トレーニング後、CLIPを使用して画像からテキストを予測したり、その逆を行うことができます。[OWL-ViT](model_doc/owlvit) は、ゼロショット物体検出のバックボーンとしてCLIPを使用しています。事前トレーニング後、物体検出ヘッドが追加され、(`クラス`、`バウンディングボックス`) ペアに対するセット予測が行われます。

### Encoder-decoder[[mm-encoder-decoder]]

光学文字認識（OCR）は、通常、画像を理解しテキストを生成するために複数のコンポーネントが関与するテキスト認識タスクです。 [TrOCR](model_doc/trocr) は、エンドツーエンドのTransformerを使用してこのプロセスを簡略化します。エンコーダーは画像を固定サイズのパッチとして処理するためのViTスタイルのモデルであり、デコーダーはエンコーダーの隠れ状態を受け入れ、テキストを自己回帰的に生成します。[Donut](model_doc/donut) はOCRベースのアプローチに依存しないより一般的なビジュアルドキュメント理解モデルで、エンコーダーとしてSwin Transformer、デコーダーとして多言語BARTを使用します。 Donutは画像とテキストの注釈に基づいて次の単語を予測することにより、テキストを読むために事前トレーニングされます。デコーダーはプロンプトを与えられたトークンシーケンスを生成します。プロンプトは各ダウンストリームタスクごとに特別なトークンを使用して表現されます。例えば、ドキュメントの解析には`解析`トークンがあり、エンコーダーの隠れ状態と組み合わされてドキュメントを構造化された出力フォーマット（JSON）に解析します。

## Reinforcement learning

<iframe style="border: 1px solid rgba(0, 0, 0, 0.1);" width="1000" height="450" src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Ffile%2FiB3Y6RvWYki7ZuKO6tNgZq%2Freinforcement-learning%3Fnode-id%3D0%253A1%26t%3DhPQwdx3HFPWJWnVf-1" allowfullscreen></iframe>

### Decoder[[rl-decoder]]

意思決定と軌跡トランスフォーマーは、状態、アクション、報酬をシーケンスモデリングの問題として捉えます。 [Decision Transformer](model_doc/decision_transformer) は、リターン・トゥ・ゴー、過去の状態、およびアクションに基づいて将来の希望リターンにつながるアクションの系列を生成します。最後の *K* タイムステップでは、3つのモダリティそれぞれがトークン埋め込みに変換され、将来のアクショントークンを予測するためにGPTのようなモデルによって処理されます。[Trajectory Transformer](model_doc/trajectory_transformer) も状態、アクション、報酬をトークン化し、GPTアーキテクチャで処理します。報酬調整に焦点を当てたDecision Transformerとは異なり、Trajectory Transformerはビームサーチを使用して将来のアクションを生成します。
