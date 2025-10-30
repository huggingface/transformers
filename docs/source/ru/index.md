<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# 🤗 Transformers

Современные технологии машинного обучения для [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), и [JAX](https://jax.readthedocs.io/en/latest/).

🤗 Transformers предоставляет API и инструменты для удобной загрузки и обучения современных предобученных моделей. Использование таких моделей позволяет сократить вычислительные затраты, уменьшить углеродный след и сэкономить время и ресурсы, необходимые для обучения модели с нуля. Эти модели поддерживают популярные задачи в различных областях:

📝 **Обработка естественного языка**: классификация текста, распознавание именованных сущностей, ответы на вопросы, языковое моделирование, генерация кода, суммаризация, перевод, выбор из нескольких вариантов и генерация текста.<br>
🖼️ **Компьютерное зрение**: классификация изображений, детекция объектов и сегментация.<br>
🗣️ **Аудио**: автоматическое распознавание речи и классификация аудиозаписей.<br>
🐙 **Мультимодальные задачи**: ответы на вопросы по таблицам, оптическое распознавание символов, извлечение информации из сканированных документов, классификация видео и ответы на визуальные вопросы.

🤗 Transformers поддерживает совместимость между PyTorch, TensorFlow и JAX, что обеспечивает гибкость на разных этапах работы с моделью. Вы можете обучить модель с помощью всего трех строк кода в одном фреймворке и использовать её для инференса в другом. Кроме того, модели можно экспортировать в такие форматы, как ONNX и TorchScript, для использования в производственных системах.

Присоединяйтесь к постоянно растущему сообществу на [Hub](https://huggingface.co/models), [forum](https://discuss.huggingface.co/), или [Discord](https://discord.com/invite/JfAtkvEtRb) уже сегодня!

## Если вы ищете индивидуальную поддержку от команды Hugging Face

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a>

## Содержание

Документация разделена на пять разделов:

- **НАЧАЛО РАБОТЫ** предоставляет краткий обзор библиотеки и инструкции по установке, чтобы вы могли быстро приступить к работе.
- **ТУТОРИАЛЫ** - отличный старт для начинающих. Этот раздел поможет вам овладеть базовыми навыками, необходимыми для работы с библиотекой.
- **РУКОВОДСТВА** объясняют, как достичь конкретной цели, например, дообучить предобученную модель для языкового моделирования или создать и опубликовать собственную модель.
- **КОНЦЕПТУАЛЬНЫЕ РУКОВОДСТВА** предлагают более глубокое объяснение концепций и идей, лежащих в основе моделей, задач и принципов дизайна библиотеки 🤗 Transformers.
- **API** описывает все классы и функции:

  - **ОСНОВНЫЕ КЛАССЫ** содержат описание самых важных классов, таких как конфигурация, модель, токенизатор и пайплайн.
  - **МОДЕЛИ** содержат описание классов и функций, связанных с каждой моделью, реализованной в библиотеке.
  - **ВНУТРЕННИЕ ИНСТРУМЕНТЫ** описывают вспомогательные классы и функции, используемые внутри библиотеки.


## Поддерживаемые модели и фреймворки

В таблице ниже представлена текущая поддержка различных моделей в библиотеке, включая наличие Python-токенизатора (называемого "медленным"), "быстрого" токенизатора на базе библиотеки 🤗 Tokenizers, а также поддержку моделей в JAX (через Flax), PyTorch и/или TensorFlow.

<!--This table is updated automatically from the auto modules with _make fix-copies_. Do not update manually!-->

|                                  Model                                   | PyTorch support | TensorFlow support | Flax Support |
|:------------------------------------------------------------------------:|:---------------:|:------------------:|:------------:|
|                        [ALBERT](model_doc/albert)                        |       ✅        |         ✅         |      ✅      |
|                         [ALIGN](model_doc/align)                         |       ✅        |         ❌         |      ❌      |
|                       [AltCLIP](model_doc/altclip)                       |       ✅        |         ❌         |      ❌      |
| [Audio Spectrogram Transformer](model_doc/audio-spectrogram-transformer) |       ✅        |         ❌         |      ❌      |
|                    [Autoformer](model_doc/autoformer)                    |       ✅        |         ❌         |      ❌      |
|                          [Bark](model_doc/bark)                          |       ✅        |         ❌         |      ❌      |
|                          [BART](model_doc/bart)                          |       ✅        |         ✅         |      ✅      |
|                       [BARThez](model_doc/barthez)                       |       ✅        |         ✅         |      ✅      |
|                       [BARTpho](model_doc/bartpho)                       |       ✅        |         ✅         |      ✅      |
|                          [BEiT](model_doc/beit)                          |       ✅        |         ❌         |      ✅      |
|                          [BERT](model_doc/bert)                          |       ✅        |         ✅         |      ✅      |
|               [Bert Generation](model_doc/bert-generation)               |       ✅        |         ❌         |      ❌      |
|                 [BertJapanese](model_doc/bert-japanese)                  |       ✅        |         ✅         |      ✅      |
|                      [BERTweet](model_doc/bertweet)                      |       ✅        |         ✅         |      ✅      |
|                      [BigBird](model_doc/big_bird)                       |       ✅        |         ❌         |      ✅      |
|               [BigBird-Pegasus](model_doc/bigbird_pegasus)               |       ✅        |         ❌         |      ❌      |
|                        [BioGpt](model_doc/biogpt)                        |       ✅        |         ❌         |      ❌      |
|                           [BiT](model_doc/bit)                           |       ✅        |         ❌         |      ❌      |
|                    [Blenderbot](model_doc/blenderbot)                    |       ✅        |         ✅         |      ✅      |
|              [BlenderbotSmall](model_doc/blenderbot-small)               |       ✅        |         ✅         |      ✅      |
|                          [BLIP](model_doc/blip)                          |       ✅        |         ✅         |      ❌      |
|                        [BLIP-2](model_doc/blip-2)                        |       ✅        |         ❌         |      ❌      |
|                         [BLOOM](model_doc/bloom)                         |       ✅        |         ❌         |      ✅      |
|                          [BORT](model_doc/bort)                          |       ✅        |         ✅         |      ✅      |
|                   [BridgeTower](model_doc/bridgetower)                   |       ✅        |         ❌         |      ❌      |
|                          [BROS](model_doc/bros)                          |       ✅        |         ❌         |      ❌      |
|                          [ByT5](model_doc/byt5)                          |       ✅        |         ✅         |      ✅      |
|                     [CamemBERT](model_doc/camembert)                     |       ✅        |         ✅         |      ❌      |
|                        [CANINE](model_doc/canine)                        |       ✅        |         ❌         |      ❌      |
|                     [Chameleon](model_doc/chameleon)                     |       ✅        |         ❌         |      ❌      |
|                  [Chinese-CLIP](model_doc/chinese_clip)                  |       ✅        |         ❌         |      ❌      |
|                          [CLAP](model_doc/clap)                          |       ✅        |         ❌         |      ❌      |
|                          [CLIP](model_doc/clip)                          |       ✅        |         ✅         |      ✅      |
|                       [CLIPSeg](model_doc/clipseg)                       |       ✅        |         ❌         |      ❌      |
|                          [CLVP](model_doc/clvp)                          |       ✅        |         ❌         |      ❌      |
|                       [CodeGen](model_doc/codegen)                       |       ✅        |         ❌         |      ❌      |
|                    [CodeLlama](model_doc/code_llama)                     |       ✅        |         ❌         |      ✅      |
|                        [Cohere](model_doc/cohere)                        |       ✅        |         ❌         |      ❌      |
|              [Conditional DETR](model_doc/conditional_detr)              |       ✅        |         ❌         |      ❌      |
|                      [ConvBERT](model_doc/convbert)                      |       ✅        |         ✅         |      ❌      |
|                      [ConvNeXT](model_doc/convnext)                      |       ✅        |         ✅         |      ❌      |
|                    [ConvNeXTV2](model_doc/convnextv2)                    |       ✅        |         ✅         |      ❌      |
|                           [CPM](model_doc/cpm)                           |       ✅        |         ✅         |      ✅      |
|                       [CPM-Ant](model_doc/cpmant)                        |       ✅        |         ❌         |      ❌      |
|                          [CTRL](model_doc/ctrl)                          |       ✅        |         ✅         |      ❌      |
|                           [CvT](model_doc/cvt)                           |       ✅        |         ✅         |      ❌      |
|                           [DAC](model_doc/dac)                           |       ✅        |         ❌         |      ❌      |
|                   [Data2VecAudio](model_doc/data2vec)                    |       ✅        |         ❌         |      ❌      |
|                    [Data2VecText](model_doc/data2vec)                    |       ✅        |         ❌         |      ❌      |
|                   [Data2VecVision](model_doc/data2vec)                   |       ✅        |         ✅         |      ❌      |
|                          [DBRX](model_doc/dbrx)                          |       ✅        |         ❌         |      ❌      |
|                       [DeBERTa](model_doc/deberta)                       |       ✅        |         ✅         |      ❌      |
|                    [DeBERTa-v2](model_doc/deberta-v2)                    |       ✅        |         ✅         |      ❌      |
|          [Decision Transformer](model_doc/decision_transformer)          |       ✅        |         ❌         |      ❌      |
|               [Deformable DETR](model_doc/deformable_detr)               |       ✅        |         ❌         |      ❌      |
|                          [DeiT](model_doc/deit)                          |       ✅        |         ✅         |      ❌      |
|                        [DePlot](model_doc/deplot)                        |       ✅        |         ❌         |      ❌      |
|                [Depth Anything](model_doc/depth_anything)                |       ✅        |         ❌         |      ❌      |
|                          [DETA](model_doc/deta)                          |       ✅        |         ❌         |      ❌      |
|                          [DETR](model_doc/detr)                          |       ✅        |         ❌         |      ❌      |
|                      [DialoGPT](model_doc/dialogpt)                      |       ✅        |         ✅         |      ✅      |
|                         [DiNAT](model_doc/dinat)                         |       ✅        |         ❌         |      ❌      |
|                        [DINOv2](model_doc/dinov2)                        |       ✅        |         ❌         |      ✅      |
|                    [DistilBERT](model_doc/distilbert)                    |       ✅        |         ✅         |      ✅      |
|                           [DiT](model_doc/dit)                           |       ✅        |         ❌         |      ✅      |
|                       [DonutSwin](model_doc/donut)                       |       ✅        |         ❌         |      ❌      |
|                           [DPR](model_doc/dpr)                           |       ✅        |         ✅         |      ❌      |
|                           [DPT](model_doc/dpt)                           |       ✅        |         ❌         |      ❌      |
|               [EfficientFormer](model_doc/efficientformer)               |       ✅        |         ✅         |      ❌      |
|                  [EfficientNet](model_doc/efficientnet)                  |       ✅        |         ❌         |      ❌      |
|                       [ELECTRA](model_doc/electra)                       |       ✅        |         ✅         |      ✅      |
|                       [EnCodec](model_doc/encodec)                       |       ✅        |         ❌         |      ❌      |
|               [Encoder decoder](model_doc/encoder-decoder)               |       ✅        |         ✅         |      ✅      |
|                         [ERNIE](model_doc/ernie)                         |       ✅        |         ❌         |      ❌      |
|                       [ErnieM](model_doc/ernie_m)                        |       ✅        |         ❌         |      ❌      |
|                           [ESM](model_doc/esm)                           |       ✅        |         ✅         |      ❌      |
|              [FairSeq Machine-Translation](model_doc/fsmt)               |       ✅        |         ❌         |      ❌      |
|                        [Falcon](model_doc/falcon)                        |       ✅        |         ❌         |      ❌      |
|                  [FalconMamba](model_doc/falcon_mamba)                   |       ✅        |         ❌         |      ❌      |
|         [FastSpeech2Conformer](model_doc/fastspeech2_conformer)          |       ✅        |         ❌         |      ❌      |
|                       [FLAN-T5](model_doc/flan-t5)                       |       ✅        |         ✅         |      ✅      |
|                      [FLAN-UL2](model_doc/flan-ul2)                      |       ✅        |         ✅         |      ✅      |
|                      [FlauBERT](model_doc/flaubert)                      |       ✅        |         ✅         |      ❌      |
|                         [FLAVA](model_doc/flava)                         |       ✅        |         ❌         |      ❌      |
|                          [FNet](model_doc/fnet)                          |       ✅        |         ❌         |      ❌      |
|                      [FocalNet](model_doc/focalnet)                      |       ✅        |         ❌         |      ❌      |
|                  [Funnel Transformer](model_doc/funnel)                  |       ✅        |         ✅         |      ❌      |
|                          [Fuyu](model_doc/fuyu)                          |       ✅        |         ❌         |      ❌      |
|                         [Gemma](model_doc/gemma)                         |       ✅        |         ❌         |      ✅      |
|                        [Gemma2](model_doc/gemma2)                        |       ✅        |         ❌         |      ❌      |
|                           [GIT](model_doc/git)                           |       ✅        |         ❌         |      ❌      |
|                           [GLM](model_doc/glm)                           |       ✅        |         ❌         |      ❌      |
|                          [GLPN](model_doc/glpn)                          |       ✅        |         ❌         |      ❌      |
|                       [GPT Neo](model_doc/gpt_neo)                       |       ✅        |         ❌         |      ✅      |
|                      [GPT NeoX](model_doc/gpt_neox)                      |       ✅        |         ❌         |      ❌      |
|             [GPT NeoX Japanese](model_doc/gpt_neox_japanese)             |       ✅        |         ❌         |      ❌      |
|                         [GPT-J](model_doc/gptj)                          |       ✅        |         ✅         |      ✅      |
|                       [GPT-Sw3](model_doc/gpt-sw3)                       |       ✅        |         ✅         |      ✅      |
|                   [GPTBigCode](model_doc/gpt_bigcode)                    |       ✅        |         ❌         |      ❌      |
|               [GPTSAN-japanese](model_doc/gptsan-japanese)               |       ✅        |         ❌         |      ❌      |
|                       [Granite](model_doc/granite)                       |       ✅        |         ❌         |      ❌      |
|                  [GraniteMoeMoe](model_doc/granitemoe)                   |       ✅        |         ❌         |      ❌      |
|                    [Graphormer](model_doc/graphormer)                    |       ✅        |         ❌         |      ❌      |
|                [Grounding DINO](model_doc/grounding-dino)                |       ✅        |         ❌         |      ❌      |
|                      [GroupViT](model_doc/groupvit)                      |       ✅        |         ✅         |      ❌      |
|                       [HerBERT](model_doc/herbert)                       |       ✅        |         ✅         |      ✅      |
|                         [Hiera](model_doc/hiera)                         |       ✅        |         ❌         |      ❌      |
|                        [Hubert](model_doc/hubert)                        |       ✅        |         ✅         |      ❌      |
|                        [I-BERT](model_doc/ibert)                         |       ✅        |         ❌         |      ❌      |
|                       [IDEFICS](model_doc/idefics)                       |       ✅        |         ✅         |      ❌      |
|                      [Idefics2](model_doc/idefics2)                      |       ✅        |         ❌         |      ❌      |
|                      [Idefics3](model_doc/idefics3)                      |       ✅        |         ❌         |      ❌      |
|                      [ImageGPT](model_doc/imagegpt)                      |       ✅        |         ❌         |      ❌      |
|                      [Informer](model_doc/informer)                      |       ✅        |         ❌         |      ❌      |
|                  [InstructBLIP](model_doc/instructblip)                  |       ✅        |         ❌         |      ❌      |
|             [InstructBlipVideo](model_doc/instructblipvideo)             |       ✅        |         ❌         |      ❌      |
|                         [Jamba](model_doc/jamba)                         |       ✅        |         ❌         |      ❌      |
|                        [JetMoe](model_doc/jetmoe)                        |       ✅        |         ❌         |      ❌      |
|                       [Jukebox](model_doc/jukebox)                       |       ✅        |         ❌         |      ❌      |
|                      [KOSMOS-2](model_doc/kosmos-2)                      |       ✅        |         ❌         |      ❌      |
|                      [LayoutLM](model_doc/layoutlm)                      |       ✅        |         ✅         |      ❌      |
|                    [LayoutLMv2](model_doc/layoutlmv2)                    |       ✅        |         ❌         |      ❌      |
|                    [LayoutLMv3](model_doc/layoutlmv3)                    |       ✅        |         ✅         |      ❌      |
|                     [LayoutXLM](model_doc/layoutxlm)                     |       ✅        |         ❌         |      ❌      |
|                           [LED](model_doc/led)                           |       ✅        |         ✅         |      ❌      |
|                         [LeViT](model_doc/levit)                         |       ✅        |         ❌         |      ❌      |
|                          [LiLT](model_doc/lilt)                          |       ✅        |         ❌         |      ❌      |
|                         [LLaMA](model_doc/llama)                         |       ✅        |         ❌         |      ✅      |
|                        [Llama2](model_doc/llama2)                        |       ✅        |         ❌         |      ✅      |
|                        [Llama3](model_doc/llama3)                        |       ✅        |         ❌         |      ✅      |
|                         [LLaVa](model_doc/llava)                         |       ✅        |         ❌         |      ❌      |
|                    [LLaVA-NeXT](model_doc/llava_next)                    |       ✅        |         ❌         |      ❌      |
|              [LLaVa-NeXT-Video](model_doc/llava_next_video)              |       ✅        |         ❌         |      ❌      |
|               [LLaVA-Onevision](model_doc/llava_onevision)               |       ✅        |         ❌         |      ❌      |
|                    [Longformer](model_doc/longformer)                    |       ✅        |         ✅         |      ❌      |
|                        [LongT5](model_doc/longt5)                        |       ✅        |         ❌         |      ✅      |
|                          [LUKE](model_doc/luke)                          |       ✅        |         ❌         |      ❌      |
|                        [LXMERT](model_doc/lxmert)                        |       ✅        |         ✅         |      ❌      |
|                        [M-CTC-T](model_doc/mctct)                        |       ✅        |         ❌         |      ❌      |
|                       [M2M100](model_doc/m2m_100)                        |       ✅        |         ❌         |      ❌      |
|                    [MADLAD-400](model_doc/madlad-400)                    |       ✅        |         ✅         |      ✅      |
|                         [Mamba](model_doc/mamba)                         |       ✅        |         ❌         |      ❌      |
|                        [mamba2](model_doc/mamba2)                        |       ✅        |         ❌         |      ❌      |
|                        [Marian](model_doc/marian)                        |       ✅        |         ✅         |      ✅      |
|                      [MarkupLM](model_doc/markuplm)                      |       ✅        |         ❌         |      ❌      |
|                   [Mask2Former](model_doc/mask2former)                   |       ✅        |         ❌         |      ❌      |
|                    [MaskFormer](model_doc/maskformer)                    |       ✅        |         ❌         |      ❌      |
|                        [MatCha](model_doc/matcha)                        |       ✅        |         ❌         |      ❌      |
|                         [mBART](model_doc/mbart)                         |       ✅        |         ✅         |      ✅      |
|                      [mBART-50](model_doc/mbart50)                       |       ✅        |         ✅         |      ✅      |
|                          [MEGA](model_doc/mega)                          |       ✅        |         ❌         |      ❌      |
|                 [Megatron-BERT](model_doc/megatron-bert)                 |       ✅        |         ❌         |      ❌      |
|                 [Megatron-GPT2](model_doc/megatron_gpt2)                 |       ✅        |         ✅         |      ✅      |
|                       [MGP-STR](model_doc/mgp-str)                       |       ✅        |         ❌         |      ❌      |
|                          [Mimi](model_doc/mimi)                          |       ✅        |         ❌         |      ❌      |
|                       [Mistral](model_doc/mistral)                       |       ✅        |         ✅         |      ✅      |
|                       [Mixtral](model_doc/mixtral)                       |       ✅        |         ❌         |      ❌      |
|                        [Mllama](model_doc/mllama)                        |       ✅        |         ❌         |      ❌      |
|                         [mLUKE](model_doc/mluke)                         |       ✅        |         ❌         |      ❌      |
|                           [MMS](model_doc/mms)                           |       ✅        |         ✅         |      ✅      |
|                    [MobileBERT](model_doc/mobilebert)                    |       ✅        |         ✅         |      ❌      |
|                  [MobileNetV1](model_doc/mobilenet_v1)                   |       ✅        |         ❌         |      ❌      |
|                  [MobileNetV2](model_doc/mobilenet_v2)                   |       ✅        |         ❌         |      ❌      |
|                     [MobileViT](model_doc/mobilevit)                     |       ✅        |         ✅         |      ❌      |
|                   [MobileViTV2](model_doc/mobilevitv2)                   |       ✅        |         ❌         |      ❌      |
|                         [Moshi](model_doc/moshi)                         |       ✅        |         ❌         |      ❌      |
|                         [MPNet](model_doc/mpnet)                         |       ✅        |         ✅         |      ❌      |
|                           [MPT](model_doc/mpt)                           |       ✅        |         ❌         |      ❌      |
|                           [MRA](model_doc/mra)                           |       ✅        |         ❌         |      ❌      |
|                           [MT5](model_doc/mt5)                           |       ✅        |         ✅         |      ✅      |
|                      [MusicGen](model_doc/musicgen)                      |       ✅        |         ❌         |      ❌      |
|               [MusicGen Melody](model_doc/musicgen_melody)               |       ✅        |         ❌         |      ❌      |
|                           [MVP](model_doc/mvp)                           |       ✅        |         ❌         |      ❌      |
|                           [NAT](model_doc/nat)                           |       ✅        |         ❌         |      ❌      |
|                      [Nemotron](model_doc/nemotron)                      |       ✅        |         ❌         |      ❌      |
|                         [Nezha](model_doc/nezha)                         |       ✅        |         ❌         |      ❌      |
|                          [NLLB](model_doc/nllb)                          |       ✅        |         ❌         |      ❌      |
|                      [NLLB-MOE](model_doc/nllb-moe)                      |       ✅        |         ❌         |      ❌      |
|                        [Nougat](model_doc/nougat)                        |       ✅        |         ✅         |      ✅      |
|                 [Nyströmformer](model_doc/nystromformer)                 |       ✅        |         ❌         |      ❌      |
|                          [OLMo](model_doc/olmo)                          |       ✅        |         ❌         |      ❌      |
|                [OLMo November 2024](model_doc/olmo_1124)                 |       ✅        |         ❌         |      ❌      |
|                         [OLMoE](model_doc/olmoe)                         |       ✅        |         ❌         |      ❌      |
|                   [OmDet-Turbo](model_doc/omdet-turbo)                   |       ✅        |         ❌         |      ❌      |
|                     [OneFormer](model_doc/oneformer)                     |       ✅        |         ❌         |      ❌      |
|                    [OpenAI GPT](model_doc/openai-gpt)                    |       ✅        |         ✅         |      ❌      |
|                      [OpenAI GPT-2](model_doc/gpt2)                      |       ✅        |         ✅         |      ✅      |
|                    [OpenLlama](model_doc/open-llama)                     |       ✅        |         ❌         |      ❌      |
|                           [OPT](model_doc/opt)                           |       ✅        |         ✅         |      ✅      |
|                       [OWL-ViT](model_doc/owlvit)                        |       ✅        |         ❌         |      ❌      |
|                         [OWLv2](model_doc/owlv2)                         |       ✅        |         ❌         |      ❌      |
|                     [PaliGemma](model_doc/paligemma)                     |       ✅        |         ❌         |      ❌      |
|                  [PatchTSMixer](model_doc/patchtsmixer)                  |       ✅        |         ❌         |      ❌      |
|                      [PatchTST](model_doc/patchtst)                      |       ✅        |         ❌         |      ❌      |
|                       [Pegasus](model_doc/pegasus)                       |       ✅        |         ✅         |      ✅      |
|                     [PEGASUS-X](model_doc/pegasus_x)                     |       ✅        |         ❌         |      ❌      |
|                     [Perceiver](model_doc/perceiver)                     |       ✅        |         ❌         |      ❌      |
|                     [Persimmon](model_doc/persimmon)                     |       ✅        |         ❌         |      ❌      |
|                           [Phi](model_doc/phi)                           |       ✅        |         ❌         |      ❌      |
|                          [Phi3](model_doc/phi3)                          |       ✅        |         ❌         |      ❌      |
|                        [Phimoe](model_doc/phimoe)                        |       ✅        |         ❌         |      ❌      |
|                       [PhoBERT](model_doc/phobert)                       |       ✅        |         ✅         |      ✅      |
|                    [Pix2Struct](model_doc/pix2struct)                    |       ✅        |         ❌         |      ❌      |
|                       [Pixtral](model_doc/pixtral)                       |       ✅        |         ❌         |      ❌      |
|                        [PLBart](model_doc/plbart)                        |       ✅        |         ❌         |      ❌      |
|                    [PoolFormer](model_doc/poolformer)                    |       ✅        |         ❌         |      ❌      |
|                     [Pop2Piano](model_doc/pop2piano)                     |       ✅        |         ❌         |      ❌      |
|                    [ProphetNet](model_doc/prophetnet)                    |       ✅        |         ❌         |      ❌      |
|                           [PVT](model_doc/pvt)                           |       ✅        |         ❌         |      ❌      |
|                        [PVTv2](model_doc/pvt_v2)                         |       ✅        |         ❌         |      ❌      |
|                       [QDQBert](model_doc/qdqbert)                       |       ✅        |         ❌         |      ❌      |
|                         [Qwen2](model_doc/qwen2)                         |       ✅        |         ❌         |      ❌      |
|                   [Qwen2Audio](model_doc/qwen2_audio)                    |       ✅        |         ❌         |      ❌      |
|                     [Qwen2MoE](model_doc/qwen2_moe)                      |       ✅        |         ❌         |      ❌      |
|                      [Qwen2VL](model_doc/qwen2_vl)                       |       ✅        |         ❌         |      ❌      |
|                           [RAG](model_doc/rag)                           |       ✅        |         ✅         |      ❌      |
|                         [REALM](model_doc/realm)                         |       ✅        |         ❌         |      ❌      |
|               [RecurrentGemma](model_doc/recurrent_gemma)                |       ✅        |         ❌         |      ❌      |
|                      [Reformer](model_doc/reformer)                      |       ✅        |         ❌         |      ❌      |
|                        [RegNet](model_doc/regnet)                        |       ✅        |         ✅         |      ✅      |
|                       [RemBERT](model_doc/rembert)                       |       ✅        |         ✅         |      ❌      |
|                        [ResNet](model_doc/resnet)                        |       ✅        |         ✅         |      ✅      |
|                     [RetriBERT](model_doc/retribert)                     |       ✅        |         ❌         |      ❌      |
|                       [RoBERTa](model_doc/roberta)                       |       ✅        |         ✅         |      ✅      |
|          [RoBERTa-PreLayerNorm](model_doc/roberta-prelayernorm)          |       ✅        |         ✅         |      ✅      |
|                      [RoCBert](model_doc/roc_bert)                       |       ✅        |         ❌         |      ❌      |
|                      [RoFormer](model_doc/roformer)                      |       ✅        |         ✅         |      ✅      |
|                       [RT-DETR](model_doc/rt_detr)                       |       ✅        |         ❌         |      ❌      |
|                [RT-DETR-ResNet](model_doc/rt_detr_resnet)                |       ✅        |         ❌         |      ❌      |
|                          [RWKV](model_doc/rwkv)                          |       ✅        |         ❌         |      ❌      |
|                           [SAM](model_doc/sam)                           |       ✅        |         ✅         |      ❌      |
|                  [SeamlessM4T](model_doc/seamless_m4t)                   |       ✅        |         ❌         |      ❌      |
|                [SeamlessM4Tv2](model_doc/seamless_m4t_v2)                |       ✅        |         ❌         |      ❌      |
|                     [SegFormer](model_doc/segformer)                     |       ✅        |         ✅         |      ❌      |
|                        [SegGPT](model_doc/seggpt)                        |       ✅        |         ❌         |      ❌      |
|                           [SEW](model_doc/sew)                           |       ✅        |         ❌         |      ❌      |
|                         [SEW-D](model_doc/sew-d)                         |       ✅        |         ❌         |      ❌      |
|                        [SigLIP](model_doc/siglip)                        |       ✅        |         ❌         |      ❌      |
|        [Speech Encoder decoder](model_doc/speech-encoder-decoder)        |       ✅        |         ❌         |      ✅      |
|                 [Speech2Text](model_doc/speech_to_text)                  |       ✅        |         ✅         |      ❌      |
|                      [SpeechT5](model_doc/speecht5)                      |       ✅        |         ❌         |      ❌      |
|                      [Splinter](model_doc/splinter)                      |       ✅        |         ❌         |      ❌      |
|                   [SqueezeBERT](model_doc/squeezebert)                   |       ✅        |         ❌         |      ❌      |
|                      [StableLm](model_doc/stablelm)                      |       ✅        |         ❌         |      ❌      |
|                    [Starcoder2](model_doc/starcoder2)                    |       ✅        |         ❌         |      ❌      |
|                    [SuperPoint](model_doc/superpoint)                    |       ✅        |         ❌         |      ❌      |
|                   [SwiftFormer](model_doc/swiftformer)                   |       ✅        |         ✅         |      ❌      |
|                    [Swin Transformer](model_doc/swin)                    |       ✅        |         ✅         |      ❌      |
|                 [Swin Transformer V2](model_doc/swinv2)                  |       ✅        |         ❌         |      ❌      |
|                       [Swin2SR](model_doc/swin2sr)                       |       ✅        |         ❌         |      ❌      |
|           [SwitchTransformers](model_doc/switch_transformers)            |       ✅        |         ❌         |      ❌      |
|                            [T5](model_doc/t5)                            |       ✅        |         ✅         |      ✅      |
|                        [T5v1.1](model_doc/t5v1.1)                        |       ✅        |         ✅         |      ✅      |
|             [Table Transformer](model_doc/table-transformer)             |       ✅        |         ❌         |      ❌      |
|                         [TAPAS](model_doc/tapas)                         |       ✅        |         ✅         |      ❌      |
|                         [TAPEX](model_doc/tapex)                         |       ✅        |         ✅         |      ✅      |
|       [Time Series Transformer](model_doc/time_series_transformer)       |       ✅        |         ❌         |      ❌      |
|                   [TimeSformer](model_doc/timesformer)                   |       ✅        |         ❌         |      ❌      |
|        [Trajectory Transformer](model_doc/trajectory_transformer)        |       ✅        |         ❌         |      ❌      |
|                  [Transformer-XL](model_doc/transfo-xl)                  |       ✅        |         ✅         |      ❌      |
|                         [TrOCR](model_doc/trocr)                         |       ✅        |         ❌         |      ❌      |
|                          [TVLT](model_doc/tvlt)                          |       ✅        |         ❌         |      ❌      |
|                           [TVP](model_doc/tvp)                           |       ✅        |         ❌         |      ❌      |
|                          [UDOP](model_doc/udop)                          |       ✅        |         ❌         |      ❌      |
|                           [UL2](model_doc/ul2)                           |       ✅        |         ✅         |      ✅      |
|                          [UMT5](model_doc/umt5)                          |       ✅        |         ❌         |      ❌      |
|                     [UniSpeech](model_doc/unispeech)                     |       ✅        |         ❌         |      ❌      |
|                 [UniSpeechSat](model_doc/unispeech-sat)                  |       ✅        |         ❌         |      ❌      |
|                       [UnivNet](model_doc/univnet)                       |       ✅        |         ❌         |      ❌      |
|                       [UPerNet](model_doc/upernet)                       |       ✅        |         ❌         |      ❌      |
|                           [VAN](model_doc/van)                           |       ✅        |         ❌         |      ❌      |
|                   [VideoLlava](model_doc/video_llava)                    |       ✅        |         ❌         |      ❌      |
|                      [VideoMAE](model_doc/videomae)                      |       ✅        |         ❌         |      ❌      |
|                          [ViLT](model_doc/vilt)                          |       ✅        |         ❌         |      ❌      |
|                      [VipLlava](model_doc/vipllava)                      |       ✅        |         ❌         |      ❌      |
|        [Vision Encoder decoder](model_doc/vision-encoder-decoder)        |       ✅        |         ✅         |      ✅      |
|       [VisionTextDualEncoder](model_doc/vision-text-dual-encoder)        |       ✅        |         ✅         |      ✅      |
|                   [VisualBERT](model_doc/visual_bert)                    |       ✅        |         ❌         |      ❌      |
|                           [ViT](model_doc/vit)                           |       ✅        |         ✅         |      ✅      |
|                    [ViT Hybrid](model_doc/vit_hybrid)                    |       ✅        |         ❌         |      ❌      |
|                        [VitDet](model_doc/vitdet)                        |       ✅        |         ❌         |      ❌      |
|                       [ViTMAE](model_doc/vit_mae)                        |       ✅        |         ✅         |      ❌      |
|                      [ViTMatte](model_doc/vitmatte)                      |       ✅        |         ❌         |      ❌      |
|                       [ViTMSN](model_doc/vit_msn)                        |       ✅        |         ❌         |      ❌      |
|                          [VITS](model_doc/vits)                          |       ✅        |         ❌         |      ❌      |
|                         [ViViT](model_doc/vivit)                         |       ✅        |         ❌         |      ❌      |
|                      [Wav2Vec2](model_doc/wav2vec2)                      |       ✅        |         ✅         |      ✅      |
|                 [Wav2Vec2-BERT](model_doc/wav2vec2-bert)                 |       ✅        |         ❌         |      ❌      |
|            [Wav2Vec2-Conformer](model_doc/wav2vec2-conformer)            |       ✅        |         ❌         |      ❌      |
|              [Wav2Vec2Phoneme](model_doc/wav2vec2_phoneme)               |       ✅        |         ✅         |      ✅      |
|                         [WavLM](model_doc/wavlm)                         |       ✅        |         ❌         |      ❌      |
|                       [Whisper](model_doc/whisper)                       |       ✅        |         ✅         |      ✅      |
|                        [X-CLIP](model_doc/xclip)                         |       ✅        |         ❌         |      ❌      |
|                         [X-MOD](model_doc/xmod)                          |       ✅        |         ❌         |      ❌      |
|                          [XGLM](model_doc/xglm)                          |       ✅        |         ✅         |      ✅      |
|                           [XLM](model_doc/xlm)                           |       ✅        |         ✅         |      ❌      |
|                [XLM-ProphetNet](model_doc/xlm-prophetnet)                |       ✅        |         ❌         |      ❌      |
|                   [XLM-RoBERTa](model_doc/xlm-roberta)                   |       ✅        |         ✅         |      ✅      |
|                [XLM-RoBERTa-XL](model_doc/xlm-roberta-xl)                |       ✅        |         ❌         |      ❌      |
|                         [XLM-V](model_doc/xlm-v)                         |       ✅        |         ✅         |      ✅      |
|                         [XLNet](model_doc/xlnet)                         |       ✅        |         ✅         |      ❌      |
|                         [XLS-R](model_doc/xls_r)                         |       ✅        |         ✅         |      ✅      |
|                 [XLSR-Wav2Vec2](model_doc/xlsr_wav2vec2)                 |       ✅        |         ✅         |      ✅      |
|                         [YOLOS](model_doc/yolos)                         |       ✅        |         ❌         |      ❌      |
|                          [YOSO](model_doc/yoso)                          |       ✅        |         ❌         |      ❌      |
|                         [Zamba](model_doc/zamba)                         |       ✅        |         ❌         |      ❌      |
|                      [ZoeDepth](model_doc/zoedepth)                      |       ✅        |         ❌         |      ❌      |

<!-- End table-->
