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

State-of-the-art Machine Learning for [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and [JAX](https://jax.readthedocs.io/en/latest/).

🤗 Transformers provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities, such as:

📝 **Natural Language Processing**: text classification, named entity recognition, question answering, language modeling, summarization, translation, multiple choice, and text generation.<br>
🖼️ **Computer Vision**: image classification, object detection, and segmentation.<br>
🗣️ **Audio**: automatic speech recognition and audio classification.<br>
🐙 **Multimodal**: table question answering, optical character recognition, information extraction from scanned documents, video classification, and visual question answering.

🤗 Transformers support framework interoperability between PyTorch, TensorFlow, and JAX. This provides the flexibility to use a different framework at each stage of a model's life; train a model in three lines of code in one framework, and load it for inference in another. Models can also be exported to a format like ONNX and TorchScript for deployment in production environments.

Join the growing community on the [Hub](https://huggingface.co/models), [forum](https://discuss.huggingface.co/), or [Discord](https://discord.com/invite/JfAtkvEtRb) today!

## If you are looking for custom support from the Hugging Face team

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a>

## Contents

The documentation is organized into five sections:

- **GET STARTED** provides a quick tour of the library and installation instructions to get up and running.
- **TUTORIALS** are a great place to start if you're a beginner. This section will help you gain the basic skills you need to start using the library.
- **HOW-TO GUIDES** show you how to achieve a specific goal, like finetuning a pretrained model for language modeling or how to write and share a custom model.
- **CONCEPTUAL GUIDES** offers more discussion and explanation of the underlying concepts and ideas behind models, tasks, and the design philosophy of 🤗 Transformers.
- **API** describes all classes and functions:

  - **MAIN CLASSES** details the most important classes like configuration, model, tokenizer, and pipeline.
  - **MODELS** details the classes and functions related to each model implemented in the library.
  - **INTERNAL HELPERS** details utility classes and functions used internally.


## Supported models and frameworks

The table below represents the current support in the library for each of those models, whether they have a Python
tokenizer (called "slow"). A "fast" tokenizer backed by the 🤗 Tokenizers library, whether they have support in Jax (via
Flax), PyTorch, and/or TensorFlow.

<!--This table is updated automatically from the auto modules with _make fix-copies_. Do not update manually!-->

<<<<<<< HEAD
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
|                  [Chinese-CLIP](model_doc/chinese_clip)                  |       ✅        |         ❌         |      ❌      |
|                          [CLAP](model_doc/clap)                          |       ✅        |         ❌         |      ❌      |
|                          [CLIP](model_doc/clip)                          |       ✅        |         ✅         |      ✅      |
|                       [CLIPSeg](model_doc/clipseg)                       |       ✅        |         ❌         |      ❌      |
|                       [CodeGen](model_doc/codegen)                       |       ✅        |         ❌         |      ❌      |
|                    [CodeLlama](model_doc/code_llama)                     |       ✅        |         ❌         |      ❌      |
|              [Conditional DETR](model_doc/conditional_detr)              |       ✅        |         ❌         |      ❌      |
|                      [ConvBERT](model_doc/convbert)                      |       ✅        |         ✅         |      ❌      |
|                      [ConvNeXT](model_doc/convnext)                      |       ✅        |         ✅         |      ❌      |
|                    [ConvNeXTV2](model_doc/convnextv2)                    |       ✅        |         ✅         |      ❌      |
|                           [CPM](model_doc/cpm)                           |       ✅        |         ✅         |      ✅      |
|                       [CPM-Ant](model_doc/cpmant)                        |       ✅        |         ❌         |      ❌      |
|                          [CTRL](model_doc/ctrl)                          |       ✅        |         ✅         |      ❌      |
|                           [CvT](model_doc/cvt)                           |       ✅        |         ✅         |      ❌      |
|                   [Data2VecAudio](model_doc/data2vec)                    |       ✅        |         ❌         |      ❌      |
|                    [Data2VecText](model_doc/data2vec)                    |       ✅        |         ❌         |      ❌      |
|                   [Data2VecVision](model_doc/data2vec)                   |       ✅        |         ✅         |      ❌      |
|                       [DeBERTa](model_doc/deberta)                       |       ✅        |         ✅         |      ❌      |
|                    [DeBERTa-v2](model_doc/deberta-v2)                    |       ✅        |         ✅         |      ❌      |
|          [Decision Transformer](model_doc/decision_transformer)          |       ✅        |         ❌         |      ❌      |
|               [Deformable DETR](model_doc/deformable_detr)               |       ✅        |         ❌         |      ❌      |
|                          [DeiT](model_doc/deit)                          |       ✅        |         ✅         |      ❌      |
|                        [DePlot](model_doc/deplot)                        |       ✅        |         ❌         |      ❌      |
|                          [DETA](model_doc/deta)                          |       ✅        |         ❌         |      ❌      |
|                          [DETR](model_doc/detr)                          |       ✅        |         ❌         |      ❌      |
|                      [DialoGPT](model_doc/dialogpt)                      |       ✅        |         ✅         |      ✅      |
|                         [DiNAT](model_doc/dinat)                         |       ✅        |         ❌         |      ❌      |
|                        [DINOv2](model_doc/dinov2)                        |       ✅        |         ❌         |      ❌      |
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
|                       [FLAN-T5](model_doc/flan-t5)                       |       ✅        |         ✅         |      ✅      |
|                      [FLAN-UL2](model_doc/flan-ul2)                      |       ✅        |         ✅         |      ✅      |
|                      [FlauBERT](model_doc/flaubert)                      |       ✅        |         ✅         |      ❌      |
|                         [FLAVA](model_doc/flava)                         |       ✅        |         ❌         |      ❌      |
|                          [FNet](model_doc/fnet)                          |       ✅        |         ❌         |      ❌      |
|                      [FocalNet](model_doc/focalnet)                      |       ✅        |         ❌         |      ❌      |
|                  [Funnel Transformer](model_doc/funnel)                  |       ✅        |         ✅         |      ❌      |
|                          [Fuyu](model_doc/fuyu)                          |       ✅        |         ❌         |      ❌      |
|                           [GIT](model_doc/git)                           |       ✅        |         ❌         |      ❌      |
|                          [GLPN](model_doc/glpn)                          |       ✅        |         ❌         |      ❌      |
|                       [GPT Neo](model_doc/gpt_neo)                       |       ✅        |         ❌         |      ✅      |
|                      [GPT NeoX](model_doc/gpt_neox)                      |       ✅        |         ❌         |      ❌      |
|             [GPT NeoX Japanese](model_doc/gpt_neox_japanese)             |       ✅        |         ❌         |      ❌      |
|                         [GPT-J](model_doc/gptj)                          |       ✅        |         ✅         |      ✅      |
|                       [GPT-Sw3](model_doc/gpt-sw3)                       |       ✅        |         ✅         |      ✅      |
|                   [GPTBigCode](model_doc/gpt_bigcode)                    |       ✅        |         ❌         |      ❌      |
|               [GPTSAN-japanese](model_doc/gptsan-japanese)               |       ✅        |         ❌         |      ❌      |
|                    [Graphormer](model_doc/graphormer)                    |       ✅        |         ❌         |      ❌      |
|                      [GroupViT](model_doc/groupvit)                      |       ✅        |         ✅         |      ❌      |
|                       [HerBERT](model_doc/herbert)                       |       ✅        |         ✅         |      ✅      |
|                        [Hubert](model_doc/hubert)                        |       ✅        |         ✅         |      ❌      |
|                        [I-BERT](model_doc/ibert)                         |       ✅        |         ❌         |      ❌      |
|                       [IDEFICS](model_doc/idefics)                       |       ✅        |         ❌         |      ❌      |
|                      [ImageGPT](model_doc/imagegpt)                      |       ✅        |         ❌         |      ❌      |
|                      [Informer](model_doc/informer)                      |       ✅        |         ❌         |      ❌      |
|                  [InstructBLIP](model_doc/instructblip)                  |       ✅        |         ❌         |      ❌      |
|                       [Jukebox](model_doc/jukebox)                       |       ✅        |         ❌         |      ❌      |
|                      [KOSMOS-2](model_doc/kosmos-2)                      |       ✅        |         ❌         |      ❌      |
|                      [LayoutLM](model_doc/layoutlm)                      |       ✅        |         ✅         |      ❌      |
|                    [LayoutLMv2](model_doc/layoutlmv2)                    |       ✅        |         ❌         |      ❌      |
|                    [LayoutLMv3](model_doc/layoutlmv3)                    |       ✅        |         ✅         |      ❌      |
|                     [LayoutXLM](model_doc/layoutxlm)                     |       ✅        |         ❌         |      ❌      |
|                           [LED](model_doc/led)                           |       ✅        |         ✅         |      ❌      |
|                         [LeViT](model_doc/levit)                         |       ✅        |         ❌         |      ❌      |
|                          [LiLT](model_doc/lilt)                          |       ✅        |         ❌         |      ❌      |
|                         [LLaMA](model_doc/llama)                         |       ✅        |         ❌         |      ❌      |
|                        [Llama2](model_doc/llama2)                        |       ✅        |         ❌         |      ❌      |
|                    [Longformer](model_doc/longformer)                    |       ✅        |         ✅         |      ❌      |
|                        [LongT5](model_doc/longt5)                        |       ✅        |         ❌         |      ✅      |
|                          [LUKE](model_doc/luke)                          |       ✅        |         ❌         |      ❌      |
|                        [LXMERT](model_doc/lxmert)                        |       ✅        |         ✅         |      ❌      |
|                        [M-CTC-T](model_doc/mctct)                        |       ✅        |         ❌         |      ❌      |
|                       [M2M100](model_doc/m2m_100)                        |       ✅        |         ❌         |      ❌      |
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
|                       [Mistral](model_doc/mistral)                       |       ✅        |         ❌         |      ❌      |
|                         [mLUKE](model_doc/mluke)                         |       ✅        |         ❌         |      ❌      |
|                           [MMS](model_doc/mms)                           |       ✅        |         ✅         |      ✅      |
|                    [MobileBERT](model_doc/mobilebert)                    |       ✅        |         ✅         |      ❌      |
|                  [MobileNetV1](model_doc/mobilenet_v1)                   |       ✅        |         ❌         |      ❌      |
|                  [MobileNetV2](model_doc/mobilenet_v2)                   |       ✅        |         ❌         |      ❌      |
|                     [MobileViT](model_doc/mobilevit)                     |       ✅        |         ✅         |      ❌      |
|                   [MobileViTV2](model_doc/mobilevitv2)                   |       ✅        |         ❌         |      ❌      |
|                         [MPNet](model_doc/mpnet)                         |       ✅        |         ✅         |      ❌      |
|                           [MPT](model_doc/mpt)                           |       ✅        |         ❌         |      ❌      |
|                           [MRA](model_doc/mra)                           |       ✅        |         ❌         |      ❌      |
|                           [MT5](model_doc/mt5)                           |       ✅        |         ✅         |      ✅      |
|                      [MusicGen](model_doc/musicgen)                      |       ✅        |         ❌         |      ❌      |
|                           [MVP](model_doc/mvp)                           |       ✅        |         ❌         |      ❌      |
|                           [NAT](model_doc/nat)                           |       ✅        |         ❌         |      ❌      |
|                         [Nezha](model_doc/nezha)                         |       ✅        |         ❌         |      ❌      |
|                          [NLLB](model_doc/nllb)                          |       ✅        |         ❌         |      ❌      |
|                      [NLLB-MOE](model_doc/nllb-moe)                      |       ✅        |         ❌         |      ❌      |
|                        [Nougat](model_doc/nougat)                        |       ✅        |         ✅         |      ✅      |
|                 [Nyströmformer](model_doc/nystromformer)                 |       ✅        |         ❌         |      ❌      |
|                     [OneFormer](model_doc/oneformer)                     |       ✅        |         ❌         |      ❌      |
|                    [OpenAI GPT](model_doc/openai-gpt)                    |       ✅        |         ✅         |      ❌      |
|                      [OpenAI GPT-2](model_doc/gpt2)                      |       ✅        |         ✅         |      ✅      |
|                    [OpenLlama](model_doc/open-llama)                     |       ✅        |         ❌         |      ❌      |
|                           [OPT](model_doc/opt)                           |       ✅        |         ✅         |      ✅      |
|                       [OWL-ViT](model_doc/owlvit)                        |       ✅        |         ❌         |      ❌      |
|                         [OWLv2](model_doc/owlv2)                         |       ✅        |         ❌         |      ❌      |
|                       [Pegasus](model_doc/pegasus)                       |       ✅        |         ✅         |      ✅      |
|                     [PEGASUS-X](model_doc/pegasus_x)                     |       ✅        |         ❌         |      ❌      |
|                     [Perceiver](model_doc/perceiver)                     |       ✅        |         ❌         |      ❌      |
|                     [Persimmon](model_doc/persimmon)                     |       ✅        |         ❌         |      ❌      |
|                       [PhoBERT](model_doc/phobert)                       |       ✅        |         ✅         |      ✅      |
|                    [Pix2Struct](model_doc/pix2struct)                    |       ✅        |         ❌         |      ❌      |
|                        [PLBart](model_doc/plbart)                        |       ✅        |         ❌         |      ❌      |
|                    [PoolFormer](model_doc/poolformer)                    |       ✅        |         ❌         |      ❌      |
|                     [Pop2Piano](model_doc/pop2piano)                     |       ✅        |         ❌         |      ❌      |
|                    [ProphetNet](model_doc/prophetnet)                    |       ✅        |         ❌         |      ❌      |
|                           [PVT](model_doc/pvt)                           |       ✅        |         ❌         |      ❌      |
|                       [QDQBert](model_doc/qdqbert)                       |       ✅        |         ❌         |      ❌      |
|                           [RAG](model_doc/rag)                           |       ✅        |         ✅         |      ❌      |
|                         [REALM](model_doc/realm)                         |       ✅        |         ❌         |      ❌      |
|                      [Reformer](model_doc/reformer)                      |       ✅        |         ❌         |      ❌      |
|                        [RegNet](model_doc/regnet)                        |       ✅        |         ✅         |      ✅      |
|                       [RemBERT](model_doc/rembert)                       |       ✅        |         ✅         |      ❌      |
|                        [ResNet](model_doc/resnet)                        |       ✅        |         ✅         |      ✅      |
|                     [RetriBERT](model_doc/retribert)                     |       ✅        |         ❌         |      ❌      |
|                       [RoBERTa](model_doc/roberta)                       |       ✅        |         ✅         |      ✅      |
|          [RoBERTa-PreLayerNorm](model_doc/roberta-prelayernorm)          |       ✅        |         ✅         |      ✅      |
|                      [RoCBert](model_doc/roc_bert)                       |       ✅        |         ❌         |      ❌      |
|                      [RoFormer](model_doc/roformer)                      |       ✅        |         ✅         |      ✅      |
|                          [RWKV](model_doc/rwkv)                          |       ✅        |         ❌         |      ❌      |
|                           [SAM](model_doc/sam)                           |       ✅        |         ✅         |      ❌      |
|                  [SeamlessM4T](model_doc/seamless_m4t)                   |       ✅        |         ❌         |      ❌      |
|                     [SegFormer](model_doc/segformer)                     |       ✅        |         ✅         |      ❌      |
|                           [SEW](model_doc/sew)                           |       ✅        |         ❌         |      ❌      |
|                         [SEW-D](model_doc/sew-d)                         |       ✅        |         ❌         |      ❌      |
|        [Speech Encoder decoder](model_doc/speech-encoder-decoder)        |       ✅        |         ❌         |      ✅      |
|                 [Speech2Text](model_doc/speech_to_text)                  |       ✅        |         ✅         |      ❌      |
|                      [SpeechT5](model_doc/speecht5)                      |       ✅        |         ❌         |      ❌      |
|                      [Splinter](model_doc/splinter)                      |       ✅        |         ❌         |      ❌      |
|                   [SqueezeBERT](model_doc/squeezebert)                   |       ✅        |         ❌         |      ❌      |
|                   [SwiftFormer](model_doc/swiftformer)                   |       ✅        |         ❌         |      ❌      |
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
|                           [UL2](model_doc/ul2)                           |       ✅        |         ✅         |      ✅      |
|                          [UMT5](model_doc/umt5)                          |       ✅        |         ❌         |      ❌      |
|                     [UniSpeech](model_doc/unispeech)                     |       ✅        |         ❌         |      ❌      |
|                 [UniSpeechSat](model_doc/unispeech-sat)                  |       ✅        |         ❌         |      ❌      |
|                       [UPerNet](model_doc/upernet)                       |       ✅        |         ❌         |      ❌      |
|                           [VAN](model_doc/van)                           |       ✅        |         ❌         |      ❌      |
|                      [VideoMAE](model_doc/videomae)                      |       ✅        |         ❌         |      ❌      |
|                          [ViLT](model_doc/vilt)                          |       ✅        |         ❌         |      ❌      |
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

<!-- End table-->
=======
|             Model             | PyTorch support | TensorFlow support | Flax Support |
|:-----------------------------:|:---------------:|:------------------:|:------------:|
|            ALBERT             |       ✅        |         ✅         |      ✅      |
|             ALIGN             |       ✅        |         ❌         |      ❌      |
|            AltCLIP            |       ✅        |         ❌         |      ❌      |
| Audio Spectrogram Transformer |       ✅        |         ❌         |      ❌      |
|          Autoformer           |       ✅        |         ❌         |      ❌      |
|             Bark              |       ✅        |         ❌         |      ❌      |
|             BART              |       ✅        |         ✅         |      ✅      |
|             BEiT              |       ✅        |         ❌         |      ✅      |
|             BERT              |       ✅        |         ✅         |      ✅      |
|        Bert Generation        |       ✅        |         ❌         |      ❌      |
|            BigBird            |       ✅        |         ❌         |      ✅      |
|        BigBird-Pegasus        |       ✅        |         ❌         |      ❌      |
|            BioGpt             |       ✅        |         ❌         |      ❌      |
|              BiT              |       ✅        |         ❌         |      ❌      |
|          Blenderbot           |       ✅        |         ✅         |      ✅      |
|        BlenderbotSmall        |       ✅        |         ✅         |      ✅      |
|             BLIP              |       ✅        |         ✅         |      ❌      |
|            BLIP-2             |       ✅        |         ❌         |      ❌      |
|             BLOOM             |       ✅        |         ❌         |      ✅      |
|          BridgeTower          |       ✅        |         ❌         |      ❌      |
|             BROS              |       ✅        |         ❌         |      ❌      |
|           CamemBERT           |       ✅        |         ✅         |      ❌      |
|            CANINE             |       ✅        |         ❌         |      ❌      |
|         Chinese-CLIP          |       ✅        |         ❌         |      ❌      |
|             CLAP              |       ✅        |         ❌         |      ❌      |
|             CLIP              |       ✅        |         ✅         |      ✅      |
|            CLIPSeg            |       ✅        |         ❌         |      ❌      |
|             CLVP              |       ✅        |         ❌         |      ❌      |
|            CodeGen            |       ✅        |         ❌         |      ❌      |
|           CodeLlama           |       ✅        |         ❌         |      ❌      |
|       Conditional DETR        |       ✅        |         ❌         |      ❌      |
|           ConvBERT            |       ✅        |         ✅         |      ❌      |
|           ConvNeXT            |       ✅        |         ✅         |      ❌      |
|          ConvNeXTV2           |       ✅        |         ❌         |      ❌      |
|            CPM-Ant            |       ✅        |         ❌         |      ❌      |
|             CTRL              |       ✅        |         ✅         |      ❌      |
|              CvT              |       ✅        |         ✅         |      ❌      |
|         Data2VecAudio         |       ✅        |         ❌         |      ❌      |
|         Data2VecText          |       ✅        |         ❌         |      ❌      |
|        Data2VecVision         |       ✅        |         ✅         |      ❌      |
|            DeBERTa            |       ✅        |         ✅         |      ❌      |
|          DeBERTa-v2           |       ✅        |         ✅         |      ❌      |
|     Decision Transformer      |       ✅        |         ❌         |      ❌      |
|        Deformable DETR        |       ✅        |         ❌         |      ❌      |
|             DeiT              |       ✅        |         ✅         |      ❌      |
|             DETA              |       ✅        |         ❌         |      ❌      |
|             DETR              |       ✅        |         ❌         |      ❌      |
|             DiNAT             |       ✅        |         ❌         |      ❌      |
|            DINOv2             |       ✅        |         ❌         |      ❌      |
|          DistilBERT           |       ✅        |         ✅         |      ✅      |
|           DonutSwin           |       ✅        |         ❌         |      ❌      |
|              DPR              |       ✅        |         ✅         |      ❌      |
|              DPT              |       ✅        |         ❌         |      ❌      |
|        EfficientFormer        |       ✅        |         ✅         |      ❌      |
|         EfficientNet          |       ✅        |         ❌         |      ❌      |
|            ELECTRA            |       ✅        |         ✅         |      ✅      |
|            EnCodec            |       ✅        |         ❌         |      ❌      |
|        Encoder decoder        |       ✅        |         ✅         |      ✅      |
|             ERNIE             |       ✅        |         ❌         |      ❌      |
|            ErnieM             |       ✅        |         ❌         |      ❌      |
|              ESM              |       ✅        |         ✅         |      ❌      |
|  FairSeq Machine-Translation  |       ✅        |         ❌         |      ❌      |
|            Falcon             |       ✅        |         ❌         |      ❌      |
|           FlauBERT            |       ✅        |         ✅         |      ❌      |
|             FLAVA             |       ✅        |         ❌         |      ❌      |
|             FNet              |       ✅        |         ❌         |      ❌      |
|           FocalNet            |       ✅        |         ❌         |      ❌      |
|      Funnel Transformer       |       ✅        |         ✅         |      ❌      |
|              GIT              |       ✅        |         ❌         |      ❌      |
|             GLPN              |       ✅        |         ❌         |      ❌      |
|            GPT Neo            |       ✅        |         ❌         |      ✅      |
|           GPT NeoX            |       ✅        |         ❌         |      ❌      |
|       GPT NeoX Japanese       |       ✅        |         ❌         |      ❌      |
|             GPT-J             |       ✅        |         ✅         |      ✅      |
|            GPT-Sw3            |       ✅        |         ✅         |      ✅      |
|          GPTBigCode           |       ✅        |         ❌         |      ❌      |
|        GPTSAN-japanese        |       ✅        |         ❌         |      ❌      |
|          Graphormer           |       ✅        |         ❌         |      ❌      |
|           GroupViT            |       ✅        |         ✅         |      ❌      |
|            Hubert             |       ✅        |         ✅         |      ❌      |
|            I-BERT             |       ✅        |         ❌         |      ❌      |
|            IDEFICS            |       ✅        |         ❌         |      ❌      |
|           ImageGPT            |       ✅        |         ❌         |      ❌      |
|           Informer            |       ✅        |         ❌         |      ❌      |
|         InstructBLIP          |       ✅        |         ❌         |      ❌      |
|            Jukebox            |       ✅        |         ❌         |      ❌      |
|           LayoutLM            |       ✅        |         ✅         |      ❌      |
|          LayoutLMv2           |       ✅        |         ❌         |      ❌      |
|          LayoutLMv3           |       ✅        |         ✅         |      ❌      |
|              LED              |       ✅        |         ✅         |      ❌      |
|             LeViT             |       ✅        |         ❌         |      ❌      |
|             LiLT              |       ✅        |         ❌         |      ❌      |
|             LLaMA             |       ✅        |         ❌         |      ❌      |
|          Longformer           |       ✅        |         ✅         |      ❌      |
|            LongT5             |       ✅        |         ❌         |      ✅      |
|             LUKE              |       ✅        |         ❌         |      ❌      |
|            LXMERT             |       ✅        |         ✅         |      ❌      |
|            M-CTC-T            |       ✅        |         ❌         |      ❌      |
|            M2M100             |       ✅        |         ❌         |      ❌      |
|            Marian             |       ✅        |         ✅         |      ✅      |
|           MarkupLM            |       ✅        |         ❌         |      ❌      |
|          Mask2Former          |       ✅        |         ❌         |      ❌      |
|          MaskFormer           |       ✅        |         ❌         |      ❌      |
|             mBART             |       ✅        |         ✅         |      ✅      |
|             MEGA              |       ✅        |         ❌         |      ❌      |
|         Megatron-BERT         |       ✅        |         ❌         |      ❌      |
|            MGP-STR            |       ✅        |         ❌         |      ❌      |
|            Mistral            |       ✅        |         ❌         |      ❌      |
|          MobileBERT           |       ✅        |         ✅         |      ❌      |
|          MobileNetV1          |       ✅        |         ❌         |      ❌      |
|          MobileNetV2          |       ✅        |         ❌         |      ❌      |
|           MobileViT           |       ✅        |         ✅         |      ❌      |
|          MobileViTV2          |       ✅        |         ❌         |      ❌      |
|             MPNet             |       ✅        |         ✅         |      ❌      |
|              MPT              |       ✅        |         ❌         |      ❌      |
|              MRA              |       ✅        |         ❌         |      ❌      |
|              MT5              |       ✅        |         ✅         |      ✅      |
|           MusicGen            |       ✅        |         ❌         |      ❌      |
|              MVP              |       ✅        |         ❌         |      ❌      |
|              NAT              |       ✅        |         ❌         |      ❌      |
|             Nezha             |       ✅        |         ❌         |      ❌      |
|           NLLB-MOE            |       ✅        |         ❌         |      ❌      |
|            Nougat             |       ✅        |         ✅         |      ✅      |
|         Nyströmformer         |       ✅        |         ❌         |      ❌      |
|           OneFormer           |       ✅        |         ❌         |      ❌      |
|          OpenAI GPT           |       ✅        |         ✅         |      ❌      |
|         OpenAI GPT-2          |       ✅        |         ✅         |      ✅      |
|           OpenLlama           |       ✅        |         ❌         |      ❌      |
|              OPT              |       ✅        |         ✅         |      ✅      |
|            OWL-ViT            |       ✅        |         ❌         |      ❌      |
|            Pegasus            |       ✅        |         ✅         |      ✅      |
|           PEGASUS-X           |       ✅        |         ❌         |      ❌      |
|           Perceiver           |       ✅        |         ❌         |      ❌      |
|           Persimmon           |       ✅        |         ❌         |      ❌      |
|          Pix2Struct           |       ✅        |         ❌         |      ❌      |
|            PLBart             |       ✅        |         ❌         |      ❌      |
|          PoolFormer           |       ✅        |         ❌         |      ❌      |
|           Pop2Piano           |       ✅        |         ❌         |      ❌      |
|          ProphetNet           |       ✅        |         ❌         |      ❌      |
|              PVT              |       ✅        |         ❌         |      ❌      |
|            QDQBert            |       ✅        |         ❌         |      ❌      |
|              RAG              |       ✅        |         ✅         |      ❌      |
|             REALM             |       ✅        |         ❌         |      ❌      |
|           Reformer            |       ✅        |         ❌         |      ❌      |
|            RegNet             |       ✅        |         ✅         |      ✅      |
|            RemBERT            |       ✅        |         ✅         |      ❌      |
|            ResNet             |       ✅        |         ✅         |      ✅      |
|           RetriBERT           |       ✅        |         ❌         |      ❌      |
|            RoBERTa            |       ✅        |         ✅         |      ✅      |
|     RoBERTa-PreLayerNorm      |       ✅        |         ✅         |      ✅      |
|            RoCBert            |       ✅        |         ❌         |      ❌      |
|           RoFormer            |       ✅        |         ✅         |      ✅      |
|             RWKV              |       ✅        |         ❌         |      ❌      |
|              SAM              |       ✅        |         ✅         |      ❌      |
|           SegFormer           |       ✅        |         ✅         |      ❌      |
|              SEW              |       ✅        |         ❌         |      ❌      |
|             SEW-D             |       ✅        |         ❌         |      ❌      |
|    Speech Encoder decoder     |       ✅        |         ❌         |      ✅      |
|          Speech2Text          |       ✅        |         ✅         |      ❌      |
|         Speech2Text2          |       ❌        |         ❌         |      ❌      |
|           SpeechT5            |       ✅        |         ❌         |      ❌      |
|           Splinter            |       ✅        |         ❌         |      ❌      |
|          SqueezeBERT          |       ✅        |         ❌         |      ❌      |
|          SwiftFormer          |       ✅        |         ❌         |      ❌      |
|       Swin Transformer        |       ✅        |         ✅         |      ❌      |
|      Swin Transformer V2      |       ✅        |         ❌         |      ❌      |
|            Swin2SR            |       ✅        |         ❌         |      ❌      |
|      SwitchTransformers       |       ✅        |         ❌         |      ❌      |
|              T5               |       ✅        |         ✅         |      ✅      |
|       Table Transformer       |       ✅        |         ❌         |      ❌      |
|             TAPAS             |       ✅        |         ✅         |      ❌      |
|    Time Series Transformer    |       ✅        |         ❌         |      ❌      |
|          TimeSformer          |       ✅        |         ❌         |      ❌      |
|    Trajectory Transformer     |       ✅        |         ❌         |      ❌      |
|        Transformer-XL         |       ✅        |         ✅         |      ❌      |
|             TrOCR             |       ✅        |         ❌         |      ❌      |
|             TVLT              |       ✅        |         ❌         |      ❌      |
|             UMT5              |       ✅        |         ❌         |      ❌      |
|           UniSpeech           |       ✅        |         ❌         |      ❌      |
|         UniSpeechSat          |       ✅        |         ❌         |      ❌      |
|            UPerNet            |       ✅        |         ❌         |      ❌      |
|              VAN              |       ✅        |         ❌         |      ❌      |
|           VideoMAE            |       ✅        |         ❌         |      ❌      |
|             ViLT              |       ✅        |         ❌         |      ❌      |
|    Vision Encoder decoder     |       ✅        |         ✅         |      ✅      |
|     VisionTextDualEncoder     |       ✅        |         ✅         |      ✅      |
|          VisualBERT           |       ✅        |         ❌         |      ❌      |
|              ViT              |       ✅        |         ✅         |      ✅      |
|          ViT Hybrid           |       ✅        |         ❌         |      ❌      |
|            VitDet             |       ✅        |         ❌         |      ❌      |
|            ViTMAE             |       ✅        |         ✅         |      ❌      |
|           ViTMatte            |       ✅        |         ❌         |      ❌      |
|            ViTMSN             |       ✅        |         ❌         |      ❌      |
|             VITS              |       ✅        |         ❌         |      ❌      |
|             ViViT             |       ✅        |         ❌         |      ❌      |
|           Wav2Vec2            |       ✅        |         ✅         |      ✅      |
|      Wav2Vec2-Conformer       |       ✅        |         ❌         |      ❌      |
|             WavLM             |       ✅        |         ❌         |      ❌      |
|            Whisper            |       ✅        |         ✅         |      ✅      |
|            X-CLIP             |       ✅        |         ❌         |      ❌      |
|             X-MOD             |       ✅        |         ❌         |      ❌      |
|             XGLM              |       ✅        |         ✅         |      ✅      |
|              XLM              |       ✅        |         ✅         |      ❌      |
|        XLM-ProphetNet         |       ✅        |         ❌         |      ❌      |
|          XLM-RoBERTa          |       ✅        |         ✅         |      ✅      |
|        XLM-RoBERTa-XL         |       ✅        |         ❌         |      ❌      |
|             XLNet             |       ✅        |         ✅         |      ❌      |
|             YOLOS             |       ✅        |         ❌         |      ❌      |
|             YOSO              |       ✅        |         ❌         |      ❌      |


<!-- End table-->
>>>>>>> 2b02f70ed (very minor fix of index.md 2)
