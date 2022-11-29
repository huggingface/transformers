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

|             Model             | Tokenizer slow | Tokenizer fast | PyTorch support | TensorFlow support | Flax Support |
|:-----------------------------:|:--------------:|:--------------:|:---------------:|:------------------:|:------------:|
|            ALBERT             |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|             ALIGN             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            AltCLIP            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
| Audio Spectrogram Transformer |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          Autoformer           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             BART              |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|             BEiT              |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|             BERT              |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|        Bert Generation        |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|            BigBird            |       ✅       |       ✅       |       ✅        |         ❌         |      ✅      |
|        BigBird-Pegasus        |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            BioGpt             |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|              BiT              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          Blenderbot           |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|        BlenderbotSmall        |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|             BLIP              |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|            BLIP-2             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             BLOOM             |       ❌       |       ✅       |       ✅        |         ❌         |      ❌      |
|          BridgeTower          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           CamemBERT           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            CANINE             |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|         Chinese-CLIP          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             CLAP              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             CLIP              |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|            CLIPSeg            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            CodeGen            |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|       Conditional DETR        |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           ConvBERT            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|           ConvNeXT            |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|          ConvNeXTV2           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            CPM-Ant            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|             CTRL              |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|              CvT              |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|         Data2VecAudio         |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         Data2VecText          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        Data2VecVision         |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|            DeBERTa            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|          DeBERTa-v2           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|     Decision Transformer      |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        Deformable DETR        |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             DeiT              |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|             DETA              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             DETR              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             DiNAT             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          DistilBERT           |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|           DonutSwin           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|              DPR              |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|              DPT              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        EfficientFormer        |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|         EfficientNet          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            ELECTRA            |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|            EnCodec            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        Encoder decoder        |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|             ERNIE             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            ErnieM             |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|              ESM              |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|  FairSeq Machine-Translation  |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|            Falcon             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           FlauBERT            |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|             FLAVA             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             FNet              |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|           FocalNet            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|      Funnel Transformer       |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|              GIT              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             GLPN              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            GPT Neo            |       ❌       |       ❌       |       ✅        |         ❌         |      ✅      |
|           GPT NeoX            |       ❌       |       ✅       |       ✅        |         ❌         |      ❌      |
|       GPT NeoX Japanese       |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|             GPT-J             |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|            GPT-Sw3            |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|          GPTBigCode           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        GPTSAN-japanese        |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|          Graphormer           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           GroupViT            |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|            Hubert             |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|            I-BERT             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           ImageGPT            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           Informer            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         InstructBLIP          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            Jukebox            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|           LayoutLM            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|          LayoutLMv2           |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|          LayoutLMv3           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|              LED              |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|             LeViT             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             LiLT              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             LLaMA             |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|          Longformer           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            LongT5             |       ❌       |       ❌       |       ✅        |         ❌         |      ✅      |
|             LUKE              |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|            LXMERT             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            M-CTC-T            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            M2M100             |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|            Marian             |       ✅       |       ❌       |       ✅        |         ✅         |      ✅      |
|           MarkupLM            |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|          Mask2Former          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          MaskFormer           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        MaskFormerSwin         |       ❌       |       ❌       |       ❌        |         ❌         |      ❌      |
|             mBART             |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|             MEGA              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         Megatron-BERT         |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            MGP-STR            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|          MobileBERT           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|          MobileNetV1          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          MobileNetV2          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           MobileViT           |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|          MobileViTV2          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             MPNet             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|              MRA              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|              MT5              |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|           MusicGen            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|              MVP              |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|              NAT              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             Nezha             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           NLLB-MOE            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         Nyströmformer         |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           OneFormer           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          OpenAI GPT           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|         OpenAI GPT-2          |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|           OpenLlama           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|              OPT              |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|            OWL-ViT            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            Pegasus            |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|           PEGASUS-X           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           Perceiver           |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|          Pix2Struct           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            PLBart             |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|          PoolFormer           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          ProphetNet           |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|            QDQBert            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|              RAG              |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|             REALM             |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|           Reformer            |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|            RegNet             |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|            RemBERT            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            ResNet             |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|           RetriBERT           |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|            RoBERTa            |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|     RoBERTa-PreLayerNorm      |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|            RoCBert            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|           RoFormer            |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|             RWKV              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|              SAM              |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|           SegFormer           |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|              SEW              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             SEW-D             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|    Speech Encoder decoder     |       ❌       |       ❌       |       ✅        |         ❌         |      ✅      |
|          Speech2Text          |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|         Speech2Text2          |       ✅       |       ❌       |       ❌        |         ❌         |      ❌      |
|           SpeechT5            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|           Splinter            |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|          SqueezeBERT          |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|          SwiftFormer          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|       Swin Transformer        |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|      Swin Transformer V2      |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            Swin2SR            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|      SwitchTransformers       |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|              T5               |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|       Table Transformer       |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             TAPAS             |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|    Time Series Transformer    |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          TimeSformer          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         TimmBackbone          |       ❌       |       ❌       |       ❌        |         ❌         |      ❌      |
|    Trajectory Transformer     |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        Transformer-XL         |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|             TrOCR             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             TVLT              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             UMT5              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           UniSpeech           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         UniSpeechSat          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            UPerNet            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|              VAN              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           VideoMAE            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             ViLT              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|    Vision Encoder decoder     |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|     VisionTextDualEncoder     |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|          VisualBERT           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|              ViT              |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|          ViT Hybrid           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            ViTMAE             |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|            ViTMSN             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             ViViT             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           Wav2Vec2            |       ✅       |       ❌       |       ✅        |         ✅         |      ✅      |
|      Wav2Vec2-Conformer       |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             WavLM             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            Whisper            |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|            X-CLIP             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             X-MOD             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             XGLM              |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|              XLM              |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|        XLM-ProphetNet         |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|          XLM-RoBERTa          |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|        XLM-RoBERTa-XL         |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             XLNet             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|             YOLOS             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             YOSO              |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |

<!-- End table-->
