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

Machine Learning de última generación para PyTorch, TensorFlow y JAX.

🤗 Transformers proporciona APIs para descargar y entrenar fácilmente modelos preentrenados de última generación. El uso de modelos  preentrenados puede reducir tus costos de cómputo, tu huella de carbono y ahorrarte tiempo al entrenar un modelo desde cero. Los modelos se pueden utilizar en diferentes modalidades, tales como:

* 📝 Texto: clasificación de texto, extracción de información, respuesta a preguntas, resumir, traducción y generación de texto en más de 100 idiomas.
* 🖼️ Imágenes: clasificación de imágenes, detección de objetos y segmentación.
* 🗣️ Audio: reconocimiento de voz y clasificación de audio.
* 🐙 Multimodal: respuesta a preguntas en tablas, reconocimiento óptico de caracteres, extracción de información de documentos escaneados, clasificación de videos y respuesta visual a preguntas.

Nuestra biblioteca admite una integración perfecta entre tres de las bibliotecas de deep learning más populares: [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/) y [JAX](https://jax.readthedocs.io/en/latest/). Entrena tu modelo con tres líneas de código en un framework y cárgalo para inferencia con otro.
Cada arquitectura de 🤗 Transformers se define en un módulo de Python independiente para que se puedan personalizar fácilmente para investigación y experimentos.

## Si estás buscando soporte personalizado del equipo de Hugging Face

<a target="_blank" href="https://huggingface.co/support">
<img alt="HuggingFace Expert Acceleration Program" src="https://huggingface.co/front/thumbnails/support.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a>

## Contenidos

La documentación está organizada en cuatro partes:

- **EMPEZAR** contiene un recorrido rápido e instrucciones de instalación para comenzar a usar 🤗 Transformers.
- **TUTORIALES** es un excelente lugar para comenzar. Esta sección te ayudará a obtener las habilidades básicas que necesitas para comenzar a usar 🤗 Transformers.
- **GUÍAS PRÁCTICAS** te mostrará cómo lograr un objetivo específico, cómo hacer fine-tuning a un modelo preentrenado para el modelado de lenguaje o cómo crear un cabezal para un modelo personalizado.
- **GUÍAS CONCEPTUALES** proporciona más discusión y explicación de los conceptos e ideas subyacentes detrás de los modelos, las tareas y la filosofía de diseño de 🤗 Transformers. 

La biblioteca actualmente contiene implementaciones de JAX, PyTorch y TensorFlow, pesos de modelos preentrenados, scripts de uso y utilidades de conversión para los siguientes modelos.

### Modelos compatibles

<!--This list is updated automatically from the README with _make fix-copies_. Do not update manually! -->

1. **[ALBERT](model_doc/albert)** (de Google Research y el Instituto Tecnológico de Toyota en Chicago) publicado con el paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://huggingface.co/papers/1909.11942), por Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
1. **[ALIGN](model_doc/align)** (de Google Research) publicado con el paper [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://huggingface.co/papers/2102.05918) por Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.
1. **[BART](model_doc/bart)** (de Facebook) publicado con el paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://huggingface.co/papers/1910.13461) por Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov y Luke Zettlemoyer.
1. **[BARThez](model_doc/barthez)** (de École polytechnique) publicado con el paper [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://huggingface.co/papers/2010.12321) por Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis.
1. **[BARTpho](model_doc/bartpho)** (de VinAI Research) publicado con el paper [BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese](https://huggingface.co/papers/2109.09701) por Nguyen Luong Tran, Duong Minh Le y Dat Quoc Nguyen.
1. **[BEiT](model_doc/beit)** (de Microsoft) publicado con el paper [BEiT: BERT Pre-Training of Image Transformers](https://huggingface.co/papers/2106.08254) por Hangbo Bao, Li Dong, Furu Wei.
1. **[BERT](model_doc/bert)** (de Google) publicado con el paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://huggingface.co/papers/1810.04805) por Jacob Devlin, Ming-Wei Chang, Kenton Lee y Kristina Toutanova.
1. **[BERTweet](model_doc/bertweet)** (de VinAI Research) publicado con el paper [BERTweet: A pre-trained language model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2/) por Dat Quoc Nguyen, Thanh Vu y Anh Tuan Nguyen.
1. **[BERT For Sequence Generation](model_doc/bert-generation)** (de Google) publicado con el paper [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://huggingface.co/papers/1907.12461) por Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
1. **[BigBird-RoBERTa](model_doc/big_bird)** (de Google Research) publicado con el paper [Big Bird: Transformers for Longer Sequences](https://huggingface.co/papers/2007.14062) por Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
1. **[BigBird-Pegasus](model_doc/bigbird_pegasus)** (de Google Research) publicado con el paper [Big Bird: Transformers for Longer Sequences](https://huggingface.co/papers/2007.14062) por Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
1. **[Blenderbot](model_doc/blenderbot)** (de Facebook) publicado con el paper [Recipes for building an open-domain chatbot](https://huggingface.co/papers/2004.13637) por Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
1. **[BlenderbotSmall](model_doc/blenderbot-small)** (de Facebook) publicado con el paper [Recipes for building an open-domain chatbot](https://huggingface.co/papers/2004.13637) por Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
1. **[BORT](model_doc/bort)** (de Alexa) publicado con el paper [Optimal Subarchitecture Extraction For BERT](https://huggingface.co/papers/2010.10499) por Adrian de Wynter y Daniel J. Perry.
1. **[ByT5](model_doc/byt5)** (de Google Research) publicado con el paper [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://huggingface.co/papers/2105.13626) por Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel.
1. **[CamemBERT](model_doc/camembert)** (de Inria/Facebook/Sorbonne) publicado con el paper [CamemBERT: a Tasty French Language Model](https://huggingface.co/papers/1911.03894) por Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah y Benoît Sagot.
1. **[CANINE](model_doc/canine)** (de Google Research) publicado con el paper [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://huggingface.co/papers/2103.06874) por Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting.
1. **[ConvNeXT](model_doc/convnext)** (de Facebook AI) publicado con el paper [A ConvNet for the 2020s](https://huggingface.co/papers/2201.03545) por Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
1. **[ConvNeXTV2](model_doc/convnextv2)** (de Facebook AI) publicado con el paper [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://huggingface.co/papers/2301.00808) por Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
1. **[CLIP](model_doc/clip)** (de OpenAI) publicado con el paper [Learning Transferable Visual Models From Natural Language Supervision](https://huggingface.co/papers/2103.00020) por Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
1. **[ConvBERT](model_doc/convbert)** (de YituTech) publicado con el paper [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://huggingface.co/papers/2008.02496) por Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan.
1. **[CPM](model_doc/cpm)** (de Universidad de Tsinghua) publicado con el paper [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://huggingface.co/papers/2012.00413) por Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
1. **[CTRL](model_doc/ctrl)** (de Salesforce) publicado con el paper [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://huggingface.co/papers/1909.05858) por Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong y Richard Socher.
1. **[Data2Vec](model_doc/data2vec)** (de Facebook) publicado con el paper [Data2Vec:  A General Framework for Self-supervised Learning in Speech, Vision and Language](https://huggingface.co/papers/2202.03555) por Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
1. **[DeBERTa](model_doc/deberta)** (de Microsoft) publicado con el paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://huggingface.co/papers/2006.03654) por Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
1. **[DeBERTa-v2](model_doc/deberta-v2)** (de Microsoft) publicado con el paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://huggingface.co/papers/2006.03654) por Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
1. **[Decision Transformer](model_doc/decision_transformer)** (de Berkeley/Facebook/Google) publicado con el paper [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://huggingface.co/papers/2106.01345) por Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.
1. **[DiT](model_doc/dit)** (de Microsoft Research) publicado con el paper [DiT: Self-supervised Pre-training for Document Image Transformer](https://huggingface.co/papers/2203.02378) por Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei.
1. **[DeiT](model_doc/deit)** (de Facebook) publicado con el paper [Training data-efficient image transformers & distillation through attention](https://huggingface.co/papers/2012.12877) por Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
1. **[DETR](model_doc/detr)** (de Facebook) publicado con el paper [End-to-End Object Detection with Transformers](https://huggingface.co/papers/2005.12872) por Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.
1. **[DialoGPT](model_doc/dialogpt)** (de Microsoft Research) publicado con el paper [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://huggingface.co/papers/1911.00536) por Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
1. **[DistilBERT](model_doc/distilbert)** (de HuggingFace), publicado junto con el paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://huggingface.co/papers/1910.01108) por Victor Sanh, Lysandre Debut y Thomas Wolf. Se ha aplicado el mismo método para comprimir GPT2 en [DistilGPT2](https://github.com/huggingface/transformers-research-projects/tree/main/distillation), RoBERTa en [DistilRoBERTa](https://github.com/huggingface/transformers-research-projects/tree/main/distillation), BERT multilingüe en [DistilmBERT](https://github.com/huggingface/transformers-research-projects/tree/main/distillation) y una versión alemana de DistilBERT.
1. **[DPR](model_doc/dpr)** (de Facebook) publicado con el paper [Dense Passage Retrieval for Open-Domain Question Answering](https://huggingface.co/papers/2004.04906) por Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, y Wen-tau Yih.
1. **[DPT](master/model_doc/dpt)** (de Intel Labs) publicado con el paper [Vision Transformers for Dense Prediction](https://huggingface.co/papers/2103.13413) por René Ranftl, Alexey Bochkovskiy, Vladlen Koltun.
1. **[EfficientNet](model_doc/efficientnet)** (from Google Research) released with the paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://huggingface.co/papers/1905.11946)  by Mingxing Tan and Quoc V. Le.
1. **[EncoderDecoder](model_doc/encoder-decoder)** (de Google Research) publicado con el paper [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://huggingface.co/papers/1907.12461) por Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
1. **[ELECTRA](model_doc/electra)** (de Google Research/Universidad de Stanford) publicado con el paper [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://huggingface.co/papers/2003.10555) por Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
1. **[FlauBERT](model_doc/flaubert)** (de CNRS) publicado con el paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://huggingface.co/papers/1912.05372) por Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
1. **[FNet](model_doc/fnet)** (de Google Research) publicado con el paper [FNet: Mixing Tokens with Fourier Transforms](https://huggingface.co/papers/2105.03824) por James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon.
1. **[Funnel Transformer](model_doc/funnel)** (de CMU/Google Brain) publicado con el paper [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://huggingface.co/papers/2006.03236) por Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
1. **[GLPN](model_doc/glpn)** (de KAIST) publicado con el paper [Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth](https://huggingface.co/papers/2201.07436) por Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, Junmo Kim.
1. **[GPT](model_doc/openai-gpt)** (de OpenAI) publicado con el paper [Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised/) por Alec Radford, Karthik Narasimhan, Tim Salimans y Ilya Sutskever.
1. **[GPT-2](model_doc/gpt2)** (de OpenAI) publicado con el paper [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models/) por Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei y Ilya Sutskever.
1. **[GPT-J](model_doc/gptj)** (de EleutherAI) publicado con el repositorio [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/) por Ben Wang y Aran Komatsuzaki.
1. **[GPT Neo](model_doc/gpt_neo)** (de EleutherAI) publicado en el paper [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) por Sid Black, Stella Biderman, Leo Gao, Phil Wang y Connor Leahy.
1. **[GPTSAN-japanese](model_doc/gptsan-japanese)** released with [GPTSAN](https://github.com/tanreinama/GPTSAN) by Toshiyuki Sakamoto (tanreinama).
1. **[Hubert](model_doc/hubert)** (de Facebook) publicado con el paper [HuBERT: Self-Supervised Speech Representation Learning por Masked Prediction of Hidden Units](https://huggingface.co/papers/2106.07447) por Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.
1. **[I-BERT](model_doc/ibert)** (de Berkeley) publicado con el paper [I-BERT: Integer-only BERT Quantization](https://huggingface.co/papers/2101.01321) por Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer.
1. **[ImageGPT](model_doc/imagegpt)** (de OpenAI) publicado con el paper [Generative Pretraining from Pixels](https://openai.com/blog/image-gpt/) por Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever.
1. **[LayoutLM](model_doc/layoutlm)** (de Microsoft Research Asia) publicado con el paper [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://huggingface.co/papers/1912.13318) por Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
1. **[LayoutLMv2](model_doc/layoutlmv2)** (de Microsoft Research Asia) publicado con el paper [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://huggingface.co/papers/2012.14740) por Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou.
1. **[LayoutXLM](model_doc/layoutxlm)** (de Microsoft Research Asia) publicado con el paper [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://huggingface.co/papers/2104.08836) por Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei.
1. **[LED](model_doc/led)** (de AllenAI) publicado con el paper [Longformer: The Long-Document Transformer](https://huggingface.co/papers/2004.05150) por Iz Beltagy, Matthew E. Peters, Arman Cohan.
1. **[Longformer](model_doc/longformer)** (de AllenAI) publicado con el paper [Longformer: The Long-Document Transformer](https://huggingface.co/papers/2004.05150) por Iz Beltagy, Matthew E. Peters, Arman Cohan.
1. **[LUKE](model_doc/luke)** (de Studio Ousia) publicado con el paper [LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://huggingface.co/papers/2010.01057) por Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yuji Matsumoto.
1. **[mLUKE](model_doc/mluke)** (de Studio Ousia) publicado con el paper [mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models](https://huggingface.co/papers/2110.08151) por Ryokan Ri, Ikuya Yamada, y Yoshimasa Tsuruoka.
1. **[LXMERT](model_doc/lxmert)** (de UNC Chapel Hill) publicado con el paper [LXMERT: Learning Cross-Modality Encoder Representations from Transformers for Open-Domain Question Answering](https://huggingface.co/papers/1908.07490) por Hao Tan y Mohit Bansal.
1. **[M2M100](model_doc/m2m_100)** (de Facebook) publicado con el paper [Beyond English-Centric Multilingual Machine Translation](https://huggingface.co/papers/2010.11125) por Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, Armand Joulin.
1. **[MarianMT](model_doc/marian)** Modelos de traducción automática entrenados usando [OPUS](http://opus.nlpl.eu/) data por Jörg Tiedemann. El [Marian Framework](https://marian-nmt.github.io/) está siendo desarrollado por el equipo de traductores de Microsoft.
1. **[Mask2Former](model_doc/mask2former)** (de FAIR y UIUC) publicado con el paper [Masked-attention Mask Transformer for Universal Image Segmentation](https://huggingface.co/papers/2112.01527) por Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar.
1. **[MaskFormer](model_doc/maskformer)** (de Meta y UIUC) publicado con el paper [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://huggingface.co/papers/2107.06278) por Bowen Cheng, Alexander G. Schwing, Alexander Kirillov.
1. **[MBart](model_doc/mbart)** (de Facebook) publicado con el paper [Multilingual Denoising Pre-training for Neural Machine Translation](https://huggingface.co/papers/2001.08210) por Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.
1. **[MBart-50](model_doc/mbart)** (de Facebook) publicado con el paper [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://huggingface.co/papers/2008.00401) por Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, Angela Fan.
1. **[Megatron-BERT](model_doc/megatron-bert)** (de NVIDIA) publicado con el paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://huggingface.co/papers/1909.08053) por Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper y Bryan Catanzaro.
1. **[Megatron-GPT2](model_doc/megatron_gpt2)** (de NVIDIA) publicado con el paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://huggingface.co/papers/1909.08053) por Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper y Bryan Catanzaro.
1. **[MPNet](model_doc/mpnet)** (de Microsoft Research) publicado con el paper [MPNet: Masked and Permuted Pre-training for Language Understanding](https://huggingface.co/papers/2004.09297) por Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.
1. **[MT5](model_doc/mt5)** (de Google AI) publicado con el paper [mT5: A massively multilingual pre-trained text-to-text transformer](https://huggingface.co/papers/2010.11934) por Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.
1. **[Nyströmformer](model_doc/nystromformer)** (de la Universidad de Wisconsin - Madison) publicado con el paper [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://huggingface.co/papers/2102.03902) por Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, Vikas Singh.
1. **[OneFormer](model_doc/oneformer)** (de la SHI Labs) publicado con el paper [OneFormer: One Transformer to Rule Universal Image Segmentation](https://huggingface.co/papers/2211.06220) por Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi.
1. **[Pegasus](model_doc/pegasus)** (de Google) publicado con el paper [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://huggingface.co/papers/1912.08777) por Jingqing Zhang, Yao Zhao, Mohammad Saleh y Peter J. Liu.
1. **[Perceiver IO](model_doc/perceiver)** (de Deepmind) publicado con el paper [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://huggingface.co/papers/2107.14795) por Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira.
1. **[PhoBERT](model_doc/phobert)** (de VinAI Research) publicado con el paper [PhoBERT: Pre-trained language models for Vietnamese](https://www.aclweb.org/anthology/2020.findings-emnlp.92/) por Dat Quoc Nguyen y Anh Tuan Nguyen.
1. **[PLBart](model_doc/plbart)** (de UCLA NLP) publicado con el paper [Unified Pre-training for Program Understanding and Generation](https://huggingface.co/papers/2103.06333) por Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang.
1. **[PoolFormer](model_doc/poolformer)** (de Sea AI Labs) publicado con el paper [MetaFormer is Actually What You Need for Vision](https://huggingface.co/papers/2111.11418) por Yu, Weihao y Luo, Mi y Zhou, Pan y Si, Chenyang y Zhou, Yichen y Wang, Xinchao y Feng, Jiashi y Yan, Shuicheng.
1. **[ProphetNet](model_doc/prophetnet)** (de Microsoft Research) publicado con el paper [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://huggingface.co/papers/2001.04063) por Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang y Ming Zhou.
1. **[QDQBert](model_doc/qdqbert)** (de NVIDIA) publicado con el paper [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://huggingface.co/papers/2004.09602) por Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev y Paulius Micikevicius.
1. **[REALM](model_doc/realm.html)** (de Google Research) publicado con el paper [REALM: Retrieval-Augmented Language Model Pre-Training](https://huggingface.co/papers/2002.08909) por Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat y Ming-Wei Chang.
1. **[Reformer](model_doc/reformer)** (de Google Research) publicado con el paper [Reformer: The Efficient Transformer](https://huggingface.co/papers/2001.04451) por Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
1. **[RemBERT](model_doc/rembert)** (de Google Research) publicado con el paper [Rethinking embedding coupling in pre-trained language models](https://huggingface.co/papers/2010.12821) por Hyung Won Chung, Thibault Févry, Henry Tsai, M. Johnson, Sebastian Ruder.
1. **[RegNet](model_doc/regnet)** (de META Platforms) publicado con el paper [Designing Network Design Space](https://huggingface.co/papers/2003.13678) por Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár.
1. **[ResNet](model_doc/resnet)** (de Microsoft Research) publicado con el paper [Deep Residual Learning for Image Recognition](https://huggingface.co/papers/1512.03385) por Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
1. **[RoBERTa](model_doc/roberta)** (de Facebook), publicado junto con el paper [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://huggingface.co/papers/1907.11692) por Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
1. **[RoFormer](model_doc/roformer)** (de ZhuiyiTechnology), publicado junto con el paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://huggingface.co/papers/2104.09864) por Jianlin Su y Yu Lu y Shengfeng Pan y Bo Wen y Yunfeng Liu.
1. **[SegFormer](model_doc/segformer)** (de NVIDIA) publicado con el paper [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://huggingface.co/papers/2105.15203) por Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo.
1. **[SEW](model_doc/sew)** (de ASAPP) publicado con el paper [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://huggingface.co/papers/2109.06870) por Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.
1. **[SEW-D](model_doc/sew_d)** (de ASAPP) publicado con el paper [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://huggingface.co/papers/2109.06870) por Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.
1. **[SpeechToTextTransformer](model_doc/speech_to_text)** (de Facebook), publicado junto con el paper [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://huggingface.co/papers/2010.05171) por Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino.
1. **[SpeechToTextTransformer2](model_doc/speech_to_text_2)** (de Facebook), publicado junto con el paper [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://huggingface.co/papers/2104.06678) por Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau.
1. **[Splinter](model_doc/splinter)** (de Universidad de Tel Aviv), publicado junto con el paper [Few-Shot Question Answering by Pretraining Span Selection](https://huggingface.co/papers/2101.00438) pory Ori Ram, Yuval Kirstain, Jonathan Berant, Amir Globerson, Omer Levy.
1. **[SqueezeBert](model_doc/squeezebert)** (de Berkeley) publicado con el paper [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://huggingface.co/papers/2006.11316) por Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, y Kurt W. Keutzer.
1. **[Swin Transformer](model_doc/swin)** (de Microsoft) publicado con el paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://huggingface.co/papers/2103.14030) por Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
1. **[T5](model_doc/t5)** (de Google AI) publicado con el paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://huggingface.co/papers/1910.10683) por Colin Raffel y Noam Shazeer y Adam Roberts y Katherine Lee y Sharan Narang y Michael Matena y Yanqi Zhou y Wei Li y Peter J. Liu.
1. **[T5v1.1](model_doc/t5v1.1)** (de Google AI) publicado en el repositorio [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511) por Colin Raffel y Noam Shazeer y Adam Roberts y Katherine Lee y Sharan Narang y Michael Matena y Yanqi Zhou y Wei Li y Peter J. Liu.
1. **[TAPAS](model_doc/tapas)** (de Google AI) publicado con el paper [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://huggingface.co/papers/2004.02349) por Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno y Julian Martin Eisenschlos.
1. **[TAPEX](model_doc/tapex)** (de Microsoft Research) publicado con el paper [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://huggingface.co/papers/2107.07653) por Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, Jian-Guang Lou.
1. **[Transformer-XL](model_doc/transfo-xl)** (de Google/CMU) publicado con el paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://huggingface.co/papers/1901.02860) por Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
1. **[TrOCR](model_doc/trocr)** (de Microsoft), publicado junto con el paper [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://huggingface.co/papers/2109.10282) por Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei.
1. **[UniSpeech](model_doc/unispeech)** (de Microsoft Research) publicado con el paper [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://huggingface.co/papers/2101.07597) por Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang.
1. **[UniSpeechSat](model_doc/unispeech-sat)** (de Microsoft Research) publicado con el paper [UNISPEECH-SAT: UNIVERSAL SPEECH REPRESENTATION LEARNING WITH SPEAKER AWARE PRE-TRAINING](https://huggingface.co/papers/2110.05752) por Sanyuan Chen, Yu Wu, Chengyi Wang, Zhengyang Chen, Zhuo Chen, Shujie Liu, Jian Wu, Yao Qian, Furu Wei, Jinyu Li, Xiangzhan Yu.
1. **[VAN](model_doc/van)** (de la Universidad de Tsinghua y la Universidad de Nankai) publicado con el paper [Visual Attention Network](https://huggingface.co/papers/2202.09741) por Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu.
1. **[ViLT](model_doc/vilt)** (de NAVER AI Lab/Kakao Enterprise/Kakao Brain) publicado con el paper [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://huggingface.co/papers/2102.03334) por Wonjae Kim, Bokyung Son, Ildoo Kim.
1. **[Vision Transformer (ViT)](model_doc/vit)** (de Google AI) publicado con el paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://huggingface.co/papers/2010.11929) por Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
1. **[ViTMAE](model_doc/vit_mae)** (de Meta AI) publicado con el paper [Masked Autoencoders Are Scalable Vision Learners](https://huggingface.co/papers/2111.06377) por Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.
1. **[VisualBERT](model_doc/visual_bert)** (de UCLA NLP) publicado con el paper [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://huggingface.co/papers/1908.03557) por Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang.
1. **[WavLM](model_doc/wavlm)** (de Microsoft Research) publicado con el paper [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://huggingface.co/papers/2110.13900) por Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Furu Wei.
1. **[Wav2Vec2](model_doc/wav2vec2)** (de Facebook AI) publicado con el paper [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://huggingface.co/papers/2006.11477) por Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.
1. **[Wav2Vec2Phoneme](model_doc/wav2vec2_phoneme)** (de Facebook AI) publicado con el paper [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition](https://huggingface.co/papers/2109.11680) por Qiantong Xu, Alexei Baevski, Michael Auli.
1. **[XGLM](model_doc/xglm)** (de Facebook AI) publicado con el paper [Few-shot Learning with Multilingual Language Models](https://huggingface.co/papers/2112.10668) por Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, Xian Li.
1. **[XLM](model_doc/xlm)** (de Facebook) publicado junto con el paper [Cross-lingual Language Model Pretraining](https://huggingface.co/papers/1901.07291) por Guillaume Lample y Alexis Conneau.
1. **[XLM-ProphetNet](model_doc/xlm-prophetnet)** (de Microsoft Research) publicado con el paper [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://huggingface.co/papers/2001.04063) por Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang y Ming Zhou.
1. **[XLM-RoBERTa](model_doc/xlm-roberta)** (de Facebook AI), publicado junto con el paper [Unsupervised Cross-lingual Representation Learning at Scale](https://huggingface.co/papers/1911.02116) por Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer y Veselin Stoyanov.
1. **[XLM-RoBERTa-XL](model_doc/xlm-roberta-xl)** (de Facebook AI), publicado junto con el paper [Larger-Scale Transformers for Multilingual Masked Language Modeling](https://huggingface.co/papers/2105.00572) por Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, Alexis Conneau.
1. **[XLNet](model_doc/xlnet)** (de Google/CMU) publicado con el paper [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://huggingface.co/papers/1906.08237) por Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
1. **[XLSR-Wav2Vec2](model_doc/xlsr_wav2vec2)** (de Facebook AI) publicado con el paper [Unsupervised Cross-Lingual Representation Learning For Speech Recognition](https://huggingface.co/papers/2006.13979) por Alexis Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, Michael Auli.
1. **[XLS-R](model_doc/xls_r)** (de Facebook AI) publicado con el paper [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://huggingface.co/papers/2111.09296) por Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, Michael Auli.
1. **[YOSO](model_doc/yoso)** (de la Universidad de Wisconsin-Madison) publicado con el paper [You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling](https://huggingface.co/papers/2111.09714) por Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh.


### Frameworks compatibles

La siguiente tabla representa el soporte actual en la biblioteca para cada uno de esos modelos, ya sea que tengan un tokenizador de Python (llamado "slow"). Un tokenizador "fast" respaldado por la biblioteca 🤗 Tokenizers, ya sea que tengan soporte en Jax (a través de
Flax), PyTorch y/o TensorFlow.

<!--This table is updated automatically from the auto modules with _make fix-copies_. Do not update manually!-->

|            Modelo           | Tokenizer slow | Tokenizer fast | PyTorch support | TensorFlow support | Flax Support |
|:---------------------------:|:--------------:|:--------------:|:---------------:|:------------------:|:------------:|
|           ALBERT            |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|            BART             |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|            BEiT             |       ❌       |       ❌       |       ✅        |         ❌         |      ✅      |
|            BERT             |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|       Bert Generation       |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|           BigBird           |       ✅       |       ✅       |       ✅        |         ❌         |      ✅      |
|       BigBirdPegasus        |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         Blenderbot          |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|       BlenderbotSmall       |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|          CamemBERT          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|           Canine            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|            CLIP             |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|          ConvBERT           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|          ConvNext           |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|            CTRL             |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|        Data2VecAudio        |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        Data2VecText         |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           DeBERTa           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|         DeBERTa-v2          |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|    Decision Transformer     |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            DeiT             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            DETR             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         DistilBERT          |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|             DPR             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|             DPT             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           ELECTRA           |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|       Encoder decoder       |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
| FairSeq Machine-Translation |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|          FlauBERT           |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|            FNet             |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|     Funnel Transformer      |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            GLPN             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           GPT Neo           |       ❌       |       ❌       |       ✅        |         ❌         |      ✅      |
|            GPT-J            |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|           Hubert            |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|           I-BERT            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          ImageGPT           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          LayoutLM           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|         LayoutLMv2          |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|             LED             |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|         Longformer          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            LUKE             |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|           LXMERT            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|           M2M100            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|           Marian            |       ✅       |       ❌       |       ✅        |         ✅         |      ✅      |
|         MaskFormer          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            mBART            |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|        MegatronBert         |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         MobileBERT          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            MPNet            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|             mT5             |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|        Nystromformer        |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         OpenAI GPT          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|        OpenAI GPT-2         |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|           Pegasus           |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|          Perceiver          |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|           PLBart            |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|         PoolFormer          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|         ProphetNet          |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|           QDQBert           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             RAG             |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|            Realm            |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|          Reformer           |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|           RegNet            |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|           RemBERT           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|           ResNet            |       ❌       |       ❌       |       ✅        |         ❌         |      ✅      |
|          RetriBERT          |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|           RoBERTa           |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|          RoFormer           |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|          SegFormer          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             SEW             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            SEW-D            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|   Speech Encoder decoder    |       ❌       |       ❌       |       ✅        |         ❌         |      ✅      |
|         Speech2Text         |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|        Speech2Text2         |       ✅       |       ❌       |       ❌        |         ❌         |      ❌      |
|          Splinter           |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|         SqueezeBERT         |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|            Swin             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             T5              |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|            TAPAS            |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|            TAPEX            |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|       Transformer-XL        |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|            TrOCR            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          UniSpeech          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        UniSpeechSat         |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             VAN             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            ViLT             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|   Vision Encoder decoder    |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|    VisionTextDualEncoder    |       ❌       |       ❌       |       ✅        |         ❌         |      ✅      |
|         VisualBert          |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|             ViT             |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|           ViTMAE            |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|          Wav2Vec2           |       ✅       |       ❌       |       ✅        |         ✅         |      ✅      |
|            WavLM            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            XGLM             |       ✅       |       ✅       |       ✅        |         ❌         |      ✅      |
|             XLM             |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|         XLM-RoBERTa         |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|       XLM-RoBERTa-XL        |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        XLMProphetNet        |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|            XLNet            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            YOSO             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |

<!-- End table-->
