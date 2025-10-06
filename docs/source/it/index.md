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

Machine Learning allo stato dell'arte per PyTorch, TensorFlow e JAX.

🤗 Transformers fornisce delle API per scaricare in modo semplice e allenare modelli pre-allenati allo stato dell'arte. L'utilizzo di modelli pre-allenati può ridurre i tuoi costi computazionali, l'impatto ambientale, e farti risparmiare il tempo che utilizzeresti per allenare un modello da zero. I modelli possono essere utilizzati in diverse modalità come ad esempio:

* 📝 Testo: classificazione del testo, estrazione delle informazioni, rispondere a domande, riassumere, traduzione e generazione del testo in più di 100 lingue.
* 🖼️ Immagini: classificazione di immagini, rilevazione di oggetti e segmentazione.
* 🗣️ Audio: riconoscimento vocale e classificazione dell'audio.
* 🐙 Multimodale: rispondere a domande inerenti dati tabulari, riconoscimento ottico dei caratteri, estrazione di informazioni a partire da documenti scannerizzati, classificazione di video e risposta visuale a domande.

La nostra libreria supporta un'integrazione perfetta tra tre delle librerie per il deep learning più popolari: [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/) e [JAX](https://jax.readthedocs.io/en/latest/). Allena il tuo modello in tre righe di codice in un framework, e caricalo per l'inferenza in un altro.

Ogni architettura di 🤗 Transformers è definita in un modulo Python indipendente così da poter essere personalizzata in modo semplice per la ricerca e gli esperimenti.

## Se stai cercando supporto personalizzato dal team di Hugging Face

<a target="_blank" href="https://huggingface.co/support">
<img alt="HuggingFace Expert Acceleration Program" src="https://huggingface.co/front/thumbnails/support.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a>

## Contenuti

La documentazione è organizzata in cinque parti:

- **INIZIARE** contiene un tour rapido e le istruzioni di installazione per cominciare ad utilizzare 🤗 Transformers.
- **TUTORIALS** è un buon posto da cui iniziare se per te la nostra libreria è nuova. Questa sezione ti aiuterà ad acquisire le competenze basilari di cui hai bisogno per iniziare ad  utilizzare 🤗 Transformers.
- **GUIDE PRATICHE** ti mostrerà come raggiungere obiettivi specifici come fare fine-tuning di un modello pre-allenato per la modellizzazione del linguaggio o come creare una testa per un modello personalizzato.
- **GUIDE CONCETTUALI** fornisce discussioni e spiegazioni dei concetti sottostanti alle idee dietro ai modelli, compiti, e la filosofia di progettazione di 🤗 Transformers.
- **API** descrive ogni classe e funzione, raggruppate in:
    - **CLASSI PRINCIPALI** per le classi principali che espongono le API importanti della libreria.
    - **MODELLI** per le classi e le funzioni relative ad ogni modello implementato all'interno della libreria.
    - **HELPERS INTERNI** per le classi e le funzioni che utilizziamo internamente.

La libreria attualmente contiene implementazioni in JAX, PyTorch e TensorFlow, pesi di modelli pre-allenati, script di utilizzo e strumenti di conversione per i seguenti modelli.

### Modelli supportati

<!--This list is updated automatically from the README with _make fix-copies_. Do not update manually! -->

1. **[ALBERT](model_doc/albert)** (da Google Research e l'Istituto Tecnologico di Chicago) rilasciato con il paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://huggingface.co/papers/1909.11942), da Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
1. **[ALIGN](model_doc/align)** (from Google Research) rilasciato con il paper [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://huggingface.co/papers/2102.05918) da Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.
1. **[BART](model_doc/bart)** (da Facebook) rilasciato con il paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://huggingface.co/papers/1910.13461) da Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov e Luke Zettlemoyer.
1. **[BARThez](model_doc/barthez)** (da politecnico di École) rilasciato con il paper [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://huggingface.co/papers/2010.12321) da Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis.
1. **[BARTpho](model_doc/bartpho)** (da VinAI Research) rilasciato con il paper [BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese](https://huggingface.co/papers/2109.09701) da Nguyen Luong Tran, Duong Minh Le e Dat Quoc Nguyen.
1. **[BEiT](model_doc/beit)** (da Microsoft) rilasciato con il paper [BEiT: BERT Pre-Training of Image Transformers](https://huggingface.co/papers/2106.08254) da Hangbo Bao, Li Dong, Furu Wei.
1. **[BERT](model_doc/bert)** (da Google) rilasciato con il paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://huggingface.co/papers/1810.04805) da Jacob Devlin, Ming-Wei Chang, Kenton Lee e Kristina Toutanova.
1. **[BERTweet](model_doc/bertweet)** (da VinAI Research) rilasciato con il paper [BERTweet: A pre-trained language model for English Tweets](https://aclanthology.org/2020.emnlp-demos.2/) da Dat Quoc Nguyen, Thanh Vu e Anh Tuan Nguyen.
1. **[BERT For Sequence Generation](model_doc/bert-generation)** (da Google) rilasciato con il paper [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://huggingface.co/papers/1907.12461) da Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
1. **[BigBird-RoBERTa](model_doc/big_bird)** (da Google Research) rilasciato con il paper [Big Bird: Transformers for Longer Sequences](https://huggingface.co/papers/2007.14062) da Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
1. **[BigBird-Pegasus](model_doc/bigbird_pegasus)** (v Google Research) rilasciato con il paper [Big Bird: Transformers for Longer Sequences](https://huggingface.co/papers/2007.14062) da Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.
1. **[Blenderbot](model_doc/blenderbot)** (da Facebook) rilasciato con il paper [Recipes for building an open-domain chatbot](https://huggingface.co/papers/2004.13637) da Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
1. **[BlenderbotSmall](model_doc/blenderbot-small)** (da Facebook) rilasciato con il paper [Recipes for building an open-domain chatbot](https://huggingface.co/papers/2004.13637) da Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston.
1. **[BORT](model_doc/bort)** (da Alexa) rilasciato con il paper [Optimal Subarchitecture Extraction For BERT](https://huggingface.co/papers/2010.10499) da Adrian de Wynter e Daniel J. Perry.
1. **[ByT5](model_doc/byt5)** (da Google Research) rilasciato con il paper [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://huggingface.co/papers/2105.13626) da Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel.
1. **[CamemBERT](model_doc/camembert)** (da Inria/Facebook/Sorbonne) rilasciato con il paper [CamemBERT: a Tasty French Language Model](https://huggingface.co/papers/1911.03894) da Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah e Benoît Sagot.
1. **[CANINE](model_doc/canine)** (da Google Research) rilasciato con il paper [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://huggingface.co/papers/2103.06874) da Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting.
1. **[ConvNeXT](model_doc/convnext)** (da Facebook AI) rilasciato con il paper [A ConvNet for the 2020s](https://huggingface.co/papers/2201.03545) da Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie.
1. **[ConvNeXTV2](model_doc/convnextv2)** (da Facebook AI) rilasciato con il paper [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://huggingface.co/papers/2301.00808) da Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie.
1. **[CLIP](model_doc/clip)** (da OpenAI) rilasciato con il paper [Learning Transferable Visual Models From Natural Language Supervision](https://huggingface.co/papers/2103.00020) da Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
1. **[ConvBERT](model_doc/convbert)** (da YituTech) rilasciato con il paper [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://huggingface.co/papers/2008.02496) da Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan.
1. **[CPM](model_doc/cpm)** (dalla Università di Tsinghua) rilasciato con il paper [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://huggingface.co/papers/2012.00413) da Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.
1. **[CTRL](model_doc/ctrl)** (da Salesforce) rilasciato con il paper [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://huggingface.co/papers/1909.05858) da Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong e Richard Socher.
1. **[CvT](model_doc/cvt)** (da Microsoft) rilasciato con il paper [CvT: Introducing Convolutions to Vision Transformers](https://huggingface.co/papers/2103.15808) da Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, Lei Zhang.
1. **[Data2Vec](model_doc/data2vec)** (da Facebook) rilasciato con il paper [Data2Vec:  A General Framework for Self-supervised Learning in Speech, Vision and Language](https://huggingface.co/papers/2202.03555) da Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.
1. **[DeBERTa](model_doc/deberta)** (da Microsoft) rilasciato con il paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://huggingface.co/papers/2006.03654) da Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
1. **[DeBERTa-v2](model_doc/deberta-v2)** (da Microsoft) rilasciato con il paper [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://huggingface.co/papers/2006.03654) da Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.
1. **[Decision Transformer](model_doc/decision_transformer)** (da Berkeley/Facebook/Google) rilasciato con il paper [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://huggingface.co/papers/2106.01345) da Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch.
1. **[DiT](model_doc/dit)** (da Microsoft Research) rilasciato con il paper [DiT: Self-supervised Pre-training for Document Image Transformer](https://huggingface.co/papers/2203.02378) da Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, Furu Wei.
1. **[DeiT](model_doc/deit)** (da Facebook) rilasciato con il paper [Training data-efficient image transformers & distillation through attention](https://huggingface.co/papers/2012.12877) da Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
1. **[DETR](model_doc/detr)** (da Facebook) rilasciato con il paper [End-to-End Object Detection with Transformers](https://huggingface.co/papers/2005.12872) da Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko.
1. **[DialoGPT](model_doc/dialogpt)** (da Microsoft Research) rilasciato con il paper [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://huggingface.co/papers/1911.00536) da Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
1. **[DistilBERT](model_doc/distilbert)** (da HuggingFace), rilasciato assieme al paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://huggingface.co/papers/1910.01108) da Victor Sanh, Lysandre Debut e Thomas Wolf. La stessa tecnica è stata applicata per comprimere GPT2 in [DistilGPT2](https://github.com/huggingface/transformers-research-projects/tree/main/distillation), RoBERTa in [DistilRoBERTa](https://github.com/huggingface/transformers-research-projects/tree/main/distillation), Multilingual BERT in [DistilmBERT](https://github.com/huggingface/transformers-research-projects/tree/main/distillation) and a German version of DistilBERT.
1. **[DPR](model_doc/dpr)** (da Facebook) rilasciato con il paper [Dense Passage Retrieval for Open-Domain Question Answering](https://huggingface.co/papers/2004.04906) da Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, e Wen-tau Yih.
1. **[DPT](master/model_doc/dpt)** (da Intel Labs) rilasciato con il paper [Vision Transformers for Dense Prediction](https://huggingface.co/papers/2103.13413) da René Ranftl, Alexey Bochkovskiy, Vladlen Koltun.
1. **[EfficientNet](model_doc/efficientnet)** (from Google Research) released with the paper [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://huggingface.co/papers/1905.11946)  by Mingxing Tan and Quoc V. Le.
1. **[EncoderDecoder](model_doc/encoder-decoder)** (da Google Research) rilasciato con il paper [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://huggingface.co/papers/1907.12461) da Sascha Rothe, Shashi Narayan, Aliaksei Severyn.
1. **[ELECTRA](model_doc/electra)** (da Google Research/Stanford University) rilasciato con il paper [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://huggingface.co/papers/2003.10555) da Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
1. **[FlauBERT](model_doc/flaubert)** (da CNRS) rilasciato con il paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://huggingface.co/papers/1912.05372) da Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
1. **[FLAVA](model_doc/flava)** (da Facebook AI) rilasciato con il paper [FLAVA: A Foundational Language And Vision Alignment Model](https://huggingface.co/papers/2112.04482) da Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, e Douwe Kiela.
1. **[FNet](model_doc/fnet)** (da Google Research) rilasciato con il paper [FNet: Mixing Tokens with Fourier Transforms](https://huggingface.co/papers/2105.03824) da James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon.
1. **[Funnel Transformer](model_doc/funnel)** (da CMU/Google Brain) rilasciato con il paper [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://huggingface.co/papers/2006.03236) da Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.
1. **[GLPN](model_doc/glpn)** (da KAIST) rilasciato con il paper [Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth](https://huggingface.co/papers/2201.07436) da Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, Junmo Kim.
1. **[GPT](model_doc/openai-gpt)** (da OpenAI) rilasciato con il paper [Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised/) da Alec Radford, Karthik Narasimhan, Tim Salimans e Ilya Sutskever.
1. **[GPT-2](model_doc/gpt2)** (da OpenAI) rilasciato con il paper [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models/) da Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei e Ilya Sutskever.
1. **[GPT-J](model_doc/gptj)** (da EleutherAI) rilasciato nel repository [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/) da Ben Wang e Aran Komatsuzaki.
1. **[GPT Neo](model_doc/gpt_neo)** (da EleutherAI) rilasciato nel repository [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) da Sid Black, Stella Biderman, Leo Gao, Phil Wang e Connor Leahy.
1. **[GPT NeoX](model_doc/gpt_neox)** (da EleutherAI) rilasciato con il paper [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://huggingface.co/papers/2204.06745) da Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, Samuel Weinbach
1. **[Hubert](model_doc/hubert)** (da Facebook) rilasciato con il paper [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://huggingface.co/papers/2106.07447) da Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.
1. **[I-BERT](model_doc/ibert)** (da Berkeley) rilasciato con il paper [I-BERT: Integer-only BERT Quantization](https://huggingface.co/papers/2101.01321) da Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer.
1. **[ImageGPT](model_doc/imagegpt)** (da OpenAI) rilasciato con il paper [Generative Pretraining from Pixels](https://openai.com/blog/image-gpt/) da Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever.
1. **[LayoutLM](model_doc/layoutlm)** (da Microsoft Research Asia) rilasciato con il paper [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://huggingface.co/papers/1912.13318) da Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou.
1. **[LayoutLMv2](model_doc/layoutlmv2)** (da Microsoft Research Asia) rilasciato con il paper [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://huggingface.co/papers/2012.14740) da Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou.
1. **[LayoutLMv3](model_doc/layoutlmv3)** (da Microsoft Research Asia) rilasciato con il paper [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://huggingface.co/papers/2204.08387) da Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, Furu Wei.
1. **[LayoutXLM](model_doc/layoutlxlm)** (da Microsoft Research Asia) rilasciato con il paper [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://huggingface.co/papers/2104.08836) da Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Furu Wei.
1. **[LED](model_doc/led)** (da AllenAI) rilasciato con il paper [Longformer: The Long-Document Transformer](https://huggingface.co/papers/2004.05150) da Iz Beltagy, Matthew E. Peters, Arman Cohan.
1. **[Longformer](model_doc/longformer)** (da AllenAI) rilasciato con il paper [Longformer: The Long-Document Transformer](https://huggingface.co/papers/2004.05150) da Iz Beltagy, Matthew E. Peters, Arman Cohan.
1. **[LUKE](model_doc/luke)** (da Studio Ousia) rilasciato con il paper [LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://huggingface.co/papers/2010.01057) da Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yuji Matsumoto.
1. **[mLUKE](model_doc/mluke)** (da Studio Ousia) rilasciato con il paper [mLUKE: The Power of Entity Representations in Multilingual Pretrained Language Models](https://huggingface.co/papers/2110.08151) da Ryokan Ri, Ikuya Yamada, e Yoshimasa Tsuruoka.
1. **[LXMERT](model_doc/lxmert)** (da UNC Chapel Hill) rilasciato con il paper [LXMERT: Learning Cross-Modality Encoder Representations from Transformers for Open-Domain Question Answering](https://huggingface.co/papers/1908.07490) da Hao Tan e Mohit Bansal.
1. **[M2M100](model_doc/m2m_100)** (da Facebook) rilasciato con il paper [Beyond English-Centric Multilingual Machine Translation](https://huggingface.co/papers/2010.11125) da Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, Armand Joulin.
1. **[MarianMT](model_doc/marian)** Modello di machine learning per le traduzioni allenato utilizzando i dati [OPUS](http://opus.nlpl.eu/) di Jörg Tiedemann. Il [Framework Marian](https://marian-nmt.github.io/) è stato sviluppato dal Microsoft Translator Team.
1. **[Mask2Former](model_doc/mask2former)** (da FAIR e UIUC) rilasciato con il paper [Masked-attention Mask Transformer for Universal Image Segmentation](https://huggingface.co/papers/2112.01527) da Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar.
1. **[MaskFormer](model_doc/maskformer)** (da Meta e UIUC) rilasciato con il paper [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://huggingface.co/papers/2107.06278) da Bowen Cheng, Alexander G. Schwing, Alexander Kirillov.
1. **[MBart](model_doc/mbart)** (da Facebook) rilasciato con il paper [Multilingual Denoising Pre-training for Neural Machine Translation](https://huggingface.co/papers/2001.08210) da Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.
1. **[MBart-50](model_doc/mbart)** (da Facebook) rilasciato con il paper [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://huggingface.co/papers/2008.00401) da Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, Angela Fan.
1. **[Megatron-BERT](model_doc/megatron-bert)** (da NVIDIA) rilasciato con il paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://huggingface.co/papers/1909.08053) da Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper e Bryan Catanzaro.
1. **[Megatron-GPT2](model_doc/megatron_gpt2)** (da NVIDIA) rilasciato con il paper [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://huggingface.co/papers/1909.08053) da Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper e Bryan Catanzaro.
1. **[MPNet](model_doc/mpnet)** (da Microsoft Research) rilasciato con il paper [MPNet: Masked and Permuted Pre-training for Language Understanding](https://huggingface.co/papers/2004.09297) da Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.
1. **[MT5](model_doc/mt5)** (da Google AI) rilasciato con il paper [mT5: A massively multilingual pre-trained text-to-text transformer](https://huggingface.co/papers/2010.11934) da Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel.
1. **[Nyströmformer](model_doc/nystromformer)** (dalla Università del Wisconsin - Madison) rilasciato con il paper [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://huggingface.co/papers/2102.03902) da Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, Vikas Singh.
1. **[OneFormer](model_doc/oneformer)** (da SHI Labs) rilasciato con il paper [OneFormer: One Transformer to Rule Universal Image Segmentation](https://huggingface.co/papers/2211.06220) da Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi.
1. **[OPT](master/model_doc/opt)** (da Meta AI) rilasciato con il paper [OPT: Open Pre-trained Transformer Language Models](https://huggingface.co/papers/2205.01068) da Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen et al.
1. **[Pegasus](model_doc/pegasus)** (da Google) rilasciato con il paper [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://huggingface.co/papers/1912.08777) da Jingqing Zhang, Yao Zhao, Mohammad Saleh e Peter J. Liu.
1. **[Perceiver IO](model_doc/perceiver)** (da Deepmind) rilasciato con il paper [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://huggingface.co/papers/2107.14795) da Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira.
1. **[PhoBERT](model_doc/phobert)** (da VinAI Research) rilasciato con il paper [PhoBERT: Pre-trained language models for Vietnamese](https://www.aclweb.org/anthology/2020.findings-emnlp.92/) da Dat Quoc Nguyen e Anh Tuan Nguyen.
1. **[PLBart](model_doc/plbart)** (da UCLA NLP) rilasciato con il paper [Unified Pre-training for Program Understanding and Generation](https://huggingface.co/papers/2103.06333) da Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang.
1. **[PoolFormer](model_doc/poolformer)** (da Sea AI Labs) rilasciato con il paper [MetaFormer is Actually What You Need for Vision](https://huggingface.co/papers/2111.11418) da Yu, Weihao e Luo, Mi e Zhou, Pan e Si, Chenyang e Zhou, Yichen e Wang, Xinchao e Feng, Jiashi e Yan, Shuicheng.
1. **[ProphetNet](model_doc/prophetnet)** (da Microsoft Research) rilasciato con il paper [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://huggingface.co/papers/2001.04063) da Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang e Ming Zhou.
1. **[QDQBert](model_doc/qdqbert)** (da NVIDIA) rilasciato con il paper [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://huggingface.co/papers/2004.09602) da Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev e Paulius Micikevicius.
1. **[REALM](model_doc/realm.html)** (da Google Research) rilasciato con il paper [REALM: Retrieval-Augmented Language Model Pre-Training](https://huggingface.co/papers/2002.08909) da Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat e Ming-Wei Chang.
1. **[Reformer](model_doc/reformer)** (da Google Research) rilasciato con il paper [Reformer: The Efficient Transformer](https://huggingface.co/papers/2001.04451) da Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
1. **[RemBERT](model_doc/rembert)** (da Google Research) rilasciato con il paper [Rethinking embedding coupling in pre-trained language models](https://huggingface.co/papers/2010.12821) da Hyung Won Chung, Thibault Févry, Henry Tsai, M. Johnson, Sebastian Ruder.
1. **[RegNet](model_doc/regnet)** (da META Platforms) rilasciato con il paper [Designing Network Design Space](https://huggingface.co/papers/2003.13678) da Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár.
1. **[ResNet](model_doc/resnet)** (da Microsoft Research) rilasciato con il paper [Deep Residual Learning for Image Recognition](https://huggingface.co/papers/1512.03385) da Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
1. **[RoBERTa](model_doc/roberta)** (da Facebook), rilasciato assieme al paper [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://huggingface.co/papers/1907.11692) da Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
1. **[RoFormer](model_doc/roformer)** (da ZhuiyiTechnology), rilasciato assieme al paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://huggingface.co/papers/2104.09864) da Jianlin Su e Yu Lu e Shengfeng Pan e Bo Wen e Yunfeng Liu.
1. **[SegFormer](model_doc/segformer)** (da NVIDIA) rilasciato con il paper [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://huggingface.co/papers/2105.15203) da Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo.
1. **[SEW](model_doc/sew)** (da ASAPP) rilasciato con il paper [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://huggingface.co/papers/2109.06870) da Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.
1. **[SEW-D](model_doc/sew_d)** (da ASAPP) rilasciato con il paper [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech Recognition](https://huggingface.co/papers/2109.06870) da Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger, Yoav Artzi.
1. **[SpeechToTextTransformer](model_doc/speech_to_text)** (da Facebook), rilasciato assieme al paper [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://huggingface.co/papers/2010.05171) da Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino.
1. **[SpeechToTextTransformer2](model_doc/speech_to_text_2)** (da Facebook), rilasciato assieme al paper [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://huggingface.co/papers/2104.06678) da Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau.
1. **[Splinter](model_doc/splinter)** (dalla Università di Tel Aviv), rilasciato assieme al paper [Few-Shot Question Answering by Pretraining Span Selection](https://huggingface.co/papers/2101.00438) da Ori Ram, Yuval Kirstain, Jonathan Berant, Amir Globerson, Omer Levy.
1. **[SqueezeBert](model_doc/squeezebert)** (da Berkeley) rilasciato con il paper [SqueezeBERT: What can computer vision teach NLP about efficient neural networks?](https://huggingface.co/papers/2006.11316) da Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, e Kurt W. Keutzer.
1. **[Swin Transformer](model_doc/swin)** (da Microsoft) rilasciato con il paper [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://huggingface.co/papers/2103.14030) da Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.
1. **[T5](model_doc/t5)** (da Google AI) rilasciato con il paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://huggingface.co/papers/1910.10683) da Colin Raffel e Noam Shazeer e Adam Roberts e Katherine Lee e Sharan Narang e Michael Matena e Yanqi Zhou e Wei Li e Peter J. Liu.
1. **[T5v1.1](model_doc/t5v1.1)** (da Google AI) rilasciato nel repository [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511) da Colin Raffel e Noam Shazeer e Adam Roberts e Katherine Lee e Sharan Narang e Michael Matena e Yanqi Zhou e Wei Li e Peter J. Liu.
1. **[TAPAS](model_doc/tapas)** (da Google AI) rilasciato con il paper [TAPAS: Weakly Supervised Table Parsing via Pre-training](https://huggingface.co/papers/2004.02349) da Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno e Julian Martin Eisenschlos.
1. **[TAPEX](model_doc/tapex)** (da Microsoft Research) rilasciato con il paper [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://huggingface.co/papers/2107.07653) da Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, Jian-Guang Lou.
1. **[Trajectory Transformer](model_doc/trajectory_transformers)** (dall'Università della California a Berkeley) rilasciato con il paper [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://huggingface.co/papers/2106.02039) da Michael Janner, Qiyang Li, Sergey Levine
1. **[Transformer-XL](model_doc/transfo-xl)** (da Google/CMU) rilasciato con il paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://huggingface.co/papers/1901.02860) da Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
1. **[TrOCR](model_doc/trocr)** (da Microsoft), rilasciato assieme al paper [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://huggingface.co/papers/2109.10282) da Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei.
1. **[UniSpeech](model_doc/unispeech)** (da Microsoft Research) rilasciato con il paper [UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data](https://huggingface.co/papers/2101.07597) da Chengyi Wang, Yu Wu, Yao Qian, Kenichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang.
1. **[UniSpeechSat](model_doc/unispeech-sat)** (da Microsoft Research) rilasciato con il paper [UNISPEECH-SAT: UNIVERSAL SPEECH REPRESENTATION LEARNING WITH SPEAKER AWARE PRE-TRAINING](https://huggingface.co/papers/2110.05752) da Sanyuan Chen, Yu Wu, Chengyi Wang, Zhengyang Chen, Zhuo Chen, Shujie Liu, Jian Wu, Yao Qian, Furu Wei, Jinyu Li, Xiangzhan Yu.
1. **[VAN](model_doc/van)** (dalle Università di Tsinghua e Nankai) rilasciato con il paper [Visual Attention Network](https://huggingface.co/papers/2202.09741) da Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu.
1. **[ViLT](model_doc/vilt)** (da NAVER AI Lab/Kakao Enterprise/Kakao Brain) rilasciato con il paper [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://huggingface.co/papers/2102.03334) da Wonjae Kim, Bokyung Son, Ildoo Kim.
1. **[Vision Transformer (ViT)](model_doc/vit)** (da Google AI) rilasciato con il paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://huggingface.co/papers/2010.11929) da Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
1. **[ViTMAE](model_doc/vit_mae)** (da Meta AI) rilasciato con il paper [Masked Autoencoders Are Scalable Vision Learners](https://huggingface.co/papers/2111.06377) da Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.
1. **[VisualBERT](model_doc/visual_bert)** (da UCLA NLP) rilasciato con il paper [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://huggingface.co/papers/1908.03557) da Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang.
1. **[WavLM](model_doc/wavlm)** (da Microsoft Research) rilasciato con il paper [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://huggingface.co/papers/2110.13900) da Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Furu Wei.
1. **[Wav2Vec2](model_doc/wav2vec2)** (da Facebook AI) rilasciato con il paper [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://huggingface.co/papers/2006.11477) da Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.
1. **[Wav2Vec2Phoneme](model_doc/wav2vec2_phoneme)** (da Facebook AI) rilasciato con il paper [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition](https://huggingface.co/papers/2109.11680) da Qiantong Xu, Alexei Baevski, Michael Auli.
1. **[XGLM](model_doc/xglm)** (da Facebook AI) rilasciato con il paper [Few-shot Learning with Multilingual Language Models](https://huggingface.co/papers/2112.10668) da Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, Xian Li.
1. **[XLM](model_doc/xlm)** (v Facebook) rilasciato assieme al paper [Cross-lingual Language Model Pretraining](https://huggingface.co/papers/1901.07291) da Guillaume Lample e Alexis Conneau.
1. **[XLM-ProphetNet](model_doc/xlm-prophetnet)** (da Microsoft Research) rilasciato con il paper [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://huggingface.co/papers/2001.04063) da Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang e Ming Zhou.
1. **[XLM-RoBERTa](model_doc/xlm-roberta)** (da Facebook AI), rilasciato assieme al paper [Unsupervised Cross-lingual Representation Learning at Scale](https://huggingface.co/papers/1911.02116) da Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer e Veselin Stoyanov.
1. **[XLM-RoBERTa-XL](model_doc/xlm-roberta-xl)** (da Facebook AI), rilasciato assieme al paper [Larger-Scale Transformers for Multilingual Masked Language Modeling](https://huggingface.co/papers/2105.00572) da Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, Alexis Conneau.
1. **[XLNet](model_doc/xlnet)** (da Google/CMU) rilasciato con il paper [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://huggingface.co/papers/1906.08237) da Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
1. **[XLSR-Wav2Vec2](model_doc/xlsr_wav2vec2)** (da Facebook AI) rilasciato con il paper [Unsupervised Cross-Lingual Representation Learning For Speech Recognition](https://huggingface.co/papers/2006.13979) da Alexis Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, Michael Auli.
1. **[XLS-R](model_doc/xls_r)** (da Facebook AI) rilasciato con il paper [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale](https://huggingface.co/papers/2111.09296) da Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, Michael Auli.
1. **[YOLOS](model_doc/yolos)** (dalla Università della scienza e tecnologia di Huazhong) rilasciato con il paper [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://huggingface.co/papers/2106.00666) da Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu.
1. **[YOSO](model_doc/yoso)** (dall'Università del Wisconsin - Madison) rilasciato con il paper [You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling](https://huggingface.co/papers/2111.09714) da Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh.


### Framework supportati

La tabella seguente rappresenta il supporto attuale nella libreria per ognuno di questi modelli, si può identificare se questi hanno un Python
tokenizer (chiamato "slow"). Un tokenizer "fast" supportato dalla libreria 🤗 Tokenizers, e se hanno supporto in Jax (via Flax), PyTorch, e/o TensorFlow.

<!--This table is updated automatically from the auto modules with _make fix-copies_. Do not update manually!-->

|            Model            | Tokenizer slow | Tokenizer fast | PyTorch support | TensorFlow support | Flax Support |
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
|             CvT             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        Data2VecAudio        |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        Data2VecText         |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|       Data2VecVision        |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|           DeBERTa           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|         DeBERTa-v2          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
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
|            Flava            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            FNet             |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|     Funnel Transformer      |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            GLPN             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|           GPT Neo           |       ❌       |       ❌       |       ✅        |         ❌         |      ✅      |
|          GPT NeoX           |       ❌       |       ✅       |       ✅        |         ❌         |      ❌      |
|            GPT-J            |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
|           Hubert            |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|           I-BERT            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          ImageGPT           |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|          LayoutLM           |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|         LayoutLMv2          |       ✅       |       ✅       |       ✅        |         ❌         |      ❌      |
|         LayoutLMv3          |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
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
|             OPT             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
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
|           ResNet            |       ❌       |       ❌       |       ✅        |         ✅         |      ✅      |
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
|            Swin             |       ❌       |       ❌       |       ✅        |         ✅         |      ❌      |
|             T5              |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|            TAPAS            |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|   Trajectory Transformer    |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
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
|     Wav2Vec2-Conformer      |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            WavLM            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            XGLM             |       ✅       |       ✅       |       ✅        |         ❌         |      ✅      |
|             XLM             |       ✅       |       ❌       |       ✅        |         ✅         |      ❌      |
|         XLM-RoBERTa         |       ✅       |       ✅       |       ✅        |         ✅         |      ✅      |
|       XLM-RoBERTa-XL        |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|        XLMProphetNet        |       ✅       |       ❌       |       ✅        |         ❌         |      ❌      |
|            XLNet            |       ✅       |       ✅       |       ✅        |         ✅         |      ❌      |
|            YOLOS            |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |
|            YOSO             |       ❌       |       ❌       |       ✅        |         ❌         |      ❌      |

<!-- End table-->
