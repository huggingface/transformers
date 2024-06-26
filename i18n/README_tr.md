<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_de.md">Deutsch</a> |
        <b>Türkçe</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/README_vi.md">Tiếng Việt</a> |
    </p>
</h4>

<h3 align="center">
    <p>JAX, PyTorch ve TensorFlow için son teknoloji Makine Öğrenimi</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers, metin, görüntü ve ses gibi farklı yöntemler üzerinde görevleri gerçekleştirmek için binlerce önceden eğitilmiş model sağlar.

Bu modeller aşağıdakilere uygulanabilir:

* 📝 Metin sınıflandırma, bilgi çıkarma, soru yanıtlama, özetleme, çeviri ve metin oluşturma gibi görevler için 100'den fazla dilde metin.
* 🖼️ Görüntü sınıflandırma, nesne algılama ve segmentasyon gibi görevler için görüntüler.
* 🗣️ Konuşma tanıma ve ses sınıflandırması gibi görevler için ses.

Transformatör modelleri ayrıca aşağıdaki görevleri de yerine getirebilir: **birkaç yöntemin bir araya getirilmesi**, tablo soru cevaplama, optik karakter tanıma, taranan belgelerden bilgi çıkarma, video sınıflandırma ve görsel soru cevaplama gibi.

🤗 Transformers, bu önceden eğitilmiş modelleri belirli bir metin üzerinde hızlı bir şekilde indirip kullanmanız, bunlara kendi veri kümelerinizde ince ayar yapmanız ve ardından bunları topluluğumuzla paylaşmanız için API'ler sağlar [model hub](https://huggingface.co/models). Aynı zamanda, bir mimariyi tanımlayan her python modülü tamamen bağımsızdır ve hızlı araştırma deneylerine olanak sağlayacak şekilde değiştirilebilir.

🤗 Transformers en popüler üç derin öğrenme kütüphanesi tarafından desteklenmektedir — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) ve [TensorFlow](https://www.tensorflow.org/)— aralarında kusursuz bir entegrasyonla. Çıkarım yapmak için diğeriyle yüklemeden önce modellerinizi biriyle eğitmek kolaydır.

## Çevrimiçi demolar

Modellerimizin çoğunu doğrudan sayfalarında test edebilirsiniz. [model hub](https://huggingface.co/models). Biz bunu da [private model hosting, versioning, & an inference API](https://huggingface.co/pricing) kamu ve özel modeller için sunuyoruz.

İşte birkaç örnek:

Doğal Dil İşlemede:
- [Masked word completion with BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Named Entity Recognition with Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Text generation with Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Natural Language Inference with RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Summarization with BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Question answering with DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Translation with T5](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)

Bilgisayar Görüşünde:
- [Image classification with ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Object Detection with DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Semantic Segmentation with SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Panoptic Segmentation with Mask2Former](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)
- [Depth Estimation with Depth Anything](https://huggingface.co/docs/transformers/main/model_doc/depth_anything)
- [Video Classification with VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)
- [Universal Segmentation with OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)

Seste:
- [Automatic Speech Recognition with Whisper](https://huggingface.co/openai/whisper-large-v3)
- [Keyword Spotting with Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Audio Classification with Audio Spectrogram Transformer](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

Çok modlu görevlerde:
- [Table Question Answering with TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [Visual Question Answering with ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [Image captioning with LLaVa](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [Zero-shot Image Classification with SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384)
- [Document Question Answering with LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [Zero-shot Video Classification with X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)
- [Zero-shot Object Detection with OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
- [Zero-shot Image Segmentation with CLIPSeg](https://huggingface.co/docs/transformers/model_doc/clipseg)
- [Automatic Mask Generation with SAM](https://huggingface.co/docs/transformers/model_doc/sam)


## Transformers'ı kullanan 100 proje

Transformers, önceden eğitilmiş modellerin kullanımına yönelik bir araç setinden daha fazlasıdır: kendisi ve Hugging Face Hub etrafında inşa edilen bir proje topluluğudur. Transformers'ın geliştiricilerin, araştırmacıların, öğrencilerin, profesörlerin, mühendislerin ve herkesin hayallerindeki projeleri inşa etmelerine olanak sağlamasını istiyoruz.

Transformers'ın 100.000 yıldızını kutlamak için dikkatleri topluluğa yöneltmeye karar verdik ve [awesome-transformers](./awesome-transformers.md) 100'ü listeleyen sayfa
trafoların yakınında inşa edilen inanılmaz projeler.

Listede yer alması gerektiğini düşündüğünüz bir projeye sahipseniz veya onu kullanıyorsanız, lütfen eklemek için bir PR açın!

## Hugging Face ekibinden özel destek arıyorsanız

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## Hızlı tur

Belirli bir girdide (metin, resim, ses, ...) bir modeli hemen kullanmak için `pipeline` API'sini sağlıyoruz. Pipelines, önceden eğitilmiş bir modeli, o modelin eğitimi sırasında kullanılan ön işlemeyle birlikte gruplandırır. Olumlu ve olumsuz metinleri sınıflandırmak için pipeline'ı hızlı bir şekilde nasıl kullanacağınız aşağıda açıklanmıştır:

```python
>>> from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('Transformatör deposuna boru hattını tanıtmaktan çok mutluyuz.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

İkinci kod satırı, işlem hattı tarafından kullanılan önceden eğitilmiş modeli indirir ve önbelleğe alırken, üçüncüsü onu verilen metin üzerinde değerlendirir. Burada cevap %99,97 güvenle "olumlu"dur.

NLP'de ve aynı zamanda bilgisayarla görme ve konuşmada birçok görevin önceden eğitilmiş, kullanıma hazır bir `pipeline` vardır. Örneğin bir görüntüde tespit edilen nesneleri kolaylıkla çıkartabiliriz:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Download an image with cute cats
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# Allocate a pipeline for object detection
>>> object_detector = pipeline('object-detection')
>>> object_detector(image)
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```

Burada, görüntüde tespit edilen nesnelerin bir listesini, nesneyi çevreleyen bir kutu ve bir güven puanıyla birlikte alıyoruz. Soldaki orijinal görüntü, sağda ise tahminler gösteriliyor:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

`pipeline` API'si tarafından desteklenen görevler hakkında daha fazla bilgiyi şu adreste bulabilirsiniz: [this tutorial](https://huggingface.co/docs/transformers/task_summary).

Ek olarak `pipeline`, Önceden eğitilmiş modellerden herhangi birini size verilen görevde indirmek ve kullanmak için tek yapmanız gereken üç satır koddur. İşte PyTorch sürümü:
```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

Ve işte TensorFlow'un eşdeğer (muadil) kodu:
```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

Tokenizer, önceden eğitilmiş modelin beklediği tüm ön işlemlerden sorumludur ve doğrudan tek bir dize (yukarıdaki örneklerde olduğu gibi) veya bir liste üzerinden çağrılabilir. Aşağı akış kodunda kullanabileceğiniz veya ** argüman açma operatörünü kullanarak doğrudan modelinize aktarabileceğiniz bir sözlük çıktısı alacaktır.

Modelin kendisi düzenli [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) veya bir [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) (arka ucunuza bağlı olarak) her zamanki gibi kullanabilirsiniz. [This tutorial](https://huggingface.co/docs/transformers/training) böyle bir modelin klasik bir PyTorch veya TensorFlow eğitim döngüsüne nasıl entegre edileceğini veya yeni bir veri kümesinde hızlı bir şekilde ince ayar yapmak için `Trainer` API'mizin nasıl kullanılacağını açıklıyor.

## Neden Transformers kullanmalıyım?

1. Kullanımı kolay son teknoloji modeller:
    - Doğal dil anlama ve oluşturma, bilgisayarlı görme ve ses görevlerinde yüksek performans.
    - Eğitimciler ve uygulayıcılar için giriş engeli düşüktür.
    - Öğrenilecek yalnızca üç sınıfla kullanıcıya yönelik az sayıda soyutlama.
    - Önceden eğitilmiş tüm modellerimizi kullanmaya yönelik birleşik bir API.

1. Daha düşük bilgi işlem maliyetleri, daha küçük karbon ayak izi:
    - Araştırmacılar her zaman yeniden eğitmek yerine eğitilmiş modelleri paylaşabilirler.
    - Uygulayıcılar hesaplama süresini ve üretim maliyetlerini azaltabilir.
    - Tüm yöntemlerde 400.000'den fazla önceden eğitilmiş model içeren düzinelerce mimari.

1. Bir modelin kullanım ömrünün her bölümü için doğru çerçeveyi seçin:
    - En son teknolojiye sahip modelleri 3 satır kodla eğitin.
    - Tek bir modeli dilediğiniz gibi TF2.0/PyTorch/JAX çerçeveleri arasında taşıyın.
    - Eğitim, değerlendirme ve üretim için doğru çerçeveyi sorunsuz bir şekilde seçin.

1. Bir modeli veya örneği ihtiyaçlarınıza göre kolayca özelleştirin:
    - Orijinal yazarları tarafından yayınlanan sonuçları yeniden üretmek için her mimariye örnekler sunuyoruz.
    - Modelin iç kısımları mümkün olduğunca tutarlı bir şekilde ortaya çıkar.
    - Model dosyaları hızlı deneyler için kütüphaneden bağımsız olarak kullanılabilir.

## Neden Transformers'ı kullanmamalıyım?

- Bu kütüphane, sinir ağları için yapı taşlarından oluşan modüler bir araç kutusu değildir. Model dosyalarındaki kod, ek soyutlamalarla bilerek yeniden düzenlenmez; böylece araştırmacılar, ek soyutlamalara/dosyalara dalmadan her model üzerinde hızlı bir şekilde yineleme yapabilir.
- Eğitim API'sinin herhangi bir model üzerinde çalışması amaçlanmamıştır ancak kitaplık tarafından sağlanan modellerle çalışacak şekilde optimize edilmiştir. Genel makine öğrenimi döngüleri için başka bir kitaplık kullanmalısınız (muhtemel olarak, [Accelerate](https://huggingface.co/docs/accelerate)).
- Mümkün olduğu kadar çok kullanım senaryosu sunmaya çalışırken, senaryolarımızdaki komut dosyaları [examples folder](https://github.com/huggingface/transformers/tree/main/examples) sadece bu: örnekler. Özel sorununuz üzerinde kullanıma hazır bir şekilde çalışmamaları ve bunları ihtiyaçlarınıza uyarlamak için birkaç kod satırını değiştirmeniz gerekmesi beklenir.

## Kurulum

### pip Kullanarak

Bu depo Python 3.8+, Flax 0.4.1+, PyTorch 1.11+ ve TensorFlow 2.6+ üzerinde test edilmiştir.

🤗 Transformers uygulamasını içinde indirmeniz gerekir [virtual environment](https://docs.python.org/3/library/venv.html).Python sanal ortamlarına aşina değilseniz, şuraya göz atın: [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Öncelikle kullanacağınız Python sürümüyle sanal bir ortam oluşturun ve etkinleştirin.

Daha sonra Flax, PyTorch veya TensorFlow'dan en az birini yüklemeniz gerekecektir.
Bakınız [TensorFlow installation page](https://www.tensorflow.org/install/), [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) and/or [Flax](https://github.com/google/flax#quick-install) and [Jax](https://github.com/google/jax#installation) platformunuza özel kurulum komutuyla ilgili kurulum sayfaları.

Bu back-end'lerden biri kurulduğunda, 🤗 Transformatörler pip kullanılarak aşağıdaki gibi kurulabilir:

```bash
pip install transformers
```

Örneklerle oynamak istiyorsanız veya kodun son noktasına ihtiyacınız varsa ve yeni sürümü sabırsızlıkla bekliyorsanız, [kütüphaneyi kaynağından indirin](https://huggingface.co/docs/transformers/installation#installing-from-source).

### conda Kullanarak

🤗 Conda kullanılarak Transformers aşağıdaki gibi kurulabilir:

```shell script
conda install conda-forge::transformers
```

> **_NOT:_**  `transformers`'ı   `huggingface` üzerinden indirme kanalı kullanımdan kaldırıldı.

Conda ile nasıl kurulacağını görmek için Flax, PyTorch veya TensorFlow'un kurulum sayfalarını takip edin.

> **_NOT:_** Windows'ta önbelleğe alma özelliğinden yararlanmak için Geliştirici Modunu etkinleştirmeniz istenebilir. Bu sizin için bir seçenek değilse lütfen bize bildirin. [bu problemi](https://github.com/huggingface/huggingface_hub/issues/1062).

## Model mimarileri

**[Tüm model kontrol noktaları](https://huggingface.co/models)** 🤗 Transformers, huggingface.co'dan sorunsuz bir şekilde entegre edilmiştir.[model hub](https://huggingface.co/models), wburada doğrudan tarafından yüklenirler [kullanıcı](https://huggingface.co/users) ve [organizasyonlar](https://huggingface.co/organizations).

Anlık olarak kontrol noktaları sayısı: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers şu anda aşağıdaki mimarileri sunmaktadır: buradan [bakın](https://huggingface.co/docs/transformers/model_summary) her birinin üst düzey bir özeti için.

Her modelin Flax, PyTorch veya TensorFlow'da bir uygulaması olup olmadığını veya 🤗 Tokenizers kütüphanesi tarafından desteklenen ilişkili bir tokenizer'a sahip olup olmadığını kontrol etmek için bakınız [bu tabloya](https://huggingface.co/docs/transformers/index#supported-frameworks).

Bu uygulamalar çeşitli veri kümelerinde test edilmiştir (örnek komut dosyalarına bakın) ve orijinal uygulamaların performansıyla eşleşmelidir. Performansla ilgili daha fazla ayrıntıyı Örnekler bölümünde bulabilirsiniz. [doküman](https://github.com/huggingface/transformers/tree/main/examples).


## Daha fazlasını öğrenin

| Bölüm | Açıklama |
|-|-|
| [Doküman](https://huggingface.co/docs/transformers/) | Tam API belgeleri ve eğitimleri |
| [Görev Özeti](https://huggingface.co/docs/transformers/task_summary) | 🤗 Transformers tarafından desteklenen görevler |
| [Ön işleme eğitimi](https://huggingface.co/docs/transformers/preprocessing) | Modellere veri hazırlamak için `Tokenizer` sınıfını kullanma |
| [Eğitim ve ince ayar](https://huggingface.co/docs/transformers/training) | 🤗 Transformers tarafından sağlanan modelleri PyTorch/TensorFlow eğitim döngüsünde ve `Tokenizer` API'sinde kullanma |
| [Hızlı tur: İnce ayar/kullanım komut dosyaları](https://github.com/huggingface/transformers/tree/main/examples) | Çok çeşitli görevlerde modellerde ince ayar yapmak için örnek komut dosyaları |
| [Model paylaşımı ve yükleme](https://huggingface.co/docs/transformers/model_sharing) | İnce ayarlı modellerinizi yükleyin ve toplulukla paylaşın |

## Alıntı

Biz artık bir [kağıt](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) sahibiyiz. 🤗 Transformers kütüphanesi için alıntı yapabilirsiniz:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
