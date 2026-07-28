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
    <a href="https://huggingface.com/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/blob/main/README.md">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Português</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_it.md">Italiano</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fa.md">فارسی</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ro.md">Română</a> |
        <b>Türkçe</b> |
    </p>
</h4>

<h3 align="center">
    <p>Inference ve eğitim için son teknoloji önceden eğitilmiş modeller</p>
</h3>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers; metin, bilgisayarla görü, ses, video ve çok modlu (multimodal) modeller için, hem inference hem de eğitim amacıyla son teknoloji makine öğrenimini mümkün kılan model tanımlama framework'ü olarak çalışır.

Model tanımını merkezileştirir; böylece bu tanım tüm ekosistem genelinde ortak kabul görür. `transformers`, framework'ler arasındaki pivot noktasıdır: bir model tanımı destekleniyorsa, eğitim framework'lerinin (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), inference motorlarının (vLLM, SGLang, TGI, ...) ve `transformers`'taki model tanımından yararlanan komşu modelleme kütüphanelerinin (llama.cpp, mlx, ...) çoğuyla uyumlu olacaktır.

Yeni son teknoloji modelleri desteklemeye ve model tanımlarını basit, özelleştirilebilir ve verimli hâle getirerek kullanımlarını demokratikleştirmeye yardımcı olmayı taahhüt ediyoruz.

[Hugging Face Hub](https://huggingface.co/models) üzerinde kullanabileceğiniz 1M'den fazla Transformers [model kontrol noktası (checkpoint)](https://huggingface.co/models?library=transformers&sort=trending) bulunmaktadır.

Bir model bulmak ve hemen kullanmaya başlamak için bugün [Hub](https://huggingface.co/)'ı keşfedin.

## Kurulum

Transformers, Python 3.10+ ve [PyTorch](https://pytorch.org/get-started/locally/) 2.4+ ile çalışır.

[venv](https://docs.python.org/3/library/venv.html) veya hızlı, Rust tabanlı bir Python paket ve proje yöneticisi olan [uv](https://docs.astral.sh/uv/) ile bir sanal ortam (virtual environment) oluşturup etkinleştirin.

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

Transformers'ı sanal ortamınıza kurun.

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

Kütüphanedeki en son değişiklikleri istiyorsanız veya katkıda bulunmak istiyorsanız Transformers'ı kaynaktan kurun. Ancak *en son* sürüm kararlı olmayabilir. Bir hatayla karşılaşırsanız çekinmeden bir [issue](https://github.com/huggingface/transformers/issues) açabilirsiniz.

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
```

## Hızlı Başlangıç

[Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API'siyle Transformers'ı hemen kullanmaya başlayın. `Pipeline`; metin, ses, görü ve çok modlu görevleri destekleyen üst seviye (high-level) bir inference sınıfıdır. Girdinin ön işlenmesini üstlenir ve uygun çıktıyı döndürür.

Bir pipeline örneği oluşturun ve metin üretimi için kullanılacak modeli belirtin. Model indirilir ve önbelleğe alınır; böylece tekrar tekrar kolayca kullanabilirsiniz. Son olarak, modeli yönlendirmek için biraz metin verin.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

Bir modelle sohbet etmek için kullanım kalıbı aynıdır. Tek fark, sizinle sistem arasında bir sohbet geçmişi (yani `Pipeline`'a verilecek girdi) oluşturmanız gerektiğidir.

> [!TIP]
> [`transformers serve` çalıştığı](https://huggingface.co/docs/transformers/main/en/serving) sürece bir modelle doğrudan komut satırından da sohbet edebilirsiniz.
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```py
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

`Pipeline`'ın farklı modaliteler ve görevler için nasıl çalıştığını görmek için aşağıdaki örnekleri genişletin.

<details>
<summary>Otomatik konuşma tanıma</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>Görüntü sınıflandırma</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'label': 'macaw', 'score': 0.997848391532898},
 {'label': 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
  'score': 0.0016551691805943847},
 {'label': 'lorikeet', 'score': 0.00018523589824326336},
 {'label': 'African grey, African gray, Psittacus erithacus',
  'score': 7.85409429227002e-05},
 {'label': 'quail', 'score': 5.502637941390276e-05}]
```

</details>

<details>
<summary>Görsel soru yanıtlama</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</details>

## Neden Transformers kullanmalıyım?

1. Kullanımı kolay son teknoloji modeller:
    - Doğal dil anlama ve üretme, bilgisayarla görü, ses, video ve çok modlu görevlerde yüksek performans.
    - Araştırmacılar, mühendisler ve geliştiriciler için düşük giriş engeli.
    - Kullanıcıya yönelik az sayıda soyutlama ve öğrenilmesi gereken yalnızca üç sınıf.
    - Tüm önceden eğitilmiş modellerimizi kullanmak için birleşik bir API.

1. Daha düşük hesaplama maliyeti, daha küçük karbon ayak izi:
    - Sıfırdan eğitmek yerine eğitilmiş modelleri paylaşın.
    - Hesaplama süresini ve üretim maliyetlerini azaltın.
    - Tüm modaliteler genelinde 1M'den fazla önceden eğitilmiş kontrol noktasına sahip yüzlerce model mimarisi.

1. Bir modelin yaşam döngüsünün her aşaması için doğru framework'ü seçin:
    - Son teknoloji modelleri 3 satır kodla eğitin.
    - Tek bir modeli PyTorch/JAX/TF2.0 framework'leri arasında dilediğiniz gibi taşıyın.
    - Eğitim, değerlendirme ve üretim için doğru framework'ü seçin.

1. Bir modeli veya örneği ihtiyaçlarınıza göre kolayca özelleştirin:
    - Her mimari için, özgün yazarların yayımladığı sonuçları yeniden üretmek üzere örnekler sunuyoruz.
    - Modelin iç yapıları mümkün olduğunca tutarlı biçimde açığa çıkarılmıştır.
    - Model dosyaları, hızlı denemeler için kütüphaneden bağımsız olarak kullanılabilir.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## Ne zaman Transformers kullanmamalıyım?

- Bu kütüphane, sinir ağları için yapı taşlarından oluşan modüler bir araç kutusu değildir. Model dosyalarındaki kod, araştırmacıların ek soyutlamalara/dosyalara dalmadan her bir model üzerinde hızlıca yineleme yapabilmesi için kasıtlı olarak ek soyutlamalarla yeniden düzenlenmemiştir.
- Eğitim API'si, Transformers tarafından sağlanan PyTorch modelleriyle çalışmak üzere optimize edilmiştir. Genel amaçlı makine öğrenimi döngüleri için [Accelerate](https://huggingface.co/docs/accelerate) gibi başka bir kütüphane kullanmalısınız.
- [Örnek kodlar](https://github.com/huggingface/transformers/tree/main/examples) yalnızca *örnektir*. Sizin özel kullanım durumunuzda doğrudan çalışmayabilir ve çalışması için kodu uyarlamanız gerekebilir.

## Transformers kullanan 100 proje

Transformers, önceden eğitilmiş modelleri kullanmak için bir araç takımından çok daha fazlasıdır; etrafında ve Hugging Face Hub üzerinde kurulan bir proje topluluğudur. Transformers'ın; geliştiricilerin, araştırmacıların, öğrencilerin, profesörlerin, mühendislerin ve diğer herkesin hayalindeki projeleri hayata geçirmesine olanak tanımasını istiyoruz.

Transformers'ın 100.000 yıldızını kutlamak için, Transformers ile inşa edilmiş 100 inanılmaz projeyi listeleyen [awesome-transformers](./awesome-transformers.md) sayfasıyla topluluğu ön plana çıkarmak istedik.

Bir projeye sahipseniz veya kullandığınız bir projenin listede yer alması gerektiğini düşünüyorsanız, eklemek için lütfen bir PR açın!

## Örnek modeller

Modellerimizin çoğunu doğrudan [Hub model sayfalarında](https://huggingface.co/models) test edebilirsiniz.

Çeşitli kullanım durumlarına yönelik birkaç örnek modeli görmek için aşağıdaki her modaliteyi genişletin.

<details>
<summary>Ses</summary>

- [CLAP](https://huggingface.co/laion/clap-htsat-fused) ile ses sınıflandırma
- [Parakeet](https://huggingface.co/nvidia/parakeet-ctc-1.1b#transcribing-using-transformers-%F0%9F%A4%97), [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) ve [Moonshine-Streaming](https://huggingface.co/UsefulSensors/moonshine-streaming-medium) ile otomatik konuşma tanıma
- [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks) ile anahtar sözcük yakalama
- [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) ile konuşmadan konuşmaya üretim
- [MusicGen](https://huggingface.co/facebook/musicgen-large) ile metinden sese
- [CSM](https://huggingface.co/sesame/csm-1b) ile metinden konuşmaya

</details>

<details>
<summary>Bilgisayarla görü</summary>

- [SAM](https://huggingface.co/facebook/sam-vit-base) ile otomatik maske üretimi
- [DepthPro](https://huggingface.co/apple/DepthPro-hf) ile derinlik tahmini
- [DINO v2](https://huggingface.co/facebook/dinov2-base) ile görüntü sınıflandırma
- [SuperPoint](https://huggingface.co/magic-leap-community/superpoint) ile anahtar nokta tespiti
- [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor) ile anahtar nokta eşleştirme
- [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) ile nesne tespiti
- [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple) ile poz tahmini
- [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) ile evrensel segmentasyon
- [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large) ile video sınıflandırma

</details>

<details>
<summary>Çok modlu (Multimodal)</summary>

- [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), [Audio Flamingo](https://huggingface.co/nvidia/audio-flamingo-3-hf) ile sesten veya metinden metne
- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) ile belge soru yanıtlama
- [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) ile görüntüden veya metinden metne
- [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) ile görüntü altyazılama
- [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) ile OCR tabanlı belge anlama
- [TAPAS](https://huggingface.co/google/tapas-base) ile tablo soru yanıtlama
- [Emu3](https://huggingface.co/BAAI/Emu3-Gen) ile birleşik çok modlu anlama ve üretim
- [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) ile görüntüden metne
- [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) ile görsel soru yanıtlama
- [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) ile metinsel ifadeye dayalı görsel segmentasyon

</details>

<details>
<summary>NLP</summary>

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) ile maskelenmiş sözcük tamamlama
- [Gemma](https://huggingface.co/google/gemma-2-2b) ile adlandırılmış varlık tanıma
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) ile soru yanıtlama
- [BART](https://huggingface.co/facebook/bart-large-cnn) ile özetleme
- [T5](https://huggingface.co/google-t5/t5-base) ile çeviri
- [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B) ile metin üretimi
- [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B) ile metin sınıflandırma

</details>

## Alıntı

🤗 Transformers kütüphanesi için alıntı yapabileceğiniz bir [makalemiz](https://aclanthology.org/2020.emnlp-demos.6/) bulunmaktadır:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-demos.6/",
    pages = "38--45"
}
```
