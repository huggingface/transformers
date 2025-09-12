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
    <img alt="Hugging Face Transformers Kütüphanesi" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://huggingface.co/models"><img alt="Hub'daki Kontrol Noktaları" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub sürümü" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Katkıda Bulunan Sözleşmesi" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <b>İngilizce</b> |
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
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_tr.md">Türkçe</a>
    </p>
</h4>

<h3 align="center">
    <p>Çıkarım ve eğitim için en son teknoloji önceden eğitilmiş modeller</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

Transformers, metin, bilgisayar görüsü, ses, video ve multimodal modeller için önceden eğitilmiş bir kütüphanedir ve çıkarım ve eğitim için kullanılabilir. Transformers'ı kullanarak modelleri kendi verilerinizle ince ayar yapabilir, çıkarım uygulamaları oluşturabilir ve çoklu modalitelerde yapay zeka kullanım durumlarını gerçekleştirebilirsiniz.

[Hugging Face Hub](https://huggingface.com/models) üzerinde Transformers için 500K'dan fazla [model kontrol noktası](https://huggingface.co/models?library=transformers&sort=trending) bulunmaktadır ve bunları kullanabilirsiniz.

Bugün [Hub](https://huggingface.co/)'ı keşfedin, bir model bulun ve Transformers'ı kullanarak hemen başlayın.

## Kurulum

Transformers, Python 3.9+ ile çalışır ve [PyTorch](https://pytorch.org/get-started/locally/) 2.1+, [TensorFlow](https://www.tensorflow.org/install/pip) 2.6+ ve [Flax](https://flax.readthedocs.io/en/latest/) 0.4.1+ ile uyumludur.

[venv](https://docs.python.org/3/library/venv.html) veya [uv](https://docs.astral.sh/uv/) ile sanal bir ortam oluşturun ve etkinleştirin, uv hızlı bir Rust tabanlı Python paket ve proje yöneticisidir.

```py
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

Sanal ortamınıza Transformers'ı kurun.

```py
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

Kütüphanedeki en son değişiklikleri almak istiyorsanız veya katkıda bulunmakla ilgileniyorsanız, Transformers'ı kaynak kodundan kurun. Ancak, *en son* sürüm stabil olmayabilir. Bir hata ile karşılaşırsanız lütfen bir [issue](https://github.com/huggingface/transformers/issues) açın.

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

## Hızlı Başlangıç

[Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API'si ile Transformers'ı hemen kullanmaya başlayın. `Pipeline`, metin, ses, görüntü ve multimodal görevleri destekleyen yüksek seviyeli bir çıkarım sınıfıdır. Girişin ön işlemesini yapar ve uygun çıktıyı döndürür.

Metin oluşturma için kullanmak istediğiniz modeli belirterek bir pipeline oluşturun. Model indirilir ve önbelleğe alınır, böylece tekrar kolayca kullanabilirsiniz. Son olarak, modeli tetiklemek için bazı metinler geçirin.

```py
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
[{'generated_text': 'the secret to baking a really good cake is 1) to use the right ingredients and 2) to follow the recipe exactly. the recipe for the cake is as follows: 1 cup of sugar, 1 cup of flour, 1 cup of milk, 1 cup of butter, 1 cup of eggs, 1 cup of chocolate chips. if you want to make 2 cakes, how much sugar do you need? To make 2 cakes, you will need 2 cups of sugar.'}]
```

Bir model ile sohbet etmek için kullanım modeli aynıdır. Tek fark, sizin ve sistem arasındaki bir sohbet geçmişi oluşturmanız gerektiğidir (bu `Pipeline` için girdidir).

> [!İPUCU]
> Bir model ile komut satırından da sohbet edebilirsiniz.
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

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Aşağıdaki örnekleri genişletin ve `Pipeline`'ın farklı modaliteler ve görevler için nasıl çalıştığını görün.

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
<summary>Görsel soru cevaplama</summary>

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

1. Kullanımı kolay en son teknoloji modeller:
    - Doğal dil anlama ve oluşturma, bilgisayar görüsü, ses, video ve multimodal görevlerde yüksek performans.
    - Araştırmacılar, mühendisler ve geliştiriciler için düşük giriş engeli.
    - Kullanıcıya yönelik sadece üç sınıf öğrenmek için birkaç soyutlama.
    - Tüm önceden eğitilmiş modellerimiz için birleştirilmiş bir API.

2. Daha düşük hesaplama maliyetleri, daha küçük karbon ayak izi:
    - Sıfırdan eğitmek yerine eğitilmiş modelleri paylaşın.
    - Hesaplama süresini ve üretim maliyetlerini azaltın.
    - Tüm modalitelerde 1M'dan fazla önceden eğitilmiş kontrol noktası ile düzinece model mimarisi.

3. Bir modelin yaşam döngüsünün her aşaması için doğru çerçeveyi seçin:
    - 3 satır kodla en son teknoloji modelleri eğitin.
    - Bir modeli PyTorch/JAX/TF2.0 çerçeveleri arasında istediğiniz gibi taşıyın.
    - Eğitim, değerlendirme ve üretim için doğru çerçeveyi seçin.

4. Bir modeli veya örneği ihtiyaçlarınıza kolayca özelleştirin:
    - Her mimari için örnekler sağlıyoruz, böylece orijinal yazarları tarafından yayınlanan sonuçları yeniden üretebilirsiniz.
    - Model içleri mümkün olduğunca tutarlı bir şekilde açıklanmıştır.
    - Hızlı deneyler için model dosyaları kütüphaneden bağımsız olarak kullanılabilir.

<a target="_blank" href="https://huggingface.co/enterprise">
    <img alt="Hugging Face Enterprise Hub" src="https://github.com/user-attachments/assets/247fb16d-d251-4583-96c4-d3d76dda4925">
</a><br>

## Neden Transformers kullanmamalıyım?

- Bu kütüphane, sinir ağları için modüler bir araç kutusu değildir. Model dosyalarındaki kod, araştırmacıların ek soyutlamalar/dosyalarla uğraşmadan her bir model üzerinde hızlı bir şekilde çalışabilmesi için kasıtlı olarak ek soyutlamalarla yeniden düzenlenmemiştir.
- Eğitim API'si, Transformers tarafından sağlanan PyTorch modelleriyle çalışacak şekilde optimize edilmiştir. Genel makine öğrenimi döngüleri için başka bir kütüphane kullanmalısınız, örneğin [Accelerate](https://huggingface.co/docs/accelerate).
- [Örnek betikler](https://github.com/huggingface/transformers/tree/main/examples) sadece *örneklerdir*. Belirli kullanım durumunuzda doğrudan çalışmayabilirler ve çalışması için kodu uyarlamanız gerekebilir.

## Transformers kullanan 100 proje

Transformers, önceden eğitilmiş modelleri kullanmak için bir araç setinden daha fazlasıdır, aynı zamanda Hugging Face Hub etrafında inşa edilmiş projelerin bir topluluğudur. Transformers'ı kullanarak geliştiricilerin, araştırmacıların, öğrencilerin, profesörlerin, mühendislerin ve diğer herkesin hayal ettikleri projeleri oluşturmalarını sağlamak istiyoruz.

Transformers'ın 100.000 yıldızını kutlamak için, Transformers ile inşa edilmiş 100 inanılmaz projeyi listeleyen [awesome-transformers](./awesome-transformers.md) sayfası ile topluluğu öne çıkarmak istedik.

Listeye dahil edilmesi gerektiğini düşündüğünüz bir projeniz varsa veya kullanıyorsanız, lütfen onu eklemek için bir PR açın!

## Örnek modeller

Çoğu modelimizi doğrudan [Hub model sayfaları](https://huggingface.co/models) üzerinde test edebilirsiniz.

Her modalite için çeşitli kullanım durumları için birkaç örnek model görmek üzere aşağıdaki modaliteleri genişletin.

<details>
<summary>Ses</summary>

- [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) ile ses sınıflandırma
- [Moonshine](https://huggingface.co/UsefulSensors/moonshine) ile otomatik konuşma tanıma
- [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks) ile anahtar kelime tespiti
- [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) ile konuşmadan konuşmaya oluşturma
- [MusicGen](https://huggingface.co/facebook/musicgen-large) ile metinden sese
- [Bark](https://huggingface.co/suno/bark) ile metinden konuşmaya

</details>

<details>
<summary>Bilgisayar Görüşü</summary>

- [SAM](https://huggingface.co/facebook/sam-vit-base) ile otomatik maske oluşturma
- [DepthPro](https://huggingface.co/apple/DepthPro-hf) ile derinlik tahmini
- [DINO v2](https://huggingface.co/facebook/dinov2-base) ile görüntü sınıflandırma
- [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor) ile anahtar nokta tespiti
- [SuperGlue](https://huggingface.co/magic-leap-community/superglue) ile anahtar nokta eşleştirme
- [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd) ile nesne tespiti
- [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple) ile poz tahmini
- [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large) ile evrensel segmentasyon
- [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large) ile video sınıflandırma

</details>

<details>
<summary>Çoklu Modalite</summary>

- [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B) ile ses veya metinden metne
- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) ile belge soru cevaplama
- [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) ile görüntü veya metinden metne
- [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) ile görüntü açıklama
- [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) ile OCR tabanlı belge anlama
- [TAPAS](https://huggingface.co/google/tapas-base) ile tablo soru cevaplama
- [Emu3](https://huggingface.co/BAAI/Emu3-Gen) ile birleşik çoklu modalite anlama ve oluşturma
- [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) ile görüntüden metne
- [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) ile görsel soru cevaplama
- [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224) ile görsel referans ifadesi segmentasyonu

</details>

<details>
<summary>NLP</summary>

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) ile maske kelime tamamlama
- [Gemma](https://huggingface.co/google/gemma-2-2b) ile isimli varlık tanıma
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) ile soru cevaplama
- [BART](https://huggingface.co/facebook/bart-large-cnn) ile özetleme
- [T5](https://huggingface.co/google-t5/t5-base) ile çeviri
- [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B) ile metin oluşturma
- [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B) ile metin sınıflandırma

</details>

## Atıf

Şimdi 🤗 Transformers kütüphanesi için atıf yapabileceğiniz bir [makale](https://www.aclweb.org/anthology/2020.emnlp-demos.6/)miz var:
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
