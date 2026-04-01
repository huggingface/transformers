<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Kurulum

Transformers, [PyTorch](https://pytorch.org/get-started/locally/) ile çalışır. Python 3.10+ ve PyTorch 2.4+ üzerinde test edilmiştir.

## Sanal ortam[[virtual-environment]]

[uv](https://docs.astral.sh/uv/), son derece hızlı, Rust tabanlı bir Python paket ve proje yöneticisidir. Farklı projeleri yönetmek ve bağımlılıklar arasındaki uyumluluk sorunlarını önlemek için varsayılan olarak bir [sanal ortam](https://docs.astral.sh/uv/pip/environments/) gerektirir.

[pip](https://pip.pypa.io/en/stable/) yerine doğrudan kullanılabilir. pip kullanmayı tercih ediyorsan, aşağıdaki komutlardan `uv` kısmını kaldırman yeterli.

> [!TIP]
> uv'yi yüklemek için uv [kurulum](https://docs.astral.sh/uv/guides/install-python/) belgelerine bak.

Transformers'ı yüklemek için bir sanal ortam oluştur.

```bash
uv venv .env
source .env/bin/activate
```

## Python

Transformers'ı aşağıdaki komut ile yükle.

[uv](https://docs.astral.sh/uv/), hızlı ve Rust tabanlı bir Python paket ve proje yöneticisidir.

```bash
uv pip install transformers
```

GPU hızlandırması için [PyTorch](https://pytorch.org/get-started/locally) uygun CUDA sürücülerini yükle.

Sisteminin bir NVIDIA GPU algılayıp algılamadığını kontrol etmek için aşağıdaki komutu çalıştır.

```bash
nvidia-smi
```

Transformers'ın yalnızca CPU sürümünü yüklemek için aşağıdaki komutu çalıştır.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers
```

Yüklemenin başarılı olup olmadığını aşağıdaki komut ile test et. Verilen metin için bir etiket ve skor döndürmesi gerekir.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Kaynaktan yükleme[[source-install]]

Kaynaktan yükleme, kütüphanenin *kararlı* sürümü yerine *en son* sürümünü yükler. En güncel Transformers değişikliklerine sahip olmanı sağlar ve en son özelliklerle deney yapmak veya henüz kararlı sürümde yayınlanmamış bir hatayı düzeltmek için kullanışlıdır.

Dezavantajı, en son sürümün her zaman kararlı olmayabileceğidir. Herhangi bir sorunla karşılaşırsan, lütfen bir [GitHub Issue](https://github.com/huggingface/transformers/issues) açarak bize bildir, en kısa sürede düzeltelim.

Kaynaktan yüklemek için aşağıdaki komutu çalıştır.

```bash
uv pip install git+https://github.com/huggingface/transformers
```

Yüklemenin başarılı olup olmadığını aşağıdaki komut ile kontrol et. Verilen metin için bir etiket ve skor döndürmesi gerekir.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Düzenlenebilir yükleme[[editable-install]]

[Düzenlenebilir yükleme](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs), Transformers ile yerel geliştirme yapıyorsan kullanışlıdır. Dosyaları kopyalamak yerine Transformers'ın yerel kopyasını Transformers [deposuna](https://github.com/huggingface/transformers) bağlar. Dosyalar Python'un import yoluna eklenir.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
uv pip install -e .
```

> [!WARNING]
> Kullanmaya devam etmek için yerel Transformers klasörünü saklamalısın.

Yerel Transformers sürümünü ana depodaki en son değişikliklerle güncellemek için aşağıdaki komutu çalıştır.

```bash
cd ~/transformers/
git pull
```

## conda

[conda](https://docs.conda.io/projects/conda/en/stable/#), dilden bağımsız bir paket yöneticisidir. Transformers'ı yeni oluşturduğun sanal ortamda [conda-forge](https://anaconda.org/conda-forge/transformers) kanalından yükle.

```bash
conda install conda-forge::transformers
```

## Ayarlar[[set-up]]

Yüklemeden sonra Transformers önbellek konumunu yapılandırabilir veya kütüphaneyi çevrimdışı kullanım için ayarlayabilirsin.

### Önbellek dizini[[cache-directory]]

Önceden eğitilmiş bir modeli [`~PreTrainedModel.from_pretrained`] ile yüklediğinde, model Hub'dan indirilir ve yerel olarak önbelleğe alınır.

Her model yüklendiğinde, önbelleğe alınmış modelin güncel olup olmadığı kontrol edilir. Aynıysa yerel model yüklenir. Değilse daha yeni model indirilir ve önbelleğe alınır.

`HF_HUB_CACHE` kabuk ortam değişkeni tarafından belirlenen varsayılan dizin `~/.cache/huggingface/hub` şeklindedir. Windows'ta varsayılan dizin `C:\Users\kullaniciadi\.cache\huggingface\hub` olarak ayarlanır.

Modeli farklı bir dizinde önbelleğe almak için aşağıdaki kabuk ortam değişkenlerindeki yolu değiştir (öncelik sırasına göre listelenmiştir).

1. [HF_HUB_CACHE](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache) (varsayılan)
2. [HF_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)
3. [XDG_CACHE_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#xdgcachehome) + `/huggingface` (yalnızca `HF_HOME` ayarlanmamışsa)

### Çevrimdışı mod[[offline-mode]]

Transformers'ı çevrimdışı veya güvenlik duvarı arkasındaki bir ortamda kullanmak için indirilen ve önbelleğe alınmış dosyalara önceden ihtiyaç vardır. Hub'dan bir model deposunu [`~huggingface_hub.snapshot_download`] yöntemi ile indir.

> [!TIP]
> Hub'dan dosya indirmek için daha fazla seçenek hakkında bilgi almak için [Hub'dan dosya indirme](https://hf.co/docs/huggingface_hub/guides/download) rehberine bak. Belirli revizyonlardan dosya indirebilir, CLI'dan indirebilir ve hatta bir depodan hangi dosyaları indireceğini filtreleyebilirsin.

```py
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")
```

Bir model yüklerken Hub'a HTTP çağrılarını önlemek için `HF_HUB_OFFLINE=1` ortam değişkenini ayarla.

```bash
HF_HUB_OFFLINE=1 \
python examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext ...
```

Yalnızca önbelleğe alınmış dosyaları yüklemek için bir diğer seçenek, [`~PreTrainedModel.from_pretrained`] fonksiyonunda `local_files_only=True` ayarlamaktır.

```py
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("./path/to/local/directory", local_files_only=True)
```
