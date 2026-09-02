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

## Sanal ortam

[uv](https://docs.astral.sh/uv/), Rust tabanlı son derece hızlı bir Python paket ve proje yöneticisidir ve farklı projeleri yönetmek ile bağımlılıklar arasındaki uyumluluk sorunlarını önlemek için varsayılan olarak bir [sanal ortam](https://docs.astral.sh/uv/pip/environments/) gerektirir.

[pip](https://pip.pypa.io/en/stable/) yerine doğrudan kullanılabilir, ancak pip'i tercih ediyorsan aşağıdaki komutlardan `uv` kısmını kaldırman yeterli.

> [!TIP]
> uv'yi kurmak için uv [kurulum](https://docs.astral.sh/uv/guides/install-python/) belgelerine bak.

Transformers'ı kurmak için bir sanal ortam oluştur.

```bash
uv venv .env
source .env/bin/activate
```

## Python

Aşağıdaki komutla Transformers'ı kur.

[uv](https://docs.astral.sh/uv/), Rust tabanlı hızlı bir Python paket ve proje yöneticisidir.

```bash
uv pip install transformers
```

GPU hızlandırması için [PyTorch](https://pytorch.org/get-started/locally) ile uyumlu CUDA sürücülerini kur.

Sisteminin bir NVIDIA GPU algılayıp algılamadığını kontrol etmek için aşağıdaki komutu çalıştır.

```bash
nvidia-smi
```

Transformers'ın yalnızca CPU sürümünü kurmak için aşağıdaki komutu çalıştır.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers
```

Kurulumun başarılı olup olmadığını aşağıdaki komutla test et. Verilen metin için bir etiket ve skor döndürmesi gerekir.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Kaynaktan kurulum

Kaynaktan kurulum, kütüphanenin *kararlı* sürümü yerine *en son* sürümünü kurar. En güncel Transformers değişikliklerine sahip olmanı sağlar ve en son özelliklerle deney yapmak veya henüz kararlı sürümde resmi olarak yayınlanmamış bir hatayı düzeltmek için kullanışlıdır.

Dezavantajı, en son sürümün her zaman kararlı olmayabilmesidir. Herhangi bir sorunla karşılaşırsan, lütfen en kısa sürede düzeltebilmemiz için bir [GitHub Issue](https://github.com/huggingface/transformers/issues) aç.

Aşağıdaki komutla kaynaktan kur.

```bash
uv pip install git+https://github.com/huggingface/transformers
```

Kurulumun başarılı olup olmadığını aşağıdaki komutla kontrol et. Verilen metin için bir etiket ve skor döndürmesi gerekir.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Düzenlenebilir kurulum

[Düzenlenebilir kurulum](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs), Transformers ile yerel olarak geliştirme yapıyorsan kullanışlıdır. Dosyaları kopyalamak yerine yerel Transformers kopyanı Transformers [deposuna](https://github.com/huggingface/transformers) bağlar. Dosyalar Python'ın import yoluna eklenir.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
uv pip install -e .
```

> [!WARNING]
> Kullanmaya devam etmek için yerel Transformers klasörünü saklamalısın.

Ana depodaki en son değişikliklerle yerel Transformers sürümünü güncellemek için aşağıdaki komutu çalıştır.

```bash
cd ~/transformers/
git pull
```

## conda

[conda](https://docs.conda.io/projects/conda/en/stable/#), dilden bağımsız bir paket yöneticisidir. Yeni oluşturduğun sanal ortamda [conda-forge](https://anaconda.org/conda-forge/transformers) kanalından Transformers'ı kur.

```bash
conda install conda-forge::transformers
```

## Yapılandırma

Kurulumdan sonra Transformers önbellek konumunu yapılandırabilir veya kütüphaneyi çevrimdışı kullanım için ayarlayabilirsin.

### Önbellek dizini

Önceden eğitilmiş bir modeli [`~PreTrainedModel.from_pretrained`] ile yüklediğinde, model Hub'dan indirilir ve yerel olarak önbelleğe alınır.

Bir modeli her yüklediğinde, önbelleğe alınmış modelin güncel olup olmadığı kontrol edilir. Aynıysa yerel model yüklenir. Değilse yeni model indirilir ve önbelleğe alınır.

Varsayılan dizin, `HF_HUB_CACHE` kabuk ortam değişkeni tarafından belirlenir ve `~/.cache/huggingface/hub` şeklindedir. Windows'ta varsayılan dizin `C:\Users\kullaniciadi\.cache\huggingface\hub` şeklindedir.

Bir modeli farklı bir dizine önbelleğe almak için aşağıdaki kabuk ortam değişkenlerindeki yolu değiştir (öncelik sırasına göre listelenmiştir).

1. [HF_HUB_CACHE](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache) (varsayılan)
2. [HF_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)
3. [XDG_CACHE_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#xdgcachehome) + `/huggingface` (yalnızca `HF_HOME` ayarlanmamışsa)

### Çevrimdışı mod

Transformers'ı çevrimdışı veya güvenlik duvarı olan bir ortamda kullanmak için indirilen ve önbelleğe alınmış dosyaların önceden hazır olması gerekir. [`~huggingface_hub.snapshot_download`] yöntemiyle Hub'dan bir model deposunu indir.

> [!TIP]
> Hub'dan dosya indirmek için daha fazla seçenek hakkında [Hub'dan dosya indirme](https://hf.co/docs/huggingface_hub/guides/download) rehberine bak. Belirli sürümlerden dosya indirebilir, CLI'dan indirebilir ve hatta bir depodan hangi dosyaların indirileceğini filtreleyebilirsin.

```py
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")
```

Bir model yüklerken Hub'a HTTP çağrılarını engellemek için `HF_HUB_OFFLINE=1` ortam değişkenini ayarla.

```bash
HF_HUB_OFFLINE=1 \
python examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext ...
```

Yalnızca önbelleğe alınmış dosyaları yüklemek için başka bir seçenek de [`~PreTrainedModel.from_pretrained`] içinde `local_files_only=True` ayarlamaktır.

```py
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("./path/to/local/directory", local_files_only=True)
```
