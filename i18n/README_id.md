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
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_it.md">Italiano</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_bn.md">বাংলা</a> |
        <b>Bahasa Indonesia</b> |
    </p>
</h4>

<h3 align="center">
    <p>Pustaka pemrosesan bahasa alami canggih yang dibuat untuk JAX, PyTorch, dan TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>

🤗 Transformers menyediakan ribuan pretrained model untuk mendukung klasifikasi teks, ekstraksi informasi, menjawab pertanyaan, meringkas, menerjemahkan, dan menghasilkan teks dalam lebih dari 100 bahasa. Tujuannya adalah membuat teknologi NLP paling canggih mudah digunakan untuk semua orang.

🤗 Transformers menyediakan API yang mudah untuk mengunduh dan menggunakan pretrained model tersebut dengan cepat pada teks yang diberikan, melakukan fine-tune pada dataset Anda sendiri, kemudian membagikannya dengan komunitas melalui [model hub](https://huggingface.co/models). Pada saat yang sama, setiap modul Python sepenuhnya independen, membuatnya mudah untuk dimodifikasi dan cocok untuk eksperimen penelitian cepat.

🤗 Transformers mendukung tiga deep learning library paling populer: [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/), dan [TensorFlow](https://www.tensorflow.org/) — dan terintegrasi dengan mulus. Anda dapat melatih model Anda di satu framework, kemudian memuatnya untuk melakukan inference di framework yang lain.

## Demonstrasi online

Anda dapat langsung mencoba sebagian besar model di [model hub](https://huggingface.co/models). Kami juga menyediakan [hosting model private, manajemen versi model, serta API inferensi](https://huggingface.co/pricing).

Berikut adalah beberapa contoh:
- [Menggunakan BERT untuk mengisi kata](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France) (fill-mask)

- [Menggunakan Electra untuk pengenalan entitas bernama](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city) (named entity recognition)

- [Menggunakan GPT-2 untuk pembuatan teks](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C+) (text generation)

- [Menggunakan RoBERTa untuk inferensi bahasa alami](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal) (natural language inference)

- [Menggunakan BART untuk peringkasan teks](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct) (text summarization)

- [Menggunakan DistilBERT untuk tanya jawab](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species) (question answering)

- [Menggunakan T5 untuk penerjemahan](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin) (translation)

**[Write With Transformer](https://transformer.huggingface.co)**, yang dibuat oleh tim Hugging Face, adalah demo resmi untuk pembuatan teks.

## Jika Anda mencari layanan dukungan khusus yang disediakan oleh tim Hugging Face

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://huggingface.co/front/thumbnails/support.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>

## Panduan Memulai Cepat

Kami menyediakan API `pipeline` untuk penggunaan model secara cepat. Pipeline sudah mencakup pretrained model dan pra-pemrosesan teks yang sesuai. Berikut adalah contoh penggunaan pipeline untuk menentukan sentimen positif dan negatif:

```python
>>> from transformers import pipeline

# Gunakan sentiment analysis pipeline
>>> classifier = pipeline('sentiment-analysis')
>>> classifier('We are very happy to introduce pipeline to the transformers repository.')
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

Baris kode kedua mengunduh dan menyimpan cache pretrained model yang digunakan oleh `pipeline`, sementara baris ketiga mengevaluasinya pada teks yang diberikan. Jawaban di sini, "POSITIVE", memiliki tingkat kepercayaan 99.97%.

Banyak tugas NLP memiliki `pipeline` siap pakai. Sebagai contoh, kita bisa dengan mudah mengekstrak jawaban dari pertanyaan berdasarkan teks yang diberikan:

``` python
>>> from transformers import pipeline

# Gunakan question answering pipeline
>>> question_answerer = pipeline('question-answering')
>>> question_answerer({
...     'question': 'What is the name of the repository ?',
...     'context': 'Pipeline has been included in the huggingface/transformers repository'
... })
{'score': 0.30970096588134766, 'start': 34, 'end': 58, 'answer': 'huggingface/transformers'}

```

Selain memberikan jawaban, pretrained model juga memberikan skor kepercayaan serta posisi awal dan akhir jawaban di dalam teks yang sudah di-tokenize. Anda dapat mempelajari lebih lanjut tentang tugas-tugas yang didukung oleh API `pipeline` dari [tutorial ini](https://huggingface.co/docs/transformers/task_summary).

Mengunduh dan menggunakan pretrained model apa pun untuk tugas Anda sangatlah mudah, cukup dengan tiga baris kode. Berikut adalah contoh untuk PyTorch:

```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

Berikut adalah kode yang sesuai untuk TensorFlow:

```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

Tokenizer menyediakan pra-pemrosesan untuk semua pretrained model, dan dapat secara langsung mengonversi string tunggal (seperti contoh di atas) atau sebuah daftar (list). Ia akan menghasilkan sebuah kamus (dict) yang bisa Anda gunakan pada kode selanjutnya atau langsung diteruskan ke model menggunakan operator `**`.

Model itu sendiri adalah [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) atau [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) biasa (tergantung pada backend Anda) dan dapat digunakan seperti biasa. [Tutorial ini](https://huggingface.co/transformers/training.html) menjelaskan cara mengintegrasikan model seperti ini ke dalam loop pelatihan PyTorch atau TensorFlow yang umum, atau cara menggunakan Trainer API kami untuk melakukan fine-tune secara cepat pada dataset baru.

## Mengapa menggunakan transformers?

1. Model canggih yang mudah digunakan:
    - Performa luar biasa pada NLU dan NLG
    - Mudah untuk dipelajari dan diimplementasikan
    - Abstraksi tingkat tinggi, pengguna hanya perlu mempelajari 3 kelas utama
    - API terstandardisasi untuk semua model

1. Biaya komputasi lebih rendah, emisi karbon lebih sedikit:
    - Para peneliti dapat berbagi model yang sudah dilatih daripada melatih ulang dari awal setiap saat.
    - Para insinyur dapat mengurangi waktu komputasi dan biaya produksi.
    - Puluhan arsitektur model, lebih dari dua ribu pretrained model, dan dukungan untuk lebih dari 100 bahasa.

1. Solusi lengkap untuk setiap bagian dari siklus hidup model:
    - Latih model canggih hanya dengan 3 baris kode.
    - Model dapat dialihkan antar-framework deep learning yang berbeda dengan mudah.
    - Pilih framework yang paling sesuai untuk pelatihan, evaluasi, dan produksi, dengan integrasi yang mulus.

1. Kustomisasi model dan contoh dengan mudah untuk kebutuhan Anda:
    - Kami menyediakan beberapa contoh untuk setiap arsitektur model untuk mereproduksi hasil dari paper aslinya.
    - Arsitektur internal model yang konsisten.
    - File model dapat digunakan secara terpisah untuk kemudahan modifikasi dan eksperimen cepat.

## Kapan sebaiknya saya tidak menggunakan transformers?

- Pustaka ini bukanlah toolbox jaringan saraf yang modular. Kode di dalam file model tidak dibungkus dengan abstraksi tambahan, agar para peneliti dapat dengan cepat menelusuri dan memodifikasi kode tanpa terjebak dalam pembungkus kelas (class wrapper) yang rumit.
- `Trainer` API tidak kompatibel dengan semua jenis model; ia hanya dioptimalkan untuk model-model yang ada di pustaka ini. Untuk kebutuhan machine learning umum, silakan gunakan pustaka lain.
- Meskipun kami telah melakukan yang terbaik, script yang ada di direktori [examples hanyalah contoh](https://github.com/huggingface/transformers/tree/main/examples). Untuk masalah yang spesifik, script tersebut belum tentu bisa langsung dipakai dan mungkin perlu beberapa baris kode untuk dimodifikasi agar sesuai dengan kebutuhan Anda.

## Instalasi

### Dengan pip

Repositori ini telah diuji pada Python 3.9+, Flax 0.4.1+, PyTorch 2.1+, dan TensorFlow 2.6+.

Anda sebaiknya menginstal 🤗 Transformers di dalam [virtual environment](https://docs.python.org/3/library/venv.html). Jika Anda belum familiar dengan virtual environment Python, silakan lihat panduan [pengguna ini](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Pertama, buat dan aktifkan virtual environment dengan versi Python yang ingin Anda gunakan.

Kemudian, Anda perlu menginstal salah satu dari Flax, PyTorch, atau TensorFlow. Silakan merujuk ke [halaman instalasi TensorFlow](https://www.tensorflow.org/install/), [halaman instalasi PyTorch](https://pytorch.org/get-started/locally/#start-locally), atau [halaman instalasi Flax](https://github.com/google/flax#quick-install) untuk panduan cara menginstal framework tersebut di platform Anda.

Setelah salah satu backend tersebut berhasil diinstal, 🤗 Transformers dapat diinstal sebagai berikut:

```bash
pip install transformers
```

Jika Anda ingin mencoba contoh (examples) atau ingin menggunakan kode pengembangan terbaru sebelum rilis resmi, Anda harus [menginstalnya dari source](https://huggingface.co/docs/transformers/installation#installing-from-source).

### Menggunakan conda

🤗 Transformers dapat diinstal melalui conda sebagai berikut:

```shell script
conda install conda-forge::transformers
```

> **_Catatan:_** Menginstal `transformers` dari channel `huggingface` telah usang.

Untuk menginstal Flax, PyTorch, atau TensorFlow melalui conda, silakan merujuk ke petunjuk di halaman instalasi masing-masing.

## Arsitektur Model

**[Semua checkpoint model](https://huggingface.co/models)** yang didukung oleh 🤗 Transformers, yang diunggah oleh [pengguna](https://huggingface.co/users) dan [organisasi](https://huggingface.co/organizations), terintegrasi secara sempurna dengan [model hub](https://huggingface.co) huggingface.co.


Jumlah checkpoint saat ini: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers saat ini mendukung arsitektur-arsitektur berikut. Untuk gambaran umum model, silakan lihat [di sini](https://huggingface.co/docs/transformers/model_summary).

Untuk memeriksa apakah suatu model sudah memiliki implementasi untuk Flax, PyTorch, atau TensorFlow, atau apakah model tersebut memiliki tokenizer yang sesuai di pustaka 🤗 Tokenizers, silakan lihat [tabel ini](https://huggingface.co/docs/transformers/index#supported-frameworks).

Implementasi-implementasi ini telah diuji pada beberapa dataset (silakan lihat script contoh) dan seharusnya memiliki performa yang setara dengan implementasi aslinya. Anda dapat mempelajari detail mengenai implementasi tersebut di [bagian ini](https://huggingface.co/docs/transformers/examples) pada dokumentasi contoh.


## Pelajari Lebih Lanjut

| Bagian | Deskripsi |
|-|-|
| [Dokumentasi](https://huggingface.co/docs/transformers/) | Dokumentasi API lengkap dan tutorial |
| [Ringkasan Tugas](https://huggingface.co/docs/transformers/task_summary) | Tugas-tugas yang didukung oleh 🤗 Transformers |
| [Tutorial Pra-pemrosesan](https://huggingface.co/docs/transformers/preprocessing) | Menggunakan `Tokenizer` untuk menyiapkan data bagi model |
| [Pelatihan dan Fine-tuning](https://huggingface.co/docs/transformers/training) | Menggunakan model 🤗 Transformers dengan metode pelatihan bawaan PyTorch/TensorFlow atau dengan `Trainer` API |
| [Panduan Cepat: Fine-tuning dan Script Contoh](https://github.com/huggingface/transformers/tree/main/examples) | Script contoh untuk berbagai macam tugas |
| [Berbagi dan Mengunggah Model](https://huggingface.co/docs/transformers/model_sharing) | Unggah dan bagikan model hasil fine-tune Anda dengan komunitas |
| [Migrasi](https://huggingface.co/docs/transformers/migration) | Cara migrasi dari `pytorch-transformers` atau `pytorch-pretrained-bert` ke 🤗 Transformers |

## Sitasi

Kami telah secara resmi memublikasikan [paper](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) untuk pustaka ini. Jika Anda menggunakan pustaka 🤗 Transformers, Anda dapat mengutip:

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