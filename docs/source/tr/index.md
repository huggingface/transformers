<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Transformers

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers; metin, bilgisayar görüşü, ses, video ve multimodal modeller için hem çıkarsama (inference) hem de eğitim (training) amaçlı kullanılan, en gelişmiş makine öğrenmesi modellerinin tanımlandığı bir çerçeve olarak görev yapar.

Model tanımını merkezileştirerek ekosistem genelinde bu tanımın üzerinde uzlaşı sağlanır. `transformers`, framework'ler arasında bir köprü görevi görür: bir model tanımı destekleniyorsa, çoğu eğitim framework'ü (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), çıkarsama motoru (vLLM, SGLang, TGI, ...) ve yan modelleme kütüphaneleriyle (llama.cpp, mlx, ...) uyumlu olacaktır. Bu kütüphaneler, model tanımını doğrudan `transformers` üzerinden alır.

Yeni ve en gelişmiş modelleri desteklemeyi ve kullanımlarını demokratikleştirmeyi taahhüt ediyoruz. Bunu, model tanımlarını basit, özelleştirilebilir ve verimli hale getirerek yapıyoruz.

Hugging Face Hub üzerinde kullanabileceğin 1 milyondan fazla Transformers [model checkpoint'i](https://huggingface.co/models?library=transformers&sort=trending) bulunmaktadır.

Bir model bulmak ve Transformers ile hemen başlamak için [Hub'ı](https://huggingface.com/) bugünden keşfetmeye başla.

Transformers'daki en yeni metin, görüntü, ses ve multimodal model mimarilerini keşfetmek için [Model Zaman Çizelgesi](./models_timeline) sayfasına göz at.

## Özellikler[[features]]

Transformers, en gelişmiş önceden eğitilmiş modellerle çıkarsama veya eğitim için ihtiyacın olan her şeyi sağlar. Temel özelliklerden bazıları şunlardır:

- [Pipeline](./pipeline_tutorial): Metin üretimi, görüntü segmentasyonu, otomatik konuşma tanıma, belge soru yanıtlama ve daha birçok makine öğrenmesi görevi için basit ve optimize edilmiş çıkarsama sınıfı.
- [Trainer](./trainer): PyTorch modelleri için karışık hassasiyet (mixed precision), torch.compile ve FlashAttention desteği sunan kapsamlı bir eğitim ve dağıtık eğitim aracı.
- [generate](./llm_tutorial): Büyük dil modelleri (LLM) ve görsel dil modelleri (VLM) ile hızlı metin üretimi. Akış (streaming) ve birden fazla kod çözme stratejisi desteği içerir.

## Tasarım[[design]]

> [!TIP]
> Transformers'ın tasarım ilkeleri hakkında daha fazla bilgi edinmek için [Felsefe](./philosophy) sayfasını oku.

Transformers, yazılım geliştiriciler, makine öğrenmesi mühendisleri ve araştırmacılar için tasarlanmıştır. Temel tasarım ilkeleri şunlardır:

1. Hızlı ve kullanımı kolay: Her model yalnızca üç ana sınıftan (yapılandırma, model ve ön işleyici) oluşturulur ve [`Pipeline`] veya [`Trainer`] ile hızlıca çıkarsama veya eğitim için kullanılabilir.
2. Önceden eğitilmiş modeller: Tamamen yeni bir model eğitmek yerine önceden eğitilmiş bir model kullanarak karbon ayak izini, işlem maliyetini ve zamanını azalt. Her önceden eğitilmiş model, orijinal modele mümkün olduğunca yakın şekilde yeniden üretilir ve en gelişmiş performansı sunar.

<div class="flex justify-center">
  <a target="_blank" href="https://huggingface.co/support">
      <img alt="HuggingFace Expert Acceleration Program" src="https://hf.co/datasets/huggingface/documentation-images/resolve/81d7d9201fd4ceb537fc4cebc22c29c37a2ed216/transformers/transformers-index.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
  </a>
</div>

## Öğren[[learn]]

Transformers'a yeni başlıyorsan veya transformer modelleri hakkında daha fazla bilgi edinmek istiyorsan, [LLM kursuna](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt) başlamanı öneriyoruz. Bu kapsamlı kurs, transformer modellerinin nasıl çalıştığından çeşitli görevlerdeki pratik uygulamalara kadar her şeyi kapsar. Yüksek kaliteli veri kümeleri oluşturmaktan büyük dil modellerini ince ayar yapmaya ve akıl yürütme yeteneklerini uygulamaya kadar tüm iş akışını öğreneceksin. Kurs, öğrenirken sağlam bir temel bilgi oluşturman için hem teorik hem de uygulamalı alıştırmalar içerir.
