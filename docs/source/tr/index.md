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

Transformers; metin, bilgisayar görüşü, ses, video ve multimodal modeller için en gelişmiş makine öğrenmesi modellerinin hem çıkarsama hem de eğitim için model tanımlama çerçevesi olarak çalışır.

Model tanımını merkezileştirerek bu tanımın ekosistem genelinde kabul görmesini sağlar. `transformers`, çerçeveler arası bir köprü görevindedir: eğer bir model tanımı destekleniyorsa, eğitim çerçevelerinin büyük çoğunluğuyla (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), çıkarsama motorlarıyla (vLLM, SGLang, TGI, ...) ve `transformers`'daki model tanımını kullanan bitişik modelleme kütüphaneleriyle (llama.cpp, mlx, ...) uyumlu olacaktır.

Yeni en gelişmiş modelleri desteklemeye ve model tanımlarını basit, özelleştirilebilir ve verimli tutarak kullanımlarını demokratikleştirmeye söz veriyoruz.

Hugging Face [Hub](https://huggingface.com/models)'da kullanabileceğin 1 milyondan fazla Transformers [model kontrol noktası](https://huggingface.co/models?library=transformers&sort=trending) bulunuyor.

Bir model bulmak için [Hub](https://huggingface.com/)'ı keşfet ve hemen başlamak için Transformers'ı kullan.

Transformers'daki en son metin, görüntü, ses ve multimodal model mimarilerini keşfetmek için [Model Zaman Çizelgesi](./models_timeline)'ne göz at.

## Özellikler

Transformers, en gelişmiş önceden eğitilmiş modellerle çıkarsama veya eğitim için ihtiyacın olan her şeyi sağlar. Ana özelliklerden bazıları şunlardır:

- [Pipeline](./pipeline_tutorial): Metin üretimi, görüntü segmentasyonu, otomatik konuşma tanıma, belge soru cevaplama ve daha birçok makine öğrenmesi görevi için basit ve optimize edilmiş çıkarsama sınıfı.
- [Trainer](./trainer): PyTorch modelleri için karışık hassasiyet, torch.compile ve FlashAttention gibi özellikleri destekleyen, eğitim ve dağıtık eğitim için kapsamlı bir eğitici.
- [generate](./llm_tutorial): Büyük dil modelleri (LLM'ler) ve görsel dil modelleri (VLM'ler) ile hızlı metin üretimi; akış ve birden fazla çözümleme stratejisi desteği dahil.

## Tasarım

> [!TIP]
> Transformers'ın tasarım ilkeleri hakkında daha fazla bilgi edinmek için [Felsefe](./philosophy) sayfasını oku.

Transformers, geliştiriciler, makine öğrenmesi mühendisleri ve araştırmacılar için tasarlanmıştır. Temel tasarım ilkeleri şunlardır:

1. Hızlı ve kullanımı kolay: Her model yalnızca üç ana sınıftan (yapılandırma, model ve ön işleyici) oluşturulur ve [`Pipeline`] veya [`Trainer`] ile hızlıca çıkarsama veya eğitim için kullanılabilir.
2. Önceden eğitilmiş modeller: Tamamen yeni bir model eğitmek yerine önceden eğitilmiş bir model kullanarak karbon ayak izini, hesaplama maliyetini ve zamanı azalt. Her önceden eğitilmiş model, orijinal modele olabildiğince yakın şekilde yeniden üretilmiştir ve en gelişmiş performansı sunar.

<div class="flex justify-center">
  <a target="_blank" href="https://huggingface.co/support">
      <img alt="HuggingFace Expert Acceleration Program" src="https://hf.co/datasets/huggingface/documentation-images/resolve/81d7d9201fd4ceb537fc4cebc22c29c37a2ed216/transformers/transformers-index.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
  </a>
</div>

## Öğren

Transformers'a yeni başlıyor ya da transformer modelleri hakkında daha fazla bilgi edinmek istiyorsan, [LLM kursu](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt) ile başlamanı öneririz. Bu kapsamlı kurs, transformer modellerinin nasıl çalıştığından çeşitli görevler için pratik uygulamalara kadar her şeyi kapsar. Yüksek kaliteli veri kümelerinin derlenmesinden büyük dil modellerinin ince ayarlanmasına ve akıl yürütme yeteneklerinin uygulanmasına kadar tam iş akışını öğreneceksin. Kurs, öğrenirken sağlam bir temel bilgi oluşturmak için hem teorik hem de uygulamalı alıştırmalar içermektedir.
