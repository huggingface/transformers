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

# Hızlı Başlangıç

[[open-in-colab]]

Transformers, herkesin transformer modelleriyle öğrenmeye veya geliştirmeye başlayabilmesi için hızlı ve kullanımı kolay olacak şekilde tasarlanmıştır.

Kullanıcıya yönelik soyutlamalar yalnızca üç model oluşturma sınıfı ve iki çıkarsama veya eğitim API'si ile sınırlıdır. Bu hızlı başlangıç rehberi, Transformers'ın temel özelliklerini tanıtır ve sana şunları nasıl yapacağını gösterir:

- önceden eğitilmiş bir model yükleme
- [`Pipeline`] ile çıkarsama yapma
- [`Trainer`] ile bir modeli ince ayarlama

## Kurulum

Başlamak için bir Hugging Face [hesabı](https://hf.co/join) oluşturmanı öneririz. Hesap, Hugging Face [Hub](https://hf.co/docs/hub/index)'da sürüm kontrollü modelleri, veri kümelerini ve [Spaces](https://hf.co/spaces)'leri barındırmanı ve erişmeni sağlar. Hub, keşif ve geliştirme için işbirlikçi bir platformdur.

Bir [Kullanıcı Erişim Jetonu](https://hf.co/docs/hub/security-tokens#user-access-tokens) oluştur ve hesabına giriş yap.

<hfoptions id="authenticate">
<hfoption id="notebook">

İstendiğinde Kullanıcı Erişim Jetonunu [`~huggingface_hub.notebook_login`] fonksiyonuna yapıştır.

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
<hfoption id="CLI">

[huggingface_hub[cli]](https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started) paketinin kurulu olduğundan emin ol ve aşağıdaki komutu çalıştır. İstendiğinde Kullanıcı Erişim Jetonunu yapıştır.

```bash
hf auth login
```

</hfoption>
</hfoptions>

PyTorch'u kur.

```bash
!pip install torch
```

Ardından Transformers'ın güncel bir sürümünü ve Hugging Face ekosisteminden veri kümelerine ve görüntü modellerine erişmek, eğitimi değerlendirmek ve büyük modeller için eğitimi optimize etmek için bazı ek kütüphaneleri kur.

```bash
!pip install -U transformers datasets evaluate accelerate timm
```

## Önceden eğitilmiş modeller

Her önceden eğitilmiş model üç temel sınıftan türetilir.

| **Sınıf** | **Açıklama** |
|---|---|
| [`PreTrainedConfig`] | Dikkat başlığı sayısı veya kelime hazinesi boyutu gibi model özelliklerini belirten bir dosya. |
| [`PreTrainedModel`] | Yapılandırma dosyasındaki model özellikleri tarafından tanımlanan bir model (veya mimari). Önceden eğitilmiş bir model yalnızca ham gizli durumları döndürür. Belirli bir görev için, ham gizli durumları anlamlı bir sonuca dönüştürmek üzere uygun model başlığını kullan (örneğin, [`LlamaModel`] ile [`LlamaForCausalLM`] karşılaştırması). |
| Ön işleyici | Ham girdileri (metin, görüntü, ses, multimodal) modele sayısal girdilere dönüştüren bir sınıf. Örneğin, [`PreTrainedTokenizer`] metni tensörlere, [`ImageProcessingMixin`] ise pikselleri tensörlere dönüştürür. |

Her görev ve makine öğrenmesi çerçevesi için uygun mimariyi otomatik olarak çıkarsadığından, modelleri ve ön işleyicileri yüklemek için [AutoClass](./model_doc/auto) API'sini kullanmanı öneririz.

Hub'dan ağırlıkları ve yapılandırma dosyasını model ve ön işleyici sınıfına yüklemek için [`~PreTrainedModel.from_pretrained`] kullan.

Bir model yüklerken, modelin en uygun şekilde yüklenmesini sağlamak için aşağıdaki parametreleri yapılandır.

- `device_map="auto"` model ağırlıklarını otomatik olarak en hızlı cihazına atar.
- `dtype="auto"` model ağırlıklarını doğrudan depolandıkları veri tipinde başlatır; bu da ağırlıkların iki kez yüklenmesini önlemeye yardımcı olabilir (PyTorch, ağırlıkları varsayılan olarak `torch.float32` ile yükler).

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

Metni tokenizer ile tokenleştir ve PyTorch tensörleri olarak döndür. Çıkarsamayı hızlandırmak için varsa modeli bir hızlandırıcıya taşı.

```py
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)
```

Model artık çıkarsama veya eğitim için hazır.

Çıkarsama için, tokenleştirilmiş girdileri metin üretmek üzere [`~GenerationMixin.generate`] fonksiyonuna geçir. Token kimliklerini [`~PreTrainedTokenizerBase.batch_decode`] ile tekrar metne çevir.

```py
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
'<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'
```

> [!TIP]
> Bir modeli nasıl ince ayarlayacağını öğrenmek için [Trainer](#trainer-api) bölümüne atla.

## Pipeline

[`Pipeline`] sınıfı, önceden eğitilmiş bir modelle çıkarsama yapmanın en kolay yoludur. Metin üretimi, görüntü segmentasyonu, otomatik konuşma tanıma, belge soru cevaplama ve daha birçok görevi destekler.

> [!TIP]
> Mevcut görevlerin tam listesi için [Pipeline](./main_classes/pipelines) API referansına bak.

Bir [`Pipeline`] nesnesi oluştur ve bir görev seç. Varsayılan olarak, [`Pipeline`] belirli bir görev için varsayılan önceden eğitilmiş modeli indirir ve önbelleğe alır. Belirli bir model seçmek için model adını `model` parametresine geçir.

<hfoptions id="pipeline-tasks">
<hfoption id="metin üretimi">

Çıkarsama için uygun bir hızlandırıcıyı otomatik olarak algılamak üzere [`Accelerator`] kullan.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device=device)
```

Daha fazla metin üretmek için [`Pipeline`]'a bir başlangıç metni ver.

```py
pipeline("The secret to baking a good cake is ", max_length=50)
[{'generated_text': 'The secret to baking a good cake is 100% in the batter. The secret to a great cake is the icing.\nThis is why we\'ve created the best buttercream frosting reci'}]
```

</hfoption>
<hfoption id="görüntü segmentasyonu">

Çıkarsama için uygun bir hızlandırıcıyı otomatik olarak algılamak üzere [`Accelerator`] kullan.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic", device=device)
```

[`Pipeline`]'a bir görüntü (URL veya yerel dosya yolu) geçir.

<div class="flex justify-center">
   <img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"/>
</div>

```py
segments = pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
segments[0]["label"]
'bird'
segments[1]["label"]
'bird'
```

</hfoption>
<hfoption id="otomatik konuşma tanıma">

Çıkarsama için uygun bir hızlandırıcıyı otomatik olarak algılamak üzere [`Accelerator`] kullan.

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)
```

[`Pipeline`]'a bir ses dosyası geçir.

```py
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
```

</hfoption>
</hfoptions>

## Trainer

[`Trainer`], PyTorch modelleri için eksiksiz bir eğitim ve değerlendirme döngüsüdür. Bir eğitim döngüsünü elle yazmakla ilişkili birçok şablon kodu soyutlar, böylece daha hızlı eğitime başlayabilir ve eğitim tasarımı seçimlerine odaklanabilirsin. Bir modelden, veri kümesinden, ön işleyiciden ve veri kümesinden veri grupları oluşturmak için bir veri toplayıcıdan ihtiyacın var.

Eğitim sürecini özelleştirmek için [`TrainingArguments`] sınıfını kullan. Eğitim, değerlendirme ve daha fazlası için birçok seçenek sunar. Eğitim ihtiyaçlarını karşılamak için grup boyutu, öğrenme hızı, karışık hassasiyet, torch.compile ve daha fazlası gibi eğitim hiperparametreleri ve özelliklerini dene. Hızlı bir temel sonuç elde etmek için varsayılan eğitim parametrelerini de kullanabilirsin.

Eğitim için bir model, tokenizer ve veri kümesi yükle.

```py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("rotten_tomatoes")
```

Metni tokenleştirmek ve PyTorch tensörlerine dönüştürmek için bir fonksiyon oluştur. Bu fonksiyonu [`~datasets.Dataset.map`] yöntemiyle tüm veri kümesine uygula.

```py
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)
```

Veri grupları oluşturmak için bir veri toplayıcı yükle ve tokenizer'ı ona geçir.

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

Ardından, eğitim özellikleri ve hiperparametreleri ile [`TrainingArguments`] ayarla.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)
```

Son olarak, tüm bu bileşenleri [`Trainer`]'a geçir ve eğitimi başlatmak için [`~Trainer.train`] fonksiyonunu çağır.

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

Modelini ve tokenizer'ını [`~Trainer.push_to_hub`] ile Hub'a paylaş.

```py
trainer.push_to_hub()
```

Tebrikler, Transformers ile ilk modelini eğittin!

## Sonraki adımlar

Artık Transformers'ı ve sunduklarını daha iyi anladığına göre, en çok ilgini çekeni keşfetmeye ve öğrenmeye devam etmenin zamanı geldi.

- **Temel sınıflar**: Yapılandırma, model ve işlemci sınıfları hakkında daha fazla bilgi edin. Bu, modellerin nasıl oluşturulacağını ve özelleştirileceğini, farklı girdi türlerinin (ses, görüntü, multimodal) nasıl ön işleneceğini ve modelinin nasıl paylaşılacağını anlamana yardımcı olacaktır.
- **Çıkarsama**: [`Pipeline`]'ı daha derinlemesine keşfet, LLM'lerle çıkarsama ve sohbet, ajanlar ve makine öğrenmesi çerçeven ve donanımınla çıkarsamayı nasıl optimize edeceğini öğren.
- **Eğitim**: [`Trainer`]'ı daha ayrıntılı incele, dağıtık eğitim ve belirli donanım üzerinde eğitimi optimize etme hakkında bilgi edin.
- **Kuantizasyon**: Kuantizasyon ile bellek ve depolama gereksinimlerini azalt ve ağırlıkları daha az bit ile temsil ederek çıkarsamayı hızlandır.
- **Kaynaklar**: Belirli bir görev için bir modelin nasıl eğitileceğine ve çıkarsama yapılacağına dair uçtan uca tarifler mi arıyorsun? Görev tariflerine göz at!
