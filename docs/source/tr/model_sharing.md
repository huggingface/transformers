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

# Model Paylaşımı

Hugging Face [Hub](https://hf.co/models); her tür ve boyuttaki modelleri paylaşmak, keşfetmek ve kullanmak için bir platformdur. Açık kaynak makine öğrenimini herkes için ileriye taşımak adına modelini Hub'da paylaşmanı şiddetle tavsiye ederiz!

Bu rehber, Transformers üzerinden bir modelin Hub'da nasıl paylaşılacağını gösterecektir.

## Kurulum

Bir modeli Hub'da paylaşmak için bir Hugging Face [hesabına](https://hf.co/join) ihtiyacın var. Bir [Kullanıcı Erişim Belirteci (User Access Token)](https://hf.co/docs/hub/security-tokens#user-access-tokens) oluştur (varsayılan olarak [önbellekte](./installation#önbellek-dizini) saklanır) ve komut satırından ya da not defterinden (notebook) hesabına giriş yap.

<hfoptions id="share">
<hfoption id="huggingface-CLI">

```bash
hf auth login
```

</hfoption>
<hfoption id="notebook">

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
</hfoptions>

## Depo özellikleri

<Youtube id="XvSGPZFEjDY"/>

Her model deposu; versiyonlama, commit geçmişi ve diff görselleştirme özelliklerine sahiptir.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png"/>
</div>

Versiyonlama, [Git](https://git-scm.com/) ve [Git Large File Storage (LFS)](https://git-lfs.github.com/) üzerine kuruludur ve revizyonları (bir model sürümünü commit hash'i, etiket (tag) veya dal (branch) ile belirtme yöntemi) mümkün kılar.

Örneğin, bir commit hash'inden belirli bir model sürümünü yüklemek için [`~PreTrainedModel.from_pretrained`] içindeki `revision` parametresini kullanabilirsin.

```py
model = AutoModel.from_pretrained(
    "julien-c/EsperBERTo-small", revision="4c77982"
)
```

Model depoları, bir modele kimlerin erişebileceğini denetlemek için [erişim kısıtlamasını (gating)](https://hf.co/docs/hub/models-gated) da destekler. Erişim kısıtlaması (gating), bir araştırma modelinin herkese açık hale getirilmeden önce seçkin bir kullanıcı grubu tarafından önizlenmesine izin vermek için yaygın olarak kullanılır.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/gated-model.png"/>
</div>

Bir model deposu ayrıca kullanıcıların doğrudan Hub üzerinde modelle etkileşime girebilmesi için bir çıkarsama [widget'ı](https://hf.co/docs/hub/models-widgets) içerir.

Daha fazla bilgi için Hub [Modeller](https://hf.co/docs/hub/models) dokümantasyonuna göz atabilirsin.

## Model yükleme

İş akışı tercihine bağlı olarak bir modeli Hub'a yüklemenin birkaç yolu vardır. [`Trainer`] ile bir modeli yükleyebilir, doğrudan model üzerinde [`~PreTrainedModel.push_to_hub`] metodunu çağırabilir veya Hub web arayüzünü kullanabilirsin.

<Youtube id="Z1-XMy-GNLQ"/>

### Trainer

[`Trainer`], eğitimden sonra bir modeli doğrudan Hub'a yükleyebilir. [`TrainingArguments`] içinde `push_to_hub=True` ayarını yap ve bunu [`Trainer`]'a ilet. Eğitim tamamlandığında, modeli yüklemek için [`~transformers.Trainer.push_to_hub`] metodunu çağır.

[`~transformers.Trainer.push_to_hub`], eğitim hiperparametreleri ve sonuçları gibi faydalı bilgileri model kartına otomatik olarak ekler.

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.push_to_hub()
```

### PushToHubMixin

[`~utils.PushToHubMixin`], bir modeli veya tokenizer'ı Hub'a yükleme işlevselliği sağlar.

Bir modeli Hub'a yüklemek için doğrudan model üzerinde [`~utils.PushToHubMixin.push_to_hub`] metodunu çağır. Bu, ad alanın (namespace) altında [`~utils.PushToHubMixin.push_to_hub`] içinde belirtilen model adıyla bir depo oluşturur.

```py
model.push_to_hub("my-awesome-model")
```

Tokenizer gibi diğer nesneler de Hub'a aynı şekilde yüklenir.

```py
tokenizer.push_to_hub("my-awesome-model")
```

Hugging Face profilinde artık yeni oluşturulan model deposu görüntülenmelidir. Yüklenen tüm dosyaları görmek için **Files** sekmesine gidebilirsin.

Dosyaları Hub'a yükleme hakkında daha fazla bilgi için [Dosyaları Hub'a yükleme (Upload files to the Hub)](https://hf.co/docs/hub/how-to-upstream) rehberine başvurabilirsin.

### Hub web arayüzü

Hub web arayüzü, model yüklemek için kod gerektirmeyen (no-code) bir yaklaşımdır.

1. [**New Model**](https://huggingface.co/new) seçeneğini seçerek yeni bir depo oluştur.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png"/>
</div>

Modelin hakkında bazı bilgiler ekle:

- Deponun **sahibini (owner)** seç. Bu kendin veya üyesi olduğun organizasyonlardan biri olabilir.
- Modelin için bir ad seç; bu aynı zamanda deponun adı olacaktır.
- Modelinin herkese açık (public) mı yoksa gizli (private) mi olacağını belirle.
- Lisans kullanımını ayarla.

2. Model deposunu oluşturmak için **Create model** butonuna tıkla.

3. **Files** sekmesini seç ve depoya bir dosyayı sürükleyip bırakmak için **Add file** butonuna tıkla. Bir commit mesajı ekle ve dosyayı commit etmek için **Commit changes to main** butonuna tıkla.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png"/>
</div>

## Model kartı

[Model kartları](https://hf.co/docs/hub/model-cards#model-cards), kullanıcıları bir modelin performansı, sınırlamaları, olası yanlılıkları ve etik değerlendirmeleri hakkında bilgilendirir. Depona bir model kartı eklemen şiddetle tavsiye edilir!

Model kartı, depodaki bir `README.md` dosyasıdır. Bu dosyayı şu şekillerde ekleyebilirsin:

- bir `README.md` dosyasını manuel olarak oluşturup yükleyerek
- depodaki **Edit model card** butonuna tıklayarak

Bir model kartında nelerin bulunması gerektiğine dair bir örnek için Llama 3.1 [model kartına](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) göz atabilirsin.

[Model Kartları](https://hf.co/docs/hub/model-cards#model-cards) rehberinde bulunan diğer model kartı meta verileri (karbon emisyonları, lisans, makale bağlantısı vb.) hakkında daha fazla bilgi edinebilirsin.
