<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Eğitim betikleri

Transformers, PyTorch ve çeşitli görevler için [transformers/examples](https://github.com/huggingface/transformers/tree/main/examples) dizininde birçok örnek eğitim betiği sunar. [transformers/research projects](https://github.com/huggingface/transformers-research-projects/) ve [transformers/legacy](https://github.com/huggingface/transformers/tree/main/examples/legacy) dizinlerinde ek betikler de bulunur; ancak bunlar aktif olarak güncellenmez ve Transformers'ın belirli bir sürümünü gerektirir.

Örnek betikler yalnızca birer örnektir ve betiği kendi kullanım senaryona uyarlaman gerekebilir. Bu konuda sana yardımcı olmak için çoğu betik, verilerin nasıl önceden işlendiğini son derece açık bir şekilde sunarak gerektiğinde düzenleme yapmana olanak tanır.

Bir örnek betikte uygulamak istediğin herhangi bir özellik için lütfen bir pull request göndermeden önce bunu [forumda](https://discuss.huggingface.co/) veya bir [sorun (issue)](https://github.com/huggingface/transformers/issues) başlığında tartış. Katkıları memnuniyetle karşılasak da, okunabilirlik pahasına daha fazla işlevsellik ekleyen bir pull request'in kabul edilmesi olası değildir.

Bu rehber, [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization)'ta örnek bir metin özetleme eğitim betiğini nasıl çalıştıracağını gösterir.

## Kurulum

Örnek betiğin en güncel sürümünü çalıştırmak için yeni bir sanal ortamda Transformers'ı kaynaktan kur.

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

Transformers'ın belirli veya daha eski bir sürümündeki betiği almak için aşağıdaki komutu çalıştır.

```bash
git checkout tags/v3.5.1
```

Doğru sürümü ayarladıktan sonra dilediğin örnek klasörüne git ve örneğe özel gereksinimleri yükle.

```bash
pip install -r requirements.txt
```

## Bir betik çalıştırma

Veri kümesini maksimum örnek sayısına göre kırpmak için `max_train_samples`, `max_eval_samples` ve `max_predict_samples` parametrelerini dahil ederek daha küçük bir veri kümesiyle başla. Bu, tamamlanması saatler sürebilecek tüm veri kümesine geçmeden önce eğitimin beklendiği gibi çalıştığından emin olmana yardımcı olur.

> [!WARNING]
> Tüm örnek betikler `max_predict_samples` parametresini desteklemez. Bir betiğin bu parametreyi destekleyip desteklemediğini kontrol etmek için aşağıdaki komutu çalıştır.
>
> ```bash
> python examples/pytorch/summarization/run_summarization.py -h
> ```

Aşağıdaki örnek, [T5-small](https://huggingface.co/google-t5/t5-small) modeline [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) veri kümesi üzerinde ince ayar yapar. T5, özetleme yapması için bir istem (prompt) vermek üzere ek bir `source_prefix` parametresi gerektirir.

Örnek betik bir veri kümesini indirip önceden işler, ardından desteklenen bir model mimarisiyle [`Trainer`] kullanarak ona ince ayar yapar.

Eğitim kesintiye uğrarsa bir kontrol noktasından (checkpoint) eğitime devam etmek çok faydalıdır; çünkü her şeye yeniden başlamak zorunda kalmazsın:

* `--resume_from_checkpoint path_to_specific_checkpoint` parametresi, belirli bir kontrol noktası klasöründen eğitime devam edilmesini sağlar.

`--push_to_hub` parametresi ile modelini [Hub](https://huggingface.co/)'da paylaş. Bu işlem bir depo (repository) oluşturur ve modeli `--output_dir` içinde belirtilen klasör adına yükler. Depo adını belirtmek için `--push_to_hub_model_id` parametresini de kullanabilirsin.

Her şey düzgün çalıştığında `max_train_samples`, `max_eval_samples` ve `max_predict_samples` parametrelerini kaldır ve bir kontrol noktasından devam etmek için `--resume_from_checkpoint path_to_specific_checkpoint` parametresini ekle.

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --max_train_samples 50 \
    --max_eval_samples 50 \
    --max_predict_samples 50 \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --push_to_hub \
    --push_to_hub_model_id finetuned-t5-cnn_dailymail \
    --predict_with_generate
```

Karışık hassasiyet (mixed precision) ve dağıtık eğitim için aşağıdaki parametreleri ekle ve eğitimi [torchrun](https://pytorch.org/docs/stable/elastic/run.html) ile başlat.

* Karışık hassasiyetle eğitimi etkinleştirmek için `fp16` veya `bf16` parametrelerini ekle. XPU cihazları yalnızca `bf16` destekler.
* Eğitimde kullanılacak GPU sayısını belirlemek için `nproc_per_node` parametresini ekle.

```bash
torchrun \
    --nproc_per_node 8 pytorch/summarization/run_summarization.py \
    --fp16 \
    ...
    ...
```

PyTorch; performansı artırmak üzere tasarlanmış donanımlar olan TPU'ları, [PyTorch/XLA](https://github.com/pytorch/xla/blob/master/README.md) paketi aracılığıyla destekler. `xla_spawn.py` betiğini çalıştır ve eğitimde kullanılacak TPU çekirdeği sayısını belirlemek için `num_cores` parametresini kullan.

```bash
python xla_spawn.py --num_cores 8 pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    ...
    ...
```

## Accelerate

[Accelerate](https://huggingface.co/docs/accelerate), PyTorch eğitim döngüsüne tam görünürlük sunarken dağıtık eğitimi basitleştirmek için tasarlanmıştır. Accelerate ile bir betik üzerinden eğitim yapmayı planlıyorsan, betiğin `_no_trainer.py` sürümünü kullan.

En güncel sürüme sahip olduğundan emin olmak için Accelerate'i kaynaktan kur.

```bash
pip install git+https://github.com/huggingface/accelerate
```

Eğitim kurulumun hakkındaki birkaç soruyu yanıtlamak için [accelerate config](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) komutunu çalıştır. Bu işlem sistemin hakkında bir yapılandırma dosyası oluşturur ve kaydeder.

```bash
accelerate config
```

Sisteminin düzgün yapılandırıldığından emin olmak için [accelerate test](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-test) komutunu kullanabilirsin.

```bash
accelerate test
```

Eğitimi başlatmak için [accelerate launch](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) komutunu çalıştır.

```bash
accelerate launch run_summarization_no_trainer.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ~/tmp/tst-summarization \
```

## Özel veri kümesi

Özetleme betikleri, CSV veya JSONL dosyası oldukları sürece özel veri kümelerini destekler. Kendi veri kümeni kullanırken aşağıdaki ek parametreleri belirtmen gerekir:

* `train_file` ve `validation_file`, eğitim ve doğrulama dosyalarının yolunu belirtir.
* `text_column`, özetlenecek girdi metnidir.
* `summary_column`, çıktı olarak üretilecek hedef metindir.

Özel bir veri kümesini özetlemek için örnek bir komut aşağıda gösterilmiştir.

```bash
python examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path google-t5/t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --text_column text_column_name \
    --summary_column summary_column_name \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate \
```
