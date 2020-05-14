---
language: turkish
---
# Turkish SQuAD  Model : Question Answering

I fine-tuned Turkish-Bert-Model for Question-Answering problem with Turkish version of SQuAD; TQuAD 
* BERT-base: https://huggingface.co/dbmdz/bert-base-turkish-uncased
* TQuAD dataset:  https://github.com/TQuad/turkish-nlp-qa-dataset


# Training Code

```
!python3 run_squad.py \
  --model_type bert \
  --model_name_or_path dbmdz/bert-base-turkish-uncased\
  --do_train \
  --do_eval \
  --train_file trainQ.json \
  --predict_file dev1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir "./model"
```


# Example Usage

> Load Model
```
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForQuestionAnswering.from_pretrained("./model")
nlp=pipeline("question-answering", model=model, tokenizer=tokenizer)
```

> Apply the model
```

sait="ABASIYANIK, Sait Faik. Hikayeci (Adapazarı 23 Kasım 1906-İstanbul 11 Mayıs 1954). \
İlk öğrenimine Adapazarı’nda Rehber-i Terakki Mektebi’nde başladı. İki yıl kadar Adapazarı İdadisi’nde okudu.\
İstanbul Erkek Lisesi’nde devam ettiği orta öğrenimini Bursa Lisesi’nde tamamladı (1928). İstanbul Edebiyat \
Fakültesi’ne iki yıl devam ettikten sonra babasının isteği üzerine iktisat öğrenimi için İsviçre’ye gitti. \
Kısa süre sonra iktisat öğrenimini bırakarak Lozan’dan Grenoble’a geçti. Üç yıl başıboş bir edebiyat öğrenimi \
gördükten sonra babası tarafından geri çağrıldı (1933). Bir müddet Halıcıoğlu Ermeni Yetim Mektebi'nde Türkçe \
gurup dersleri öğretmenliği yaptı. Ticarete atıldıysa da tutunamadı. Bir ay Haber gazetesinde adliye muhabirliği\
yaptı (1942). Babasının ölümü üzerine aileden kalan emlakin geliri ile avare bir hayata başladı. Evlenemedi.\
Yazları Burgaz adasındaki köşklerinde, kışları Şişli’deki apartmanlarında annesi ile beraber geçen bu fazla \
içkili bohem hayatı ömrünün sonuna kadar sürdü."

print(nlp(question="Ne zaman avare bir hayata başladı?", context=sait))
print(nlp(question="Sait Faik hangi Lisede orta öğrenimini tamamladı?", context=sait))

```
```
# Ask your self ! type your question
print(nlp(question="...?", context=sait))
```


Check My other Model
https://huggingface.co/savasy
