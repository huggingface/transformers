
# For Turkish language, here is an easy-to-use NER application. 
 ** Türkçe için kolay bir python  NER (Bert + Transfer Learning)  (İsim Varlık Tanıma) modeli... 


Thanks to @stefan-it, I applied the followings for training


cd tr-data

for file in train.txt dev.txt test.txt labels.txt
do
  wget https://schweter.eu/storage/turkish-bert-wikiann/$file
done

cd ..
It will download the pre-processed datasets with training, dev and test splits and put them in a tr-data folder.

Run pre-training
After downloading the dataset, pre-training can be started. Just set the following environment variables:
```
export MAX_LENGTH=128
export BERT_MODEL=dbmdz/bert-base-turkish-cased 
export OUTPUT_DIR=tr-new-model
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=625
export SEED=1
```
Then run pre-training:
```
python3 run_ner.py --data_dir ./tr-data3 \
--model_type bert \
--labels ./tr-data/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR-$SEED \
--max_seq_length $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict \
--fp16
```


# Usage

```
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
model = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner=pipeline('ner', model=model, tokenizer=tokenizer)
ner("Mustafa Kemal Atatürk 19 Mayıs 1919'da Samsun'a ayak bastı.")
```
# Some results
Data1:  For the data above
Eval Results:

* precision = 0.916400580551524
* recall = 0.9342309684101502
* f1 = 0.9252298787412536
* loss = 0.11335893666411284

Test Results:
* precision = 0.9192058759362955
* recall = 0.9303010230367262
* f1 = 0.9247201697271198
* loss = 0.11182546521618497



Data2:
https://github.com/stefan-it/turkish-bert/files/4558187/nerdata.txt
The performance for the data given by @kemalaraz is as follows

savas@savas-lenova:~/Desktop/trans/tr-new-model-1$ cat eval_results.txt
* precision = 0.9461980692049029
* recall = 0.959309358847465
* f1 = 0.9527086063783312
* loss = 0.037054269206847804

savas@savas-lenova:~/Desktop/trans/tr-new-model-1$ cat test_results.txt
* precision = 0.9458370635631155
* recall = 0.9588201928530913
* f1 = 0.952284378344882
* loss = 0.035431676572445225

