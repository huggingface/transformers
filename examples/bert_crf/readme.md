# bert+crf NER
Although BERT models are powerful, to do NER tasks, CRF layer is still essential. 


## tags
Different from the run_ner.py already in the examples, we regard the tokens that are not at the starting position of a word to have the tag "X", which is in the labels set. E.g., fot the CoNLL03 task, the labels are ["X", "O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]. 

## how-to-use
```bash

# tf ckpts --> pytorch 

python transformers/convert_bert_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path resources/uncased_L-12_H-768_A-12/bert_model.ckpt --bert_config_file resources/uncased_L-12_H-768_A-12/bert_config.json --pytorch_dump_path resources/uncased_L-12_H-768_A-12/pytorch_model.bin

# run ner crf
python examples/bert_crf/run_ner_crf.py --data_dir datasets/semeval_task6 --labels datasets/semeval_task6/label.txt --model_type bert --model_name_or_path resources/uncased_L-12_H-768_A-12 --config_name resources/uncased_L-12_H-768_A-12/bert_config.json --do_train --do_eval --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --do_lower_case --overwrite_output_dir --overwrite_cache --logging_steps 100 --save_steps 100 --eval_all_checkpoints --num_train_epochs 10
``` 