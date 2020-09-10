

# Roberta Large STS-B

This model is a fine tuned RoBERTA model over STS-B.
It was trained with these params:
!python /content/transformers/examples/text-classification/run_glue.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name STS-B \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir /content/glue_data/STS-B/ \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /content/roberta-sts-b


## How to run

```python



import toolz
import torch
batch_size = 6

def roberta_similarity_batches(to_predict):
  batches = toolz.partition(batch_size, to_predict)
  similarity_scores = []  
  for batch in batches: 
    sentences = [(sentence_similarity["sent1"], sentence_similarity["sent2"])  for sentence_similarity in batch]   
    batch_scores = similarity_roberta(model, tokenizer,sentences)
    similarity_scores = similarity_scores + batch_scores[0].cpu().squeeze(axis=1).tolist()
  return similarity_scores

def similarity_roberta(model, tokenizer, sent_pairs):
  batch_token = tokenizer(sent_pairs, padding='max_length', truncation=True, max_length=500)
  res = model(torch.tensor(batch_token['input_ids']).cuda(), attention_mask=torch.tensor(batch_token["attention_mask"]).cuda())  
  return res

similarity_roberta(model, tokenizer, [('NEW YORK--(BUSINESS WIRE)--Rosen Law Firm, a global investor rights law firm, announces it is investigating potential securities claims on behalf of shareholders of Vale S.A. ( VALE ) resulting from allegations that Vale may have issued materially misleading business information to the investing public',
                                       'EQUITY ALERT: Rosen Law Firm Announces Investigation of Securities Claims Against Vale S.A. – VALE')])
                                       
```                                 
