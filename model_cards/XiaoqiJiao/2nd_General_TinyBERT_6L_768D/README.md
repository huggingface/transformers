TinyBERT
======== 
TinyBERT is 7.5x smaller and 9.4x faster on inference than BERT-base and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages. The overview of TinyBERT learning is illustrated as follows: 
<br />
<br />
<img src="tinybert_overview.png" width="800" height="210"/>
<br />
<br />

For more details about the techniques of TinyBERT, refer to our paper:

[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)


Release Notes
=============
First version: 2019/11/26

Installation
============
Run command below to install the environment(**using python3**)
```bash
pip install -r requirements.txt
```

General Distillation
====================
In general distillation, we use the original BERT-base without fine-tuning as the teacher and a large-scale text corpus as the learning data. By performing the Transformer distillation on the text from general domain, we obtain a general TinyBERT which provides a good initialization for the task-specific distillation. 

General distillation has two steps: (1) generate the corpus of json format; (2) run the transformer distillation;

Step 1: use `pregenerate_training_data.py` to produce the corpus of json format  


```
 
# ${BERT_BASE_DIR}$ includes the BERT-base teacher model.
 
python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \ 
                  --bert_model ${BERT_BASE_DIR}$ \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$ 
                             
```

Step 2: use `general_distill.py` to run the general distillation
```
 # ${STUDENT_CONFIG_DIR}$ includes the config file of student_model.
 
python general_distill.py --pregenerated_data ${CORPUS_JSON}$ \ 
                          --teacher_model ${BERT_BASE}$ \
                          --student_model ${STUDENT_CONFIG_DIR}$ \
                          --reduce_memory --do_lower_case \
                          --train_batch_size 256 \
                          --output_dir ${GENERAL_TINYBERT_DIR}$ 
```


We also provide the models of general TinyBERT here and users can skip the general distillation.

=================1st version to reproduce our results in the paper ===========================

[General_TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj) 

[General_TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1wXWR00EHK-Eb7pbyw0VP234i2JTnjJ-x)

=================2nd version (2019/11/18) trained with more (book+wiki) and no `[MASK]` corpus =======

[General_TinyBERT_v2(4layer-312dim)](https://drive.google.com/open?id=1PhI73thKoLU2iliasJmlQXBav3v33-8z)

[General_TinyBERT_v2(6layer-768dim)](https://drive.google.com/open?id=1r2bmEsQe4jUBrzJknnNaBJQDgiRKmQjF)


Data Augmentation
=================
Data augmentation aims to expand the task-specific training set. Learning more task-related examples, the generalization capabilities of student model can be further improved. We combine a pre-trained language model BERT and GloVe embeddings to do word-level replacement for data augmentation.

Use `data_augmentation.py` to run data augmentation and the augmented dataset `train_aug.tsv` is automatically saved into the corresponding ${GLUE_DIR/TASK_NAME}$
```

python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR}$ \
                            --glove_embs ${GLOVE_EMB}$ \
                            --glue_dir ${GLUE_DIR}$ \  
                            --task_name ${TASK_NAME}$

```
Before running data augmentation of GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory GLUE_DIR. And TASK_NAME can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.

Task-specific Distillation
==========================
In the task-specific distillation, we re-perform the proposed Transformer distillation to further improve TinyBERT by focusing on learning the task-specific knowledge. 

Task-specific distillation includes two steps: (1) intermediate layer distillation; (2) prediction layer distillation.

Step 1: use `task_distill.py` to run the intermediate layer distillation.
```

# ${FT_BERT_BASE_DIR}$ contains the fine-tuned BERT-base model.

python task_distill.py --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${GENERAL_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \ 
                       --output_dir ${TMP_TINYBERT_DIR}$ \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 10 \
                       --aug_train \
                       --do_lower_case  
                         
```


Step 2: use `task_distill.py` to run the prediction layer distillation.
```

python task_distill.py --pred_distill  \
                       --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${TMP_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${TINYBERT_DIR}$ \
                       --aug_train  \  
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --eval_step 100 \
                       --max_seq_length 128 \
                       --train_batch_size 32 
                       
```


We here also provide the distilled TinyBERT(both 4layer-312dim and 6layer-768dim) of all GLUE tasks for evaluation. Every task has its own folder where the corresponding model has been saved.

[TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1_sCARNCgOZZFiWTSgNbE7viW_G5vIXYg) 

[TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1Vf0ZnMhtZFUE0XoD3hTXc6QtHwKr_PwS)


Evaluation
==========================
The `task_distill.py` also provide the evalution by running the following command:

```
${TINYBERT_DIR}$ includes the config file, student model and vocab file.

python task_distill.py --do_eval \
                       --student_model ${TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${OUTPUT_DIR}$ \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128  
                                   
```

To Dos
=========================
* Evaluate TinyBERT on Chinese tasks.
* Tiny*: use NEZHA or ALBERT as the teacher in TinyBERT learning.
* Release better general TinyBERTs.
