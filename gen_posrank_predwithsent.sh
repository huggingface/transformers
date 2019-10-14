#!/bin/bash
for ((i=0;i<4;i++));do
{
python ./examples/nq_posrank_sentences_generation.py \
--example_pk_file /data/nieping/pytorch-transformers/models/wwm_train5piece/examples_${i}_.pk \
--feature_pk_file /data/nieping/pytorch-transformers/models/wwm_train5piece/features_${i}_.pk \
--results_pk_file /data/nieping/pytorch-transformers/models/wwm_train5piece/allresults_${i}_.pk \
--output_nbest_pk_file data/train_5_piece/train5piece_nbest_${i}.pk \
--output_pred_file data/train_5_piece/train5piece_pred_${i}.json \
--output_nbest_pred_with_sent_file data/train_5_piece/train5piece_nbest_predwithsent_${i}.pk
echo Done ${i}
} &
done
