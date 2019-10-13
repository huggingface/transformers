#!/bin/bash
for ((i=0;i<4;i++));do
{
python ./examples/nq_posranker_sentences_generation.py \
    --example_pk_file models/wwm_train5piece/example_${i}_.pk \
    --feature_pk_file models/wwm_train5piece/feature_${i}_.pk \
    --results_pk_file models/wwm_train5piece/allresults_${i}_.pk \
    --output_nbest_pk_file data/train_5_piece/train5piece_nbest.pk \
    --output_pred_file data/train5piece_pred.json \
    --output_nbest_pred_with_sent_file data/train_5_piece/train5piece_nbest_predwithsent.pk
    echo Done ${i}
} &
done
