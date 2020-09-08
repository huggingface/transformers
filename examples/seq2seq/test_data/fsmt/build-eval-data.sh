#!/bin/bash

# generates bleu eval data
# ./build-eval-data.sh

export OBJS=8
pairs=(ru-en en-ru en-de de-en)

(
printf "{\n"
for pair in "${pairs[@]}"
do
    export PAIR=$pair
    printf "    \"$PAIR\": {\n"
    printf "        \"src\": [\n"
    sacrebleu -t wmt19 -l $PAIR --echo src | head -$OBJS | perl -ne 'chomp; s#"#\\"#g; print qq[    "$_",\n]'
    printf "        ],\n"
    printf "        \"tgt\": [\n"
    sacrebleu -t wmt19 -l $PAIR --echo ref | head -$OBJS | perl -ne 'chomp; s#"#\\"#g; print qq[    "$_",\n]'
    printf "        ],\n"
    printf "    },\n"
done
printf "}\n"

) > fsmt_val_data.yaml
