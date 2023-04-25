FROM code_base

WORKDIR /app/transformers/examples/pytorch/summarization

ARG LANG="c"

RUN mkdir -p /app/transformers/examples/pytorch/summarization/data
#copy the data files
COPY ./data/${LANG}_data.tar /app/transformers/examples/pytorch/summarization/data

WORKDIR /app/transformers/examples/pytorch/summarization/data
#unzip c language data files
RUN tar xfv ${LANG}_data.tar

ARG TEXT_COL=code
ARG SUMMARY_COL=comments
ARG _TRAINING_FILE=${LANG}_training_dataset.json 
ARG _VALIDATION_FILE=${LANG}_validation_dataset.json

ENV e_text_col=${TEXT_COL}
ENV e_summary_col=${SUMMARY_COL}

WORKDIR /app/transformers/examples/pytorch/summarization

# ENTRYPOINT [python, run_summarization.py --model_name_or_path t5-small --do_train y --do_eval y --text_column $TEXT_COL --summary_column ${SUMMARY_COL} --train_file ./data/${_TRAINING_FILE} --validation_file ./data/${_VALIDATION_FILE} --output_dir ./output/tst-summarization --overwrite_output_dir true --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --predict_with_generate --num_train_epochs 10 --logging_strategy epoch --auto_find_batch_size True --source_prefix 'summarize: ']
RUN echo "python run_summarization.py --model_name_or_path t5-small \
                    --do_train y --do_eval y \
                    --text_column ${TEXT_COL} \
                    --summary_column ${SUMMARY_COL} \
                    --train_file ./data/${_TRAINING_FILE} \
                    --validation_file ./data/${_VALIDATION_FILE} \
                    --output_dir ./output/tst-summarization \
                    --overwrite_output_dir true \
                    --per_device_train_batch_size=4 --per_device_eval_batch_size=4 \
                    --predict_with_generate --num_train_epochs 10 \
                    --logging_strategy epoch --auto_find_batch_size True \
                    --source_prefix 'summarize: '" > run_summarization.sh
RUN chmod +x run_summarization.sh
CMD /app/transformers/examples/pytorch/summarization/run_summarization.sh
