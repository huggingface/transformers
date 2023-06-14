export PYTHONPATH="../":"${PYTHONPATH}"

python use_own_knowledge_dataset.py

ray start --head
python finetune_rag.py \
    --model_name_or_path facebook/rag-token-base \
    --model_type rag_token \
    --context_encoder_name facebook/dpr-ctx_encoder-multiset-base \
    --fp16 \
    --gpus 1  \
    --profile \
    --end2end \
    --index_name custom

ray stop
