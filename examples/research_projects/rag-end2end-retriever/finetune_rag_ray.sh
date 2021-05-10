# Sample script to finetune RAG using Ray for distributed retrieval.

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# Start a single-node Ray cluster.
ray start --head

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag_ray.sh --help to see all the possible options

python finetune_rag.py \
    --data_dir /home/gsir059/train-dataset-selected \
    --output_dir /hpc/gsir059/git-push-rag/final-push/transformers/examples/research_projects/rag/model_checkpoints \
    --model_name_or_path facebook/rag-token-base \
    --model_type rag_token \
    --fp16 \
    --gpus 2  \
    --profile \
    --do_train \
    --do_predict \
    --n_val -1  \
    --train_batch_size 2 \
    --eval_batch_size 1 \
    --max_source_length 128 \
    --max_target_length 35 \
    --val_max_target_length 35 \
    --test_max_target_length 35 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs 100 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 8 \
    --distributed_retriever ray \
    --num_retrieval_workers 4  \
    --passages_path /home/gsir059/HNSW/my_knowledge_dataset \
    --index_path /home/gsir059/HNSW/my_knowledge_dataset_hnsw_index.faiss \
    --index_name custom \
    --context_encoder_type facebook/dpr-ctx_encoder-multiset-base \
    --csv_path /home/gsir059/COVID-FILES/covid_dump.csv \
    --data_cache_dir /home/gsir059/cache \
    --index_gpus 2 \
    --gpu_order [0,1,2,3,4,5,6,7,8,9] \
    --shard_dir /home/gsir059/kb-shards \
    --indexing_freq 500
    
# Stop the Ray cluster.
ray stop
