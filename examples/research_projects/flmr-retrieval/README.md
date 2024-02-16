Example Use of FLMR

## Environment

Create virtualenv:
```
conda create -n FLMR_new python=3.10 -y
conda activate FLMR_new
```
Install Pytorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install faiss

```
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```

Test if faiss generate error
```
python -c "import faiss"
```

Install transformers from this folder
```
cd ../../..
pip install -e .
```
Install ColBERT engine
```
cd examples/research_projects/flmr-retrieval/third_party/ColBERT
pip install -e .
```

Install other dependencies
```
pip install ujson gitpython easydict ninja datasets
```

## FLMR
```
cd transformers/examples/research_projects/flmr-retrieval/
```

Download `KBVQA_data` from [here](https://huggingface.co/datasets/BByrneLab/RAVQAV2Data) and unzip the image folders.

Run the following command:

```
python example_use_flmr.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name OKVQA_GS\
            --experiment_name OKVQA_GS \
            --indexing_batch_size 64 \
            --image_root_dir /path/to/KBVQA_data/ok-vqa/ \
            --dataset_path LinWeizheDragon/OKVQA_FLMR_preprocessed_data \
            --passage_dataset_path LinWeizheDragon/OKVQA_FLMR_preprocessed_GoogleSearch_passages \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 \
            --checkpoint_path LinWeizheDragon/FLMR \
            --image_processor_name openai/clip-vit-base-patch32 \
            --query_batch_size 8 \
            --num_ROIs 9 \
```

## Use PreFLMR
```
cd transformers/examples/research_projects/flmr-retrieval/
```

Run the following command:

```
python example_use_preflmr.py \
            --use_gpu \
            --index_root_path "." \
            --index_name EVQA_PreFLMR_ViT-G \
            --experiment_name EVQA \
            --indexing_batch_size 64 \
            --image_root_dir /rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/EVQA/eval_image/ \
            --dataset_path LinWeizheDragon/EVQA_PreFLMR_preprocessed_data \
            --passage_dataset_path LinWeizheDragon/EVQA_PreFLMR_preprocessed_passages \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-G \
            --image_processor_name laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
            --query_batch_size 8 \
            --run_indexing
```