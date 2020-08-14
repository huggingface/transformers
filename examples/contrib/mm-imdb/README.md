## MM-IMDb

Based on the script [`run_mmimdb.py`](https://github.com/huggingface/transformers/blob/master/examples/contrib/mm-imdb/run_mmimdb.py).

[MM-IMDb](http://lisi1.unal.edu.co/mmimdb/) is a Multimodal dataset with around 26,000 movies including images, plots and other metadata.

### Training on MM-IMDb

```
python run_mmimdb.py \
    --data_dir /path/to/mmimdb/dataset/ \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --output_dir /path/to/save/dir/ \
    --do_train \
    --do_eval \
    --max_seq_len 512 \
    --gradient_accumulation_steps 20 \
    --num_image_embeds 3 \
    --num_train_epochs 100 \
    --patience 5
```

