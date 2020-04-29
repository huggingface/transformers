***This script evaluates the the multitask pre-trained checkpoint for ``t5-base`` (see paper [here](https://arxiv.org/pdf/1910.10683.pdf)) on the CNN/Daily Mail test dataset. Please note that the results in the paper were attained using a model fine-tuned on summarization, so that results will be worse here by approx. 0.5 ROUGE points***

### Get the CNN Data
First, you need to download the CNN data. It's about ~400 MB and can be downloaded by 
running 

```bash
python download_cnn_daily_mail.py cnn_articles_input_data.txt cnn_articles_reference_summaries.txt
```

You should confirm that each file has 11490 lines:

```bash
wc -l cnn_articles_input_data.txt # should print 11490
wc -l cnn_articles_reference_summaries.txt # should print 11490
```

### Generating Summaries

To create summaries for each article in dataset, run:
```bash
python evaluate_cnn.py cnn_articles_input_data.txt cnn_generated_articles_summaries.txt cnn_articles_reference_summaries.txt rouge_score.txt
```
The default batch size, 8, fits in 16GB GPU memory, but may need to be adjusted to fit your system.
The rouge scores "rouge1, rouge2, rougeL" are automatically created and saved in ``rouge_score.txt``.


### Finetuning
Pass model_type=t5 and model `examples/summarization/bart/finetune.py`
