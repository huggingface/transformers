### Data

CNN/DailyMail data
```bash
wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_dm.tgz
tar -xzvf cnn_dm.tgz
```

this should make a directory called cnn_dm/ with files like `test.source`.
To use your own data, copy that files format. Each article to be summarized is on its own line.

XSUM Data:
```bash
wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/xsum.tar.gz
tar -xzvf xsum.tar.gz
```


### Evaluation

To create summaries for each article in dataset, run:
```bash
python run_eval.py <path_to_test.source> test_generations.txt <model-name>  --score_path rouge_scores.txt
```
The default batch size, 4, fits in 16GB GPU memory, but may need to be adjusted to fit your system.


### Training
Run/modify `finetune.sh`
