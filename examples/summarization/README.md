### Get CNN Data
Both types of models do require CNN data and follow different procedures of obtaining so.

#### For BART models
To be able to reproduce the authors' results on the CNN/Daily Mail dataset you first need to download both CNN and Daily Mail datasets [from Kyunghyun Cho's website](https://cs.nyu.edu/~kcho/DMQA/) (the links next to "Stories") in the same folder. Then uncompress the archives by running:

```bash
wget https://s3.amazonaws.com/datasets.huggingface.co/summarization/cnn_dm.tgz
tar -xzvf cnn_dm.tgz
```

this should make a directory called cnn_dm/ with files like `test.source`.
To use your own data, copy that files format. Each article to be summarized is on its own line.

#### For T5 models
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

### Evaluation

To create summaries for each article in dataset, run:
```bash
python evaluate_cnn.py <path_to_test.source> test_generations.txt <model-name>
```
The default batch size, 8, fits in 16GB GPU memory, but may need to be adjusted to fit your system.

### Training
Run/modify `finetune_bart.sh` or `finetune_t5.sh`

## (WIP) Rouge Scores

To create summaries for each article in dataset and also calculate rouge scores run:
```bash
python evaluate_cnn.py <path_to_test.source> test_generations.txt <model-name> --reference_path <path_to_correct_summaries> --score_path <path_to_save_rouge_scores>
```
The rouge scores "rouge1, rouge2, rougeL" are automatically created and saved in ``<path_to_save_rouge_scores>``.

### Stanford CoreNLP Setup
```
ptb_tokenize () {
    cat $1 | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $2
}

sudo apt install openjdk-8-jre-headless
sudo apt-get install ant
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
cd stanford-corenlp-full-2018-10-05
export CLASSPATH=stanford-corenlp-3.9.2.jar:stanford-corenlp-3.9.2-models.jar
```
Then run `ptb_tokenize` on `test.target` and your generated hypotheses.
### Rouge Setup
Install `files2rouge` following the instructions at [here](https://github.com/pltrdy/files2rouge).
I also needed to run `sudo apt-get install libxml-parser-perl`

```python
from files2rouge import files2rouge
from files2rouge import settings
files2rouge.run(<path_to_tokenized_hypo>,
                <path_to_tokenized_target>,
               saveto='rouge_output.txt')
```
