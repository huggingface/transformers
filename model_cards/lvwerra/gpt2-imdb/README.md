# GPT2-IMDB

## What is it?
A GPT2 (`gpt2`) language model fine-tuned on the [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Training setting

The GPT2 language model was fine-tuned for 1 epoch on the IMDB dataset. All comments were joined into a single text file separated by the EOS token:

```
import pandas as pd
df = pd.read_csv("imdb-dataset.csv")
imdb_str = " <|endoftext|> ".join(df['review'].tolist())

with open ('imdb.txt', 'w') as f:
    f.write(imdb_str)
```

To train the model the `run_language_modeling.py` script in the `transformer` library was used:

```
python run_language_modeling.py 
	--train_data_file imdb.txt 
	--output_dir gpt2-imdb 
	--model_type gpt2 
	--model_name_or_path gpt2
```
