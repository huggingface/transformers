## LoTTE dataset

The <b>Lo</b>ng-<b>T</b>ail <b>T</b>opic-stratified <b>E</b>valuation (LoTTE) benchmark includes 12 domain-specific datasets derived from StackExchange questions and answers. Datasets span topics including writing, recreation, science, technology, and lifestyle. LoTTE includes two sets of queries: the first set is comprised of search-based queries from the GooAQ dataset, while the second set is comprised of forum-based queries taken directly from StackExchange.

The dataset can be downloaded from this link: [https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz) 

The dataset is organized as follows:
```
|-- lotte
  |-- writing
     |-- dev
	     |-- collection.tsv
	     |-- metadata.jsonl
	     |-- questions.search.tsv
	     |-- qas.search.jsonl
	     |-- questions.forum.tsv
	     |-- qas.forum.jsonl
	  |-- test
	     |-- collection.tsv
	     |-- ...
  |-- recreation
     |-- ...
  |-- ...
```
Here is a description of each file's contents:
-  `collection.tsv`:  A list of passages where each line is of the form by `[pid]\t[text]`
- `metadata.jsonl`: A list of JSON dictionaries for each question where each line is of the form:
```
    {
    	"dataset": dataset,
    	"question_id": question_id,
    	"post_ids": [post_id_1, post_id_2, ..., post_id_n],
    	"scores": [score_1, score_2, ..., score_n],
    	"post_urls": [url_1, url_2, ..., url_n],
    	"post_authors": [author_1, author_2, ..., author_n],
    	"post_author_urls": [url_1, url_2, ..., url_n],
    	"question_author": question_author,
    	"question_author_url", question_author_url
    }
```
- `questions.search.tsv`:  A list of search-based questions of the form `[qid]\t[text]`
- `qas.search.jsonl`: A list of JSON dictionaries for each search-based question's answer data of the form:

```
	{
		"qid": qid,
		"query": query,
		"answer_pids": answer_pids
	}
``` 
- `questions.forum.tsv`: A list of forum-based questions
- `qas.forum.tsv`: A list of JSON dictionaries for each forum-based question's answer data

We also include a script to evaluate LoTTE rankings: `evaluate_lotte_rankings.py`. Each rankings file must be in a tsv format with each line of the form `[qid]\t[pid]\t[rank]\t[score]`. Note that `qid`s must be in sequential order starting from 0, and `rank`s must be  in sequential order starting from 1. The rankings directory must have the following structure:
```
|--rankings
  |-- dev
    |-- writing.search.ranking.tsv
    |-- writing.forum.ranking.tsv
    |-- recreation.search.ranking.tsv
    |-- recreation.forum.ranking.tsv
    |-- science.search.ranking.tsv
    |-- science.forum.ranking.tsv
    |-- technology.search.ranking.tsv
    |-- technology.forum.ranking.tsv
    |-- lifestyle.search.ranking.tsv
    |-- lifestyle.forum.ranking.tsv
    |-- pooled.search.ranking.tsv
    |-- pooled.forum.ranking.tsv
  |-- test
    |-- writing.search.ranking.tsv
    |-- ...
```
Note that the file names must match exactly, though if some files are missing the script will print partial results. An example usage of the script is as follows:
```
python evaluate_lotte_rankings.py --k 5 --split test --data_path /path/to/lotte --rankings_path /path/to/rankings
```
This will produce the following output (numbers taken from the ColBERTv2 evaluation):
```
[query_type=search, dataset=writing] Success@5: 80.1
[query_type=search, dataset=recreation] Success@5: 72.3
[query_type=search, dataset=science] Success@5: 56.7
[query_type=search, dataset=technology] Success@5: 66.1
[query_type=search, dataset=lifestyle] Success@5: 84.7
[query_type=search, dataset=pooled] Success@5: 71.6

[query_type=forum, dataset=writing] Success@5: 76.3
[query_type=forum, dataset=recreation] Success@5: 70.8
[query_type=forum, dataset=science] Success@5: 46.1
[query_type=forum, dataset=technology] Success@5: 53.6
[query_type=forum, dataset=lifestyle] Success@5: 76.9
[query_type=forum, dataset=pooled] Success@5: 63.4
```

