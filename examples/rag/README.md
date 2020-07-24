# Retrieval evaluation

We evaluate the retrieval against DPR evaluation data. You can download respective files from links listed [here](https://github.com/facebookresearch/DPR/blob/master/data/download_data.py#L39-L45).

1. Download and unzip the gold data file. We use the `biencoder-nq-dev` from https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz.
2. Parse the unziped file using the `parse_dpr_relevance_data.py`
```
python examples/rag/parse_dpr_relevance_data.py --src_path path/to/unziped/biencoder-nq-dev.json --dst_path path/to/output/biencoder-nq-dev.tsv
```
3. Run evaluation:
```
python examples/rag/eval_dpr.py --predictions_path path/to/dpr_predictions.tsv --gold_data_path path/to/output/biencoder-nq-dev.tsv --model_type sequence --k 1
```

## Results
- on `wiki_dpr` we get Precision@1: 30.514198004604758.
- using the index build for the RAG paper we achieve Precision@1: 60.521872601688415.