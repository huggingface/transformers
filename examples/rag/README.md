# Intro
RAG is a seq2seq model which encapsulates two core components: a question encoder and a generator.
During a forward pass, we encode the input with the question encoder and pass it
to the retriever to extract relevant context documents. The documents are then prepended to the input.
Such contextualized inputs is passed to the generator.

The question encoder can be any `autoencoding` model, preferably :obj:`~transformers.DPRQuestionEncoder`, and the generator can be any `seq2seq` model, preferably :obj:`~transformers.BartForConditionalGeneration`.

The model can be initialized with a :obj:`~transformers.RagRetriever` for end-to-end generation or used in combination with the outputs of a retriever in multiple steps - see examples for more details.
The model is compatible any `autoencoding` model as the ``question_encoder`` and any `seq2seq` model with language model head as the ``generator``.
The model has been tested with :class:`~transformers.DPRQuestionEncoder` as the ``question_encoder`` and :class:`~transformers.BartForConditionalGeneration` or :class:`~transformers.T5ForConditionalGeneration` as the ``generator``.

RAG models were released with the paper `Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks <https://arxiv.org/abs/2005.11401>`_ by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.


# Finetuning
Our finetuning logic is based on scripts from [`examples/seq2seq`](https://github.com/huggingface/transformers/tree/master/examples/seq2seq).
Follow instructions there regarding data preprocessing. A sample finetuning command:

```
python examples/rag/finetune.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type rag_sequence \
    --fp16 \
    --gpus 8
```


# Evaluation
Apart from the parameters specifying the model to evaluate and some extra parameters, the evaluation script expects paths to two files:
- `evaluation_set` - a path to a file specifying the evaluation dataset, a single datapoint per line, e.g.
```who is the owner of reading football club```
- `gold_data_path` - a path to a file contaning ground truth answers for datapoints from the `evaluation_set`.

We expect the following formats of the gold data file:

- for e2e evaluation, we support two formats of the gold file:
    - `qa` - where a single line in the following format: input [tab] output_list, e.g.:
    ```
    who is the owner of reading football club	['Xiu Li Dai', 'Dai Yongge', 'Dai Xiuli', 'Yongge Dai']
    ```
    - `ans` - where a single line of the gold file contains the expected output string, e.g.:
    ```
    Xiu Li Dai
    ```

- for retrieval evaluation, we expect a tab-separated list of Wikipedia page titles constituting positive contexts for a given query, e.g. given a question `who sings does he love me with reba`, a line with ground truth retrieval data could look as follows:
```
Does He Love You	Does He Love You	Red Sandy Spika dress of Reba McEntire	Greatest Hits Volume Two (Reba McEntire album)	Shoot for the Moon (album)
```

## Retrieval evaluation

We demonstrate how to evaluate retrieval against DPR evaluation data. You can download respective files from links listed [here](https://github.com/facebookresearch/DPR/blob/master/data/download_data.py#L39-L45).

1. Download and unzip the gold data file. We use the `biencoder-nq-dev` from https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz.
2. Parse the unziped file using the `parse_dpr_relevance_data.py`
```
python examples/rag/parse_dpr_relevance_data.py --src_path path/to/unziped/biencoder-nq-dev.json --evaluation_set path/to/output/biencoder-nq-dev.questions --gold_data_path path/to/output/biencoder-nq-dev.pages
```
3. Run evaluation:
```
python examples/rag/eval_rag.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \ # model name or path of the model we're evaluating
    --model_type rag_sequence \ # RAG model type (rag_token or rag_sequence)
    --evaluation_set path/to/output/biencoder-nq-dev.questions \ # an input dataset for evaluation
    --gold_data_path path/to/output/biencoder-nq-dev.pages \ # a dataset containing ground truth answers for samples from the evaluation_set
    --predictions_path path/to/retrieval_preds.tsv  \ # name of file in which predictions will be stored
    --eval_mode retrieval \ # indicates whether we're performing retrieval evaluation or e2e evaluation
    --recalculate # if predictions_filename already exists, and this option is set - we regenerate the answers, otherwise we reuse the predicsion file to calculate metrics.
```


## End-to-end evaluation
```
python examples/rag/eval_rag.py \
	--model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type rag_sequence \
    --evaluation_set path/to/test.source \
    --gold_data_path path/to/gold_data \
    --predictions_path path/to/e2e_preds.txt \
    --eval_mode e2e \ # indicates whether we're performing retrieval evaluation or e2e evaluation (default)
    --n_docs 5 \ # You can experiment with retrieving different number of documents at evaluation time
    --print_predictions
```
