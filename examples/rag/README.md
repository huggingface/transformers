# Intro
RAG (for Retrieval Augmented Generation) is a seq2seq model which encapsulates two core components: a question encoder and a generator. During a forward pass, we encode the input with the question encoder and pass it
to the retriever to extract relevant context documents. The documents are then prepended to the input. Such contextualized input is passed to the generator. See [the paper](https://arxiv.org/pdf/2005.11401.pdf) for mored details.

We implement two variants of the model, both presented in the paper - `RagSequenceForGeneration. and `RagTokenForGeneration`. In both cases we use `DPRQuestionEncoder` as the question encoder. As for the generator, two compatible architectures have been tested: `BartForConditionalGeneration`  and `T5ForConditionalGeneration`.

Key files:
- `modeling_rag.py`, `tokenization_rag.py`, `configuration_rag.py` the core model implementation
- `retrieval_rag.py` - a distributed retriever built on top of the `torch.distributed` communication package. The retriever is an interface between the model and the faiss index of the encoded documents. During training, all workers initialize their own instance of the retriever, however, only the main worker loads the index into memory, which prevents OOMs on machines with multiple GPUs (we store the index in RAM). The index itself is based on the `nlp.Datasets`. We also implement a variant compatible with indices built using the original DPR implementation (https://github.com/facebookresearch/DPR)
- `eval_rag.py` - an evaluation script which allows to perform the evaluation end to end (measures the exact match and F1 on the downstream task) as well as the evaluation of the retrieval component alone (measures precision@k).
- `finetune.py` - a training script for finetuning RAG models.


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
Apart for parameters specifying the model that's being evaluated and some extra parameters, the evaluation script expects paths to two files:
- `evaluation_set` - a path file specifying the input dataset for evaluation, a single datapoint per line, e.g.
```who is the owner of reading football club```
- `gold_data_path` - a path to a file contaning ground truth answers for samples from the `evaluation_set`.

We expect the following formats of the gold data file:

- for e2e evaluation, we support two formats of gold files:
    - `qa` - where a single line in the following format: input [tab] output_list, e.g.:
    ```
    who is the owner of reading football club	['Xiu Li Dai', 'Dai Yongge', 'Dai Xiuli', 'Yongge Dai']
    ```
    - `ans` - where a single line of the gold file contains the expected output string,
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
    --predictions_filename retrieval_preds.tsv  \ # name of file in which predictions will be stored
    --eval_mode retrieval  \ # indicates whether we're performing retrieval evaluation or e2e evaluation
    --recalculate # if predictions_filename already exists, and this option is set - we regenerate the answers, otherwise we reuse the predicsion file to calculate metrics.
```


## End-to-end evaluation
```
python examples/rag/eval_rag.py \
    --model_name_or_path /private/home/piktus/rag_huggingface/data/repro-rag-sequence-63/ \
    --model_type rag_sequence \
    --evaluation_set path/to/test.source \
    --gold_data_path path/to/gold_data \
    --predictions_filename e2e_preds.txt \
    --eval_mode e2e  \ # indicates whether we're performing retrieval evaluation or e2e evaluation (default)
    --n_docs 5 \ # You can experiment with retrieving different number of documents at evaluation time
    --print_predictions
```