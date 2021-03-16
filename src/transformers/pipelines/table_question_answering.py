import collections

import numpy as np

from ..file_utils import add_end_docstrings, is_torch_available, requires_pandas
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline, PipelineException


if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


class TableQuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    Handles arguments for the TableQuestionAnsweringPipeline
    """

    def __call__(self, table=None, query=None, sequential=False, padding=True, truncation=True):
        # Returns tqa_pipeline_inputs of shape:
        # [
        #   {"table": pd.DataFrame, "query": List[str]},
        #   ...,
        #   {"table": pd.DataFrame, "query" : List[str]}
        # ]
        requires_pandas(self)
        import pandas as pd

        if table is None:
            raise ValueError("Keyword argument `table` cannot be None.")
        elif query is None:
            if isinstance(table, dict) and table.get("query") is not None and table.get("table") is not None:
                tqa_pipeline_inputs = [table]
            elif isinstance(table, list) and len(table) > 0:
                if not all(isinstance(d, dict) for d in table):
                    raise ValueError(
                        f"Keyword argument `table` should be a list of dict, but is {(type(d) for d in table)}"
                    )

                if table[0].get("query") is not None and table[0].get("table") is not None:
                    tqa_pipeline_inputs = table
                else:
                    raise ValueError(
                        f"If keyword argument `table` is a list of dictionaries, each dictionary should have a `table` "
                        f"and `query` key, but only dictionary has keys {table[0].keys()} `table` and `query` keys."
                    )
            else:
                raise ValueError(
                    f"Invalid input. Keyword argument `table` should be either of type `dict` or `list`, but "
                    f"is {type(table)})"
                )
        else:
            tqa_pipeline_inputs = [{"table": table, "query": query}]

        for tqa_pipeline_input in tqa_pipeline_inputs:
            if not isinstance(tqa_pipeline_input["table"], pd.DataFrame):
                if tqa_pipeline_input["table"] is None:
                    raise ValueError("Table cannot be None.")

                tqa_pipeline_input["table"] = pd.DataFrame(tqa_pipeline_input["table"])

        return tqa_pipeline_inputs, sequential, padding, truncation


@add_end_docstrings(PIPELINE_INIT_ARGS)
class TableQuestionAnsweringPipeline(Pipeline):
    """
    Table Question Answering pipeline using a :obj:`ModelForTableQuestionAnswering`. This pipeline is only available in
    PyTorch.

    This tabular question answering pipeline can currently be loaded from :func:`~transformers.pipeline` using the
    following task identifier: :obj:`"table-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task.
    See the up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=table-question-answering>`__.
    """

    default_input_names = "table,query"

    def __init__(self, args_parser=TableQuestionAnsweringArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args_parser = args_parser

        if self.framework == "tf":
            raise ValueError("The TableQuestionAnsweringPipeline is only available in PyTorch.")

        self.check_model_type(MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING)

        self.aggregate = bool(getattr(self.model.config, "aggregation_labels")) and bool(
            getattr(self.model.config, "num_aggregation_labels")
        )

    def batch_inference(self, **inputs):
        with torch.no_grad():
            return self.model(**inputs)

    def sequential_inference(self, **inputs):
        """
        Inference used for models that need to process sequences in a sequential fashion, like the SQA models which
        handle conversational query related to a table.
        """
        with torch.no_grad():
            all_logits = []
            all_aggregations = []
            prev_answers = None
            batch_size = inputs["input_ids"].shape[0]

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            token_type_ids = inputs["token_type_ids"].to(self.device)
            token_type_ids_example = None

            for index in range(batch_size):
                # If sequences have already been processed, the token type IDs will be created according to the previous
                # answer.
                if prev_answers is not None:
                    prev_labels_example = token_type_ids_example[:, 3]  # shape (seq_len,)
                    model_labels = np.zeros_like(prev_labels_example.cpu().numpy())  # shape (seq_len,)

                    token_type_ids_example = token_type_ids[index]  # shape (seq_len, 7)
                    for i in range(model_labels.shape[0]):
                        segment_id = token_type_ids_example[:, 0].tolist()[i]
                        col_id = token_type_ids_example[:, 1].tolist()[i] - 1
                        row_id = token_type_ids_example[:, 2].tolist()[i] - 1

                        if row_id >= 0 and col_id >= 0 and segment_id == 1:
                            model_labels[i] = int(prev_answers[(col_id, row_id)])

                    token_type_ids_example[:, 3] = torch.from_numpy(model_labels).type(torch.long).to(self.device)

                input_ids_example = input_ids[index]
                attention_mask_example = attention_mask[index]  # shape (seq_len,)
                token_type_ids_example = token_type_ids[index]  # shape (seq_len, 7)
                outputs = self.model(
                    input_ids=input_ids_example.unsqueeze(0),
                    attention_mask=attention_mask_example.unsqueeze(0),
                    token_type_ids=token_type_ids_example.unsqueeze(0),
                )
                logits = outputs.logits

                if self.aggregate:
                    all_aggregations.append(outputs.logits_aggregation)

                all_logits.append(logits)

                dist_per_token = torch.distributions.Bernoulli(logits=logits)
                probabilities = dist_per_token.probs * attention_mask_example.type(torch.float32).to(
                    dist_per_token.probs.device
                )

                coords_to_probs = collections.defaultdict(list)
                for i, p in enumerate(probabilities.squeeze().tolist()):
                    segment_id = token_type_ids_example[:, 0].tolist()[i]
                    col = token_type_ids_example[:, 1].tolist()[i] - 1
                    row = token_type_ids_example[:, 2].tolist()[i] - 1
                    if col >= 0 and row >= 0 and segment_id == 1:
                        coords_to_probs[(col, row)].append(p)

                prev_answers = {key: np.array(coords_to_probs[key]).mean() > 0.5 for key in coords_to_probs}

            logits_batch = torch.cat(tuple(all_logits), 0)

            return (logits_batch,) if not self.aggregate else (logits_batch, torch.cat(tuple(all_aggregations), 0))

    def __call__(self, *args, **kwargs):
        r"""
        Answers queries according to a table. The pipeline accepts several types of inputs which are detailed below:

        - ``pipeline(table, query)``
        - ``pipeline(table, [query])``
        - ``pipeline(table=table, query=query)``
        - ``pipeline(table=table, query=[query])``
        - ``pipeline({"table": table, "query": query})``
        - ``pipeline({"table": table, "query": [query]})``
        - ``pipeline([{"table": table, "query": query}, {"table": table, "query": query}])``

        The :obj:`table` argument should be a dict or a DataFrame built from that dict, containing the whole table:

        Example::

            data = {
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            }

        This dictionary can be passed in as such, or can be converted to a pandas DataFrame:

        Example::

            import pandas as pd
            table = pd.DataFrame.from_dict(data)


        Args:
            table (:obj:`pd.DataFrame` or :obj:`Dict`):
                Pandas DataFrame or dictionary that will be converted to a DataFrame containing all the table values.
                See above for an example of dictionary.
            query (:obj:`str` or :obj:`List[str]`):
                Query or list of queries that will be sent to the model alongside the table.
            sequential (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the
                inference to be done sequentially to extract relations within sequences, given their conversational
                nature.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).

            truncation (:obj:`bool`, :obj:`str` or :class:`~transformers.TapasTruncationStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls truncation. Accepts the following values:

                * :obj:`True` or :obj:`'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument
                  :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not
                  provided. This will truncate row by row, removing rows from the table.
                * :obj:`False` or :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with
                  sequence lengths greater than the model maximum admissible input size).


        Return:
            A dictionary or a list of dictionaries containing results: Each result is a dictionary with the following
            keys:

            - **answer** (:obj:`str`) -- The answer of the query given the table. If there is an aggregator, the answer
              will be preceded by :obj:`AGGREGATOR >`.
            - **coordinates** (:obj:`List[Tuple[int, int]]`) -- Coordinates of the cells of the answers.
            - **cells** (:obj:`List[str]`) -- List of strings made up of the answer cell values.
            - **aggregator** (:obj:`str`) -- If the model has an aggregator, this returns the aggregator.
        """
        pipeline_inputs, sequential, padding, truncation = self._args_parser(*args, **kwargs)
        batched_answers = []
        for pipeline_input in pipeline_inputs:
            table, query = pipeline_input["table"], pipeline_input["query"]
            if table.empty:
                raise ValueError("table is empty")
            if not query:
                raise ValueError("query is empty")
            inputs = self.tokenizer(
                table, query, return_tensors=self.framework, truncation="drop_rows_to_fit", padding=padding
            )

            outputs = self.sequential_inference(**inputs) if sequential else self.batch_inference(**inputs)

            if self.aggregate:
                logits, logits_agg = outputs[:2]
                predictions = self.tokenizer.convert_logits_to_predictions(inputs, logits.detach(), logits_agg)
                answer_coordinates_batch, agg_predictions = predictions
                aggregators = {i: self.model.config.aggregation_labels[pred] for i, pred in enumerate(agg_predictions)}

                no_agg_label_index = self.model.config.no_aggregation_label_index
                aggregators_prefix = {
                    i: aggregators[i] + " > " for i, pred in enumerate(agg_predictions) if pred != no_agg_label_index
                }
            else:
                logits = outputs[0]
                predictions = self.tokenizer.convert_logits_to_predictions(inputs, logits.detach())
                answer_coordinates_batch = predictions[0]
                aggregators = {}
                aggregators_prefix = {}

            answers = []
            for index, coordinates in enumerate(answer_coordinates_batch):
                cells = [table.iat[coordinate] for coordinate in coordinates]
                aggregator = aggregators.get(index, "")
                aggregator_prefix = aggregators_prefix.get(index, "")
                answer = {
                    "answer": aggregator_prefix + ", ".join(cells),
                    "coordinates": coordinates,
                    "cells": [table.iat[coordinate] for coordinate in coordinates],
                }
                if aggregator:
                    answer["aggregator"] = aggregator

                answers.append(answer)
            if len(answer) == 0:
                raise PipelineException("Empty answer")
            batched_answers.append(answers if len(answers) > 1 else answers[0])
        return batched_answers if len(batched_answers) > 1 else batched_answers[0]
